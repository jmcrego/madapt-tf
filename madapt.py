# -*- coding: utf-8 -*-
import time 
tic = time.time()
import sys
import opennmt as onmt
import pyonmttok
from inputer import similars, inputs
from predLrIt import  predLrIt
from opennmt import config as config_util
from opennmt.models import catalog
import numpy as np
import random
import tensorflow as tf
import tensorflow_addons as tfa
import copy
import argparse
import logging
toc = time.time()
#tf.config.run_functions_eagerly(True)

def output_example(idx, hyp, opt, lr=0.0, it=0, sim=0.0, nsim=0, similar_tgt=[]):
    print("{}\t{}\t{:.9f}\t{}\t{:.4f}\t{}\t{}\t{}".format(idx, it, lr, opt, sim, nsim, hyp, '\t'.join(similar_tgt)))
    sys.stdout.flush()
    
def utf_decoding(pred, token):
    for tokens, length in zip(pred["tokens"].numpy(), pred["length"].numpy()):
        sentence = b" ".join(tokens[0][: length[0]])
        sentence = sentence.decode("utf-8")
    logging.debug('sentence: {}'.format(sentence))
    if token is not None:
        return token.detokenize(sentence.split())
    else:
        return sentence

class mAdapt():
    
    def __init__(self, config_file, checkpoint_path, fpred=None):
        self.predLrIt = predLrIt(fpred) if fpred is not None else None
        self.config = config_util.load_config(config_file) # Load and merge run configurations.
        tic = time.time()
        self.beam_width = self.config['params']['beam_width'] if 'params' in self.config and 'beam_width' in self.config['params'] else 1
        logging.debug('beam_width: {}'.format(self.beam_width))
        ### model
        self.mod = config_util.load_model(self.config["model_dir"]) #load model structure
        self.mod.initialize(self.config['data'], params=self.config['params']) #indicates data configuration and other hyperparameters
        ### optimizer
        self.opt = onmt.optimizers.utils.make_optimizer('SGD', learning_rate=0.001) #self.mod.get_optimizer() #i get the optimizer used to train the base model since ckpt.restore needs it... it will never be used
        self.mod.create_variables(self.opt) #creates the parameters of the model/optimizer (otherwise they may be created at first usage)
        ckpt = tf.train.Checkpoint(model=self.mod, optimizer=self.opt) #ckpt only used when loading model/optimizer (no need to save or restore anymore)
        ckpt.restore(checkpoint_path) #copies the variables of the model/optimizer from the checkpoint to self.mod
        toc = time.time()
        logging.debug('Load/initialize base model/optimizer took {:.2f} sec'.format(toc-tic))        
        tic = time.time()
        self.is_base = True
        self.mod_base = copy.deepcopy(self.mod) #copy of the base model
        toc = time.time()
        logging.debug('Copy base model took {:.2f} sec'.format(toc-tic))

        self.tgt_tokenizer = pyonmttok.Tokenizer(**self.config['data']['target_tokenization']) if 'target_tokenization' in self.config['data'] else None
        self.t_train = 0.
        self.t_infer = 0.
        self.t_mod_restore = 0.
        self.t_opt_restore = 0.
        self.n_train = 0
        self.n_infer = 0
        self.n_opt_restore = 0
        self.n_mod_restore = 0
        ### i use fake datasets to build only once the graph functions (tf.function)
        tic =  time.time()
        dataset_inference = self.build_dataset(['fake1', 'fake2'], ['fake1', 'fake2'], False)
        self.predict_fn = tf.function(self.predict, input_signature=(dataset_inference.element_spec,))
        dataset_training = self.build_dataset(['fake1', 'fake2'], ['fake1', 'fake2'], True)
        self.training_step_fn = tf.function(self.training_step, input_signature=(dataset_training.element_spec,))
        ### i force to build the graph of the entire model
        self.predict_fn.get_concrete_function()
        self.training_step_fn.get_concrete_function()        
        toc =  time.time()
        logging.debug('Building tf graph took {:.3f} sec'.format(toc-tic))

        
    def __call__(self, idx, src, sim, lr, it, optim):
        if self.predLrIt is not None:
            self.mode_inference(idx, src, sim, optim) # mode inference: predicts lr and it, adapts accordingly and translates 
        else:
            self.mode_examples(idx, src, sim, lr, it, optim) # mode generation of examples

            
    def mode_inference(self, idx, src, sim, optim):
        logging.debug('mode inference idx={}'.format(idx))
        similar_src, similar_tgt, similar_scr = sim
        lr, it = self.predLrIt(src, similar_src, similar_tgt, similar_scr)
        if len(similar_src) and lr > 0. and it > 0:  ### must microadapt
            dataset_training = self.build_dataset(similar_src, similar_tgt, True)
            self.restore_base(optim, lr)
            for curr_it in range(it):
                tic = time.time()
                loss = self.training_step_fn(next(iter(dataset_training)))
                self.is_base = False
                toc = time.time()
                logging.debug('training step took {:.3f} sec with loss={}'.format(toc-tic,loss))
                self.t_train += toc - tic
                self.n_train += 1
        else:
            self.restore_base(None, None)
        ### translate
        tic = time.time()
        dataset_inference = self.build_dataset([src], [src], False)
        target_tokens, target_lengths = self.predict_fn(next(iter(dataset_inference)))
        toc = time.time()
        logging.debug('inference took {:.3f} sec'.format(toc-tic))
        self.t_infer += toc - tic
        self.n_infer += 1
        hyp = utf_decoding({'tokens':target_tokens, 'length':target_lengths}, self.tgt_tokenizer)
        logging.debug('hyp: {}'.format(hyp))
        output_example(idx, hyp, '-', lr=0.0, it=0, sim=0., nsim=0, similar_tgt=[])
                
        
    def mode_examples(self, idx, src, sim, lr, it, optim):
        logging.debug('mode examples idx={}'.format(idx))
        similar_src, similar_tgt, similar_scr = sim
        if len(similar_src) == 0:
            return
        dataset_inference = self.build_dataset([src], [src], False)
        ### inference with base model ###    
        self.restore_base(None, None)
        tic = time.time()
        target_tokens, target_lengths = self.predict_fn(next(iter(dataset_inference)))
        toc = time.time()
        logging.debug('inference took {:.3f} sec'.format(toc-tic))
        self.t_infer += toc - tic
        self.n_infer += 1
        hyp = utf_decoding({'tokens':target_tokens, 'length':target_lengths}, self.tgt_tokenizer)
        output_example(idx, hyp, '-', lr=0.0, it=0, sim=0., nsim=0, similar_tgt=[])
        ### micro-adaptation ###
        if len(lr) and len(it):
            dataset_training = self.build_dataset(similar_src, similar_tgt, True)
            for curr_lr in lr:
                self.restore_base(optim, curr_lr)
                for curr_it in range(it[-1]):
                    tic = time.time()
                    loss = self.training_step_fn(next(iter(dataset_training)))
                    self.is_base = False
                    toc = time.time()
                    logging.debug('training step took {:.3f} sec with loss={}'.format(toc-tic,loss))
                    self.t_train += toc - tic
                    self.n_train += 1
                    if curr_it + 1 in it:
                        tic = time.time()
                        target_tokens, target_lengths = self.predict_fn(next(iter(dataset_inference)))
                        toc = time.time()
                        logging.debug('inference took {:.3f} sec'.format(toc-tic))
                        self.t_infer += toc - tic
                        self.n_infer += 1
                        hyp = utf_decoding({'tokens':target_tokens, 'length':target_lengths}, self.tgt_tokenizer)
                        output_example(idx, hyp, optim, lr=curr_lr, it=curr_it+1, sim=similar_scr[0], nsim=len(similar_src), similar_tgt=similar_tgt)

                        
    def build_dataset(self, source, target, training):
        logging.debug('build_dataset source={} target={} training={}'.format(source,target,training))
        dataset_ = tf.data.Dataset.zip( (tf.data.Dataset.from_tensor_slices(source), tf.data.Dataset.from_tensor_slices(target)) )
        dataset_tensors = dataset_.apply(onmt.data.training_pipeline(batch_size=10, batch_type="examples",process_fn=lambda source, target: self.mod.examples_inputter.make_features((source, target), training=training)))
        return dataset_tensors

    def training_step(self, batch):
        source = batch[0]
        target = batch[1]
        loss = self.mod.train(source, target, self.opt)
        return loss

    def predict(self, batch):
        source = batch[0]
        # Run the encoder.
        source_length = source["length"]
        batch_size = tf.shape(source_length)[0]
        source_inputs = self.mod.features_inputter(source)
        encoder_outputs, _, _ = self.mod.encoder(source_inputs, source_length)
        # Prepare the decoding strategy.
        if self.beam_width > 1:
            encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, self.beam_width)
            source_length = tfa.seq2seq.tile_batch(source_length, self.beam_width)
            decoding_strategy = onmt.utils.BeamSearch(self.beam_width)
        else:
            decoding_strategy = onmt.utils.GreedySearch()
        decoder_state = self.mod.decoder.initial_state(memory=encoder_outputs, memory_sequence_length=source_length) # Run dynamic decoding.
        decoded = self.mod.decoder.dynamic_decode(self.mod.labels_inputter, tf.fill([batch_size], onmt.START_OF_SENTENCE_ID), end_id=onmt.END_OF_SENTENCE_ID, initial_state=decoder_state, decoding_strategy=decoding_strategy, maximum_iterations=200)
        target_lengths = decoded.lengths
        target_tokens = self.mod.labels_inputter.ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
        return target_tokens, target_lengths

    def restore_base(self, opt_name, lr):
        #if self.mod != self.mod_base: #comparing the value of two objects
        if not self.is_base: #restore the base model
            tic = time.time()
            self.mod = copy.deepcopy(self.mod_base)
            self.is_base = True
            toc = time.time()
            logging.debug('restore model base took {:.3f} sec'.format(toc-tic))
            self.t_mod_restore += toc - tic
            self.n_mod_restore += 1
        if opt_name is not None and lr is not None:
            tic = time.time()
            self.opt = onmt.optimizers.utils.make_optimizer(opt_name, learning_rate=lr)
            toc = time.time()
            logging.debug('restore optimizer {} {} took {:.3f} sec'.format(opt_name,lr,toc-tic))
            self.t_opt_restore += toc - tic
            self.n_opt_restore += 1

    def report(self):
        logging.info('n infers: {}'.format(self.n_infer))
        if self.n_infer:
            logging.info('time infer: {:.3f} sec, {:.3f} sec/sent'.format(self.t_infer, self.t_infer/self.n_infer))
        logging.info('n trains: {}'.format(self.n_train))
        if self.n_train:
            logging.info('time train: {:.3f} sec, {:.3f} sec/step'.format(self.t_train,self.t_train/self.n_train))
        if self.n_mod_restore:
            logging.info('n mod restores: {}'.format(self.n_mod_restore))
            logging.info('time mod restore: {:.3f} sec, {:.3f} sec/restore'.format(self.t_mod_restore, self.t_mod_restore/self.n_mod_restore))
        if self.n_opt_restore:
            logging.info('n opt restores: {}'.format(self.n_opt_restore))
            logging.info('time opt restore: {:.3f} sec, {:.3f} sec/restore'.format(self.t_opt_restore, self.t_opt_restore/self.n_opt_restore))


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cfg", required=True, nargs="+", help="list of configuration files, containing: model_dir, data, params, train, eval, infer, ...")
    parser.add_argument("--ckpt", required=True, help="path to the checkpoint or checkpoint directory to load. If not set, the latest checkpoint from the model directory is loaded")
    parser.add_argument("--src", required=True, help="path to the file containing source test sentences")
    parser.add_argument("--sim", required=False, help="path to the file containing similar source/target/score sentences")
    parser.add_argument("--pred", default=None, required=False, help="path to the LR IT prediction model")
    parser.add_argument("--it", type=int, nargs="+", default=[], help="run inference after learning during these many iterations")
    parser.add_argument("--lr", type=float, nargs="+", default=[], help="run inference after learning using these many lr values")
    parser.add_argument("--optim", default="SGD", help="optimizer name")
    parser.add_argument("--seed", type=int, default="12345", help="seed for randomness")
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    if (len(args.it) and not len(args.lr)) or (len(args.lr) and not len(args.it)):
        logging.error('missing --it OR --lr option')
        sys.exit()
    
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO' if not args.debug else 'DEBUG'), filename=None)
    logging.debug('Load libraries took {:.2f} sec'.format(toc-tic))
    args.it.sort() # it sorted in ascending order
    t_ini = time.time()

    src = inputs(args.src)
    sim = similars(args.sim)
    ma = mAdapt(args.cfg, args.ckpt, args.pred)
    for idx in range(len(src)):
        ma(idx+1, src(idx), sim(idx), args.lr, args.it, args.optim)
    ma.report()
    t_fin = time.time()
    logging.info('time total: {:.3f} sec'.format(t_fin-t_ini))


