# -*- coding: utf-8 -*-

class predLrIt():

    def __init__(self, fpred=None):
        self.fpred = fpred

    def __call__(self, src, similar_src, similar_tgt, similar_scr):
        nsim = len(similar_src)
        scr = similar_scr[0] if len(similar_scr) else 0.
        lr = 0.
        it = 0
        return lr, it
