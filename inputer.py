# -*- coding: utf-8 -*-

class inputs():
    def __init__(self, fsrc):
        self.fsrc = fsrc
        self.SRC = []
        if fsrc is not None:
            with open(fsrc, 'r') as f:
                self.SRC = f.readlines()

    def __call__(self, idx):
        l = self.SRC[idx].rstrip()
        return l

    def __len__(self):
        return len(self.SRC)

class similars():

    def __init__(self, fsim, max_n=0, min_s=0.):
        self.fsim = fsim
        self.max_n = max_n
        self.min_s = min_s
        self.SIM = []
        if fsim is not None:
            with open(fsim, 'r') as f:
                self.SIM = f.readlines()

    def __call__(self, idx):
        src = []
        tgt = []
        scr = []
        if self.fsim is not None and idx < len(self.SIM):
            l = self.SIM[idx].rstrip()
            if len(l): #line contains similars
                l = l.split('\t')
                if len(l) % 3 != 0:
                    logging.error('bad formatted similars line {}: {}'.format(idx, l))
                    sys.exit()
                for k in range(0,len(l),3):
                    s = float(l[k])
                    if s >= self.min_s:
                        break
                    scr.append(s)
                    src.append(l[k+1])
                    tgt.append(l[k+2])
                    if self.max_n > 0 and len(src) >= self.max_n:
                        break 
        return src, tgt, scr
