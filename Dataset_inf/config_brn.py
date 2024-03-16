import random
import sparse
import pickle
import itertools
import numpy as np

from math import ceil
from pathlib import Path

from Dataset_inf.prep_inf import _roi_to_coord


def is_ckbd(frac, h, h_st, w):
    r"""
    Check if a image tile should be included in the 
        checkerboard data setting 
    
    Args:
        frac: The fraction of checkerboard data setting
        h: Row id
        h_st: Starting Row id
        w: Column id

    """
    
    is_contain = False
    if frac in (1, 12, 14) and (h - h_st) % 3 == w % 3 == 1:
        is_contain = True
    elif frac == 3 and (h - h_st) % 3 == w % 3:
        is_contain = True
    elif frac == 5 and ((h - h_st) % 3 + w % 3) % 2 == 0:
        is_contain = True
    elif frac == 8 and not (h - h_st) % 3 == w % 3 == 1:
        is_contain = True
    return is_contain


class cfg_mouse(object):
    def __init__(self, root, subs, orgs,
                 frac=1, seed=None):
        r"""
        Configure the critical data class for mouse dataset,
            e.g., the proportion of training data, 
            coords, gene names, bbox color w.r.t. roi

        Args:
            root: Root to the mouse dataset 
            subs: Name of the subset of the data
            orgs: Name of organs
            frac: Proportion of the training data, 
                  used only for randomized training
            seed: Random seed for fixing the selected data used in training  

        """

        assert subs in ('all', 'random', 'checkerboard')
        assert orgs == 'organs'
        self.offset, self.roi_num = 0, 2
        self.raw_sz, self.vis_sz = 128, 256

        # 36 * 2048, 52 * 2048
        self.hei, self.wid = 73728, 106496
        self.roi, self.ovlp = 2048, self.raw_sz * 2
        self.h_num = ceil(self.hei / self.roi)
        self.w_num = ceil(self.wid / self.roi)

        # Valid index range
        self.h_st, self.h_ed = 0, self.h_num
        self.w_st, self.w_ed = 0, self.w_num
        self.h = [self.h_st, self.h_ed]
        self.w = [self.w_st, self.w_ed]

        # small bounding box for sparse training data
        # defined by checkerboard
        self.sm_roi = None
        # Neutral color bbox for input gene or image
        self.neu_clr = (5, 60, 128)
        # Training data color
        self.trn_clr = (255, 0, 0)
        self.pths, self.pths_arr = [], {}
        for key in ['trn', 'val', 'vis', 'wsi', 'trn_clr']:
            self.pths_arr[key] = [[None for _ in range(self.w_ed)]
                                  for _ in range(self.h_ed)]
        assert subs == 'all'
        self.pths = list(root.glob('*.npz'))

        self._prep_org(orgs)
        self._prep_arr(root, subs, frac)
        print(self.org_dct)

    def _prep_valid_cid(self, organ, crd):
        r"""
        Prepare the cell-level regions to be highlighted, 
            to avoid capaturing empty regions, we combine 
            manual and random selection
        

        Args:
            organ: Name of the organ
            crd: List of the row and column idx for organ regions

        """

        n = self.roi * self.roi_num // self.vis_sz
        out, clim = [], n - 1
        if organ == 'lung' and crd == (26, 7):
            fix = [(8, 5), (9, 7)]
        elif organ == 'hair' and crd == (47, 23):
            # This roi has black region when col >= 12
            fix, clim = [(6, 7), (6, 4)], 12
        elif organ == 'hair' and crd == (48, 20):
            fix = [(6, 12), (11, 11)]
        else:
            fix = []
        cmb = itertools.product(range(n), range(n))
        # Ignore boundary tile due to bbox overlap
        for _, (r, c) in enumerate(cmb):
            if 1 < r < n - 1:
                if 1 < c < clim:
                    if (r, c) not in fix:
                        out.append((r, c))
        return fix, out

    def _prep_org(self, orgs):
        r"""
        Prepare the organ-wise dict including: row and col id,
            highly expressed gene list, image ensemble pos (left or right)
            and bbox color
        
        Args:
            orgs: List of organ names

        """

        _organ = {
            # organ [((top0, left0), (top1, left1))
            'Isocortex_l': [((8, 9), (13, 5)), ('Rasgrp1', 'Rph3a', 'Baiap2', 'Slc17a7'), 'left', (0, 0, 255)],
            'Isocortex_r':  [((8, 40), (13, 44)), ('Rasgrp1', 'Rph3a', 'Baiap2', 'Slc17a7'), 'right', (0, 0, 255)],
            'Hippo_l':   [((10, 19), (11, 15)), ('Slc17a7', 'Atp1b2', 'Wipf3', 'Grin2a'), 'left', (0, 255, 0)],
            'Hippo_r':  [((10, 32), (11, 36)), ('Slc17a7', 'Atp1b2', 'Wipf3', 'Grin2a'), 'right', (0, 255, 0)],
            'Hypo_l':  [((26, 22), (29, 23)), ('Atp1b2', 'Nnat', 'Slc6a1', 'Zcchc12'), 'left', (51, 153, 255)],
            'Hypo_r':   [((26, 29), (29, 28)), ('Atp1b2', 'Nnat', 'Slc6a1', 'Zcchc12'), 'right', (51, 153, 255)],
        }
        if orgs in _organ:
            _organ = {orgs: _organ[orgs]}

        self.org_dct = {'all': {'h': self.h, 'w': self.w,
                                'cid': None, 'clr': None, 'vpos': None}}
        for key, (crds, glst, vpos, clr) in _organ.items():
            for cid, crd in enumerate(crds):
                ckey = f'{key}_{cid}'
                self.org_dct[ckey] = dict()
                self.org_dct[ckey]['h'] = (crd[0], crd[0] + self.roi_num)
                self.org_dct[ckey]['w'] = (crd[1], crd[1] + self.roi_num)
                # Get randomly and selectively sampled crop ID in roi
                fix, cnt = self._prep_valid_cid(key, crd)
                self.org_dct[ckey]['cid'] = fix + [cnt[i] for i in
                                                   sorted(random.sample(range(len(cnt)), 8 - len(fix)))]
                self.org_dct[ckey]['clr'] = clr
                self.org_dct[ckey]['vpos'] = vpos
                self.org_dct[ckey]['glst'] = glst

                cmb = itertools.product(range(2), range(2))
                for hshf, wshf in cmb:
                    # select boudry to add bbox for the
                    # follow-up roi bbox merge
                    bpos = (hshf, 2 + wshf)
                    h, w = crd[0] + hshf, crd[1] + wshf
                    vdct = {'clr': clr, 'blen': 128, 'bpos': bpos,
                            'vpos': vpos, 'gene': glst}
                    self.pths_arr['vis'][h][w] = vdct

    def _prep_arr(self, root, subs, frac):
        r"""
        Prepare the image tile array for calc the stats,
            None if the image tile is not in the data else
            path to the gene expression array
        
        Args:
            root: Root to the mouse dataset 
            subs: Name of the subset of the data
            frac: Proportion of the training data, 
                  used only for randomized training

        """

        for h in range(self.h_st, self.h_ed):
            for w in range(self.w_st, self.w_ed):
                crd, crdo = _roi_to_coord(h, w,
                                          self.hei, self.wid,
                                          self.roi, self.ovlp)
                roi_nm = '_'.join(map(str, crd + crdo))
                pth = root / f'{roi_nm}.npz'
                is_valid = pth.is_file() and crd[1] - crd[0] == self.roi and \
                    crd[3] - crd[2] == self.roi
                if not is_valid:
                    continue

                self.pths_arr['wsi'][h][w] = pth
                if subs in ('all', 'random'):
                    if pth in self.pths:
                        self.pths_arr['trn'][h][w] = pth
                    else:
                        self.pths_arr['val'][h][w] = pth
                elif subs == 'checkerboard':
                    is_contain = is_ckbd(frac, h, self.h_st, w)
                    if is_contain:
                        self.pths.append(pth)
                        self.pths_arr['trn'][h][w] = pth
                    else:
                        self.pths_arr['val'][h][w] = pth

                # Add training data bbox except for all
                if self.pths_arr['trn'][h][w] is not None and subs != 'all':
                    self.pths_arr['trn_clr'][h][w] = \
                        (self.trn_clr, 128, list(range(4)))

        _tlst = []
        for key in ('trn', 'val', 'wsi'):
            _arr = np.array(self.pths_arr[key])[:]
            _lst = _arr[_arr != None].tolist()
            assert (len(_lst) == len(set(_lst)))
            _tlst.append(len(_lst))
        assert _tlst[0] + _tlst[1] == _tlst[2]
        print(subs, frac, _tlst)
