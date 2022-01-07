import numpy as np
import glob
import sys
import os.path as osp
from tqdm import tqdm
import multiprocessing as mp


def to_txt(filepath):
    bboxes = np.load(filepath) # TK4
    bboxes[:, :, 2:] -= bboxes[:, :, :2] # TK4
    T, K = bboxes.shape[:2]
    assert T >= 16, filepath
    valid = (bboxes[:, :, 2] * bboxes[:, :, 3]) > 10
    assert valid.sum() > 0

    out = np.full((valid.sum(), 10), fill_value=-1, dtype=np.float32)
    i = 0
    for t in range(T):
        for k in range(K):
            if not valid[t, k]:
                continue
            bbox = bboxes[t, k] # 4
            out[i, 0] = t + 1 # frame num
            out[i, 2:6] = bbox # x1, y1, w, h
            out[i, 6] = 1. # score
            
            i += 1
    out_file = filepath[:-3] + 'txt'
    np.savetxt(out_file, out, delimiter=',')


if __name__ == '__main__':
    root = sys.argv[1]
    files = glob.glob(osp.join(root, '20bn-something-something-v2', '*.npy'))
    print(f'Found {len(files)} files')

    pool = mp.Pool(32)
    list(tqdm(pool.imap(to_txt, files), total=len(files)))
