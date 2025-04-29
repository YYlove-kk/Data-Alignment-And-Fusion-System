# ---------------------------------------------
# File: kpca_reduce.py
# ---------------------------------------------
import argparse
import glob
import numpy as np
import os
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA


def fit_transform(vec_paths, out_dir, dim=256, gamma=1/256):
    # 1. 收集所有向量
    mats = [np.load(p) for p in vec_paths]
    X = np.vstack(mats)           # (N, D)

    # 2. 随机 Fourier 特征近似高斯核
    rbf = RBFSampler(gamma=gamma, n_components=2048, random_state=0)
    Z = rbf.fit_transform(X)       # (N,2048)

    # 3. PCA -> 最多降到 min(样本数, 特征数, 256)
    n_components = min(dim, Z.shape[0], Z.shape[1])
    pca = PCA(n_components=n_components, whiten=True).fit(Z)
    Zred = pca.transform(Z)        # (N,n_components)

    # 4. 写分片文件
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    for path in vec_paths:
        n = np.load(path).shape[0] if path.endswith(".npz") else 1
        out_path = os.path.join(out_dir, os.path.basename(path))
        np.save(out_path, Zred[idx:idx+n])
        idx += n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    inDir = args.input_dir
    outDir = args.output_dir

    paths = glob.glob(f"{inDir}/*.npy")
    fit_transform(paths, outDir)