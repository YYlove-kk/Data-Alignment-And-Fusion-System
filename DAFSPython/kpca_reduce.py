# ---------------------------------------------
# File: kpca_reduce.py
# ---------------------------------------------
import argparse
import glob
import numpy as np
import os
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA



def fit_transform(input_dir, output_dir, dim=256, gamma=1/256):
    # 遍历每个病人的文件夹
    patient_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for patient_dir in patient_dirs:
        # 构建当前病人文件夹下文本和图像嵌入的路径
        txt_path = os.path.join(input_dir, patient_dir, f"{patient_dir}_z_t.npy")
        img_path = os.path.join(input_dir, patient_dir, f"{patient_dir}_z_i.npy")

        # 检查文件是否存在
        if not os.path.exists(txt_path) or not os.path.exists(img_path):
            print(f"Missing files for patient {patient_dir}. Skipping.")
            continue

        # 读取嵌入向量
        txt_vec = np.load(txt_path)
        img_vec = np.load(img_path)

        # 确保它们是二维向量
        if txt_vec.ndim == 1:
            txt_vec = txt_vec[np.newaxis, :]
        if img_vec.ndim == 1:
            img_vec = img_vec[np.newaxis, :]

        # 拼接文本和图像向量
        X = np.vstack([txt_vec, img_vec])  # shape: (2, D)

        # RBF 映射
        rbf = RBFSampler(gamma=gamma, n_components=2048, random_state=0)
        Z = rbf.fit_transform(X)

        # PCA降维
        n_components = min(dim, Z.shape[0], Z.shape[1])  # 通常是 2
        pca = PCA(n_components=n_components, whiten=True).fit(Z)
        Zred = pca.transform(Z)  # shape: (2, n_components)

        # 创建输出目录
        out_patient_dir = os.path.join(output_dir, patient_dir)
        os.makedirs(out_patient_dir, exist_ok=True)

        # 保存降维后的向量
        np.save(os.path.join(out_patient_dir, f"{patient_dir}_z_t.npy"), Zred[0])  # 保存文本向量
        np.save(os.path.join(out_patient_dir, f"{patient_dir}_z_i.npy"), Zred[1])  # 保存图像向量



if __name__ == "__main__":

    # 获取所有病人文件夹路径
    inDir = os.path.join("data", "align", "output")
    outDir = os.path.join("data", "align", "reduce")

    fit_transform(inDir, outDir)