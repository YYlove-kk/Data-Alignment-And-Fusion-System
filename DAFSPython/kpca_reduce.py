import numpy as np
import os
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
import glob

def kpca_transform(input_dir, output_dir, batch_size=10000, dim=256, gamma=1/256):
    # 获取 input_dir 目录下所有 .npy 文件路径
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))

    # 创建一个 RBF 核映射器 (使用随机特征近似)
    rbf = RBFSampler(gamma=gamma, n_components=2048, random_state=0)

    # 初始化输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历所有 .npy 文件
    for npy_path in npy_files:
        # 获取文件名作为标识
        source_file = os.path.basename(npy_path).replace(".npy", "")

        # 读取联合嵌入的向量
        joint_vec = np.load(npy_path)

        # 确保数据是二维的，形状应为 (n_samples, n_features)
        if joint_vec.ndim == 1:
            joint_vec = joint_vec[np.newaxis, :]

        # 将数据分批处理，避免内存溢出
        n_samples = joint_vec.shape[0]
        n_batches = (n_samples // batch_size) + (1 if n_samples % batch_size > 0 else 0)

        # 创建一个存储降维后向量的列表
        reduced_vectors = []

        for i in range(n_batches):
            # 计算当前批次的起始和结束索引
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            # 获取当前批次的数据
            batch_data = joint_vec[start_idx:end_idx]

            # 通过 RBF 映射进行特征转换
            rbf_data = rbf.fit_transform(batch_data)

            # 进行 PCA 降维
            pca = PCA(n_components=dim, whiten=True)
            reduced_batch = pca.fit_transform(rbf_data)

            # 将当前批次的降维结果添加到结果列表中
            reduced_vectors.append(reduced_batch)

        # 合并所有批次的降维结果
        final_reduced_vec = np.vstack(reduced_vectors)

        # 保存降维后的向量为 .npy 文件
        np.save(os.path.join(output_dir, f"{source_file}.npy"), final_reduced_vec)

        print(f"Processed {source_file} and saved to {output_dir}/{source_file}.npy")

if __name__ == "__main__":
    # 输入输出目录
    input_dir = "data/align/joint"  # 输入的 .npy 文件所在目录
    output_dir = "data/align/reduced"  # 输出的降维结果保存目录

    # 执行降维处理
    kpca_transform(input_dir, output_dir)
