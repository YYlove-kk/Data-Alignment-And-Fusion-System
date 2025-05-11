import numpy as np
import os
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
import glob

def kpca_transform(input_dir, output_dir, batch_size=10000, dim=256, gamma=1/256):
    # 获取 image 和 text 子文件夹中的 .npy 文件路径
    image_files = glob.glob(os.path.join(input_dir, "image", "*.npy"))
    text_files = glob.glob(os.path.join(input_dir, "text", "*.npy"))

    # 创建一个 RBF 核映射器 (使用随机特征近似)
    rbf = RBFSampler(gamma=gamma, n_components=2048, random_state=0)

    # 初始化输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理 image 文件
    all_image_data = []
    all_image_files = []

    for npy_path in image_files:
        joint_vec = np.load(npy_path)

        # 确保数据是二维的，形状应为 (n_samples, n_features)
        if joint_vec.ndim == 1:
            joint_vec = joint_vec[np.newaxis, :]

        all_image_data.append(joint_vec)
        all_image_files.append(npy_path)

    # 拼接所有 image 数据
    all_image_data = np.vstack(all_image_data)  # 拼接成一个大的数据集

    # 进行 RBF 特征映射
    rbf_image_data = rbf.fit_transform(all_image_data)

    # 进行 PCA 降维
    pca = PCA(n_components=dim, whiten=True)
    reduced_image_data = pca.fit_transform(rbf_image_data)

    # 步骤 2: 按原 image 文件顺序拆分降维后的数据
    split_index = 0
    for npy_path in all_image_files:
        joint_vec = np.load(npy_path)
        num_samples = joint_vec.shape[0]

        # 获取对应降维后的数据切片
        reduced_file_data = reduced_image_data[split_index:split_index+num_samples]

        # 获取文件名作为标识
        source_file = os.path.basename(npy_path).replace(".npy", "")

        # 保存降维后的数据
        np.save(os.path.join(output_dir, f"{source_file}.npy"), reduced_file_data)
        print(f"处理完 image {source_file} 并保存到 {output_dir}/{source_file}.npy")

        # 更新切片起始位置
        split_index += num_samples

    # 处理 text 文件
    all_text_data = []
    all_text_files = []

    for npy_path in text_files:
        joint_vec = np.load(npy_path)

        # 确保数据是二维的，形状应为 (n_samples, n_features)
        if joint_vec.ndim == 1:
            joint_vec = joint_vec[np.newaxis, :]

        all_text_data.append(joint_vec)
        all_text_files.append(npy_path)

    # 拼接所有 text 数据
    all_text_data = np.vstack(all_text_data)  # 拼接成一个大的数据集

    # 进行 RBF 特征映射
    rbf_text_data = rbf.fit_transform(all_text_data)

    # 进行 PCA 降维
    reduced_text_data = pca.fit_transform(rbf_text_data)

    # 步骤 2: 按原 text 文件顺序拆分降维后的数据
    split_index = 0
    for npy_path in all_text_files:
        joint_vec = np.load(npy_path)
        num_samples = joint_vec.shape[0]

        # 获取对应降维后的数据切片
        reduced_file_data = reduced_text_data[split_index:split_index+num_samples]

        # 获取文件名作为标识
        source_file = os.path.basename(npy_path).replace(".npy", "")

        # 保存降维后的数据
        np.save(os.path.join(output_dir, f"{source_file}.npy"), reduced_file_data)
        print(f"处理完 text {source_file} 并保存到 {output_dir}/{source_file}.npy")

        # 更新切片起始位置
        split_index += num_samples

if __name__ == "__main__":
    # 输入输出目录
    input_dir = "../data/align/raw"  # 输入的 .npy 文件所在目录，包含 image 和 text 子文件夹
    output_dir = "../data/align/reduced"  # 输出的降维结果保存目录

    # 执行降维处理
    kpca_transform(input_dir, output_dir)
