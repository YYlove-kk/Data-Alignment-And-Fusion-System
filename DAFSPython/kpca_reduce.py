import numpy as np
import os
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA, SparsePCA
import glob


def kpca_transform(input_dir, output_dir, dim=256, gamma=1 / 256, sample_threshold_ratio=1):
    # 获取 image 和 text 子文件夹中的 .npy 文件路径
    image_files = glob.glob(os.path.join(input_dir, "image", "*.npy"))
    text_files = glob.glob(os.path.join(input_dir, "text", "*.npy"))

    # 初始化输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### 图像部分：先降维再聚合
    print("正在处理 image 向量：先降维为 n×256，再聚合为 1×256...")

    rbf = RBFSampler(gamma=gamma, n_components=2048, random_state=0)

    all_image_data = []
    image_sample_lens = []
    all_image_files = []

    for npy_path in image_files:
        joint_vec = np.load(npy_path)
        if joint_vec.ndim == 1:
            joint_vec = joint_vec[np.newaxis, :]
        all_image_data.append(joint_vec)
        image_sample_lens.append(joint_vec.shape[0])
        all_image_files.append(npy_path)

    if not all_image_data:
        print(f"[警告] 未找到有效的 image 向量文件，跳过 image 降维处理。")
    else:
        all_image_data = np.vstack(all_image_data)  # 所有 image 样本拼接

        rbf_image_data = rbf.fit_transform(all_image_data)
        pca = PCA(n_components=dim, whiten=True)
        reduced_image_data = pca.fit_transform(rbf_image_data)

        # 拆分后聚合为 1×256 保存
        split_index = 0
        for npy_path, num_samples in zip(all_image_files, image_sample_lens):
            file_data = reduced_image_data[split_index:split_index + num_samples]
            reduced_vec = np.mean(file_data, axis=0)  # 聚合为 1×256

            source_file = os.path.basename(npy_path).replace(".npy", "")
            np.save(os.path.join(output_dir, "image",f"{source_file}.npy"), reduced_vec)
            print(f"处理完 image {source_file} 并保存为 1×256 到 {output_dir}/image/{source_file}.npy")

            split_index += num_samples

    ### 文本部分：选择适当的降维方法
    print("正在处理 text 向量：检查降维需求...")

    all_text_data = []
    all_text_files = []

    for npy_path in text_files:
        joint_vec = np.load(npy_path)
        if joint_vec.ndim == 1:
            joint_vec = joint_vec[np.newaxis, :]
        all_text_data.append(joint_vec)
        all_text_files.append(npy_path)

    all_text_data = np.vstack(all_text_data)
    # 检查文本数据是否包含 NaN 值
    if np.isnan(all_text_data).any():
        print("警告：文本数据中包含 NaN 值！")
        # 处理 NaN 值，如使用 0 填充，或者删除包含 NaN 的样本
        all_text_data = np.nan_to_num(all_text_data)  # 使用 0 填充 NaN

    # 标准化文本数据
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    all_text_data = scaler.fit_transform(all_text_data)


    ### 文本部分：选择适当的降维方法
    print("正在处理 text 向量：自动判断使用 KPCA 还是 SparsePCA...")

    all_text_data = []
    all_text_files = []

    for npy_path in text_files:
        vec = np.load(npy_path)
        if vec.ndim == 1:
            vec = vec[np.newaxis, :]
        all_text_data.append(vec)
        all_text_files.append(npy_path)

    if not all_text_data:
        print("[警告] 未找到有效的 text 向量文件，跳过文本降维处理。")
    else:
        all_text_data = np.vstack(all_text_data)

        # 处理 NaN
        if np.isnan(all_text_data).any():
            print("警告：文本数据中包含 NaN，将使用 0 填充。")
            all_text_data = np.nan_to_num(all_text_data)

        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        all_text_data = scaler.fit_transform(all_text_data)

        num_samples = all_text_data.shape[0]
        print(f"文本样本数：{num_samples}，目标降维维度：{dim}")

        # 自动判断使用 KPCA 还是 SparsePCA
        if num_samples < dim * sample_threshold_ratio:
            print("样本数较少，使用 SparsePCA 降维...")
            model = SparsePCA(n_components=dim, random_state=0)
            reduced_text_data = model.fit_transform(all_text_data)
        else:
            print("样本数足够，使用 KPCA 降维...")
            rbf_text_data = rbf.fit_transform(all_text_data)
            if np.isnan(rbf_text_data).any():
                print("警告：RBF 输出含 NaN，跳过文本降维。")
                reduced_text_data = None
            else:
                pca = PCA(n_components=dim, whiten=True)
                reduced_text_data = pca.fit_transform(rbf_text_data)
                if np.isnan(reduced_text_data).any():
                    print("警告：降维结果含 NaN，跳过文本降维。")
                    reduced_text_data = None

        # 保存文本降维结果
        if reduced_text_data is not None:
            os.makedirs(os.path.join(output_dir, "text"), exist_ok=True)

            for i, npy_path in enumerate(all_text_files):
                reduced_vec = reduced_text_data[i]
                source_file = os.path.basename(npy_path).replace(".npy", "")
                if np.isnan(reduced_vec).any():
                    print(f"[跳过保存] {source_file} 降维后结果中含 NaN")
                else:
                    np.save(os.path.join(output_dir, "text", f"{source_file}.npy"), reduced_vec)
                    print(f"保存文本降维结果：{source_file} -> 1×{dim}")

if __name__ == "__main__":
    # 输入输出目录
    input_dir = "../data/align/raw"  # 输入：包含 image 和 text 子文件夹
    output_dir = "../data/align/reduced"  # 输出：降维后统一保存到此目录

    # 执行降维处理
    kpca_transform(input_dir, output_dir)

    ### 模态匹配检查：确保 image 和 text 降维后的文件一一对应
    print("\n开始检查降维后 image 和 text 文件是否一一对应...")

    # 获取降维后输出目录下所有文件（不带扩展名）
    reduced_files = os.listdir(output_dir)
    reduced_files = [f for f in reduced_files if f.endswith('.npy')]
    reduced_ids = set(os.path.splitext(f)[0] for f in reduced_files)

    image_ids = set(f[:-8] for f in reduced_ids if f.endswith('_img.npy'))  # 去掉 '_img.npy'
    text_ids = set(f[:-8] for f in reduced_ids if f.endswith('_txt.npy'))  # 去掉 '_txt.npy'


    # 取出共有的 ID（以 "xxx_patientId" 为准）
    matched_ids = image_ids & text_ids
    extra_image_ids = image_ids - matched_ids
    extra_text_ids = text_ids - matched_ids

    # 删除多余 image 文件
    for eid in extra_image_ids:
        path = os.path.join(output_dir, eid + ".npy")
        os.remove(path)
        print(f"[已删除多余 image 文件] {path}")

    # 删除多余 text 文件
    for tid in extra_text_ids:
        path = os.path.join(output_dir, tid + ".npy")
        os.remove(path)
        print(f"[已删除多余 text 文件] {path}")

    print(f"降维文件一致性检查完成：共有 {len(matched_ids)} 对 image-text 文件匹配成功。")

