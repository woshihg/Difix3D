import numpy as np
from scipy.optimize import lsq_linear
from sklearn.linear_model import RANSACRegressor, LinearRegression

from PIL import Image
import matplotlib.pyplot as plt

def align_depth_maps(relative_depth: np.ndarray, absolute_depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    通过 RANSAC 拟合线性模型 (gt = a * pred + b) 来对齐相对深度图和绝对深度图。
    这种方法对异常值具有很强的鲁棒性。

    Args:
        relative_depth (np.ndarray): 预测的相对深度图（无单位）。
        absolute_depth (np.ndarray): 稀疏或带噪声的绝对深度图（以米为单位）。
                                     假设值为 0 或负数是无效的。

    Returns:
        tuple[np.ndarray, float, float]:
        - aligned_depth (np.ndarray): 应用缩放和偏移后的完整深度图。
        - scale (float): 计算出的最佳缩放因子。
        - offset (float): 计算出的最佳偏移量。
    """
    # 1. 找到绝对深度图中有效值的位置 (大于一个很小的值以避免浮点误差)
    valid_mask = absolute_depth > 1e-8

    # 2. 如果没有足够的重叠数据点，则无法计算，返回原始深度图
    if np.sum(valid_mask) < 10: # RANSAC 需要更多点
        print("警告: 有效重叠点过少，跳过对齐。")
        return relative_depth, 1.0, 0.0

    # 3. 准备 RANSAC 的数据
    # y 是我们的目标值 (绝对深度)
    y = absolute_depth[valid_mask]
    # X 是我们的特征 (相对深度)
    X = relative_depth[valid_mask].reshape(-1, 1)

    try:
        # 4. 使用 RANSAC 拟合线性模型
        ransac = RANSACRegressor(
            LinearRegression(),
            max_trials=1000,      # 增加尝试次数以获得更好的结果
            min_samples=2,        # 拟合一条直线最少需要2个点
            residual_threshold=0.1, # 10cm 的残差阈值，用于区分内点和外点
            random_state=42
        )
        ransac.fit(X, y)

        # 检查拟合是否成功
        if ransac.estimator_ is None:
            print("警告: RANSAC 未能找到一致的模型。")
            return relative_depth, 1.0, 0.0

        # 提取学到的缩放和偏移
        scale = ransac.estimator_.coef_[0]
        offset = ransac.estimator_.intercept_

    except ValueError as e:
        print(f"RANSAC 拟合时发生错误: {e}。跳过对齐。")
        return relative_depth, 1.0, 0.0

    # 5. 应用学到的变换到整个相对深度图
    aligned_depth = relative_depth * scale + offset

    return aligned_depth, scale, offset

if __name__ == '__main__':
    # --- 使用真实深度图 ---
    # 1. 定义文件路径
    abs_depth_path = "/home/woshihg/realsense/depth/0000.png"
    rel_depth_path = "/home/woshihg/PycharmProjects/Difix3D/get_point3D/ppd/0000.png"

    # 2. 加载深度图
    try:
        abs_depth_img = Image.open(abs_depth_path)
        rel_depth_img = Image.open(rel_depth_path)
    except FileNotFoundError as e:
        print(f"错误: 无法找到文件 - {e}")
        exit()

    # 3. 将图像转换为 numpy 数组
    # 相对深度图是无单位的，不应除以1000
    relative_depth_map = np.array(rel_depth_img, dtype=np.float32)
    # 假设绝对深度图是 16-bit PNG，单位是毫米
    absolute_depth_map = np.array(abs_depth_img, dtype=np.float32) / 1000.0  # 转换为米

    print(f"已加载相对深度图，形状: {relative_depth_map.shape}, 数据类型: {relative_depth_map.dtype}")
    print(f"已加载绝对深度图，形状: {absolute_depth_map.shape}, 数据类型: {absolute_depth_map.dtype}")

    # --- 计算最佳变换并应用 ---
    aligned_depth_map, optimal_scale, optimal_offset = align_depth_maps(relative_depth_map, absolute_depth_map)

    print(f"\n计算出的最佳变换: scale={optimal_scale}, offset={optimal_offset:.4f}")

    # --- 验证结果 ---
    # 仅在绝对深度图的有效区域计算误差
    valid_mask = absolute_depth_map > 1e-8
    if not np.any(valid_mask):
        print("警告: 绝对深度图中没有有效的深度值用于计算误差。")
    else:
        # 对齐前的误差比较意义不大，因为尺度不同
        final_error = np.mean(np.abs(aligned_depth_map[valid_mask] - absolute_depth_map[valid_mask]))
        print(f"\n对齐后在有效区域的平均绝对误差: {final_error:.4f} 米")

    # --- 保存对齐后的深度图 ---
    # 按照读取的规范（16位PNG，单位毫米）保存
    # 1. 将单位从米转换回毫米
    aligned_depth_mm = aligned_depth_map * 1000.0
    # 2. 转换为16位无符号整数
    aligned_depth_uint16 = aligned_depth_mm.astype(np.uint16)
    # 3. 保存为PNG文件
    Image.fromarray(aligned_depth_uint16).save("0000.png")
    print("\n对齐后的深度图已保存为 '0000.png' (16-bit, 毫米单位)")
