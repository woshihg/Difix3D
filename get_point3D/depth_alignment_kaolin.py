import numpy as np
from scipy.optimize import lsq_linear
from sklearn.linear_model import RANSACRegressor, LinearRegression

from PIL import Image
import matplotlib.pyplot as plt

# 新增：PyTorch + Kaolin 导入
import torch
from torch import nn
try:
    from kaolin.metrics.pointcloud import chamfer_distance
except Exception:
    chamfer_distance = None

def backproject_depth_to_points(depth: torch.Tensor, intrinsics: dict) -> torch.Tensor:
    """
    深度图（HxW） -> 点云 (N,3)（相机坐标系）
    depth: torch.Tensor, shape (H, W)
    intrinsics: dict with fx, fy, cx, cy
    返回 float32 tensor (N,3)
    """
    device = depth.device
    H, W = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    ys = torch.arange(0, H, device=device, dtype=torch.float32)
    xs = torch.arange(0, W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # HxW
    z = depth

    valid = (z > 0.001) & (z <= 1.0)  # 过滤距离 > 1m 和极小值
    if valid.sum() == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)

    x = (grid_x[valid] - cx) * z[valid] / fx
    y = (grid_y[valid] - cy) * z[valid] / fy
    pts = torch.stack([x, y, z[valid]], dim=1)  # N x 3
    return pts

def axis_angle_to_rotation_matrix(vec: torch.Tensor) -> torch.Tensor:
    """
    vec: (3,) axis-angle (rodriques), 返回 3x3 旋转矩阵
    """
    theta = torch.norm(vec) + 1e-8
    axis = vec / theta
    a = axis[0]; b = axis[1]; c = axis[2]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.zeros((3,3), device=vec.device, dtype=vec.dtype)
    R[0,0] = cos_t + a*a*(1-cos_t)
    R[0,1] = a*b*(1-cos_t) - c*sin_t
    R[0,2] = a*c*(1-cos_t) + b*sin_t
    R[1,0] = b*a*(1-cos_t) + c*sin_t
    R[1,1] = cos_t + b*b*(1-cos_t)
    R[1,2] = b*c*(1-cos_t) - a*sin_t
    R[2,0] = c*a*(1-cos_t) - b*sin_t
    R[2,1] = c*b*(1-cos_t) + a*sin_t
    R[2,2] = cos_t + c*c*(1-cos_t)
    return R

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
        # 4. 使用 RANSAC 拟合线性模型（作为初始化）
        ransac = RANSACRegressor(
            LinearRegression(),
            max_trials=1000,
            min_samples=2,
            residual_threshold=0.1,
            random_state=42
        )
        ransac.fit(X, y)

        if ransac.estimator_ is None:
            print("警告: RANSAC 未能找到一致的模型。")
            return relative_depth, 1.0, 0.0

        scale_init = float(ransac.estimator_.coef_[0])
        offset_init = float(ransac.estimator_.intercept_)

    except ValueError as e:
        print(f"RANSAC 拟合时发生错误: {e}。跳过对齐。")
        return relative_depth, 1.0, 0.0

    # --- 使用 Kaolin + PyTorch 优化 scale, offset 和 相机外参 ---
    if chamfer_distance is None:
        print("警告: kaolin 未找到，跳过基于 Chamfer 的优化，仅使用 RANSAC 结果。")
        aligned_depth = relative_depth * scale_init + offset_init
        return aligned_depth, scale_init, offset_init

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 内参（题主给定）
    intrinsics = {'fx': 927.17, 'fy': 927.37, 'cx': 651.32, 'cy': 349.62}

    # 将 numpy -> torch
    rel_depth_t = torch.from_numpy(relative_depth.astype(np.float32)).to(device)
    abs_depth_t = torch.from_numpy(absolute_depth.astype(np.float32)).to(device)

    # 创建绝对深度点云（绝对相机外参为单位矩阵）
    pts_abs = backproject_depth_to_points(abs_depth_t, intrinsics)  # N2 x 3
    if pts_abs.shape[0] == 0:
        print("警告: 绝对深度点云为空，跳过优化。")
        aligned_depth = relative_depth * scale_init + offset_init
        return aligned_depth, scale_init, offset_init

    # 下采样以加速计算
    max_pts = 8192
    if pts_abs.shape[0] > max_pts:
        idx = torch.randperm(pts_abs.shape[0], device=device)[:max_pts]
        pts_abs_s = pts_abs[idx]
    else:
        pts_abs_s = pts_abs

    # 初始化可优化变量
    scale_param = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32, device=device))
    offset_param = nn.Parameter(torch.tensor(offset_init, dtype=torch.float32, device=device))
    rot_vec = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))  # axis-angle
    trans = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))    # translation

    optimizer = torch.optim.Adam([scale_param, offset_param, rot_vec, trans], lr=1e-2, weight_decay=0.0)

    # 优化循环
    iters = 300
    for it in range(iters):
        optimizer.zero_grad()

        # 应用线性变换到相对深度
        depth_aligned = rel_depth_t * scale_param + offset_param
        depth_aligned = torch.clamp(depth_aligned, min=0.0)

        # 投影为点云并过滤 >1m
        pts_rel = backproject_depth_to_points(depth_aligned, intrinsics)  # N1 x 3
        if pts_rel.shape[0] == 0:
            print("警告: 相对深度投影为空，停止优化。")
            break

        # 将相对点云变换到世界（绝对）坐标系： R * p + t
        R = axis_angle_to_rotation_matrix(rot_vec)
        pts_rel_world = (R @ pts_rel.t()).t() + trans.unsqueeze(0)  # N1 x 3

        # 下采样两边点云，保持平衡
        n1 = pts_rel_world.shape[0]
        n2 = pts_abs_s.shape[0]
        s1 = min(n1, max_pts)
        s2 = min(n2, max_pts)
        if n1 > s1:
            idx1 = torch.randperm(n1, device=device)[:s1]
            samp1 = pts_rel_world[idx1]
        else:
            samp1 = pts_rel_world
        if n2 > s2:
            idx2 = torch.randperm(n2, device=device)[:s2]
            samp2 = pts_abs_s[idx2]
        else:
            samp2 = pts_abs_s

        # chamfer_distance 接受 batch 维度： (B, N, 3)
        p1 = samp1.unsqueeze(0)
        p2 = samp2.unsqueeze(0)

        cd = chamfer_distance(p1, p2, w1=1.0, w2=1.0, squared=True)
        # chamfer_distance 可能返回单张张量或 tuple，统一处理
        if isinstance(cd, tuple) or isinstance(cd, list):
            # 返回类似 (dist1, dist2)
            loss = (cd[0].mean() + cd[1].mean()) * 0.5
        else:
            loss = cd.mean()

        loss.backward()
        optimizer.step()

        if (it + 1) % 50 == 0 or it == 0:
            print(f"Iter {it+1}/{iters}, loss={loss.item():.6f}, scale={scale_param.item():.6f}, offset={offset_param.item():.6f}")

    # 优化结束，取最终值
    final_scale = float(scale_param.detach().cpu().item())
    final_offset = float(offset_param.detach().cpu().item())
    final_rot = rot_vec.detach().cpu().numpy()
    final_trans = trans.detach().cpu().numpy()

    print(f"优化结束: scale={final_scale}, offset={final_offset}, rot_vec={final_rot}, trans={final_trans}")

    # 将最终 scale/offset 应用到 numpy 深度图并返回
    aligned_depth = relative_depth * final_scale + final_offset

    return aligned_depth, final_scale, final_offset

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
    relative_depth_map = np.array(rel_depth_img, dtype=np.float32)
    absolute_depth_map = np.array(abs_depth_img, dtype=np.float32) / 1000.0  # 转换为米

    print(f"已加载相对深度图，形状: {relative_depth_map.shape}, 数据类型: {relative_depth_map.dtype}")
    print(f"已加载绝对深度图，形状: {absolute_depth_map.shape}, 数据类型: {absolute_depth_map.dtype}")

    # --- 计算最佳变换并应用 ---
    aligned_depth_map, optimal_scale, optimal_offset = align_depth_maps(relative_depth_map, absolute_depth_map)

    print(f"\n计算出的最佳变换: scale={optimal_scale}, offset={optimal_offset:.4f}")

    # --- 验证结果 ---
    valid_mask = absolute_depth_map > 1e-8
    if not np.any(valid_mask):
        print("警告: 绝对深度图中没有有效的深度值用于计算误差。")
    else:
        final_error = np.mean(np.abs(aligned_depth_map[valid_mask] - absolute_depth_map[valid_mask]))
        print(f"\n对齐后在有效区域的平均绝对误差: {final_error:.4f} 米")

    # --- 保存对齐后的深度图 ---
    aligned_depth_mm = aligned_depth_map * 1000.0
    aligned_depth_uint16 = aligned_depth_mm.astype(np.uint16)
    Image.fromarray(aligned_depth_uint16).save("0000.png")
    print("\n对齐后的深度图已保存为 '0000.png' (16-bit, 毫米单位)")
