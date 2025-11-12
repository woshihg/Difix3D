import numpy as np
import open3d as o3d
import struct
import os


def read_colmap_points3d_bin(path):
    """
    读取 COLMAP 格式的 points3D.bin 文件并返回点和颜色。

    参数:
    - path (str): points3D.bin 文件的路径。

    返回:
    - points (numpy.ndarray): Nx3 的点云坐标数组。
    - colors (numpy.ndarray): Nx3 的点云颜色数组 (范围在 0-1 之间)。
    """
    points = []
    colors = []

    with open(path, "rb") as f:
        # 读取文件头的点云总数 (uint64)
        num_points = struct.unpack('<Q', f.read(8))[0]
        print(f"文件中包含 {num_points} 个点。正在读取...")

        # 循环读取每个点的信息
        for _ in range(num_points):
            # 读取 POINT3D_ID, X, Y, Z (uint64, double, double, double)
            point_id, x, y, z = struct.unpack('<Qddd', f.read(8 + 3 * 8))

            # 读取 R, G, B (uint8, uint8, uint8)
            r, g, b = struct.unpack('<BBB', f.read(3 * 1))

            # 读取 Error (double)
            error, = struct.unpack('<d', f.read(8))

            # 读取 Track 长度 (uint64)
            track_len, = struct.unpack('<Q', f.read(8))

            # 跳过 Track 数据 (每个 track 元素是 image_id 和 point2D_idx, 均为 uint32)
            f.seek(track_len * 8, os.SEEK_CUR)

            points.append([x, y, z])
            # 将颜色值从 0-255 归一化到 0-1
            colors.append([r / 255.0, g / 255.0, b / 255.0])

    print("文件读取完成。")
    return np.array(points), np.array(colors)


def visualize_point_clouds(points1, colors1, points2):
    """
    使用 Open3D 同时可视化两个点云。
    第一个点云使用其原始颜色，第二个点云被染成绿色。

    参数:
    - points1 (numpy.ndarray): 主点云的坐标数组。
    - colors1 (numpy.ndarray): 主点云的颜色数组 (0-1)。
    - points2 (numpy.ndarray): 参考点云的坐标数组。
    """
    if len(points1) == 0 and len(points2) == 0:
        print("两个点云都为空，无法可视化。")
        return

    print("正在创建可视化窗口...")
    # 创建第一个 Open3D 点云对象
    pcd1 = o3d.geometry.PointCloud()
    if len(points1) > 0:
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd1.colors = o3d.utility.Vector3dVector(colors1)

    # 创建第二个 Open3D 点云对象（参考点云）
    pcd2 = o3d.geometry.PointCloud()
    if len(points2) > 0:
        pcd2.points = o3d.utility.Vector3dVector(points2)
        # 将参考点云染成绿色
        pcd2.paint_uniform_color([0, 1, 0])

    # 创建一个可视化窗口并添加点云
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="双点云对比可视化", width=1280, height=720)
    if len(pcd1.points) > 0:
        vis.add_geometry(pcd1)
    if len(pcd2.points) > 0:
        vis.add_geometry(pcd2)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # 可以根据需要调整点的大小
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # 设置背景为深灰色

    print("\n可视化窗口已打开。")
    print("  - 主点云: 原始颜色")
    print("  - 参考点云: 绿色")
    print("操作指南:")
    print("  - 鼠标左键 + 拖动: 旋转视角")
    print("  - 鼠标滚轮: 缩放视角")
    print("  - 鼠标右键 + 拖动: 平移视角")
    print("  - 按 'q' 或关闭窗口退出。")

    # 运行可视化
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    # --- 1. 设置要可视化的文件名 ---
    # 主点云文件（例如，由 depth_anything 生成或清理后的点云）
    input_file = "points3D.bin"
    # 参考点云文件（例如，原始 COLMAP 的稀疏重建结果）
    ref_file = "points3D_real.bin"

    # --- 2. 检查文件是否存在 ---
    if not os.path.exists(input_file) or not os.path.exists(ref_file):
        print(f"错误: 找不到文件。请确保以下文件都存在:")
        print(f"  - 主文件: '{input_file}'")
        print(f"  - 参考文件: '{ref_file}'")
    else:
        # --- 3. 读取并可视化点云 ---
        print(f"正在读取主点云: {input_file}")
        xyz1, rgb1 = read_colmap_points3d_bin(input_file)

        print(f"\n正在读取参考点云: {ref_file}")
        xyz2, _ = read_colmap_points3d_bin(ref_file) # 参考点云的颜色将被忽略

        visualize_point_clouds(xyz1, rgb1, xyz2)
