import open3d as o3d
import numpy as np
import struct
import os


def read_points3d_bin(bin_path):
    """
    读取 COLMAP 格式的 points3D.bin 文件并返回点和颜色。
    """
    points = []
    colors = []
    with open(bin_path, "rb") as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        print(f"文件中包含 {num_points} 个点。")

        for _ in range(num_points):
            # 读取每个点的数据
            # 结构: POINT3D_ID(Q), X(d), Y(d), Z(d), R(B), G(B), B(B), ERROR(d), TRACK_LEN(Q)
            point_data = f.read(8 + 8 * 3 + 3 + 8 + 8)
            if not point_data:
                break

            # 解包 XYZ 和 RGB
            xyz = struct.unpack_from('<ddd', point_data, 8)
            rgb = struct.unpack_from('<BBB', point_data, 8 + 8 * 3)

            points.append(xyz)
            colors.append([c / 255.0 for c in rgb])  # 颜色归一化到 [0, 1]

            # 跳过 track 数据
            track_len = struct.unpack_from('<Q', point_data, 8 + 8 * 3 + 3 + 8)[0]
            f.seek(track_len * 8, os.SEEK_CUR)

    print(f"成功读取 {len(points)} 个点。")
    return np.array(points), np.array(colors)


def write_points3d_bin(bin_path, points, colors):
    """
    将点和颜色数据写入 COLMAP 格式的 points3D.bin 文件。
    """
    num_points = len(points)
    with open(bin_path, "wb") as f:
        # 写入点的总数
        f.write(struct.pack('<Q', num_points))

        for i in range(num_points):
            point = points[i]
            color = (colors[i] * 255).astype(np.uint8)

            # 写入 POINT3D_ID, XYZ, RGB, ERROR, TRACK_LEN
            # POINT3D_ID 从 1 开始
            # ERROR 设为 1.0, TRACK_LEN 设为 0
            f.write(struct.pack('<Q', i + 1))  # POINT3D_ID
            f.write(struct.pack('<ddd', point[0], point[1], point[2]))  # XYZ
            f.write(struct.pack('<BBB', color[0], color[1], color[2]))  # RGB
            f.write(struct.pack('<d', 1.0))  # ERROR
            f.write(struct.pack('<Q', 0))  # TRACK_LEN = 0, no track data


def main(input_bin_path, output_bin_path):
    """
    主函数：读取点云，去除离群点，并可视化和保存结果。
    """
    if not os.path.exists(input_bin_path):
        print(f"错误: 输入文件 '{input_bin_path}' 不存在。")
        return

    # 1. 读取点云数据
    xyz, rgb = read_points3d_bin(input_bin_path)
    if xyz.shape[0] == 0:
        print("未读取到任何点，程序退出。")
        return

    # 2. 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print("\n原始点云统计信息:")
    print(pcd)

    # 3. 可视化原始点云
    print("\n显示原始点云... (按 'q' 关闭窗口)")
    o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

    # 4. 执行离群点去除
    # remove_statistical_outlier:
    #   - nb_neighbors: 指定在计算平均距离时考虑的邻居数量。
    #   - std_ratio: 标准差的倍数。点与其邻居的平均距离如果大于全局平均距离加上这个倍数的标准差，则被视为离群点。
    print("\n正在进行离群点去除...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=150, std_ratio=2.0)

    # 通过索引选择保留下来的点
    pcd_cleaned = pcd.select_by_index(ind)

    print("\n清理后的点云统计信息:")
    print(pcd_cleaned)
    num_removed = len(pcd.points) - len(pcd_cleaned.points)
    print(f"移除了 {num_removed} 个离群点。")

    # 5. 可视化清理后的点云
    print("\n显示清理后的点云... (按 'q' 关闭窗口)")
    o3d.visualization.draw_geometries([pcd_cleaned], window_name="Cleaned Point Cloud")

    # 6. 保存清理后的点云
    if output_bin_path:
        cleaned_points = np.asarray(pcd_cleaned.points)
        cleaned_colors = np.asarray(pcd_cleaned.colors)
        write_points3d_bin(output_bin_path, cleaned_points, cleaned_colors)
        print(f"\n清理后的点云已保存到: '{output_bin_path}'")


if __name__ == '__main__':
    # 输入和输出文件路径
    input_file = "points3D.bin"
    output_file = "points3D_cleaned.bin"

    main(input_bin_path=input_file, output_bin_path=output_file)
