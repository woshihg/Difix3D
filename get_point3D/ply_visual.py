import open3d as o3d
import os
import argparse


def visualize_ply(ply_path):
    """
    加载并可视化一个 PLY 点云文件。
    """
    # 检查文件是否存在
    if not os.path.exists(ply_path):
        print(f"错误: 文件 '{ply_path}' 不存在。")
        return

    print(f"正在加载点云文件: '{ply_path}'")
    try:
        # 读取 .ply 文件
        pcd = o3d.io.read_point_cloud(ply_path)

        # 检查点云是否为空
        if not pcd.has_points():
            print("错误: 点云文件中不包含任何点。")
            return

        print("加载成功。正在打开可视化窗口...")
        print("提示: 您可以在窗口中使用鼠标进行缩放和旋转。按 'q' 关闭窗口。")

        # 可视化点云
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"PLY Viewer: {os.path.basename(ply_path)}",
            width=1280,
            height=720
        )

    except Exception as e:
        print(f"读取或显示文件时发生错误: {e}")


if __name__ == '__main__':

    # 调用可视化函数
    visualize_ply("/home/woshihg/PycharmProjects/Difix3D/get_point3D/ppd/pointcloud.ply")
