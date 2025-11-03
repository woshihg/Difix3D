# python
import numpy as np
import cv2
import os
import struct


def generate_points3d_from_data(
        poses_npy_path,
        rgb_dir,
        depth_dir,
        intrinsics,
        depth_scale,
        output_path,
        num_frames_to_process,
        sampling_stride=1,
        frame_indices=None
):
    """
    从NPY位姿文件、RGB图像和深度图像生成COLMAP格式的points3D.bin二进制文件。

    新增参数:
    - frame_indices (list[int] | None): 要处理的帧索引列表（以0开始）。若为 None，则按前 num_frames_to_process 帧处理。
    """
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    print(f"从 {poses_npy_path} 加载位姿...")
    poses_c2w = np.load(poses_npy_path)
    num_total_frames = poses_c2w.shape[0]
    print(f"成功加载 {num_total_frames} 个位姿。")

    if num_frames_to_process > num_total_frames:
        print(f"警告: 请求处理 {num_frames_to_process} 帧，但只找到 {num_total_frames} 个位姿。将处理所有可用位姿。")
        num_frames_to_process = num_total_frames

    image_names = [f"{i:04d}.png" for i in range(num_total_frames)]
    point3D_id_counter = 1
    num_written_points = 0

    # 计算要处理的帧索引列表
    if frame_indices is None:
        indices_to_process = list(range(min(num_frames_to_process, num_total_frames)))
    else:
        # 过滤并保证为整数、在范围内且去重、按顺序
        try:
            raw = list(frame_indices)
        except TypeError:
            raise ValueError("frame_indices 必须是可迭代的整数序列或 None")
        indices_to_process = sorted({int(i) for i in raw if 0 <= int(i) < num_total_frames})
        if not indices_to_process:
            print("警告: 提供的 frame_indices 在有效范围内为空，退出处理。")
            return

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', 0))

        print(f"开始处理 {len(indices_to_process)} 帧图像...")
        for count, idx in enumerate(indices_to_process):
            image_name = image_names[idx]
            rgb_path = os.path.join(rgb_dir, image_name)
            depth_path = os.path.join(depth_dir, image_name)

            print(f"  处理第 {count + 1}/{len(indices_to_process)}: 索引={idx} 文件={image_name}")

            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"    警告: 找不到文件 {rgb_path} 或 {depth_path}，跳过。")
                continue

            rgb_image = cv2.imread(rgb_path)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if rgb_image is None or depth_image is None:
                print(f"    警告: 无法读取图像文件，跳过。")
                continue

            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            height, width, _ = rgb_image.shape

            pose_c2w = poses_c2w[idx]
            R_c2w = pose_c2w[:3, :3]
            t_c2w = pose_c2w[:3, 3].reshape(3, 1)

            for v in range(0, height, sampling_stride):
                for u in range(0, width, sampling_stride):
                    depth = depth_image[v, u]

                    if depth == 0:
                        continue

                    z_c = depth / depth_scale
                    x_c = (u - cx) * z_c / fx
                    y_c = (v - cy) * z_c / fy
                    p_cam = np.array([x_c, y_c, z_c]).reshape(3, 1)

                    p_world = R_c2w @ p_cam + t_c2w

                    X, Y, Z = p_world.flatten()
                    R, G, B = rgb_image[v, u]

                    image_id = idx + 1
                    point2D_idx = 0

                    f.write(struct.pack('<Qddd', point3D_id_counter, X, Y, Z))
                    f.write(struct.pack('<BBB', R, G, B))
                    f.write(struct.pack('<d', 1.0))
                    f.write(struct.pack('<Q', 1))
                    f.write(struct.pack('<II', image_id, point2D_idx))

                    point3D_id_counter += 1
                    num_written_points += 1

        f.seek(0)
        f.write(struct.pack('<Q', num_written_points))

    print(f"\n处理完成！总共生成了 {num_written_points} 个点。")
    print(f"二进制文件已保存至: {output_path}")


if __name__ == '__main__':
    # 输入文件和文件夹路径
    poses_file = r"/home/woshihg/realsense/camera_poses.npy"
    rgb_folder = r"/home/woshihg/realsense/rgb"
    depth_folder = r"/home/woshihg/realsense/depth"

    output_file = r"points3D.bin"

    intrinsics = {'fx': 927.17, 'fy': 927.37, 'cx': 651.32, 'cy': 349.62}
    depth_scale = 1000.0

    # --- 可选两种用法 ---
    # 1) 按前 N 帧处理（与原先行为一致）
    frames_to_use = 1
    selected_frame_indices = [1,16,135,255]  # 若为 None 则使用上面的 frames_to_use

    # 2) 或者指定具体帧索引列表（以0开始），例如只处理编号为 0 和 5 的图片：
    # selected_frame_indices = [0, 5]

    if "path/to/your" in str(poses_file) or "path/to/your" in str(rgb_folder) or "path/to/your" in str(depth_folder):
        print("错误: 请在运行前修改脚本中的 `poses_file`, `rgb_folder`, 和 `depth_folder` 为您的真实数据路径。")
    else:
        print("开始使用真实数据生成二进制点云文件...")
        generate_points3d_from_data(
            poses_npy_path=poses_file,
            rgb_dir=rgb_folder,
            depth_dir=depth_folder,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            output_path=output_file,
            num_frames_to_process=frames_to_use,
            sampling_stride=1,
            frame_indices=selected_frame_indices
        )
