import numpy as np
import os
from colmap.scripts.python.read_write_model import write_cameras_binary, write_images_binary, Camera, Image

# 1. 加载位姿
poses = np.load('/home/woshihg/realsense/camera_poses.npy')  # (256, 4, 4)
# 打印一个位姿以确认
print("示例位姿矩阵:\n", poses[0])


# 2. 假设图片名为 0.png, 1.png, ..., 255.png
image_names = [f"{i:04d}.png" for i in range(256)]

# 3. 相机内参（需根据实际情况填写）
width, height = 1280, 720
fx = 927.16973877
fy = 927.36688232
cx, cy = 651.31506348, 349.62133789
camera_id = 1

# 4. 构造 Camera 对象
camera = Camera(
    id=camera_id,
    model="PINHOLE",
    width=width,
    height=height,
    params=np.array([fx, fy, cx, cy])
)

# 5. 构造 Image 对象列表
from scipy.spatial.transform import Rotation as R
images = []
for i, pose in enumerate(poses):
    # COLMAP 需要 world-to-cam，假设pose是cam-to-world，则需逆
    w2c = np.linalg.inv(pose)
    Rmat = w2c[:3, :3]
    tvec = w2c[:3, 3]
    qvec = R.from_matrix(Rmat).as_quat()  # xyzw
    # COLMAP顺序是 qw, qx, qy, qz
    qvec = np.roll(qvec, 1)
    images.append(
        Image(
            id=i+1,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_names[i],
            xys=np.zeros((0, 2)),  # 没有点
            point3D_ids=np.array([], dtype=int)
        )
    )

# 6. 写入bin文件
os.makedirs('/home/woshihg/PycharmProjects/Difix3D/dataset/TEST/colmap/sparse/0', exist_ok=True)
write_cameras_binary({camera_id: camera}, '/home/woshihg/PycharmProjects/Difix3D/dataset/TEST/colmap/sparse/0/cameras.bin')
write_images_binary({im.id: im for im in images}, '/home/woshihg/PycharmProjects/Difix3D/dataset/TEST/colmap/sparse/0/images.bin')
