import numpy as np

points3D = []
point3D_ids = []
point3D_colors = []
point3D_id_to_point3D_idx = dict()
point3D_id_to_images = dict()
point3D_errors = []

input_file = 'points3D.txt'  # 替换为你的 points3D.txt 文件路径
with open(input_file, 'r') as f:
    for line in iter(lambda: f.readline().strip(), ''):
        if not line or line.startswith('#'):
            continue

        data = line.split()
        point3D_id = np.uint64(data[0])

        point3D_ids.append(point3D_id)
        point3D_id_to_point3D_idx[point3D_id] = len(points3D)
        points3D.append(map(np.float64, data[1:4]))
        point3D_colors.append(map(np.uint8, data[4:7]))
        point3D_errors.append(np.float64(data[7]))
        test = data[8:]
        print(test)
        test = map(np.uint32, data[8:])
        print(test)
        test = np.array(test)
        # print(test)

        # load (image id, point2D idx) pairs
        point3D_id_to_images[point3D_id] = \
            np.array(map(np.uint32, data[8:])).reshape(-1, 2)
