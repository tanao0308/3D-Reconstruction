import cv2
import numpy as np

# 读取相机内参矩阵
K = np.loadtxt("camera-intrinsics.txt", delimiter=' ')

# 读取深度图像
depth_image = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
depth_image /= 1000.0  # 如果深度图是以毫米为单位的，需要转换为米

# 逆相机内参矩阵
K_inv = np.linalg.inv(K)

# 获取图像的尺寸
height, width = depth_image.shape

# 创建网格
u, v = np.meshgrid(np.arange(width), np.arange(height))

# 将图像坐标转换为齐次坐标
u_homogeneous = np.stack((u, v, np.ones_like(u)), axis=-1)  # 形状为 (height, width, 3)

# 计算相机坐标
camera_space_points = depth_image[..., np.newaxis] * (u_homogeneous @ K_inv.T)  # 形状为 (height, width, 3)

# 打印一些三维点用于验证
print(camera_space_points[100, 100, :])  # 打印 (100, 100) 位置的三维点

