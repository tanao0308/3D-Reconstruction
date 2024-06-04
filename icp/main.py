# -*- coding: utf-8 -*-

from depth2cloud import depth_to_point_cloud
from icp import icp, animate_icp
import cv2
import numpy as np

d0, d1 = 0, 20
depth_map, point_cloud = [0, 0], [0, 0]
depth_map[0] = cv2.imread("../data/frame-%06d.depth.png"%(d0), -1).astype(float) 
depth_map[1] = cv2.imread("../data/frame-%06d.depth.png"%(d1), -1).astype(float) 

# 相机内参
intr = np.loadtxt("../data/camera-intrinsics.txt", delimiter=' ')  # 读取相机内参矩阵
fx, fy = intr[0, 0], intr[1, 1]
cx, cy = intr[0, 2], intr[1, 2]
 
# 将深度图转换为点云
point_cloud[0] = depth_to_point_cloud(depth_map[0], fx, fy, cx, cy)
point_cloud[1] = depth_to_point_cloud(depth_map[1], fx, fy, cx, cy)


print("finish creating clouds.")
# 打印点云信息
# print("点云形状:", point_cloud.shape)
# print("前5个点云点:\n", point_cloud[:5])

R, T, transformed_points_list = icp(point_cloud[0], point_cloud[1], 10)
animate_icp(point_cloud[0], point_cloud[1], transformed_points_list)


