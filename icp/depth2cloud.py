# -*- coding: utf-8 -*-

import numpy as np

def depth_to_point_cloud(depth_map, fx, fy, cx, cy, scale=1.0):
    """
    将深度图转换为点云。

    参数：
    depth_map (ndarray): 深度图，形状为(H, W)，每个值表示距离。
    fx (float): 相机在x方向的焦距。
    fy (float): 相机在y方向的焦距。
    cx (float): 相机的主点x坐标。
    cy (float): 相机的主点y坐标。
    scale (float): 深度值缩放因子，默认为1.0。

    返回：
    points (ndarray): 点云，形状为(N, 3)。
    """
    H, W = depth_map.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    
    # 计算点云的x, y, z坐标
    z = depth_map / scale
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    
    # 将x, y, z坐标组合成点云
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    points = extract(points);

    return points

def extract(array, fra = 0.005):
    """
    从给定的数组中抽取1/10的元素。

    参数:
    array (numpy.ndarray): 输入数组

    返回:
    numpy.ndarray: 抽取后的子数组
    """
    # 确定需要抽取的元素数量
    num_elements_to_extract = int(len(array) * fra)

    # 随机抽取元素
    extracted_indices = np.random.choice(len(array), num_elements_to_extract, replace=False)
    extracted_elements = array[extracted_indices]

    return extracted_elements


# 示例使用
if __name__ == "__main__":
    # 生成一个示例深度图（这里使用随机值作为示例）
    depth_map = np.random.rand(480, 640).astype(np.float32) * 5.0  # 深度范围在0到5米之间

    # 相机内参
    intr = np.loadtxt("../data/camera-intrinsics.txt", delimiter=' ')  # 读取相机内参矩阵
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
 
    # 将深度图转换为点云
    point_cloud = depth_to_point_cloud(depth_map, fx, fy, cx, cy)

    # 打印点云信息
    print("点云形状:", point_cloud.shape)
    print("前5个点云点:\n", point_cloud[:5])

