# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
from scipy.spatial import KDTree

def icp(source_points, target_points, max_iterations=10, tolerance=1e-6):
    """
    实现ICP算法以对齐source_points到target_points，并记录每次迭代的点云变换。

    参数:
    source_points (numpy.ndarray): 需要对齐的源点云
    target_points (numpy.ndarray): 目标点云
    max_iterations (int): 最大迭代次数
    tolerance (float): 迭代停止的误差阈值

    返回:
    R (numpy.ndarray): 旋转矩阵
    T (numpy.ndarray): 平移向量
    tran_points_list (list): 每次迭代后的变换点云列表
    """
    # 初始化旋转矩阵R为单位矩阵
    R = np.eye(3)
    # 初始化平移向量T为零向量
    T = np.zeros(3)
    # 使用目标点云构建KD树以便快速查找最近邻
    target_kdtree = KDTree(target_points)

    # 用于存储每次迭代后变换后的点云
    tran_points_list = []

    for iteration in range(max_iterations):
        print("iterating", iteration)
        # 将源点云应用当前的旋转和平移变换
        pre_points = np.dot(source_points, R.T) + T
        tran_points_list.append(pre_points)

        # 查找变换后的点云中每个点的最近邻目标点
        dis, idx = target_kdtree.query(pre_points)
        fut_points = target_points[idx] # 未来的目标点云

        # 计算误差（变换后点到目标点的平均距离）
        error = np.mean(dis)
        # 如果误差小于阈值，停止迭代
        if error < tolerance:
            break

        # 计算源点和目标点的质心
        centroid_source = np.mean(pre_points, axis=0)
        centroid_target = np.mean(fut_points, axis=0)

        # 将点云转换到质心坐标系
        source_centered = pre_points - centroid_source
        target_centered = fut_points - centroid_target

        # 计算相关矩阵H
        H = np.dot(source_centered.T, target_centered)
        # 通过奇异值分解（SVD）计算最佳旋转矩阵
        U, S, Vt = np.linalg.svd(H)
        R_iter = np.dot(Vt.T, U.T)

        # 确保旋转矩阵的正定性
        if np.linalg.det(R_iter) < 0:
            Vt[2, :] *= -1
            R_iter = np.dot(Vt.T, U.T)

        # 计算最佳平移向量
        T_iter = centroid_target - np.dot(centroid_source, R_iter.T)

        # 更新总的旋转矩阵和平移向量
        R = np.dot(R_iter, R)
        T = np.dot(R_iter, T) + T_iter

    # 返回最终的旋转矩阵、平移向量和每次迭代的变换点云列表
    return R, T, tran_points_list

def animate_icp(source_points, target_points, tran_points_list):
    """
    使用Matplotlib生成ICP算法每次迭代的点云形状动画。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='r', label='Target')
    scatter_source = ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='b', label='Source')

    def update(frame):
        ax.cla()
        ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='r', label='Target')
        pre_points = tran_points_list[frame]
        scatter_source = ax.scatter(pre_points[:, 0], pre_points[:, 1], pre_points[:, 2], c='b', label='Source')
        ax.legend()
        ax.set_title('Iteration: {}'.format(frame + 1))
        return scatter_source,

    ani = FuncAnimation(fig, update, frames=len(tran_points_list), interval=200, blit=False)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 示例使用
    source_points = np.random.rand(100, 3)
    target_points = source_points + np.array([0.5, 0.5, 0.5])

    R, T, tran_points_list = icp(source_points, target_points)
    animate_icp(source_points, target_points, tran_points_list)

