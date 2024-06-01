# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
import fusion

data = "data"

if __name__ == "__main__":
    print("Estimating voxel volume bounds...")
    n_imgs = 10  # 要处理的图像数量
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')  # 读取相机内参矩阵
    vol_bnds = np.zeros((3,2))  # 初始化体素体积边界（3维，分别是x, y, z的最小值和最大值）
    
    for i in range(n_imgs):
        # 读取深度图像和相机位姿
        depth_im = cv2.imread(data+"/frame-%06d.depth.png"%(i), -1).astype(float)  # 读取深度图像，并转换为浮点型
        depth_im /= 1000.  # 深度图像以16位PNG保存，单位是毫米，这里转换为米
        depth_im[depth_im == 65.535] = 0  # 将无效深度值设置为0（特定于7-scenes数据集）
        cam_pose = np.loadtxt(data+"/frame-%06d.pose.txt"%(i))  # 读取相机位姿（4x4刚体变换矩阵）
    
        # 计算相机视锥并扩展凸包
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)  # 获取视锥体顶点
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))  # 更新最小边界
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))  # 更新最大边界
    
    # 初始化体素体积
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)  # 初始化TSDF体积，体素大小为0.02米
    
    # 融合TSDF
    for i in range(n_imgs):
        print("Fusing frame %d/%d"%(i+1, n_imgs))  # 打印当前帧信息
    
        depth_im = cv2.imread(data+"/frame-%06d.depth.png"%(i), -1).astype(float) # 读取深度图像
        depth_im[depth_im == 65535] = 0  # 将无效深度值设置为0
        depth_im /= 1000.  # 深度图像单位转换为米
        cam_pose = np.loadtxt(data+"/frame-%06d.pose.txt"%(i))  # 读取相机位姿
    
        tsdf_vol.integrate(depth_im, cam_intr, cam_pose, obs_weight=1.)  # 融合
    
    # 从体素体积获取网格并保存到磁盘（可用Meshlab查看）
    print("Saving mesh to mesh.ply...")
    verts, faces, norms = tsdf_vol.get_mesh()  # 获取网格数据
    fusion.meshwrite("mesh.ply", verts, faces, norms)  # 保存网格到文件
    
