# -*- coding: utf-8 -*-

import numpy as np

from numba import njit, prange
from skimage import measure


class TSDFVolume:
    """Volumetric TSDF Fusion.
    """
    def __init__(self, vol_bnds, voxel_size):
        """
        vol_bnds (ndarray): 一个形状为 (3, 2) 的数组，指定了 xyz 方向的边界（最小值/最大值），单位为米。
        voxel_size (float): 体素的尺寸，单位为米。
        """
        vol_bnds = np.asarray(vol_bnds)
    
        # 定义体素网格参数
        self.vol_bnds = np.asarray(vol_bnds)
        self.voxel_size = float(voxel_size)
        self.trunc_margin = 5 * self.voxel_size  # SDF 的截断边界，将dis/trunc_margin > 1的截断为1
    
        # 调整体素网格的边界并确保数据是 C 语言的内存布局（行优先）
        self.vol_dim = np.ceil((self.vol_bnds[:,1] - self.vol_bnds[:,0]) / self.voxel_size).copy(order='C').astype(int) # 体素网格长宽高
        self.vol_bnds[:,1] = self.vol_bnds[:,0] + self.vol_dim * self.voxel_size # 更新体素网格边界
        self.vol_origin = self.vol_bnds[:,0].copy(order='C').astype(np.float32) # 原点的世界坐标
    
        print("Voxel grid size: {} x {} x {} - Number of points: {:,}".format(
            self.vol_dim[0], self.vol_dim[1], self.vol_dim[2],
            self.vol_dim[0] * self.vol_dim[1] * self.vol_dim[2])
        )
    
        self.tsdfvol = np.ones(self.vol_dim).astype(np.float32) #  TSDF 值
        self._weightvol = np.zeros(self.vol_dim).astype(np.float32) # 权重值
    
        xv, yv, zv = np.meshgrid(
            range(self.vol_dim[0]),
            range(self.vol_dim[1]),
            range(self.vol_dim[2]),
            indexing='ij'
        ) # 三个长宽高三维矩阵，xv存储每个体素x轴int坐标
        self.vox_coords = np.concatenate([
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            zv.reshape(1, -1)
        ], axis=0).astype(int).T # 长度为长宽高相乘的三维点集
    
    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        # 将体素网格坐标转换为世界坐标。
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    
        # 使用并行循环将每个体素网格坐标转换为世界坐标
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
		# 将相机坐标转换为像素坐标。
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    @staticmethod
    @njit(parallel=True)
    def integratetsdf(tsdfvol, dist, w_old, obs_weight):
		# 融合TSDF值
        tsdfvol_int = np.empty_like(tsdfvol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdfvol)):
            w_new[i] = w_old[i] + obs_weight
            tsdfvol_int[i] = (w_old[i] * tsdfvol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdfvol_int, w_new

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """
        将一个深度图像帧集成到 TSDF 体积中。
    
        参数:
		    depth_im (ndarray): 深度图像，形状为 (H, W)。
            cam_intr (ndarray): 相机内参矩阵，形状为 (3, 3)。
            cam_pose (ndarray): 相机位姿（即外参），形状为 (4, 4)。
            obs_weight (float): 当前观测的权重。较高的值表示较高的权重。
        """
        cam_pts = self.vox2world(self.vol_origin, self.vox_coords, self.voxel_size) # 将体素网格坐标转换为世界坐标
        # print(cam_pts.shape) (n, 3)
        cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))  # 将点从世界坐标系转换到相机坐标系

        pix_z = cam_pts[:, 2]  # 提取 z 坐标
        pix = self.cam2pix(cam_pts, cam_intr)  # 将相机坐标系下的点投影到像素坐标系
        pix_x, pix_y = pix[:, 0], pix[:, 1]  # 分别提取 x 和 y 像素坐标
        im_h, im_w = depth_im.shape  # 获取深度图像的高度和宽度
        # 剔除视锥体外的像素
        valid_pix = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < im_w,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < im_h,
                                   pix_z > 0))))

        depth_val = np.zeros(pix_x.shape)  # 初始化深度值数组
        # 布尔数组在索引位置，用于指示哪些像素位置是有效的
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]  # 获取有效像素的深度值

        # 集成 TSDF
        depth_diff = depth_val - pix_z  # 计算深度差 depth_val: 图像中像素的深度信息，pix_z: 点的深度信息
        valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self.trunc_margin)  # 筛选出有效点
        dist = np.minimum(1, depth_diff / self.trunc_margin)  # 计算 TSDF 距离值
        valid_vox_x = self.vox_coords[valid_pts, 0]  # 获取有效体素的 x 坐标
        valid_vox_y = self.vox_coords[valid_pts, 1]  # 获取有效体素的 y 坐标
        valid_vox_z = self.vox_coords[valid_pts, 2]  # 获取有效体素的 z 坐标
        w_old = self._weightvol[valid_vox_x, valid_vox_y, valid_vox_z]  # 获取旧的权重值
        tsdf_vals = self.tsdfvol[valid_vox_x, valid_vox_y, valid_vox_z]  # 获取旧的 TSDF 值
        valid_dist = dist[valid_pts]  # 获取有效的 TSDF 距离值
        tsdfvol_new, w_new = self.integratetsdf(tsdf_vals, valid_dist, w_old, obs_weight)  # 计算新的 TSDF 值和权重
        self._weightvol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new  # 更新权重值
        self.tsdfvol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdfvol_new  # 更新 TSDF 值
    
    def getvolume(self):
        return self.tsdfvol

    def get_mesh(self):
        """
		tsdfvol：三维体数据，通常为TSDF（有符号距离函数）体素数据。
		level：指定提取等值面的值。
		verts：提取的表面网格的顶点坐标。
		faces：网格的面，由顶点索引组成的三角形列表。
		norms：每个顶点的法向量，用于渲染和光照计算。
		vals：每个顶点的等值。
        """
        tsdfvol = self.getvolume()
        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdfvol, level=0)
        verts = verts*self.voxel_size+self.vol_origin    # voxel grid coordinates to world coordinates

        return verts, faces, norms


def rigid_transform(xyz, transform):
    # 对一个形状为 (N, 3) 的点云应用刚性变换。
    
    # 将点云转换为齐次坐标，将每个点 [x, y, z] 转换为 [x, y, z, 1]
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    
    # 返回变换后的点云，去掉齐次坐标的最后一维
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([
        (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
        (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
        np.array([0,max_depth,max_depth,max_depth,max_depth])
    ]) # 五个角点的坐标
    # 将视锥体角点从相机坐标系转换到世界坐标系
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def meshwrite(filename, verts, faces, norms):
    # 将网格写成.ply格式
    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n"%(faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
            verts[i,0], verts[i,1], verts[i,2],
            norms[i,0], norms[i,1], norms[i,2],
            200, 200, 200
    ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

    ply_file.close()

