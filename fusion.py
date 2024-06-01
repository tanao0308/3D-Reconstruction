# -*- coding: utf-8 -*-

import numpy as np

from numba import njit, prange
from skimage import measure


class TSDFVolume:
    """Volumetric TSDF Fusion.
    """
    def __init__(self, vol_bnds, voxel_size):
        """Constructor.

        Args:
            vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
                xyz bounds (min/max) in meters.
            voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self.trunc_margin = 5 * self._voxel_size    # truncation on SDF

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
        self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
        self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)

        print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
            self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
            self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
        )

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol = np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol = np.zeros(self._vol_dim).astype(np.float32)

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._vol_dim[0]),
            range(self._vol_dim[1]),
            range(self._vol_dim[2]),
            indexing='ij'
        )
        self.vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
        ], axis=0).astype(int).T

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
        """Convert camera coordinates to pixel coordinates.
        """
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
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume.
        """
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """
        将一个深度图像帧集成到 TSDF 体积中。
    
        参数:
		    depth_im (ndarray): 深度图像，形状为 (H, W)。
            cam_intr (ndarray): 相机内参矩阵，形状为 (3, 3)。
            cam_pose (ndarray): 相机位姿（即外参），形状为 (4, 4)。
            obs_weight (float): 当前观测的权重。较高的值表示较高的权重。
        """
        cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size) # 将体素网格坐标转换为世界坐标
        print(cam_pts.shape)
        cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))  # 将点从世界坐标系转换到相机坐标系
        pix_z = cam_pts[:, 2]  # 提取 z 坐标
        pix = self.cam2pix(cam_pts, cam_intr)  # 将相机坐标系下的点投影到像素坐标系
        pix_x, pix_y = pix[:, 0], pix[:, 1]  # 分别提取 x 和 y 坐标

        im_h, im_w = depth_im.shape  # 获取深度图像的高度和宽度
 
        # 剔除视锥体外的像素
        valid_pix = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < im_w,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < im_h,
                                   pix_z > 0))))

        depth_val = np.zeros(pix_x.shape)  # 初始化深度值数组
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]  # 获取有效像素的深度值
    
        # 集成 TSDF
        depth_diff = depth_val - pix_z  # 计算深度差
        valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self.trunc_margin)  # 筛选出有效点
        dist = np.minimum(1, depth_diff / self.trunc_margin)  # 计算 TSDF 距离值
        valid_vox_x = self.vox_coords[valid_pts, 0]  # 获取有效体素的 x 坐标
        valid_vox_y = self.vox_coords[valid_pts, 1]  # 获取有效体素的 y 坐标
        valid_vox_z = self.vox_coords[valid_pts, 2]  # 获取有效体素的 z 坐标
        w_old = self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]  # 获取旧的权重值
        tsdf_vals = self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]  # 获取旧的 TSDF 值
        valid_dist = dist[valid_pts]  # 获取有效的 TSDF 距离值
        tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)  # 计算新的 TSDF 值和权重
        self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new  # 更新权重值
        self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new  # 更新 TSDF 值
    
    def get_volume(self):
        return self._tsdf_vol

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume.
        """
        tsdf_vol = self.get_volume()

        # Marching cubes
        verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts*self._voxel_size + self._vol_origin

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes.
        """
        tsdf_vol = self.get_volume()

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts*self._voxel_size+self._vol_origin    # voxel grid coordinates to world coordinates

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
    """Save a 3D mesh to a polygon .ply file.
    """
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

