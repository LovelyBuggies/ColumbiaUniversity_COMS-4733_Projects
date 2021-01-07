# Stencil code based on that of Andy Zeng

import numpy as np

from numba import njit, prange
from skimage import measure
from transforms import *

class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """

    def __init__(self, volume_bounds, voxel_size):
        """Initialize tsdf volume instance variables.

        Args:
            volume_bounds (numpy.array [3, 2]): rows index [x, y, z] and cols index [min_bound, max_bound].
                Note: units are in meters.
            voxel_size (float): The side length of each voxel in meters.

        Raises:
            ValueError: If volume bounds are not the correct shape.
            ValueError: If voxel size is not positive.
        """
        volume_bounds = np.asarray(volume_bounds)
        if volume_bounds.shape != (3, 2):
            raise ValueError('volume_bounds should be of shape (3, 2).')

        if voxel_size <= 0.0:
            raise ValueError('voxel size must be positive.')

        # Define voxel volume parameters
        self._volume_bounds = volume_bounds
        self._voxel_size = float(voxel_size)
        self._truncation_margin = 2 * self._voxel_size  # truncation on SDF (max alowable distance away from a surface)

        # Adjust volume bounds and ensure C-order contiguous and calculate voxel bounds taking the voxel size into consideration
        self._voxel_bounds = np.ceil((self._volume_bounds[:,1]-self._volume_bounds[:,0])/self._voxel_size).copy(order='C').astype(int)
        self._volume_bounds[:,1] = self._volume_bounds[:,0]+self._voxel_bounds*self._voxel_size

        # volume min bound is the origin of the volume in world coordinates
        self._volume_origin = self._volume_bounds[:,0].copy(order='C').astype(np.float32)

        print('Voxel volume size: {} x {} x {} - # voxels: {:,}'.format(
            self._voxel_bounds[0],
            self._voxel_bounds[1],
            self._voxel_bounds[2],
            self._voxel_bounds[0]*self._voxel_bounds[1]*self._voxel_bounds[2]))

        # Initialize pointers to voxel volume in memory
        self._tsdf_volume = np.ones(self._voxel_bounds).astype(np.float32)

        # for computing the cumulative moving average of observations per voxel
        self._weight_volume = np.zeros(self._voxel_bounds).astype(np.float32)
        color_bounds = np.append(self._voxel_bounds, 3)
        self._color_volume = np.zeros(color_bounds).astype(np.float32)  # rgb order

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
                range(self._voxel_bounds[0]),
                range(self._voxel_bounds[1]),
                range(self._voxel_bounds[2]),
                indexing='ij')
        self._voxel_coords = np.concatenate([
                xv.reshape(1, -1),
                yv.reshape(1, -1),
                zv.reshape(1, -1)], axis=0).astype(int).T

    @staticmethod
    @njit(parallel=True)
    def voxel_to_world(volume_origin, voxel_coords, voxel_size):
        """Convert from voxel coordinates to world coordinates
            (in effect scaling voxel_coords by voxel_size).

        Args:
            volume_origin (numpy.array [3, ]): The origin of the voxel grid in world coordinate space.
            voxel_coords (numpy.array [n, 3]): Each row gives the 3D coordinates of a voxel.
            voxel_size (float): The side length of each voxel in meters.

        Returns:
            numpy.array [n, 3]: World coordinate representation of each of the n 3D points.
        """
        volume_origin = volume_origin.astype(np.float32)
        voxel_coords = voxel_coords.astype(np.float32)
        world_points = np.empty_like(voxel_coords, dtype=np.float32)

        # NOTE: prange is used instead of range(...) to take advantage of parallelism.
        for i in prange(voxel_coords.shape[0]):
            for j in range(3):
                # TODO: compute world_points
                world_points[i][j] = voxel_coords[i][j] * voxel_size + volume_origin[j]
        return world_points

    @staticmethod
    @njit(parallel=True)
    def integrate_volume_helper(tsdf_old, dist, w_old, observation_weight):
        """[summary]

        Args:
            tsdf_old (numpy.array [v, ]): v is equal to the number of voxels to be
                integrated at this timestep. Old tsdf values that need to be
                updated based on the current observation.
            margin_distance (float): The tsdf trancation margin.
            w_old (numpy.array [v, ]): old weight values.
            observation_weight (float): Weight to give each new observation.

        Returns:
            numpy.array [v, ]: new tsdf values for entries in tsdf_old
            numpy.array [v, ]: new weights to be used in the future.
        """
        # tsdf_new = np.empty_like(tsdf_old, dtype=np.float32)
        # w_new = np.empty_like(w_old, dtype=np.float32)
        # for i in prange(len(tsdf_old)):
        #     # TODO: compute tsdf_new and w_new
        #     w_new[i] = w_old[i] + observation_weight
        #     tsdf_new[i] = (w_old[i] * tsdf_old[i] + observation_weight * dist[i]) / w_new[i]
        w_new = w_old + observation_weight
        tsdf_new = (w_old * tsdf_old + observation_weight * dist) / w_new
        return tsdf_new, w_new

    def integrate(self, color_image, depth_image, camera_intrinsic, camera_pose, observation_weight=1.):
        """Integrate an RGB-D observation into the TSDF volume, by updating the weight volume,
            tsdf volume, and color volume.

        Args:
            color_image (numpy.array [h, w, 3]): An rgb image.
            depth_image (numpy.array [h, w]): A z depth image.
            camera_intrinsic (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
            camera_pose (numpy.array [4, 4]): SE3 transform representing pose (camera to world)
            observation_weight (float, optional):  The weight to assign for the current
                observation. Defaults to 1.
        """
        image_height, image_width = depth_image.shape
        color_image = color_image.astype(np.float32)
        # TODO: Convert voxel grid coordinates to world points
        cam_pts = self.voxel_to_world(self._volume_origin, self._voxel_coords, self._voxel_size)
        # TODO: Transform points in the volume to the camera coordinate system. Get voxel_z depth and u, v projections.
        cam_pts = transform_point3s(np.linalg.inv(camera_pose), cam_pts)
        # cam_pts = transform_point3s(cam_pts, camera_pose)
        pix_z = cam_pts[:, 2]
        pix = camera_to_image(camera_intrinsic, cam_pts)
        pix_x, pix_y = np.asarray(pix[:, 0], dtype=np.int), np.asarray(pix[:, 1], dtype=np.int)
        # TODO: Get valid pixels by eliminating pixels not in the image bounds, etc.
        valid_pix = np.logical_and(pix_x >= 0,
                                   np.logical_and(pix_x < image_width,
                                                  np.logical_and(pix_y >= 0,
                                                                 np.logical_and(pix_y < image_height, pix_z > 0))))
        # TODO: Get depths for valid coordinates u, v from the depth image.
        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_image[pix_y[valid_pix], pix_x[valid_pix]]
        # Integrate TSDF
        # TODO: Filter out zero depth values and cases where observed depth + truncation margin >= voxel_z
        # TODO: Truncate and normalize
        depth_diff = depth_val - pix_z  # coarse SDF
        valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._truncation_margin)
        dist = np.minimum(1, depth_diff / self._truncation_margin)
        valid_vox_x = self._voxel_coords[valid_pts, 0]
        valid_vox_y = self._voxel_coords[valid_pts, 1]
        valid_vox_z = self._voxel_coords[valid_pts, 2]
        w_old = self._weight_volume[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_vals = self._tsdf_volume[valid_vox_x, valid_vox_y, valid_vox_z]
        valid_dist = dist[valid_pts]
        # TODO: Find new weight volume and tsdf volume (hint: call helper).
        tsdf_vol_new, w_new = self.integrate_volume_helper(tsdf_vals, valid_dist, w_old, observation_weight)
        self._weight_volume[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
        self._tsdf_volume[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

        # TODO: Integrate color using old and new weights.
        old_color = self._color_volume[valid_vox_x, valid_vox_y, valid_vox_z]
        new_color = color_image[pix_y[valid_pts], pix_x[valid_pts]]
        # for i in prange(len(new_color)):  #  avoid using loop
        #     new_color[i] = np.minimum(255., (w_old[i] * old_color[i] + observation_weight * new_color[i]) / w_new[i])
        #
        # print(np.multiply(w_old, old_color))
        w_old = w_old.reshape((-1, 1))
        w_new = w_new.reshape((-1, 1))
        new_color = np.minimum(255., np.round((np.multiply(w_old, old_color) + observation_weight * new_color) / w_new))
        self._color_volume[valid_vox_x, valid_vox_y, valid_vox_z] = new_color
    def get_volume(self):
        """Get the tsdf and color volumes.

        Returns:
            numpy.array [l, w, h]: l, w, h are the dimensions of the voxel grid in voxel space.
                Each entry contains the integrated tsdf value.
            numpy.array [l, w, h, 3]: l, w, h are the dimensions of the voxel grid in voxel space.
                3 is the channel number in the order r, g, then b.
        """
        return self._tsdf_volume, self._color_volume

    def get_mesh(self):
        """ Run marching cubes over the constructed tsdf volume to get a mesh representation.

        Returns:
            numpy.array [n, 3]: each row represents a 3D point.
            numpy.array [k, 3]: each row is a list of point indices used to render triangles.
            numpy.array [n, 3]: each row represents the normal vector for the corresponding 3D point.
            numpy.array [n, 3]: each row represents the color of the corresponding 3D point.
        """
        tsdf_volume, color_vol = self.get_volume()
        # print(tsdf_volume.shape)  # [150, 150, 80]

        # Marching cubes
        voxel_points, triangles, normals, _ = measure.marching_cubes_lewiner(tsdf_volume, level=0)
        points_ind = np.round(voxel_points).astype(int)
        points = self.voxel_to_world(self._volume_origin, voxel_points, self._voxel_size)

        # Get vertex colors.
        rgb_vals = color_vol[points_ind[:,0], points_ind[:,1], points_ind[:,2]]
        colors_r = rgb_vals[:, 0]
        colors_g = rgb_vals[:, 1]
        colors_b = rgb_vals[:, 2]
        colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
        colors = colors.astype(np.uint8)

        return points, triangles, normals, colors
