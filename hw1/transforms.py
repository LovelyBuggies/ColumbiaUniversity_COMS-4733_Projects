from numba import njit, prange
import numpy as np


def transform_is_valid(t, tolerance=1e-3):
    """Check if array is a valid transform.

    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.

    Returns:
        bool: True if array is a valid transform else False.
    """

    # whether the shape is good
    if len(t.shape) != 2 or t.shape[0] != 4 or t.shape[1] != 4:
        return False

    # whether the last row is good
    if abs(t[3, 3] - 1) > tolerance or (abs(t[3, :3]) > tolerance).any():
        return False

    # whether the matrix determinant is good
    if abs(np.linalg.det(t[:3, :3]) - 1) > tolerance:
        return False

    # whether the rotation part is good
    if (abs(np.sum(np.power(t[:3, :3], 2), axis=1) - 1) > tolerance).any() or (abs(np.sum(np.power(t[:3, :3], 2), axis=0) - 1) > tolerance).any():
        return False

    return True


def transform_concat(t1, t2):
    """[summary]

    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.

    Returns:
        numpy.array [4, 4]: t1 * t2.
    """
    if not transform_is_valid(t1):
        raise ValueError("Invalid input transform t1")
    if not transform_is_valid(t2):
        raise ValueError("Invalid input transform t2")

    return np.dot(t1, t2)


def transform_point3s(t, ps):
    """Transform 3D points from one space to another.

    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).

    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.

    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """
    if not transform_is_valid(t):
        raise ValueError("Invalid input transform t")
    if len(ps.shape) != 2 or ps.shape[1] != 3:
        raise ValueError("Invalid input points ps")

    return np.dot(ps, np.transpose(t[:3, :3])) + np.tile(np.transpose(t[:3, 3]), (ps.shape[0], 1))


def transform_inverse(t):
    """Find the inverse of the transform.

    Args:
        t (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: If t is not a valid transform.

    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """
    if not transform_is_valid(t):
        raise ValueError("Invalid input transform t")
    t_inv = np.zeros((4, 4))
    t_inv[:3, :3] = np.linalg.inv(t[:3, :3])
    t_inv[:3, 3] = -t[:3, 3]
    t_inv[3, 3] = 1
    return t_inv


@njit(parallel=True)
def camera_to_image(intrinsics, camera_points):
    """Project points in camera space to the image plane.

    Args:
        intrinsics (numpy.array [3, 3]): Pinhole intrinsics.
        camera_points (numpy.array [n, 3]): n 3D points (x, y, z) in camera coordinates.

    Raises:
        ValueError: If intrinsics are not the correct shape.
        ValueError: If camera points are not the correct shape.

    Returns:
        numpy.array [n, 2]: n 2D projections of the input points on the image plane.
    """
    if intrinsics.shape != (3, 3):
        raise ValueError("Invalid input intrinsics")
    if len(camera_points.shape) != 2 or camera_points.shape[1] != 3:
        raise ValueError("Invalid camera point")

    res = np.transpose(np.dot(intrinsics, np.transpose(camera_points)))
    for i in range(len(res)):
        res[i] /= res[i][2]

    return np.rint(res[:, :2])


def depth_to_point_cloud(intrinsics, depth_image):
    """Back project a depth image to a point cloud.
        Note: point clouds are unordered, so any permutation of points in the list is acceptable.
        Note: Only output those points whose depth > 0.

    Args:
        intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
        depth_image (numpy.array [h, w]): each entry is a z depth value.

    Returns:
        numpy.array [n, 3]: each row represents a different valid 3D point.
    """
    dummy = list()
    for i in range(len(depth_image)):
        for j in range(len(depth_image[i])):
            if depth_image[i, j] != 0:
                dummy.append(np.array([j, i, 1]) * depth_image[i, j])

    res = np.transpose(np.array(dummy))
    res = np.dot(np.linalg.inv(intrinsics), res)
    return np.transpose(res)
