""" Classes and functions for 2D to 3D registration """

import numpy as np
import cv2
import cv2.aruco as aruco # pylint: disable=import-error

from sksurgerycore.algorithms.tracking_smoothing import RollingMean, \
                RollingMeanRotation, quaternion_to_matrix

from sksurgerycore.transforms.matrix import (construct_rotm_from_euler,
                                             construct_rigid_transformation,
                                             )

class Registration2D3D():
    """
    Performs registration of sets of 2D points to 3D points.
    """

    def __init__(self, three_d_points, projection_matrix, distortion,
                 buffer_size=1):
        """
        Initialises the registration class.

        :params three_d_points: a 16 x n array of 3D tag coordinates.
            First column is the tag ID. Next 3 columns are the
            tag centre coordinates. The last 12 columns are the
            3D coordinates of the 4 tag corners. If None or np.array(None)
            will init ok, but all calls to get_matrix will return false and
            np.identity
        :params projection_matrix: a camera projection matrix
        :params distortion: a camera distortion matrix
        :params buffer_size: calculate the registration using the average
                of the last buffer_size frames, must not be less than 1

        :raises: ValueError if three_d_points is not nx16
        """
        self._no_points = False
        if three_d_points is None or three_d_points.any() is None:
            self._no_points = True
            return

        if three_d_points.shape[1] != 16:
            raise ValueError("Three_d_points should have 16 columns")

        self._three_d_points = three_d_points
        self._projection_matrix = projection_matrix
        self._distortion = distortion

        self._rvec_rolling_mean = RollingMeanRotation(buffer_size)
        self._tvec_rolling_mean = RollingMean(3, buffer_size)

    def get_matrix(self, marker_ids, marker_corners):
        """
        Calculate and return a tracking matrix (4x4)

        :params marker_ids: an array of n markers ids corresponding to
        :params marker_corners: an nx8 array of marker corners

        :return: success (boolean) and a tracking matrix (id if failed)
        """

        output_matrix = np.identity(4, dtype=np.float64)
        if self._no_points:
            return False, output_matrix

        rvec, tvec = self._register(marker_ids, marker_corners)
        self._rvec_rolling_mean.pop(rvec)
        self._tvec_rolling_mean.pop(tvec)

        mean_rvec = self._rvec_rolling_mean.getmean()
        mean_tvec = self._tvec_rolling_mean.getmean()

        if np.isnan(mean_rvec).any() or np.isnan(mean_tvec).any():
            return False, output_matrix

        output_matrix[0:3, 0:3] = quaternion_to_matrix(mean_rvec)

        output_matrix[0:3, 3] = mean_tvec
        return True, output_matrix

    def _register(self, marker_ids, marker_corners):
        """
        Performs the 2D to 3D registration.

        :params marker_ids: an array of n markers ids corresponding to
        :params marker_corners: an nx8 array of marker corners

        :return: rotation and translation vectors, NaN if failed.
        """

        points3d = []
        points2d = []
        count = 0

        rvec = np.full(3, np.NaN)
        tvec = np.full(3, np.NaN)

        for index, identities in enumerate(marker_ids):
            for three_d_point in self._three_d_points:
                if identities[0] == three_d_point[0]:
                    count += 1
                    points3d.extend(three_d_point[4:])
                    points2d.extend(marker_corners[index])
                    break

        if count > 0:
            points3d = np.array(points3d).reshape((count*4), 3)
            points2d = np.array(points2d).reshape((count*4), 2)

            _, rvec, tvec = cv2.solvePnP(points3d, points2d,
                                         self._projection_matrix,
                                         self._distortion)
        return rvec, tvec

def _marker_size(marker_points):
    """
    Internal function to calculate the size of a marker
    :param marker_points: 2d or 3d points that make up the marker,
        1 row per point

    :return: The size of the marker
    """
    maxs = np.max(marker_points, axis = 0)
    mins = np.min(marker_points, axis = 0)
    size = np.linalg.norm(maxs - mins)
    return size

def estimate_poses_no_calibration(marker_corners):
    """
    Returns tracking data for a camera with no calibration data.
    x and y are the screen pixel coordinates.
    z is based on the size of the tag in pixels, there is no
    rotation. No account is taken of the size of the
    model marker pattern, so it will be bit flakey.
    """
    tracking = np.full((4,4), np.nan, dtype=np.float32)
    quality = 0.0

    x_means = []
    y_means = []
    sizes = []

    if len(marker_corners) > 0:
        for marker in marker_corners:

            means = np.mean(marker[0], axis=0)
            x_means.append(means[0])
            y_means.append(means[1])
            sizes.append(_marker_size(marker[0]))

        x_mean = np.mean(x_means)
        y_mean = np.mean(y_means)
        size = np.mean(sizes)

        tracking = np.array([[1.0, 0.0, 0.0, x_mean],
                             [0.0, 1.0, 0.0, y_mean],
                             [0.0, 0.0, 1.0, -size],
                             [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        quality = 1.0

    return tracking, quality

def estimate_poses_with_calibration(marker_corners2d, marker_ids,
                board, camera_projection_matrix, camera_distortion):
    """
    Estimate the pose of a single tag or a multi-tag rigid body
    when the camera calibration is known.
    :param marker_corners2d: a list of 2d marker corners, 1 row per tag,
        8 columns per tag
    :param model_points: Matched list of of corresponding model points,
        1 row per tag, 15 columns per tag: corner points and centre point
    :param camera_projection_matrix: a 3x3 camera projection matrix
    :param camera_distortion: camera distortion vector

    :return : a tracking matrix and a quality
    """

    if len(marker_corners2d) == 1:
        marker_width = board.objPoints[0][1][0] - board.objPoints[0][0][0]
        rvecs, tvecs, _ = \
            aruco.estimatePoseSingleMarkers(marker_corners2d, marker_width,
                                            camera_projection_matrix,
                                            camera_distortion)
        tracking = []
        t_index = 0
        for rvec in rvecs:
            rot_mat = construct_rotm_from_euler(rvec[0][0],rvec[0][1],
                                                rvec[0][2], 'xyz',
                                                is_in_radians=True)
            tracking.append(construct_rigid_transformation(rot_mat,
                                                       tvecs[t_index][0]))
            t_index += 1
        return tracking, 1.0

    print(marker_ids)
    raise NotImplementedError
    #
    #rvec = np.empty((1,3), dtype = np.float32)
    #tvec = np.empty((1,3), dtype = np.float32)
    #rvecs, tvecs, _ = \
    #    aruco.estimatePoseBoard(marker_corners2d, np.array([marker_ids]),
    #                                board,
    #                                camera_projection_matrix,
    #                                camera_distortion,
    #                                rvec,
    #                                tvec)
    #tracking = []
    #t_index = 0
    #print ("\nRVECS = ", rvecs)
    #print ("\nRVEC = ", rvec)
    #print ("\nTVEC = ", tvec)
    #for rvec in rvecs:
    #rot_mat = construct_rotm_from_euler(rvec[0],rvec[1],rvec[2], 'xyz',
    #                                      is_in_radians=True)
    #tracking.append(construct_rigid_transformation(rot_mat,
    #                                               tvec))
    #return tracking, 1.0
