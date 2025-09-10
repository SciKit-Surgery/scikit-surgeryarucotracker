# coding=utf-8

"""scikit-surgeryarucotracker tests"""

import numpy as np
import cv2 as cv
from cv2 import aruco
from sksurgerycore.algorithms.tracking_smoothing import RollingMean, \
                RollingMeanRotation, quaternion_to_matrix

from sksurgeryarucotracker.algorithms.compare_matrices \
        import matrices_equivalent
from sksurgeryarucotracker.arucotracker import ArUcoTracker

# pylint: disable=no-member

def ccw_to_cw(ccw_points):
    """
    Converts counter clockwise points to clockwise points
    """
    cw_points = ccw_points.copy()
    cw_points[:, 4:7] = ccw_points[:, 13:16]
    cw_points[:, 7:10] = ccw_points[:, 10:13]
    cw_points[:, 10:13] = ccw_points[:, 7:10]
    cw_points[:, 13:16] = ccw_points[:, 4:7]
    return cw_points

class Registration2D3D():
    """
    Performs registration of sets of 2D points to 3D points. This
    is copied from BARD'S registration2D3D. The intent is to
    ensure that tracking behaves in the same way as BARD, which
    is reasonably well tested by users.
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

        boardshape = three_d_points.shape
        try:
            _ = boardshape[1]
        except IndexError:
            three_d_points = three_d_points.reshape(1, boardshape[0])
            boardshape=three_d_points.shape

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

        rvec = np.full(3, np.nan)
        tvec = np.full(3, np.nan)

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

            if count > 1:
                _, rvec, tvec = cv.solvePnP(points3d, points2d,
                                             self._projection_matrix,
                                             self._distortion)
            else:
                _, rvec, tvec = cv.solvePnP(points3d, points2d,
                                             self._projection_matrix,
                                             self._distortion)
        return rvec, tvec

def test_arucotracker_vs_solve_pnp():
    """
    This checks whether the tracking result using skarucotracker
    matches that when using the solvePnP approach as originally
    implemented by scikit-surgeryBARD
    """
    calib_mtx = np.array([[608.67179504, 0.00000000, 323.12263928],
                          [0.00000000, 606.13421375, 231.29247171],
                          [0.0, 0.0, 1.0]], dtype = np.float64)
    distortion = np.array([-0.02191634, -0.14300148, -0.00395124,
                           -0.00044941, 0.19656551], dtype = np.float64)
    videofile = 'data/multipattern.avi'

    config = {'video source' : 'none',
              'camera projection' : calib_mtx,
              'camera distortion' : distortion,
              'aruco dictionary' : 'DICT_ARUCO_ORIGINAL',
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      }
                      ]
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()
    capture = cv.VideoCapture(videofile)
    _, image = capture.read()
    (port_handles, _timestamps, _framenumbers,
     tracking, quality) = tracker.get_frame(image)

    assert 'reference' in port_handles

    reference_index = port_handles.index('reference')

    assert np.isclose(quality[reference_index], 0.91666666)

    ref_regression = np.array([
       [-0.85007277,  0.51807849, -0.09471516, -8.06235428e+01],
       [-0.46167221, -0.64647658,  0.60739345, 5.07177658e+01],
       [ 0.25344635,  0.56005599,  0.78873458, 2.50268387e+02],
       [ 0.0000e+00, 0.00000e+00, 0.00000e+00, 1.00000000e+00]],
       dtype=np.float32)


    aruco_reference_tracking = tracking[reference_index]
    assert matrices_equivalent(aruco_reference_tracking, ref_regression)

    tracker.stop_tracking()
    tracker.close()

    #now try again using our own implementation using cv.solvepnp as
    #formerly implemented in BARD
    three_d_points = np.loadtxt('data/reference.txt')

    reference_register = Registration2D3D(three_d_points,
                                          calib_mtx, distortion,
                                          buffer_size=1)

    marker_corners, ids, _ = aruco.detectMarkers(image,
                    aruco.getPredefinedDictionary(
                            aruco.DICT_ARUCO_ORIGINAL))

    success, modelreference2camera = \
                reference_register.get_matrix(
                    ids, marker_corners)

    assert success
    assert matrices_equivalent(modelreference2camera, ref_regression)

    capture.release()


def test_vs_solve_pnp_with_smth():
    """
    This checks whether the tracking result using skarucotracker
    matches that when using the solvePnP approach as originally
    implemented by scikit-surgeryBARAD, with the addition of
    tracking smoothing
    """
    calib_mtx = np.array([[608.67179504, 0.00000000, 323.12263928],
                          [0.00000000, 606.13421375, 231.29247171],
                          [0.0, 0.0, 1.0]], dtype = np.float64)
    distortion = np.array([-0.02191634, -0.14300148, -0.00395124,
                           -0.00044941, 0.19656551], dtype = np.float64)
    videofile = 'data/multipattern.avi'

    config = {'video source' : 'none',
              'camera projection' : calib_mtx,
              'camera distortion' : distortion,
              'aruco dictionary' : 'DICT_ARUCO_ORIGINAL',
              'smoothing buffer' : 3,
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      }
                      ]
              }

    capture = cv.VideoCapture(videofile)
    #set up skarucotracker
    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    #set up solvepnp
    three_d_points = np.loadtxt('data/reference.txt')

    reference_register = Registration2D3D(three_d_points,
                                          calib_mtx, distortion,
                                          buffer_size=3)

    #first frame
    for _frame in range(10):
        _, image = capture.read()
        (port_handles, _timestamps, _framenumbers,
        tracking, _quality) = tracker.get_frame(image)

        assert 'reference' in port_handles
        reference_index = port_handles.index('reference')

        aruco_reference_tracking = tracking[reference_index]


        #now try again using our own implementation using cv.solvepnp as
        #formerly implemented in BARD
        marker_corners, ids, _ = aruco.detectMarkers(image,
                       aruco.getPredefinedDictionary(
                                aruco.DICT_ARUCO_ORIGINAL))
        #according to documentation the order of aruco.detectMarkers
        #is clockwise,
        #whereas aruco.create_board is ccw?

        success, modelreference2camera = \
                    reference_register.get_matrix(
                        ids, marker_corners)

        assert success
        assert np.allclose(modelreference2camera, aruco_reference_tracking)

    tracker.stop_tracking()
    tracker.close()
    capture.release()


def test_vs_solve_pnp_singletag():
    """
    This checks whether the tracking result using skarucotracker
    matches that when using the solvePnP approach as originally
    implemented by scikit-surgeryBARAD, with the addition of
    tracking smoothing
    """
    calib_mtx = np.array([[608.67179504, 0.00000000, 323.12263928],
                          [0.00000000, 606.13421375, 231.29247171],
                          [0.0, 0.0, 1.0]], dtype = np.float64)
    distortion = np.array([-0.02191634, -0.14300148, -0.00395124,
                           -0.00044941, 0.19656551], dtype = np.float64)
    videofile = 'data/output.avi'

    config = {'video source' : 'none',
              'camera projection' : calib_mtx,
              'camera distortion' : distortion,
              'aruco dictionary' : 'DICT_ARUCO_ORIGINAL',
              'smoothing buffer' : 5,
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/tag_0.txt',
                        'aruco dictionary' : 'DICT_4X4_50'
                      }
                      ]
              }

    config2 = {'video source' : 'none',
              'camera projection' : calib_mtx,
              'camera distortion' : distortion,
              'aruco dictionary' : 'DICT_4X4_50',
              'marker size' : 33,
              'smoothing buffer' : 5,
              }


    capture = cv.VideoCapture(videofile)
    #set up skarucotracker
    tracker = ArUcoTracker(config)
    tracker.start_tracking()
    tracker2 = ArUcoTracker(config2)
    tracker2.start_tracking()

    #set up solvepnp
    three_d_points = np.loadtxt('data/tag_0.txt')
    three_d_points = three_d_points.reshape(1, 16)
    three_d_points = ccw_to_cw(three_d_points)

    reference_register = Registration2D3D(three_d_points,
                                          calib_mtx, distortion,
                                          buffer_size=5)

    for _frame in range(10):
        _, image = capture.read()
        (port_handles, _timestamps, _framenumbers,
        tracking, _quality) = tracker.get_frame(image)

        assert 'reference' in port_handles
        reference_index = port_handles.index('reference')

        aruco_reference_tracking = tracking[reference_index]

        (port_handles, _timestamps, _framenumbers,
        tracking, _quality) = tracker2.get_frame(image)

        assert 'DICT_4X4_50:0' in port_handles
        reference_index = port_handles.index('DICT_4X4_50:0')

        aruco_reference_tracking2= tracking[reference_index]

        #now try again using our own implementation using cv.solvepnp as
        #formerly implemented in BARD
        marker_corners, ids, _ = aruco.detectMarkers(image,
                       aruco.getPredefinedDictionary(
                                aruco.DICT_4X4_50))

        success, modelreference2camera = \
                    reference_register.get_matrix(
                        ids, marker_corners)
        assert success
        assert np.allclose(modelreference2camera, aruco_reference_tracking)
        assert np.allclose(aruco_reference_tracking2, aruco_reference_tracking)

    tracker.stop_tracking()
    tracker.close()
    capture.release()
