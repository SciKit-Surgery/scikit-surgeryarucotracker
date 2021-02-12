""" Classes and functions for maintaining ArUco rigid bodies """

import numpy
import cv2.aruco as aruco # pylint: disable=import-error
from sksurgeryarucotracker.algorithms.registration_2d3d import \
                estimate_poses_no_calibration, estimate_poses_with_calibration


def load_board_from_file(filename, dictionary = aruco.DICT_ARUCO_ORIGINAL):
    """
    loads marker pattern from filename.
    :return: an aruco.board
    :raise ValueError: If the file does not have 16 or 13 columns
    """
    markers = numpy.loadtxt(filename)
    dictionary = aruco.getPredefinedDictionary(dictionary)

    #format of ID first, then 15 or 12 columns
    boardshape=markers.shape
    if boardshape[0] < 1:
        raise ValueError("Marker pattern appears to have no markers")
    if boardshape[1] != 16 and boardshape[1] != 13:
        raise ValueError("Marker pattern should have either 5 or 4 3D points")

    marker_ids = markers[:,0].astype('int')
    markerpoints = markers[:,1:13]
    if boardshape[1] == 16:
        markerpoints = markers[:,4:16].astype('float32')

    return aruco.Board_create(markerpoints, dictionary, marker_ids)


def single_tag_board(tag_size, marker_id,
                ar_dictionary = aruco.DICT_ARUCO_ORIGINAL):
    """
    Create a board consisting of a single ArUco tag

    :param: tag size in mm
    :param: marker id
    :param: dictionary to use
    """
    tag = numpy.array([[
        -tag_size/2.0, -tag_size/2.0, 0.,
        tag_size/2.0, -tag_size/2.0, 0.,
        tag_size/2.0, tag_size/2.0, 0.,
        -tag_size/2.0, tag_size/2.0, 0.]], dtype=numpy.float32)
    dictionary = aruco.getPredefinedDictionary(ar_dictionary)
    marker_ids = numpy.array([marker_id])
    return aruco.Board_create(tag, dictionary, marker_ids)


def scale_tags(board, measured_pattern_width):
    """
    We can scale the tag on a board,
    which is very useful if you've got the tag
    on your mobile phone.

    :param: the board to scale
    :param measured_pattern_width: Width of the tag in mm
    """

    model_pattern_width = min(numpy.ptp(board.objPoints[0][:][:,0]),
                              numpy.ptp(board.objPoints[0][:][:,1]))
    scale_factor = measured_pattern_width/model_pattern_width
    board.objPoints[0] *= scale_factor

    return board

class TwoDTags():
    """
    Stores two linked arrays, on of tag IDs and the other
    2D points
    """

    def __init__(self):
        self.points = []
        self.ids = []

    def append_tag(self, tag_id, points):
        """ Adds a tag to the two point list
        :param tag_id: The id of the tag
        :param points: 4 points defining the tag corners
        """
        self.points.append(points)
        self.ids.append(tag_id)


class ArUcoRigidBody():
    """
    Class to handle the loading and registering of ArUco Rigid Bodies
    """

    def __init__(self, rigid_body_name):
        """
        Initialises the RigidBody  class.
        """
        self._ar_board = None
        self._tags_2d = TwoDTags()
        self.name = rigid_body_name
        self._default_tags = None

    def set_2d_points(self, two_d_points, tag_ids):
        """
        takes a list of 2 points, and if the id's match 3D points,
        assigns them to a list of 2d points

        :param two_d_points: array of marker corners, 4 for each tag
        :param tag_ids: id for each tag

        :return: tag ids for any assigned tags
        """
        tags_assigned = []
        for index, tag_id in enumerate(tag_ids):
            if tag_id in self._ar_board.ids:
                self._tags_2d.append_tag(tag_id, two_d_points[index])
                tags_assigned.append(tag_id)
        self._default_tags = two_d_points
        return tags_assigned

    def load_3d_points(self, filename):
        """
        Loads the 3D point geometry from a file

        :param filename: Path of file containing tag data

        """
        self._ar_board = load_board_from_file(filename,
                        dictionary = aruco.DICT_ARUCO_ORIGINAL)

    def add_single_tag(self, tag_size, marker_id):
        """
        We can use this to track single ArUco tags rather than
        patterns as long as we know the tag size in mm

        :param: tag size in mm
        :param: marker id/_
        """
        self._ar_board = single_tag_board(tag_size, marker_id)

    def scale_3d_tags(self, measured_pattern_width):
        """
        We can scale the tag, which is very useful if you've got the tag
        on your mobile phone.

        :param measured_pattern_width: Width of the tag in mm
        """
        self._ar_board = scale_tags(self._ar_board, measured_pattern_width)

    def get_pose(self, camera_projection_matrix, camera_distortion):
        """
        Estimate the pose of the rigid body, with or without
        camera calibration

        :param: camera_projection_matrix 3x3 projection matrix. If
            None we estimate pose based on pattern size
        :param: 1x5 camera distortion vector
        """
        #points3d, points2d = self._match_point_lists()
        if camera_projection_matrix is None:
            return estimate_poses_no_calibration(self._tags_2d.points)

        return estimate_poses_with_calibration(
                        self._tags_2d.points, self._tags_2d.ids,
                            self._ar_board,
                            camera_projection_matrix, camera_distortion)


    def _match_point_lists(self):
        """Turns 2d and 3d points into matched point list"""
        points3d = []
        points2d = []

        for index3d, tag_id in enumerate(self._ar_board.ids):
            try:
                index2d = self._tags_2d.ids.index(tag_id)
                points3d.append(self._ar_board.objPoints[index3d])
                points2d.append(self._tags_2d.points[index2d])
            except ValueError:
                pass

        return points3d, points2d
