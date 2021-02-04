""" Classes and functions for maintaining ArUco rigid bodies """

from numpy import loadtxt, ptp, copy

class ArUcoRigidBody():
    """
    Class to handle the loading and registering of ArUco Rigid Bodies
    """

    def __init__(self):
        """
        Initialises the RigidBody  class.
        """
        self._tags_3d = None

    def set_2d_points(self, two_d_points):
        """
        takes a list of 2 points, and if the id's match 3D points,
        assigns them to a list of 2d points

        :params two_d_points: 3 x n list of n 2d points, column 0 is the
            tag id
        """
        print(self)
        print(two_d_points)

    def load_3d_points(self, filename):
        """
        Loads the 3D point geometry from a file

        :params filename: Path of file containing tag data

        """
        self._tags_3d = loadtxt(filename)

    def scale_3d_tags(self, measured_pattern_width):
        """
        We can scale the tag, which is very useful if you've got the tag
        on your mobile phone.

        :params measured_pattern_width: Width of the tag in mm
        """
        model_pattern_width = min(ptp(self._tags_3d[:, 2::3]),
                                  ptp(self._tags_3d[:, 1::3]))
        scale_factor = measured_pattern_width/model_pattern_width
        tag_ids = copy(self._tags_3d[:, 0])
        self._tags_3d *= scale_factor
        self._tags_3d[:, 0] = tag_ids
