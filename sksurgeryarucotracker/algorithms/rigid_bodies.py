""" Classes and functions for maintaining ArUco rigid bodies """

from numpy import loadtxt, ptp, copy

class ArUcoRigidBody():
    """
    Class to handle the loading and registering of ArUco Rigid Bodies
    """

    def __init__(self, rigid_body_name):
        """
        Initialises the RigidBody  class.
        """
        self._tags_3d = []
        self._tags_2d = []
        self._tag_size = []
        self._name = rigid_body_name

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
            if tag_id in self._tags_3d[:, 0]:
                self._tags_2d.append([tag_id, two_d_points[index]])
                tags_assigned.append(tag_id)

        return tags_assigned

    def load_3d_points(self, filename):
        """
        Loads the 3D point geometry from a file

        :param filename: Path of file containing tag data

        """
        self._tags_3d = loadtxt(filename)

    def add_single_tag(self, tag_size, marker_id):
        """
        We can use this to track single ArUco tags rather than
        patterns as long as we know the tag size in mm

        :param: tag size in mm
        :param: marker id
        """
        self._tag_size = tag_size
        self._tags_3d.append([marker_id, 
                        tag_size/2.0, tag_size/2.0, 0.,
                        0., 0., 0.,
                        tag_size, 0., 0.,
                        tag_size, tag_size, 0.,
                        0., tag_size, 0.])
        print (self._tags_3d)


    def scale_3d_tags(self, measured_pattern_width):
        """
        We can scale the tag, which is very useful if you've got the tag
        on your mobile phone.

        :param measured_pattern_width: Width of the tag in mm
        """
        model_pattern_width = min(ptp(self._tags_3d[:, 2::3]),
                                  ptp(self._tags_3d[:, 1::3]))
        scale_factor = measured_pattern_width/model_pattern_width
        tag_ids = copy(self._tags_3d[:, 0])
        self._tags_3d *= scale_factor
        self._tags_3d[:, 0] = tag_ids

    def get_pose(self, camera_projection_matrix, camera_distortion):
        """
        Estimate the pose of the rigid body, with or without
        camera calibration

        :param: camera_projection_matrix 3x3 projection matrix. If
            None we estimate pose based on pattern size
        :param: 1x5 camera distortion vector
        """
        if camera_projection_matrix is None:
            return self._get_poses_without_calibration()
        
        return self._get_poses_with_calibration(camera_projection_matrix,
                        camera_distortion)


    def _get_poses_without_calibration(self):
        print(self)
        return None, None

    def _get_poses_with_calibration(self, camera_projection_matrix,
                    camera_distortion):
        print(self, camera_projection_matrix, camera_distortion)
        return None, None
