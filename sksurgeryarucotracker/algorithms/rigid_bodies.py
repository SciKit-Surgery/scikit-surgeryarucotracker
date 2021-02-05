""" Classes and functions for maintaining ArUco rigid bodies """

import numpy

class ThreeDTags():
    """
    Stores two linked arrays, on of tag IDs and the other 
    3D points
    """

    def __init__(self):
        self.points = numpy.empty((0,15), dtype=numpy.float64)
        self.ids = []

    def load_from_file(self, filename):
        tags_3d = numpy.loadtxt(filename)
        self.points = tags_3d[:,1:16]
        self.ids = tags_3d[:,0]

    def add_single_tag(self, tag_size, marker_id):
        """
        We can use this to track single ArUco tags rather than
        patterns as long as we know the tag size in mm

        :param: tag size in mm
        :param: marker id
        """
        default_tag = numpy.array([[
                        tag_size/2.0, tag_size/2.0, 0.,
                        0., 0., 0.,
                        tag_size, 0., 0.,
                        tag_size, tag_size, 0.,
                        0., tag_size, 0.]], dtype=numpy.float64)
        self.points = numpy.append(self.points, default_tag, axis=0)
        self.ids.append(marker_id)

    def scale_tags(self, measured_pattern_width):
        model_pattern_width = min(numpy.ptp(self.points[:, 1::2]),
                                  numpy.ptp(self.points[:, 0::4]))
        scale_factor = measured_pattern_width/model_pattern_width
        self.points *= scale_factor


class TwoDTags():
    """
    Stores two linked arrays, on of tag IDs and the other 
    2D points
    """

    def __init__(self):
        self.points = numpy.empty((0,8), dtype=numpy.float64)
        self.ids = []

    def append_tag(self, tag_id, points):
        self.points = numpy.append(self.points, [points.ravel()], axis=0)
        self.ids.append(tag_id)


class ArUcoRigidBody():
    """
    Class to handle the loading and registering of ArUco Rigid Bodies
    """

    def __init__(self, rigid_body_name):
        """
        Initialises the RigidBody  class.
        """
        self._tags_3d = ThreeDTags()
        self._tags_2d = TwoDTags()
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
            if tag_id in self._tags_3d.ids:
                self._tags_2d.append_tag(tag_id, two_d_points[index])
                tags_assigned.append(tag_id)

        return tags_assigned

    def load_3d_points(self, filename):
        """
        Loads the 3D point geometry from a file

        :param filename: Path of file containing tag data

        """
        self._tags_3d.load_from_file(filename)

    def add_single_tag(self, tag_size, marker_id):
        """
        We can use this to track single ArUco tags rather than
        patterns as long as we know the tag size in mm

        :param: tag size in mm
        :param: marker id
        """
        self._tags_3d.add_single_tag(tag_size, marker_id)
        self._tag_size = tag_size


    def scale_3d_tags(self, measured_pattern_width):
        """
        We can scale the tag, which is very useful if you've got the tag
        on your mobile phone.

        :param measured_pattern_width: Width of the tag in mm
        """
        self._tags_3d.scale_tags(measured_pattern_width)
       
    def get_pose(self, camera_projection_matrix, camera_distortion):
        """
        Estimate the pose of the rigid body, with or without
        camera calibration

        :param: camera_projection_matrix 3x3 projection matrix. If
            None we estimate pose based on pattern size
        :param: 1x5 camera distortion vector
        """
        points3d, points2d = self._match_point_lists()
        if camera_projection_matrix is None:
            return self._get_poses_without_calibration()

        return self._get_poses_with_calibration(camera_projection_matrix,
                        camera_distortion)

    def _match_point_lists(self):
        """Turns 2d and 3d points into matched point list"""
        points3d = []
        points2d = []
        count = 0
        
        print (self._tags_2d)
        for tag in self._tags_3d.ids:
            try: 
                index2d = self._tags_2d.ids.index(tag)
                count += 1 
                #points3d.extend with 3 d point
                points2d.extend(self._tags_2d.points[index2d])
            except ValueError:
                pass
        
        return points3d, points2d


    def _get_poses_without_calibration(self):
            #print(self)
        return None, None

    def _get_poses_with_calibration(self, camera_projection_matrix,
                    camera_distortion):
            #print(self, camera_projection_matrix, camera_distortion)
        return None, None
