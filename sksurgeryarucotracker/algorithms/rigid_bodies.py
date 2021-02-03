""" Classes and functions for 2D to 3D registration """


class ArUcoRigidBody():
    """
    Class to handle the loading and registering of ArUco Rigid Bodies
    """

    def __init__(self):
        """
        Initialises the RigidBody  class.
        """
        self._3d_points = None

    def set_2d_points(self, two_d_points):
        """
        takes a list of 2 points, and if the id's match 3D points,
        assigns them to a list of 2d points
        """
        print(self)
        print(two_d_points)

    def load_3d_points(self, filename):
        """
        loads the points from a file
        """
        print(self)
        print(filename)
