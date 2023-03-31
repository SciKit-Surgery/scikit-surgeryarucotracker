"""A class to bring up a debugging window"""

from tkinter import Tk, Canvas, NW
from sksurgeryarucotracker.algorithms.bitmap_to_photo_image \
        import bitmap_to_photo

class Debugger():
    """
    Creates debugging window, to show and image of the
    video feed.
    """
    def __init__(self, in_use, subsample):
        """
        :param in_use: Boolean, if false we're not using the debugger
        :param subsample: we can subsample the image to speed things up
        """
        self.in_use = in_use
        self.initialised = False
        self.debug_window = None
        self.canvas = None
        self.debug_window = None
        self.debug_image = None
        self.subsample = subsample

    def __del__(self):
        """
        Destroys the TK window
        """
        if self.debug_window is not None:
            self.debug_window.destroy()

    def setup_window(self, frame):
        """
        Do this after init as we need a frame of
        video to set the window size
        """
        if self.in_use:
            self.debug_window = Tk()
            self.debug_window.title('Debug Window')
            self.canvas = Canvas(self.debug_window,
                        width = frame.shape[1], height = frame.shape[0])
            self.canvas.pack()
        self.initialised = True

    def update(self, frame):
        """
        Updates the video image
        """
        if not self.in_use:
            return

        if not self.initialised:
            self.setup_window(frame)

        self.debug_image = bitmap_to_photo(frame, self.subsample)
        self.canvas.create_image(0, 0, anchor=NW, image=self.debug_image)
        self.debug_window.update()
