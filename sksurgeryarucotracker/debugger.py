"""A class to bring up a debugging window"""

from tkinter import Tk, Canvas, NW, PhotoImage
from sksurgeryarucotracker.algorithms.bitmapToPhotoImage \
        import bitmapToPhoto

class debugger():
    def __init__(self, in_use):
        self.in_use = in_use
        self.initialised = False
        self.debug_window = None
        self.canvas = None
        self.debug_window = None

    def setup_window(self, frame):
        if self.in_use:
            self.debug_window = Tk()
            self.debug_window.title('Debug Window')
            self.canvas = Canvas(self.debug_window,
                        width = frame.shape[1], height = frame.shape[0])
            self.canvas.pack()
        self.initialised = True

    
    def update(self, frame):
        if not self.in_use:
            return
    
        if not self.initialised:
            self.setup_window(frame)

        self.debug_image = bitmapToPhoto(frame)
        print(frame.shape)
        print(self.debug_image)
        self.canvas.create_image(0, 0, anchor=NW, image=self.debug_image)



