"""To convert an RGB Numpy bitmap to a PhotoImage suitable for display
with TKInter from
https://stackoverflow.com/questions/1581799/
how-to-draw-a-bitmap-real-quick-in-python-using-tk-only
"""
from datetime import datetime
from tkinter import PhotoImage

def rgb2hex(red,green,blue):
    """
    converts an rgb code to hex
    """
    return f'#{blue:02x}{green:02x}{red:02x}'

def bitmap_to_photo(bitmap, subsample = 1):
    """
    converts a 3 channel numpy array image into
    a tkinter photoImage suitable for putting into
    tk widget
    """
    start_time =  datetime.now()
    ss_bitmap = bitmap[1::subsample, 1::subsample]
    print(bitmap.shape)
    image_width = ss_bitmap.shape[1]
    image_height = ss_bitmap.shape[0]

    photo_image = PhotoImage(width=image_width, height=image_height)

    imgstring = " ".join(("{"+" ".join(rgb2hex(*ss_bitmap[row,col,:])
        for col in range(image_width)) + "}")
            for row in range(image_height)) \

    print(len(imgstring))
    print ("elapsed time = ", datetime.now() - start_time)
    photo_image.put(imgstring, (0,0,image_width-1 , image_height -1))
    print ("elapsed time = ", datetime.now() - start_time)
    photo_image = photo_image.zoom(subsample)
    print ("elapsed time = ", datetime.now() - start_time)

    return photo_image
