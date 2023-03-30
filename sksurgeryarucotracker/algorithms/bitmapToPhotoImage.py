"""To convert an RGB Numpy bitmap to a PhotoImage suitable for display 
with TKInter from 
https://stackoverflow.com/questions/1581799/
how-to-draw-a-bitmap-real-quick-in-python-using-tk-only
"""
from datetime import datetime
from tkinter import PhotoImage

rgb2hex = lambda r,g,b: '#%02x%02x%02x' %(255-r,255-g,255-g)
#rgb2hex = lambda r,g,b: '#%02x%02x%02x' %(g,b,r)

def bitmapToPhoto(bitmap):
    start_time =  datetime.now()
    print(bitmap.shape)
    imageWidth = bitmap.shape[1]
    imageHeight = bitmap.shape[0]

    photoImage = PhotoImage(width=imageWidth, height=imageHeight)

   # imgstring = [ rgb2hex(*bitmap[i,j,:]) for i in range(bitmap.shape[0]) 
    #        for j in range(bitmap.shape[1]) ]
    
    row = 0
    col = 0 
    #mystring = "{" + rgb2hex(*bitmap[i,j,:]) 
    #firstRow = "{"+" ".join(rgb2hex(*bitmap[i,j,:]) for i in range(imageWidth))+"}"
    imgstring = " ".join(("{"+" ".join(rgb2hex(*bitmap[row,col,:]) for col in range(imageWidth)) + "}")
            for row in range(imageHeight))
    print(len(imgstring))
    print ("elapsed time = ", datetime.now() - start_time)
    photoImage.put(imgstring, (0,0,imageWidth-1 , imageHeight -1))
    print ("elapsed time = ", datetime.now() - start_time)

    return photoImage
