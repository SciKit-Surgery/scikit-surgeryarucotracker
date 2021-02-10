#  -*- coding: utf-8 -*-

"""Convenience function to make ArUco marker patterns
"""
from numpy import array, float32, loadtxt, floor
import cv2.aruco as aruco # pylint: disable=import-error
from cv2 import imshow

def load_markers_from_file(filename, dictionary = aruco.DICT_ARUCO_ORIGINAL):
    """
    loads marker pattern from filename.
    :return: an aruco.board
    """
    markers = loadtxt(filename)
    dictionary = aruco.getPredefinedDictionary(dictionary)

    #let's assume we've got stuff in the existing format of ID first, 
    #then 15 columns
    boardshape=markers.shape
    if boardshape[0] < 1:
        raise ValueError("Marker pattern appears to have no markers")
    if boardshape[1] != 16 and boardshape[1] != 13:
        raise ValueError("Marker pattern should have either 5 or 4 3D points")
    
    marker_ids = markers[:,0].astype('int')
    markerpoints = markers[:,1:13]
    if boardshape[1] == 16:
        markerpoints = markers[:,4:16].astype('float32')
    
    print (markerpoints)
    board = aruco.Board_create(markerpoints, dictionary, marker_ids)
    
    return board
    
def MakeMarkerPattern_WholeBoard(board):
    """
    draw a pattern of markers using cv2.aruco drawPlanarBoard
    """
    
    outsize = (32,32)
    marginSize = 3 
    borderBits = 1
    
    #this is struggling, lets try doing it ourselves one marker at a time
    image = aruco.drawPlanarBoard(board, outsize, marginSize, borderBits) 

    return image

def MakeMarkerPattern_MarkerByMarker(board):
    """
    draw a pattern one marker at a time
    """
    print (board.dictionary)
    print (board.ids)
    print (board.objPoints)
    return 1

if __name__ == '__main__':
    board = load_markers_from_file('../../data/pointer.txt')
    image = MakeMarkerPattern_MarkerByMarker(board)
    imshow (image)
