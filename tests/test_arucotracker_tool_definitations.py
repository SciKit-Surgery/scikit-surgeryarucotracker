# coding=utf-8

"""scikit-surgeryarucotracker tests"""

import numpy as np
from sksurgeryarucotracker.arucotracker import ArUcoTracker


def test_with_tool_descriptions():
    """
    connect track and close with multitags and defined
    rigid bodies,
    reqs: 03, 04 ,05, 07
    """
    #test first with no tool descriptions
    config = {'video source' : 'data/multipattern.avi'}

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    #with nothing set we'll only detect tag ID 0, the others
    #are from a different dictionary.
    (port_handles, timestamps, framenumbers,
     tracking, quality) = tracker.get_frame()
    assert len(port_handles) == len(timestamps)
    assert len(port_handles) == len(framenumbers)
    assert len(port_handles) == len(tracking)
    assert len(port_handles) == len(quality)
    assert len(port_handles) == 1
    assert 'DICT_4X4_50:0' in port_handles

    tracker.stop_tracking()
    tracker.close()

    #try again with the right dictionary for the bard marker tags
    config = {'video source' : 'data/multipattern.avi',
              'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'}

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    #with nothing set we'll only detect tag ID 0, the others
    #are from a different dictionary.
    (port_handles, timestamps, framenumbers,
     tracking, quality) = tracker.get_frame()
    assert len(port_handles) == len(timestamps)
    assert len(port_handles) == len(framenumbers)
    assert len(port_handles) == len(tracking)
    assert len(port_handles) == len(quality)
    assert len(port_handles) == 17
    assert 'DICT_ARUCO_ORIGINAL:1' in port_handles

    #we should load the tag info and check that all tags are found
    tracker.stop_tracking()
    tracker.close()

    config = {'video source' : 'data/multipattern.avi',
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      },
                      {
                        'name' : 'pointer',
                        'filename' : 'data/pointer.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      }
                      ]
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    (port_handles, timestamps, framenumbers,
     tracking, quality) = tracker.get_frame()
    assert len(port_handles) == len(timestamps)
    assert len(port_handles) == len(framenumbers)
    assert len(port_handles) == len(tracking)
    assert len(port_handles) == len(quality)
    assert len(port_handles) == 4 #there is an extraneous marker (1000)
    assert 'reference' in port_handles
    assert 'pointer' in port_handles
    assert 'DICT_4X4_50:0' in port_handles

    reference_index = port_handles.index('reference')

    ref_regression = np.array([[ 1., 0., 0., 135.38637],
                               [ 0., 1., 0., 272.5],
                               [ 0., 0., 1., -57.1915 ],
                               [ 0., 0., 0., 1. ]], dtype=np.float32)

    assert np.allclose(tracking[reference_index], ref_regression)
    assert np.isclose(quality[reference_index], 0.91666666)
    pointer_index = port_handles.index('pointer')
    assert np.isclose(quality[pointer_index], 0.83333333)

    tracker.stop_tracking()
    tracker.close()

    #check that tag 0 is not detected when we use only
    #DICT_ARUCO_ORIGINAL
    config = {'video source' : 'data/multipattern.avi',
              'aruco dictionary' : 'DICT_ARUCO_ORIGINAL',
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      },
                      {
                        'name' : 'pointer',
                        'filename' : 'data/pointer.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      }
                      ]
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    (port_handles, timestamps, framenumbers,
     tracking, quality) = tracker.get_frame()
    assert len(port_handles) == len(timestamps)
    assert len(port_handles) == len(framenumbers)
    assert len(port_handles) == len(tracking)
    assert len(port_handles) == len(quality)
    assert len(port_handles) == 3 #there is an extraneous marker (1000)
    assert 'reference' in port_handles
    assert 'pointer' in port_handles
    assert 'DICT_4X4_50:0' not in port_handles
