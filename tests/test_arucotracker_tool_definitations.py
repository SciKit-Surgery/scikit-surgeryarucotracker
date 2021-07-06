# coding=utf-8

"""scikit-surgeryarucotracker tests"""

import numpy as np
from sksurgeryarucotracker.arucotracker import ArUcoTracker
from sksurgeryarucotracker.algorithms.compare_matrices \
        import matrices_equivalent


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

    assert matrices_equivalent(tracking[reference_index], ref_regression)
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


def test_iteration_over_empty_dicts():
    """
    Tests that when no markers are detected in an early
    dictionary, we continue to iterate over the remainder
    issue #29
    """

    config = {'video source' : 'data/multipattern.avi',
              'rigid bodies' : [
                      {
                        'name' : 'bad_pointer',
                        'filename' : 'data/pointer.txt',
                        'aruco dictionary' : 'DICT_7X7_250'
                      },
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      },
                      {
                        'name' : 'good_pointer',
                        'filename' : 'data/pointer.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      },
                      ]
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    (port_handles, timestamps, framenumbers,
     tracking, quality) = tracker.get_frame()
    print(port_handles)
    assert len(port_handles) == len(timestamps)
    assert len(port_handles) == len(framenumbers)
    assert len(port_handles) == len(tracking)
    assert len(port_handles) == len(quality)
    assert len(port_handles) == 5
    assert 'reference' in port_handles
    assert 'bad_pointer' in port_handles
    assert 'good_pointer' in port_handles
    assert 'DICT_4X4_50:0' in port_handles

    ref_regression = np.array([[ 1., 0., 0., 135.38637],
                               [ 0., 1., 0., 272.5],
                               [ 0., 0., 1., -57.1915 ],
                               [ 0., 0., 0., 1. ]], dtype=np.float32)

    pointer_index = port_handles.index('bad_pointer')
    assert np.all(np.isnan(tracking[pointer_index][0:3, 0:4]))

    reference_index = port_handles.index('reference')
    assert matrices_equivalent(tracking[reference_index], ref_regression)


def test_with_tool_desc_and_calib():
    """
    connect track and close with multitags, defined
    rigid bodies, and camera calibration
    reqs: 03, 04 ,05, 07
    """
    config = {'video source' : 'data/multipattern.avi',
              'calibration' : 'data/calibration.txt',
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL',
                        'tag width' : 49.50
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

    assert tracker.has_capture()
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
    pointer_index = port_handles.index('pointer')

    assert np.isclose(quality[reference_index], 0.91666666)
    assert np.isclose(quality[pointer_index], 0.83333333)

    ref_regression = np.array([
        [-0.84701057,  0.51884094, -0.11565978, -8.22903442e+01],
        [-0.48733129, -0.67100208,  0.55880625,  4.85032501e+01],
        [0.21232361,  0.52967943,  0.82119327,  2.43992401e+02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],
        dtype=np.float32)

    assert matrices_equivalent(tracking[reference_index], ref_regression)

    tracker.stop_tracking()
    tracker.close()


def test_tool_desc_and_float64():
    """
    connect track and close with multitags, defined
    rigid bodies, and camera calibration, defined with
    different data types, issue #31
    reqs: 03, 04 ,05, 07
    """
    config = {'video source' : 'data/multipattern.avi',
              'calibration' : 'data/calibration.txt',
              'aruco dictionary' : 'DICT_ARUCO_ORIGINAL',
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      },
                      ]
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    (port_handles, _timestamps, _framenumbers,
     tracking, quality) = tracker.get_frame()
    assert 'reference' in port_handles

    reference_index = port_handles.index('reference')

    assert np.isclose(quality[reference_index], 0.91666666)

    ref_regression = np.array([
        [-0.84701057,  0.51884094, -0.11565978, -8.22903442e+01],
        [-0.48733129, -0.67100208,  0.55880625,  4.85032501e+01],
        [0.21232361,  0.52967943,  0.82119327,  2.43992401e+02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],
        dtype=np.float32)

    assert matrices_equivalent(tracking[reference_index], ref_regression)

    tracker.stop_tracking()
    tracker.close()
    port_handles = None
    tracking = None
    quality = None

    #again, but this time set calibration and distortion separately
    calib_mtx = np.array([[560.0, 0.0, 320.0],
                          [0.0, 560.0, 240.0],
                          [0.0, 0.0, 1.0]], dtype = np.float32)
    distortion = np.array([0.1, 0.0, 0.0, 0.0, 0.0], dtype = np.float32)
    config = {'video source' : 'data/multipattern.avi',
              'camera projection' : calib_mtx,
              'camera distortion' : distortion,
              'aruco dictionary' : 'DICT_ARUCO_ORIGINAL',
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      },
                      ]
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    (port_handles, _timestamps, _framenumbers,
     tracking, quality) = tracker.get_frame()
    assert 'reference' in port_handles

    reference_index = port_handles.index('reference')

    assert np.isclose(quality[reference_index], 0.91666666)
    assert matrices_equivalent(tracking[reference_index], ref_regression)

    tracker.stop_tracking()
    tracker.close()
    port_handles = None
    tracking = None
    quality = None

    #again, but this time set calibration and distortion as float64
    calib_mtx = np.array([[560.0, 0.0, 320.0],
                          [0.0, 560.0, 240.0],
                          [0.0, 0.0, 1.0]], dtype = np.float64)
    distortion = np.array([0.1, 0.0, 0.0, 0.0, 0.0], dtype = np.float64)
    config = {'video source' : 'data/multipattern.avi',
              'camera projection' : calib_mtx,
              'camera distortion' : distortion,
              'aruco dictionary' : 'DICT_ARUCO_ORIGINAL',
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      },
                      ]
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    (port_handles, _timestamps, _framenumbers,
     tracking, quality) = tracker.get_frame()
    assert 'reference' in port_handles

    reference_index = port_handles.index('reference')

    assert np.isclose(quality[reference_index], 0.91666666)
    assert matrices_equivalent(tracking[reference_index], ref_regression)

    tracker.stop_tracking()
    tracker.close()
    port_handles = None
    tracking = None
    quality = None

    #again, but this time set calibration and distortion as float
    calib_mtx = np.array([[560.0, 0.0, 320.0],
                          [0.0, 560.0, 240.0],
                          [0.0, 0.0, 1.0]], dtype = float)
    distortion = np.array([0.1, 0.0, 0.0, 0.0, 0.0], dtype = float)
    config = {'video source' : 'data/multipattern.avi',
              'camera projection' : calib_mtx,
              'camera distortion' : distortion,
              'aruco dictionary' : 'DICT_ARUCO_ORIGINAL',
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_ARUCO_ORIGINAL'
                      },
                      ]
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    (port_handles, _timestamps, _framenumbers,
     tracking, quality) = tracker.get_frame()
    assert 'reference' in port_handles

    reference_index = port_handles.index('reference')

    assert np.isclose(quality[reference_index], 0.91666666)
    assert matrices_equivalent(tracking[reference_index], ref_regression)

    tracker.stop_tracking()
    tracker.close()


def test_with_no_tags_no_calib():
    """
    connect track and close with multitags, defined
    rigid bodies, but no tags
    visible, should return an NaN matrix for each
    undetected body
    reqs: 03, 04 ,05, 07
    """
    config = {
              'video source' : 'none',
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_4X4_50'
                      },
                      {
                        'name' : 'pointer',
                        'filename' : 'data/pointer.txt',
                        'aruco dictionary' : 'DICT_4X4_50'
                      }
                      ]
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()
    image = np.zeros((640, 480, 3), dtype=np.uint8)
    (port_handles, timestamps, framenumbers,
     tracking, quality) = tracker.get_frame(image)
    assert len(port_handles) == len(timestamps)
    assert len(port_handles) == len(framenumbers)
    assert len(port_handles) == len(tracking)
    assert len(port_handles) == len(quality)
    assert len(port_handles) == 2 #there is an extraneous marker (1000)
    assert 'reference' in port_handles
    assert 'pointer' in port_handles

    reference_index = port_handles.index('reference')
    pointer_index = port_handles.index('pointer')

    assert quality[reference_index] == 0
    assert quality[pointer_index] == 0

    assert np.all(np.isnan(tracking[reference_index][0:3, 0:4]))
    assert np.all(np.isnan(tracking[pointer_index][0:3, 0:4]))

    tracker.stop_tracking()
    tracker.close()

def test_with_no_tags_and_calib():
    """
    connect track and close with multitags, defined
    rigid bodies, but no tags
    visible, should return an NaN matrix for each
    undetected body
    reqs: 03, 04 ,05, 07
    """
    config = {
              'calibration' : 'data/calibration.txt',
              'video source' : 'none',
              'rigid bodies' : [
                      {
                        'name' : 'reference',
                        'filename' : 'data/reference.txt',
                        'aruco dictionary' : 'DICT_4X4_50'
                      },
                      {
                        'name' : 'pointer',
                        'filename' : 'data/pointer.txt',
                        'aruco dictionary' : 'DICT_4X4_50'
                      }
                      ]
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()
    image = np.zeros((640, 480, 3), dtype=np.uint8)
    (port_handles, timestamps, framenumbers,
     tracking, quality) = tracker.get_frame(image)
    assert len(port_handles) == len(timestamps)
    assert len(port_handles) == len(framenumbers)
    assert len(port_handles) == len(tracking)
    assert len(port_handles) == len(quality)
    assert len(port_handles) == 2
    assert 'reference' in port_handles
    assert 'pointer' in port_handles

    reference_index = port_handles.index('reference')
    pointer_index = port_handles.index('pointer')

    assert quality[reference_index] == 0
    assert quality[pointer_index] == 0

    assert np.all(np.isnan(tracking[reference_index][0:3, 0:4]))
    assert np.all(np.isnan(tracking[pointer_index][0:3, 0:4]))

    tracker.stop_tracking()
    tracker.close()
