# coding=utf-8

"""scikit-surgeryarucotracker tests"""

import pytest
from cv2 import VideoCapture
from sksurgeryarucotracker.arucotracker import ArUcoTracker

def test_on_video_with_single_tag():
    """
    connect track and close with single tag,
    reqs: 03, 04 ,05
    """
    config = {'video source' : 'data/output.avi'}

    tracker = ArUcoTracker(config)
    tracker.start_tracking()
    for frame in range(10):
        (port_handles, timestamps, framenumbers,
         tracking, quality) = tracker.get_frame()
        assert len(port_handles) == len(timestamps)
        assert len(port_handles) == len(framenumbers)
        assert len(port_handles) == len(tracking)
        assert len(port_handles) == len(quality)
        assert len(port_handles) == 1
        assert port_handles[0] == 0
        assert framenumbers[0] == frame
        assert quality[0] == 1.0

    tracker.stop_tracking()
    tracker.close()


def test_no_video_single_tag():
    """
    raises a value error when no video and no image passed.
    works on static images
    reqs: 03, 04 ,05
    """
    config = {'video source' : 'none'}

    tracker = ArUcoTracker(config)
    tracker.start_tracking()
    with pytest.raises(ValueError):
        tracker.get_frame()
    capture = VideoCapture('data/output.avi')
    for frame in range(2):
        _, image = capture.read()
        (port_handles, timestamps, framenumbers,
         tracking, quality) = tracker.get_frame(image)
        assert len(port_handles) == len(timestamps)
        assert len(port_handles) == len(framenumbers)
        assert len(port_handles) == len(tracking)
        assert len(port_handles) == len(quality)
        assert len(port_handles) == 1
        assert port_handles[0] == 0
        assert framenumbers[0] == frame
        assert quality[0] == 1.0

    tracker.stop_tracking()
    tracker.close()


def test_on_video_with_debug():
    """
    connect track and close with single tag, with debug flag on
    debug should open a separate window showing the image capture
    reqs: 03, 04 ,05
    """
    config = {'video source' : 'data/output.avi',
              'debug' : True}

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    for frame in range(2):
        (port_handles, timestamps, framenumbers,
         tracking, quality) = tracker.get_frame()
        assert len(port_handles) == len(timestamps)
        assert len(port_handles) == len(framenumbers)
        assert len(port_handles) == len(tracking)
        assert len(port_handles) == len(quality)
        assert len(port_handles) == 1
        assert port_handles[0] == 0
        assert framenumbers[0] == frame
        assert quality[0] == 1.0

    tracker.stop_tracking()
    tracker.close()


def test_on_static_muti_tag():
    """
    connect track and close with multi tag,
    reqs: 03, 04 ,05, 07
    """
    config = {'video source' : 'data/12markers.avi',
              'aruco dictionary' : 'DICT_6X6_250'}

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    (port_handles, timestamps, framenumbers,
     tracking, quality) = tracker.get_frame()
    assert len(port_handles) == len(timestamps)
    assert len(port_handles) == len(framenumbers)
    assert len(port_handles) == len(tracking)
    assert len(port_handles) == len(quality)
    assert len(port_handles) == 12
    for tagid in range(1,13):
        assert tagid in port_handles

    tracker.stop_tracking()
    tracker.close()


def test_setting_capture_properties():
    """
    connect track and close with multi tag,
    reqs: 03, 04 ,05, 07
    """
    config = {'video source' : 'data/12markers.avi',
              'aruco dictionary' : 'DICT_6X6_250',
              "capture properties" :
                            {"CAP_PROP_FRAME_WIDTH" : 1280,
                             "CAP_PROP_FRAME_HEIGHT" : 1024}
              }

    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    (_port_handles, _timestamps, _framenumbers,
     _tracking, _quality) = tracker.get_frame()

    tracker.stop_tracking()
    tracker.close()


def test_on_video_with_calib():
    """
    connect track and close with single tag,
    and calibrated camera
    reqs: 03, 04 ,05
    """
    config1 = {'video source' : 'data/output.avi',
               'calibration' : 'data/calibration.txt',
               'marker size' : 50,
              }

    tracker = ArUcoTracker(config1)
    tracker.start_tracking()
    for _ in range(10):
        (_port_handles, _timestamps, _framenumbers,
         _tracking, _quality) = tracker.get_frame()

    tracker.stop_tracking()
    tracker.close()


def test_throw_on_bad_source():
    """
    Tests that OS error is thrown when an invalid source is used.,
    reqs:
    """
    config = {'video source' : 'data/nofile.xxxx',
              'aruco dictionary' : 'DICT_6X6_250'}

    with pytest.raises(OSError):
        _tracker = ArUcoTracker(config)


def test_throw_on_bad_calibration():
    """
    Tests that Value error is thrown when an invalid calibration is used.
    reqs:
    """
    config = {'video source' : 'data/output.avi',
              'calibration' : 'data/bad_calibration.txt'
             }

    with pytest.raises(ValueError):
        _tracker = ArUcoTracker(config)


def test_invalid_dictionary():
    """
    Tests that Import error is thrown when an invalid aruco dictionary is used.
    reqs:
    """
    config = {'video source' : 'data/12markers.avi',
              'aruco dictionary' : 'DICT_7X6_250'}

    with pytest.raises(ImportError):
        _tracker = ArUcoTracker(config)


def test_getframe_no_tracking():
    """
    Tests that value error is thrown when get frame called without tracking.
    reqs:
    """
    config = {'video source' : 'data/output.avi'}

    tracker = ArUcoTracker(config)
    with pytest.raises(ValueError):
        (_port_handles, _timestamps, _framenumbers,
         _tracking, _quality) = tracker.get_frame()
    tracker.close()


def test_get_tool_descriptions():
    """
    Tests that get too descriptions.
    If no tool descriptions have been set it should return
    "No tools defined", otherwise it should return a list of
    tool names
    reqs:
    """
    config = {'video source' : 'data/output.avi'}

    tracker = ArUcoTracker(config)
    assert tracker.get_tool_descriptions() == "No tools defined"
    tracker.close()


def test_start_tracking_throws():
    """
    Tests that value error is thrown when start tracking called when not ready.
    reqs:
    """
    config = {'video source' : 'data/12markers.avi'}
    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    with pytest.raises(ValueError):
        tracker.start_tracking()

    tracker.close()


def test_stop_tracking_throws():
    """
    Tests that value error is thrown when stop tracking called when
    not tracking.
    reqs:
    """
    config = {'video source' : 'data/12markers.avi'}
    tracker = ArUcoTracker(config)

    with pytest.raises(ValueError):
        tracker.stop_tracking()

    tracker.close()
