# coding=utf-8

"""scikit-surgeryarucotracker tests"""

import pytest
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
    assert 0 in port_handles

    tracker.stop_tracking()
    tracker.close()

    #try again with the right dictionary
    #then try again after setting some rigid bodies
    #look at image quality, we could have so that quality = detected tags / 
    #tags on body

