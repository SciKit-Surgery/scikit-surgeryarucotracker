# coding=utf-8

"""scikit-surgeryarucotracker tests"""

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
    #this configuration should also pick up the single tag 0. We can set a
    #loop in get frames at

    #for each dictionary in self._ar_dicts so we could have mutltiple
    #dictionaries

    #marker_corners, marker_ids, _ = \
                    #            aruco.detectMarkers(frame, self._ar_dict)


    tracker = ArUcoTracker(config)
    tracker.start_tracking()

    (port_handles, timestamps, framenumbers,
     tracking, quality) = tracker.get_frame()
    print(port_handles)
    assert len(port_handles) == len(timestamps)
    assert len(port_handles) == len(framenumbers)
    assert len(port_handles) == len(tracking)
    assert len(port_handles) == len(quality)
    assert len(port_handles) == 4 #there is an extraneous marker (1000)
    assert 'reference' in port_handles
    assert 'pointer' in port_handles
    assert 'DICT_4X4_50:0' in port_handles
    print(tracking)

    #then try again after setting some rigid bodies
    #look at image quality, we could have so that quality = detected tags /
    #tags on body
