scikit-surgeryarucotracker
===============================

.. image:: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgeryarucotracker/raw/master/project-icon.png
   :height: 128px
   :width: 128px
   :target: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgeryarucotracker
   :alt: Logo

.. image:: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgeryarucotracker/badges/master/pipeline.svg
   :target: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgeryarucotracker/pipelines
   :alt: GitLab-CI test status

.. image:: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgeryarucotracker/badges/master/coverage.svg
    :target: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgeryarucotracker/commits/master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/scikit-surgeryarucotracker/badge/?version=latest
    :target: http://scikit-surgeryarucotracker.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



Author: Stephen Thompson

scikit-surgeryarucotracker provides a simple Python interface between OpenCV's ARuCo marker tracking libraries and other Python packages designed around scikit-surgerytrackers. It allows you to treat an object tracked using ARuCo markers in the same way as an object tracked usinng other tracking hardware (e.g. NDI - scikit-surgerynditracker). 

scikit-surgeryarucotracker is part of the `SNAPPY`_ software project, developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, part of `University College London (UCL)`_.

scikit-surgeryarucotracker is tested with Python 3.6 and may support other Python versions.

Installing
----------

::

    pip install scikit-surgeryarucotracker

Using
-----
Configuration is done using Python libraries. Tracking data is returned in NumPy arrays.

::

    from sksurgerarucotracker.tracker import ARuCoTracker
    SETTINGS = {
        "video source" : 0
            }
    TRACKER = ARuCo()
    TRACKER.connect(SETTINGS)

    TRACKER.start_tracking()
    print(TRACKER.get_frame()
    TRACKER.stop_tracking()
    TRACKER.close()

Developing
----------

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgeryarucotracker


Running the tests
^^^^^^^^^^^^^^^^^

You can run the unit tests by installing and running tox:

::

    pip install tox
    tox

Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.


Useful links
^^^^^^^^^^^^

* `Source code repository`_
* `Documentation`_


Licensing and copyright
-----------------------

Copyright 2019 University College London.
scikit-surgeryarucotracker is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgeryarucotracker
.. _`Documentation`: https://scikit-surgeryarucotracker.readthedocs.io
.. _`SNAPPY`: https://weisslab.cs.ucl.ac.uk/WEISS/PlatformManagement/SNAPPY/wikis/home
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgeryarucotracker/blob/master/CONTRIBUTING.rst
.. _`license file`: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgeryarucotracker/blob/master/LICENSE

