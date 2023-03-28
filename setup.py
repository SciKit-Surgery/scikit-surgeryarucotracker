# coding=utf-8
"""
Setup for scikit-surgeryarucotracker
"""

from setuptools import setup, find_packages
import versioneer

# Get the long description
with open('README.rst') as f:
    long_description = f.read()

setup(
    name='scikit-surgeryarucotracker',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='scikit-surgeryarucotracker is a simple tracking interface using ARuCo markers',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/SciKit-Surgery/scikit-surgeryarucotracker',
    author='Stephen Thompson',
    author_email='s.thompson@ucl.ac.uk',
    license='BSD-3 license',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',


        'License :: OSI Approved :: BSD License',


        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    keywords='medical imaging',

    packages=find_packages(
        exclude=[
            'doc',
            'tests',
            'data',
        ]
    ),

    install_requires=[
        'numpy',
        'pillow',
        'opencv-contrib-python-headless',
        'scikit-surgerycore>=0.6.9'
    ],

    entry_points={
        'console_scripts': [
            'scikit-surgeryarucotracker=sksurgeryarucotracker.__main__:main',
        ],
    },
)
