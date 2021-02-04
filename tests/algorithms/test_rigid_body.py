#  -*- coding: utf-8 -*-
"""Tests for the 2D to 3D registration module"""
import sksurgeryarucotracker.algorithms.rigid_bodies as rgbd

def test_rigid_body_init():
    """
    Test class initialises
    """

    rigid_body = rgbd.ArUcoRigidBody()

    rigid_body.load_3d_points('data/reference.txt')

    rigid_body.scale_3d_tags(measured_pattern_width = 10)
