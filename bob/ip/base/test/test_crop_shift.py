#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed Apr 6 14:16:13 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test the crop and shift operations
"""

import numpy
import bob.ip.base

A_org           = numpy.array(range(0,100), numpy.float64).reshape((10,10))
A_crop_2x2_2x2  = numpy.array([[22, 23], [32, 33]], numpy.float64)
A3_org          = numpy.array([A_org, A_org, A_org])
A3_crop_2x2_2x2 = numpy.array([A_crop_2x2_2x2, A_crop_2x2_2x2, A_crop_2x2_2x2])
B_org           = numpy.array(range(1,5), numpy.float64).reshape((2,2))
B_shift_1x1     = numpy.array([[0,0],[0,1]], numpy.float64)
B3_org          = numpy.array([B_org, B_org, B_org])
B3_shift_1x1    = numpy.array([B_shift_1x1, B_shift_1x1, B_shift_1x1])

def test_crop_2D():
  B = numpy.ndarray((2,2), numpy.float64)
  bob.ip.base.crop(A_org, (2,2), (2,2), dst=B)
  assert (B == A_crop_2x2_2x2).all()
  C = bob.ip.base.crop(A_org, (2, 2), (2, 2))
  assert (C == A_crop_2x2_2x2).all()

def test_shift_2D():
  B = numpy.ndarray((2,2), numpy.float64)
  bob.ip.base.shift(B_org, (-1,-1), B)
  assert (B == B_shift_1x1).all()
  C = bob.ip.base.shift(B_org, (-1, -1))
  assert (C == B_shift_1x1).all()

def test_crop_3D():
  B = numpy.ndarray((3,2,2), numpy.float64)
  bob.ip.base.crop(A3_org, (2, 2), (2, 2), dst=B)
  assert (B == A3_crop_2x2_2x2).all()
  C = bob.ip.base.crop(A3_org, (2, 2), (2, 2))
  assert (C == A3_crop_2x2_2x2).all()

def test_shift_3D():
  B = numpy.ndarray((3,2,2), numpy.float64)
  bob.ip.base.shift(B3_org, (-1, -1), B)
  assert (B == B3_shift_1x1).all()
  C = bob.ip.base.shift(B_org, (-1, -1))
  assert (C == B3_shift_1x1).all()
