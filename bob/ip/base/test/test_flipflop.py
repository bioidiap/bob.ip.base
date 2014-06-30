#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed Apr 6 14:16:13 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test the flip and flop operations
"""

import numpy
import bob.ip.base

A_org       = numpy.array(range(1,5), numpy.float64).reshape((2,2))
A_ans_flip  = numpy.array([[3, 4], [1, 2]], numpy.float64)
A_ans_flop  = numpy.array([[2, 1], [4, 3]], numpy.float64)
A3_org      = numpy.array(range(1,13), numpy.float64).reshape((3,2,2))
A3_ans_flip = numpy.array([[[3, 4], [1, 2]], [[7, 8], [5, 6]], [[11, 12], [9,10]]], numpy.float64).reshape((3,2,2))
A3_ans_flop = numpy.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]], [[10, 9], [12,11]]], numpy.float64).reshape((3,2,2))

def test_flip_2D():
  B = numpy.ndarray((2,2), numpy.float64)
  bob.ip.base.flip(A_org, B)
  assert (B == A_ans_flip).all()
  C = bob.ip.base.flip(A_org)
  assert (C == A_ans_flip).all()

def test_flop_2D():
  B = numpy.ndarray((2,2), numpy.float64)
  bob.ip.base.flop(A_org, B)
  assert (B == A_ans_flop).all()
  C = bob.ip.base.flop(A_org)
  assert (C == A_ans_flop).all()

def test_flip_3D():
  B = numpy.ndarray((3,2,2), numpy.float64)
  bob.ip.base.flip(A3_org, B)
  assert (B == A3_ans_flip).all()
  C = bob.ip.base.flip(A3_org)
  assert (C == A3_ans_flip).all()

def test_flop_3D():
  B = numpy.ndarray((3,2,2), numpy.float64)
  bob.ip.base.flop(A3_org, B)
  assert (B == A3_ans_flop).all()
  C = bob.ip.base.flop(A3_org)
  assert (C == A3_ans_flop).all()
