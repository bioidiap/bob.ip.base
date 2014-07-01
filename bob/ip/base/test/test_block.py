#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Niklas Johansson <niklas.johansson@idiap.ch>
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed Apr 6 14:16:13 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test the block extractor
"""

import numpy
import bob.ip.base

A_org    = numpy.array(range(1,17), 'float64').reshape((4,4))
A_ans_0_3D  = numpy.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13, 14]], [[11, 12], [15, 16]]], 'float64')
A_ans_0_4D  = numpy.array([[[[1, 2], [5, 6]], [[3, 4], [7, 8]]], [[[9, 10], [13, 14]], [[11, 12], [15, 16]]]], 'float64')

def test_block():
  shape_4D = (2, 2, 2, 2)
  shape = bob.ip.base.block_output_shape(A_org, (2, 2), (0, 0), False)
  assert shape == shape_4D

  B = numpy.ndarray(shape_4D, 'float64')
  bob.ip.base.block(A_org, (2, 2), (0, 0), output = B)
  assert (B == A_ans_0_4D).all()
  C = bob.ip.base.block(A_org, (2, 2), (0, 0), flat = False)
  assert (C == A_ans_0_4D).all()

  shape_3D = (4, 2, 2)
  shape = bob.ip.base.block_output_shape(A_org, (2, 2), (0, 0), flat = True)
  assert shape == shape_3D

  B = numpy.ndarray(shape_3D, 'float64')
  bob.ip.base.block(A_org, (2, 2), (0, 0), output = B)
  assert (B == A_ans_0_3D).all()


