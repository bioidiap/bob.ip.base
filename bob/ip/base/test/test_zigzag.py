#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Niklas Johansson <niklas.johansson@idiap.ch>
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed Apr 6 14:16:13 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test the zigzag extractor
"""

import numpy
import bob.ip.base

A_org    = numpy.array(range(1,17), 'float64').reshape((4,4))
A_ans_3  = numpy.array((1, 2, 5), 'float64')
A_ans_6  = numpy.array((1, 2, 5, 9, 6, 3), 'float64')
A_ans_10 = numpy.array((1, 2, 5, 9, 6, 3, 4, 7, 10, 13), 'float64')

def test_zigzag_1():

  B = numpy.zeros((3,), numpy.float64)
  bob.ip.base.zigzag(A_org, B)
  assert  (B == A_ans_3).all()
#  C = zigzag(A_org, 3)
#  assert (C == A_ans_3).all()

def test_zigzag_2():

  B = numpy.zeros((6,), numpy.float64)
  bob.ip.base.zigzag(A_org, B)
  assert (B == A_ans_6).all()
#  C = zigzag(A_org, 6)
#  assert (C == A_ans_6).all()

def test_zigzag_3():

  B = numpy.zeros((10,), numpy.float64)
  bob.ip.base.zigzag(A_org, B)
  assert (B == A_ans_10).all()
#  C = zigzag(A_org, 10)
#  assert (C == A_ans_10).all()
