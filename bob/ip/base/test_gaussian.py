#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Jul 22 19:34:20 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the Gaussian
"""

import numpy
import nose.tools
from . import Gaussian, BorderType

eps = 1e-4

def test_parametrization():
  # Parametrization tests
  op = Gaussian(1,1,0.5,0.5)
  nose.tools.eq_(op.radius_y, 1)
  nose.tools.eq_(op.radius_x, 1)
  nose.tools.eq_(op.sigma_y, 0.5)
  nose.tools.eq_(op.sigma_x, 0.5)
  nose.tools.eq_(op.conv_border, BorderType.Mirror)
  op.radius_y = 2
  op.radius_x = 2
  op.sigma_y = 1.
  op.sigma_x = 1.
  op.conv_border = BorderType.Circular
  nose.tools.eq_(op.radius_y, 2)
  nose.tools.eq_(op.radius_x, 2)
  nose.tools.eq_(op.sigma_y, 1.)
  nose.tools.eq_(op.sigma_x, 1.)
  nose.tools.eq_(op.conv_border, BorderType.Circular)
  op.reset(1,1,0.5,0.5, BorderType.Mirror)
  nose.tools.eq_(op.radius_y, 1)
  nose.tools.eq_(op.radius_x, 1)
  nose.tools.eq_(op.sigma_y, 0.5)
  nose.tools.eq_(op.sigma_x, 0.5)
  nose.tools.eq_(op.conv_border, BorderType.Mirror)

def test_processing():
  # Processing tests
  op = Gaussian(1,1,0.5,0.5)
  a_uint8 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=numpy.uint8)
  a_float64 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=numpy.float64)
  a_ref = numpy.array([[1.5325, 2.4260, 3.42602, 4.3195], [5.1065, 6., 7., 7.8935], [8.6805, 9.5740, 10.5740, 11.4675]])
  a_out = numpy.ndarray(dtype=numpy.float64, shape=(3,4))

  op(a_uint8, a_out)
  assert numpy.allclose(a_out, a_ref, eps, eps)
  op(a_float64, a_out)
  assert numpy.allclose(a_out, a_ref, eps, eps)
  a_out2 = op(a_float64)
  assert numpy.allclose(a_out2, a_ref, eps, eps)

def test_comparison():
  # Comparisons tests
  op1 = Gaussian(1,1,0.5,0.5)
  op1b = Gaussian(1,1,0.5,0.5)
  op2 = Gaussian(1,1,0.5,0.5, BorderType.Circular)
  op3 = Gaussian(1,1,0.5,1.)
  op4 = Gaussian(1,1,1.,0.5)
  op5 = Gaussian(1,2,0.5,0.5)
  op6 = Gaussian(2,1,0.5,0.5)
  assert op1 == op1
  assert op1 == op1b
  assert (op1 == op2) is False
  assert (op1 == op3) is False
  assert (op1 == op4) is False
  assert (op1 == op5) is False
  assert (op1 == op6) is False
  assert (op1 != op1) is False
  assert (op1 != op1b) is False
  assert op1 != op2
  assert op1 != op3
  assert op1 != op4
  assert op1 != op5
  assert op1 != op6
