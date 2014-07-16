#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Jul 22 17:30:00 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the bob.ip.base.SelfQuotientImage
"""

import numpy
import math
import nose.tools

import bob.sp
import bob.ip.base

eps = 1e-4

def test_parametrization():
  # Parametrization tests
  op = bob.ip.base.SelfQuotientImage(2,1,1,2.)
  nose.tools.eq_(op.scales, 2)
  nose.tools.eq_(op.size_min, 1)
  nose.tools.eq_(op.size_step, 1)
  nose.tools.eq_(op.sigma, 2.)
  nose.tools.eq_(op.border, bob.sp.BorderType.Mirror)
  op.scales = 3
  op.size_min = 2
  op.size_step = 2
  op.sigma = 1.
  op.border = bob.sp.BorderType.Circular
  nose.tools.eq_(op.scales, 3)
  nose.tools.eq_(op.size_min, 2)
  nose.tools.eq_(op.size_step, 2)
  nose.tools.eq_(op.sigma, 1.)
  nose.tools.eq_(op.border, bob.sp.BorderType.Circular)


def test_processing():
  # Processing tests
  op = bob.ip.base.SelfQuotientImage(1,1,1,math.sqrt(0.5))
  a_uint8 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=numpy.uint8)
  a_float64 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=numpy.float64)
  a_ones = numpy.ones(shape=(3,4), dtype=numpy.float64)
  a_wg_ref = numpy.array([
    [1.21194, 2, 3, 3.78806],
    [3.79444, 7.45636, 8.45636, 9.20556],
    [9.21194, 10, 11, 11.7881]],
    dtype = numpy.float64
  )
  a_sqi_ref = numpy.log(1.+a_float64) - numpy.log(1.+a_wg_ref)
  a_out = numpy.ndarray(dtype=numpy.float64, shape=(3,4))

  op(a_uint8, a_out)
  assert numpy.allclose(a_out, a_sqi_ref, eps, eps)
  op(a_float64, a_out)
  assert numpy.allclose(a_out, a_sqi_ref, eps, eps)
  a_out2 = op(a_float64)
  assert numpy.allclose(a_out2, a_sqi_ref, eps, eps)


def test_comparison():
  # Comparisons tests
  op1 = bob.ip.base.SelfQuotientImage(1,1,1,0.5)
  op1b = bob.ip.base.SelfQuotientImage(1,1,1,0.5)
  op2 = bob.ip.base.SelfQuotientImage(1,1,1,0.5, bob.sp.BorderType.Circular)
  op3 = bob.ip.base.SelfQuotientImage(1,1,1,1.)
  op4 = bob.ip.base.SelfQuotientImage(1,1,2,0.5)
  op5 = bob.ip.base.SelfQuotientImage(1,2,1,0.5)
  op6 = bob.ip.base.SelfQuotientImage(2,1,1,0.5)
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
