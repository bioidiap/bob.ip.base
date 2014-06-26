#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Jul 22 18:50:40 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the MultiscaleRetinex
"""

import numpy
import nose.tools

from .. import MultiscaleRetinex, BorderType

eps = 1e-4

def test_parametrization():

  # Parametrization tests
  op = MultiscaleRetinex(2,1,1,2.)
  nose.tools.eq_(op.n_scales, 2)
  nose.tools.eq_(op.size_min, 1)
  nose.tools.eq_(op.size_step, 1)
  nose.tools.eq_(op.sigma, 2.)
  nose.tools.eq_(op.conv_border, BorderType.Mirror)
  op.n_scales = 3
  op.size_min = 2
  op.size_step = 2
  op.sigma = 1.
  op.conv_border = BorderType.Circular
  nose.tools.eq_(op.n_scales, 3)
  nose.tools.eq_(op.size_min, 2)
  nose.tools.eq_(op.size_step, 2)
  nose.tools.eq_(op.sigma, 1.)
  nose.tools.eq_(op.conv_border, BorderType.Circular)
  op.reset(1,1,1,0.5, BorderType.Mirror)
  nose.tools.eq_(op.n_scales, 1)
  nose.tools.eq_(op.size_min, 1)
  nose.tools.eq_(op.size_step, 1)
  nose.tools.eq_(op.sigma, 0.5)
  nose.tools.eq_(op.conv_border, BorderType.Mirror)

@nose.tools.nottest
def test_processing():

  # Processing tests
  # TODO
  op = MultiscaleRetinex(1,1,1,0.5)
  a_uint8 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=numpy.uint8)
  a_float64 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=numpy.float64)
  a_ones = numpy.ones(shape=(3,4), dtype=numpy.float64)
  a_g_ref = numpy.array(TODO)
  a_msr_ref = numpy.log(1.+a_float64) - numpy.log(1.+a_wg_ref)
  a_out = numpy.ndarray(dtype=numpy.float64, shape=(3,4))

  op(a_uint8, a_out)
  assert numpy.allclose(a_out, a_sqi_ref, eps, eps)
  op(a_float64, a_out)
  assert numpy.allclose(a_out, a_sqi_ref, eps, eps)
  a_out2 = op(a_float64)
  assert numpy.allclose(a_out2, a_sqi_ref, eps, eps)

def test_comparison():

  # Comparisons tests
  op1 = MultiscaleRetinex(1,1,1,0.5)
  op1b = MultiscaleRetinex(1,1,1,0.5)
  op2 = MultiscaleRetinex(1,1,1,0.5, BorderType.Circular)
  op3 = MultiscaleRetinex(1,1,1,1.)
  op4 = MultiscaleRetinex(1,1,2,0.5)
  op5 = MultiscaleRetinex(1,2,1,0.5)
  op6 = MultiscaleRetinex(2,1,1,0.5)
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
