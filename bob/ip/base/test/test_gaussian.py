#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Jul 22 19:34:20 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the Gaussian
"""

import numpy
import math
import nose.tools

import bob.ip.base
import bob.sp

import bob.io.base
import bob.io.base.test_utils

eps = 1e-4

def test_parametrization():
  # Parametrization tests
  op = bob.ip.base.Gaussian((0.5,0.6), (1,2))
  nose.tools.eq_(op.radius[0], 1)
  nose.tools.eq_(op.radius[1], 2)
  nose.tools.eq_(op.sigma[0], 0.5)
  nose.tools.eq_(op.sigma[1], 0.6)
  nose.tools.eq_(op.border, bob.sp.BorderType.Mirror)
  op.radius = (2,4)
  op.sigma = (1.,1.5)
  op.border = bob.sp.BorderType.Circular
  nose.tools.eq_(op.radius[0], 2)
  nose.tools.eq_(op.radius[1], 4)
  nose.tools.eq_(op.sigma[0], 1.)
  nose.tools.eq_(op.sigma[1], 1.5)
  nose.tools.eq_(op.border, bob.sp.BorderType.Circular)


def test_processing():
  # Processing tests
  op = bob.ip.base.Gaussian((0.5,0.5),(1,1))
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


def _normalize(image):
  a = numpy.min(image)
  b = numpy.max(image)
  scale = 255./(b-a)
  return numpy.round((image - a) * scale).astype(numpy.uint8)

def test_image():
  # copied from the old C++ tests
  image = bob.io.base.load(bob.io.base.test_utils.datafile("image.hdf5", "bob.ip.base")).astype(numpy.float64)
  processed = numpy.ndarray(image.shape, numpy.float64)

  gaussian = bob.ip.base.Gaussian((math.sqrt(2.5),math.sqrt(2.5)), (1,1))
  gaussian(image, processed)
  normalized = _normalize(processed)

  # Compare to reference image
  reference = bob.io.base.load(bob.io.base.test_utils.datafile("image_Gaussian.hdf5", "bob.ip.base", "data/filter")).astype(numpy.float64)
  assert numpy.allclose(normalized, reference)


def test_comparison():
  # Comparisons tests
  op1 = bob.ip.base.Gaussian((0.5,0.5), (1,1))
  op1b = bob.ip.base.Gaussian((0.5,0.5), (1,1))
  op2 = bob.ip.base.Gaussian((0.5,0.5), (1,1), bob.sp.BorderType.Circular)
  op3 = bob.ip.base.Gaussian((0.5,1.), (1,1))
  op4 = bob.ip.base.Gaussian((1.,0.5), (1,1))
  op5 = bob.ip.base.Gaussian((0.5,0.5), (1,2))
  op6 = bob.ip.base.Gaussian((0.5,0.5), (2,1))
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
