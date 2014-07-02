#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Aug 24 18:48:00 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the TanTriggs filter
"""

import numpy
import nose.tools
import bob.ip.base
import bob.sp

import bob.io.base
import bob.io.base.test_utils
import bob.io.image


regenerate_reference = False

eps = 1e-4

def test_parametrization():
  # Parametrization tests
  op = bob.ip.base.TanTriggs(0.2,1.,2.,2,10.,0.1)
  nose.tools.eq_(op.gamma, 0.2)
  nose.tools.eq_(op.sigma0, 1.)
  nose.tools.eq_(op.sigma1, 2.)
  nose.tools.eq_(op.radius, 2)
  nose.tools.eq_(op.threshold, 10.)
  nose.tools.eq_(op.alpha, 0.1)
  nose.tools.eq_(op.border, bob.sp.BorderType.Mirror)
  op.gamma = 0.1
  op.sigma0 = 2.
  op.sigma1 = 3.
  op.radius = 3
  op.threshold = 8.
  op.alpha = 0.2
  op.border = bob.sp.BorderType.Circular
  nose.tools.eq_(op.gamma, 0.1)
  nose.tools.eq_(op.sigma0, 2.)
  nose.tools.eq_(op.sigma1, 3.)
  nose.tools.eq_(op.radius, 3)
  nose.tools.eq_(op.threshold, 8.)
  nose.tools.eq_(op.alpha, 0.2)
  nose.tools.eq_(op.border,  bob.sp.BorderType.Circular)


def _normalize(image):
  a = numpy.min(image)
  b = numpy.max(image)
  scale = 255./(b-a)
  print scale
  return numpy.round((image - a) * scale).astype(numpy.uint8)

def test_processing():
  # Processing tests, as copied performed in the C++ part)

  # Load original image
  image = bob.io.base.load(bob.io.base.test_utils.datafile("image.pgm", "bob.ip.base"))

  # First test
  tt = bob.ip.base.TanTriggs()
  processed = tt(image)
  normalized = _normalize(processed)

  if regenerate_reference:
    bob.io.base.save(normalized, bob.io.base.test_utils.datafile("image_tantriggs.hdf5", "bob.ip.base", "data/preprocessing"))
  reference_image = bob.io.base.load(bob.io.base.test_utils.datafile("image_tantriggs.hdf5", "bob.ip.base", "data/preprocessing"))
  assert numpy.allclose(normalized, reference_image)

  # Second test (comparison with matlab implementation from X. Tan)
  tt2 = bob.ip.base.TanTriggs(0.2, 1., 2., 6, 10., 0.1, bob.sp.BorderType.Mirror)
  tt2(image, processed)
  bob.io.base.save(processed, "/scratch/mguenther/mine.hdf5")
  normalized = _normalize(processed)
  reference_image = bob.io.base.load(bob.io.base.test_utils.datafile("image_tantriggs_MATLABREF.hdf5", "bob.ip.base", "data/preprocessing"))
  assert numpy.mean(numpy.abs(normalized.astype(numpy.float64) - reference_image.astype(numpy.float64))) / 255. < 6e-2


def test_comparison():
  # Comparisons tests
  op1 = bob.ip.base.TanTriggs(0.2,1.,2.,2,10.,0.1)
  op1b = bob.ip.base.TanTriggs(0.2,1.,2.,2,10.,0.1)
  op2 = bob.ip.base.TanTriggs(0.2,1.,2.,2,10.,0.1, bob.sp.BorderType.Circular)
  op3 = bob.ip.base.TanTriggs(0.2,1.,2.,2,10.,0.2)
  op4 = bob.ip.base.TanTriggs(0.2,1.,2.,2,8.,0.1)
  op5 = bob.ip.base.TanTriggs(0.2,1.,2.,3,10.,0.1)
  op6 = bob.ip.base.TanTriggs(0.2,1.,3.,2,10.,0.1)
  op7 = bob.ip.base.TanTriggs(0.2,1.5,2.,2,10.,0.1)
  op8 = bob.ip.base.TanTriggs(0.1,1.,2.,2,10.,0.1)
  assert op1 == op1
  assert op1 == op1b
  assert (op1 == op2) is False
  assert (op1 == op3) is False
  assert (op1 == op4) is False
  assert (op1 == op5) is False
  assert (op1 == op6) is False
  assert (op1 == op7) is False
  assert (op1 == op8) is False
  assert (op1 != op1) is False
  assert (op1 != op1b) is False
  assert op1 != op2
  assert op1 != op3
  assert op1 != op4
  assert op1 != op5
  assert op1 != op6
  assert op1 != op7
  assert op1 != op8
