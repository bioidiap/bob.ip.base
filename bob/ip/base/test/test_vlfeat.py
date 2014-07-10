#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Mon Jan 23 20:46:07 2012 +0100
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests our SIFT features extractor based on VLFeat
"""

import os
import numpy
import functools
import nose.tools
from nose.plugins.skip import SkipTest

import bob.io.base
import bob.io.base.test_utils

import bob.ip.base


def vlsift_found(test):
  '''Decorator to check if the VLSIFT class is present before enabling a test'''

  @functools.wraps(test)
  def wrapper(*args, **kwargs):
    try:
      from bob.ip.base import VLSIFT
      return test(*args, **kwargs)
    except ImportError:
      raise SkipTest('VLFeat was not available at compile time')

  return wrapper

@vlsift_found
def test_VLSift_parametrization():
  # Creates a VLSIFT object in order to perform parametrization tests
  op = bob.ip.base.VLSIFT((48, 64), 3, 5, -1, 0.03, 10., 3.)
  nose.tools.eq_(op.size[0], 48)
  nose.tools.eq_(op.size[1], 64)
  nose.tools.eq_(op.scales, 3)
  nose.tools.eq_(op.octaves, 5)
  nose.tools.eq_(op.octave_min, -1)
  nose.tools.eq_(op.peak_threshold, 0.03)
  nose.tools.eq_(op.edge_threshold, 10.)
  nose.tools.eq_(op.magnif, 3.)
  op.size = (64, 96)
  op.scales = 4
  op.octaves = 6
  op.octave_min = 0
  op.peak_threshold = 0.02
  op.edge_threshold = 8.
  op.magnif = 2.
  nose.tools.eq_(op.size[0], 64)
  nose.tools.eq_(op.size[1], 96)
  nose.tools.eq_(op.scales, 4)
  nose.tools.eq_(op.octaves, 6)
  nose.tools.eq_(op.octave_min, 0)
  nose.tools.eq_(op.peak_threshold, 0.02)
  nose.tools.eq_(op.edge_threshold, 8.)
  nose.tools.eq_(op.magnif, 2.)

@vlsift_found
def test_VLSiftKeypointsPython():
  # Computes SIFT feature using VLFeat binding
  img =  bob.io.base.load(bob.io.base.test_utils.datafile('vlimg_ref.hdf5', 'bob.ip.base', "data/sift"))
  mysift1 = bob.ip.base.VLSIFT(img.shape, 3, 5, 0)
  # Define keypoints: (y, x, sigma, orientation)
  kp=numpy.array([[75., 50., 1., 1.], [100., 100., 3., 0.]], dtype=numpy.float64)
  # Compute SIFT descriptors at the given keypoints
  out_vl = mysift1(img, kp)
  # Compare to reference
  ref_vl = bob.io.base.load(bob.io.base.test_utils.datafile('vlimg_ref_siftKP.hdf5', 'bob.ip.base', "data/sift"))
  for kp in range(kp.shape[0]):
    # First 4 values are the keypoint descriptions
    assert numpy.allclose(out_vl[kp][4:], ref_vl[kp,:], 1e-6, 1e-3)

@vlsift_found
def test_comparison():
  # Comparisons tests
  op1 = bob.ip.base.VLSIFT((48, 64), 3, 5, -1, 0.03, 10., 3.)
  op1b = bob.ip.base.VLSIFT((48, 64), 3, 5, -1, 0.03, 10., 3.)
  op2 = bob.ip.base.VLSIFT((48, 64), 3, 5, -1, 0.03, 10., 2.)
  op3 = bob.ip.base.VLSIFT((48, 64), 3, 5, -1, 0.03, 8., 3.)
  op4 = bob.ip.base.VLSIFT((48, 64), 3, 5, -1, 0.02, 10., 3.)
  op5 = bob.ip.base.VLSIFT((48, 64), 3, 5, 0, 0.03, 10., 3.)
  op6 = bob.ip.base.VLSIFT((48, 64), 3, 4, -1, 0.03, 10., 3.)
  op7 = bob.ip.base.VLSIFT((48, 64), 2, 5, -1, 0.03, 10., 3.)
  op8 = bob.ip.base.VLSIFT((48, 96), 3, 5, -1, 0.03, 10., 3.)
  op9 = bob.ip.base.VLSIFT((128, 64), 3, 5, -1, 0.03, 10., 3.)
  assert op1 == op1
  assert op1 == op1b
  assert (op1 == op2) is False
  assert (op1 == op3) is False
  assert (op1 == op4) is False
  assert (op1 == op5) is False
  assert (op1 == op6) is False
  assert (op1 == op7) is False
  assert (op1 == op8) is False
  assert (op1 == op9) is False
  assert (op1 != op1) is False
  assert (op1 != op1b) is False
  assert op1 != op2
  assert op1 != op3
  assert op1 != op4
  assert op1 != op5
  assert op1 != op6
  assert op1 != op7
  assert op1 != op8
  assert op1 != op9


@vlsift_found
def test_VLDSiftPython():
  # Dense SIFT reference using VLFeat 0.9.13
  # (First 3 descriptors, Gaussian window)
  ref_vl_beg = bob.io.base.load(bob.io.base.test_utils.datafile("vldsift_gref_beg.hdf5", 'bob.ip.base', "data/sift"))
  ref_vl_end = bob.io.base.load(bob.io.base.test_utils.datafile("vldsift_gref_end.hdf5", 'bob.ip.base', "data/sift"))

  # Computes dense SIFT feature using VLFeat binding
  img = bob.io.base.load(bob.io.base.test_utils.datafile('vlimg_ref.hdf5', 'bob.ip.base', "data/sift")).astype(numpy.float32)
  mydsift1 = bob.ip.base.VLDSIFT(img.shape)
  out_vl = mydsift1(img)
  # Compare to reference (first 200 descriptors)
  offset = out_vl.shape[0]-200
  for i in range(200):
    assert numpy.allclose(out_vl[i,:], ref_vl_beg[i,:], 1e-8, 1e-6)
    assert numpy.allclose(out_vl[offset+i,:], ref_vl_end[i,:], 1e-8, 1e-6)
