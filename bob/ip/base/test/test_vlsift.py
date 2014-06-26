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
import bob.io.image
from bob.io.base.test_utils import datafile

def vlsift_found(test):
  '''Decorator to check if the VLSIFT class is present before enabling a test'''

  @functools.wraps(test)
  def wrapper(*args, **kwargs):
    try:
      from . import VLSIFT
      return test(*args, **kwargs)
    except ImportError:
      raise SkipTest('VLFeat was not available at compile time')

  return wrapper

def load_image(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes.
  filename = os.path.join("sift", relative_filename)
  array = bob.io.base.load(datafile(filename, __name__))
  return array

def equal(x, y, epsilon):
  return (abs(x - y) < epsilon)

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

@vlsift_found
def test_VLSift_parametrization():

  from . import VLSIFT

  # Creates a VLSIFT object in order to perform parametrization tests
  op = VLSIFT(48, 64, 3, 5, -1, 0.03, 10., 3.)
  nose.tools.eq_(op.height, 48)
  nose.tools.eq_(op.width, 64)
  nose.tools.eq_(op.n_intervals, 3)
  nose.tools.eq_(op.n_octaves, 5)
  nose.tools.eq_(op.octave_min, -1)
  nose.tools.eq_(op.peak_thres, 0.03)
  nose.tools.eq_(op.edge_thres, 10.)
  nose.tools.eq_(op.magnif, 3.)
  op.height = 64
  op.width = 96
  op.n_intervals = 4
  op.n_octaves = 6
  op.octave_min = 0
  op.peak_thres = 0.02
  op.edge_thres = 8.
  op.magnif = 2.
  nose.tools.eq_(op.height, 64)
  nose.tools.eq_(op.width, 96)
  nose.tools.eq_(op.n_intervals, 4)
  nose.tools.eq_(op.n_octaves, 6)
  nose.tools.eq_(op.octave_min, 0)
  nose.tools.eq_(op.peak_thres, 0.02)
  nose.tools.eq_(op.edge_thres, 8.)
  nose.tools.eq_(op.magnif, 2.)

@vlsift_found
def test_VLSiftKeypointsPython():

  from . import VLSIFT

  # Computes SIFT feature using VLFeat binding
  img = load_image('vlimg_ref.pgm')
  mysift1 = VLSIFT(img.shape[0],img.shape[1], 3, 5, 0)
  # Define keypoints: (y, x, sigma, orientation)
  kp=numpy.array([[75., 50., 1., 1.], [100., 100., 3., 0.]], dtype=numpy.float64)
  # Compute SIFT descriptors at the given keypoints
  out_vl = mysift1(img, kp)
  # Compare to reference
  ref_vl = bob.io.base.load(datafile(os.path.join('sift','vlimg_ref_siftKP.hdf5'), __name__))
  for kp in range(kp.shape[0]):
    # First 4 values are the keypoint descriptions
    assert equals(out_vl[kp][4:], ref_vl[kp,:], 1e-3)

@vlsift_found
def test_comparison():

  from . import VLSIFT

  # Comparisons tests
  op1 = VLSIFT(48, 64, 3, 5, -1, 0.03, 10., 3.)
  op1b = VLSIFT(48, 64, 3, 5, -1, 0.03, 10., 3.)
  op2 = VLSIFT(48, 64, 3, 5, -1, 0.03, 10., 2.)
  op3 = VLSIFT(48, 64, 3, 5, -1, 0.03, 8., 3.)
  op4 = VLSIFT(48, 64, 3, 5, -1, 0.02, 10., 3.)
  op5 = VLSIFT(48, 64, 3, 5, 0, 0.03, 10., 3.)
  op6 = VLSIFT(48, 64, 3, 4, -1, 0.03, 10., 3.)
  op7 = VLSIFT(48, 64, 2, 5, -1, 0.03, 10., 3.)
  op8 = VLSIFT(48, 96, 3, 5, -1, 0.03, 10., 3.)
  op9 = VLSIFT(128, 64, 3, 5, -1, 0.03, 10., 3.)
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
