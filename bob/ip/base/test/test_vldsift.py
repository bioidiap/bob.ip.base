#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Mon Jan 23 20:46:07 2012 +0100
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests our Dense SIFT features extractor based on VLFeat
"""

import os
import numpy
import functools
from nose.plugins.skip import SkipTest

import bob.io.base
import bob.io.image
from bob.io.base.test_utils import datafile

def vldsift_found(test):
  '''Decorator to check if the VLDSIFT class is present before enabling a test'''

  @functools.wraps(test)
  def wrapper(*args, **kwargs):
    try:
      from . import VLDSIFT
      return test(*args, **kwargs)
    except ImportError:
      raise SkipTest('VLFeat was not available at compile time')

  return wrapper


def load_image(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes.
  filename = os.path.join("sift", relative_filename)
  array = bob.io.base.load(datafile(filename, __name__))
  return array.astype('float32')

def equal(x, y, epsilon):
  return (abs(x - y) < epsilon)

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

@vldsift_found
def test_VLDSiftPython():

  raise SkipTest("TODO: Patch for bug #189 still not applied on branch 2.0")

  from . import VLDSIFT

  # Dense SIFT reference using VLFeat 0.9.13
  # (First 3 descriptors, Gaussian window)
  filename_beg = datafile(os.path.join("sift", "vldsift_gref_beg.hdf5"), __name__)
  ref_vl_beg = bob.io.base.load(filename_beg)
  filename_end = datafile(os.path.join("sift", "vldsift_gref_end.hdf5"), __name__)
  ref_vl_end = bob.io.base.load(filename_end)

  # Computes dense SIFT feature using VLFeat binding
  img = load_image('vlimg_ref.pgm')
  mydsift1 = VLDSIFT(img.shape[0],img.shape[1])
  out_vl = mydsift1(img)
  # Compare to reference (first 200 descriptors)
  offset = out_vl.shape[0]-200
  for i in range(200):
    assert equals(out_vl[i,:], ref_vl_beg[i,:], 2e-6)
    assert equals(out_vl[offset+i,:], ref_vl_end[i,:], 2e-6)
