#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tue Apr 26 17:25:41 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the affine transformations
"""

import math
import numpy
import nose.tools
from nose.plugins.skip import SkipTest

import bob.io.base
import bob.io.base.test_utils
import bob.io.image

import bob.ip.base

regenerate_reference = False

# test the GeomNorm class
def test_geom_norm_simple():
  # tests the geom-norm functionality (copied from old C++ tests)

  test_image = bob.io.base.load(bob.io.base.test_utils.datafile("image_r10.pgm", "bob.ip.base", "data/affine"))
  processed = numpy.ndarray((40, 40))

  # Define a Geometric normalizer
  # * rotation angle: 10 degrees
  # * scaling factor: 0.65
  # * Cropping area: 40x40
  geom_norm = bob.ip.base.GeomNorm(-10., 0.65, (40, 40), (0, 0));

  # Process giving the upper left corner as the rotation center (and the offset of the cropping area)
  geom_norm(test_image, processed, (54, 27));

  # compute normalized image
  normalized = numpy.round(processed).astype(numpy.uint8)

  # This is the actual test image that was copied from the old implementation of Bob
  reference_file = bob.io.base.test_utils.datafile("image_r10_geom_norm.pgm", "bob.ip.base", "data/affine")
  if regenerate_reference:
    bob.io.base.save(normalized, reference_file)

  reference_image = bob.io.base.load(reference_file)

  assert numpy.allclose(normalized, reference_image)


def test_geom_norm_with_mask():
  # tests the geom-norm functionality with masks (copied from old C++ tests)

  test_image = bob.io.base.load(bob.io.base.test_utils.datafile("image_r70.pgm", "bob.ip.base", "data/affine"))
  processed = numpy.ndarray((160, 160))

  test_mask = test_image != 0;
  processed_mask = numpy.ndarray(processed.shape, numpy.bool);

  # Define a Geometric normalizer
  # * rotation angle: 70 degrees
  # * scaling factor: 1.2
  # * Cropping area: 160x160
  # cropping offset: 80x80
  geom_norm = bob.ip.base.GeomNorm(-70., 1.2, (160, 160), (80, 80));

  # Process giving the masks and the center of the eye positions
  geom_norm(test_image, test_mask, processed, processed_mask, (64, 69));

  # compute normalized image
  normalized = numpy.round(processed).astype(numpy.uint8)
  mask = processed_mask.astype(numpy.uint8)*255

  # This is the actual test image that was copied from the old implementation of Bob
  reference_file = bob.io.base.test_utils.datafile("image_r70_geom_norm.pgm", "bob.ip.base", "data/affine")
  if regenerate_reference:
    bob.io.base.save(normalized, reference_file)
  reference_image = bob.io.base.load(reference_file)
  assert numpy.allclose(normalized, reference_image)

  # This is the actual test mask that was copied from the old implementation of Bob
  mask_file = bob.io.base.test_utils.datafile("image_r70_mask.pgm", "bob.ip.base", "data/affine")
  if regenerate_reference:
    bob.io.base.save(mask, mask_file)
  reference_mask = bob.io.base.load(mask_file)
  assert numpy.allclose(mask, reference_mask)


def test_geom_norm_position():
  # generate geometric normalizer that rotates by 45 degrees, scales by 2 and moves to new center (40,80)
  # (the image resolution aka cropping area 160x160 is not required for this test...)
  geom_norm = bob.ip.base.GeomNorm(45., 2., (160, 160), (40, 80));

  # define positions to be rotated
  position = (15,25)

  # we take an offset of 20,20 to rotate the point
  # the centered point is hence (-5,5)
  rotated = geom_norm(position,(20,20))

  # check the new position
  # new y-value should be 0 plus offset
  # new x value is the length of the centered vector (i.e. 5*sqrt(2)) times the scaling factor 2 plus offset
  assert numpy.allclose(rotated, (40, 80. + 5. * math.sqrt(2.) * 2))


