#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Mon Apr 18 16:08:34 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests histogram computation
"""

import os
import numpy
import random

import bob.io.base
import bob.io.image
from bob.io.base.test_utils import datafile

from . import histogram, histogram_, histogram_equalization

def load_gray(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes.
  filename = os.path.join('histo', relative_filename)
  array = bob.io.base.load(datafile(filename, __name__))
  return array[0,:,:]

def random_int(array, min_value, max_value):
  for i in range(0,array.shape()[0]):
    for j in range(0,array.shape()[1]):
      array[i, j] = random.randint(min_value, max_value)

def random_float(array, min_value, max_value):
  for i in range(0,array.shape()[0]):
    for j in range(0,array.shape()[1]):
      array[i, j] = random.uniform(min_value, max_value)

def test_uint8_histoPython():

  # Compute the histogram of a uint8 image
  input_image = load_gray('image.ppm')


  histo1 = histogram(input_image)
  histo2 = histogram(input_image, 255)
  histo3 = histogram(input_image, 0, 255)
  histo4 = histogram(input_image, 0, 255, 256)

  histo5 = numpy.ndarray((256,), 'uint64')
  histo6 = numpy.ndarray((256,), 'uint64')
  histo7 = numpy.ndarray((256,), 'uint64')
  histo8 = numpy.ndarray((256,), 'uint64')

  histogram_(input_image, histo5)
  histogram_(input_image, histo6, 255)
  histogram_(input_image, histo7, 0, 255)
  histogram_(input_image, histo8, 0, 255, 256)

  # Save the computed data
  #bob.io.base.save(histo1, os.path.join('histo','image_histo.hdf5'))

  histo_ref = bob.io.base.load(datafile(os.path.join('histo','image_histo.hdf5'), __name__))

  assert input_image.size == histo1.sum()
  assert input_image.size == histo2.sum()
  assert input_image.size == histo3.sum()
  assert input_image.size == histo4.sum()
  assert input_image.size == histo5.sum()
  assert input_image.size == histo6.sum()
  assert input_image.size == histo7.sum()
  assert input_image.size == histo8.sum()
  assert (histo_ref == histo1).all()
  assert (histo_ref == histo2).all()
  assert (histo_ref == histo3).all()
  assert (histo_ref == histo4).all()
  assert (histo_ref == histo5).all()
  assert (histo_ref == histo6).all()
  assert (histo_ref == histo7).all()
  assert (histo_ref == histo8).all()

def test_uint16_histoPython():

  # Compute the histogram of a uint16 random array

  # Generate random uint16 array
  #input_array = numpy.ndarray((50, 70), 'uint16')
  #random_int(input_array, 0, 65535)
  #bob.io.base.save(input_array, os.path.join('histo','input_uint16.hdf5'))

  input_array = bob.io.base.load(datafile(os.path.join('histo','input_uint16.hdf5'), __name__))

  histo1 = histogram(input_array)
  histo2 = histogram(input_array, 65535)
  histo3 = histogram(input_array, 0, 65535)
  histo4 = histogram(input_array, 0, 65535, 65536)

  histo5 = numpy.ndarray((65536,), 'uint64')
  histo6 = numpy.ndarray((65536,), 'uint64')
  histo7 = numpy.ndarray((65536,), 'uint64')
  histo8 = numpy.ndarray((65536,), 'uint64')

  histogram_(input_array, histo5)
  histogram_(input_array, histo6, 65535)
  histogram_(input_array, histo7, 0, 65535)
  histogram_(input_array, histo8, 0, 65535, 65536)

  # Save computed data
  #bob.io.base.save(histo1, os.path.join('histo','input_uint16.histo.hdf5'))

  histo_ref = bob.io.base.load(datafile(os.path.join('histo','input_uint16.histo.hdf5'), __name__))

  assert input_array.size == histo1.sum()
  assert input_array.size == histo2.sum()
  assert input_array.size == histo3.sum()
  assert input_array.size == histo4.sum()
  assert input_array.size == histo5.sum()
  assert input_array.size == histo6.sum()
  assert input_array.size == histo7.sum()
  assert input_array.size == histo8.sum()
  assert (histo_ref == histo1).all()
  assert (histo_ref == histo2).all()
  assert (histo_ref == histo3).all()
  assert (histo_ref == histo4).all()
  assert (histo_ref == histo5).all()
  assert (histo_ref == histo6).all()
  assert (histo_ref == histo7).all()
  assert (histo_ref == histo8).all()

def test_float_histoPython():

   # Compute the histogram of a float random array

  # Generate random float32 array
  #input_array = numpy.ndarray((50, 70), 'float32')
  #random_float(input_array, 0, 1)
  #bob.io.base.save(input_array, os.path.join('histo','input_float.hdf5'))

  input_array = bob.io.base.load(datafile(os.path.join('histo','input_float.hdf5'), __name__))

  histo2 = numpy.ndarray((10,), 'uint64')

  histo1 = histogram(input_array, 0, 1, 10)
  histogram_(input_array, histo2, 0, 1, 10)

  # Save computed data
  #bob.io.base.save(histo1,os.path.join('histo','input_float.histo.hdf5'))

  histo_ref = bob.io.base.load(datafile(os.path.join('histo','input_float.histo.hdf5'), __name__))

  assert input_array.size == histo1.sum()
  assert input_array.size == histo2.sum()
  assert (histo_ref == histo1).all()
  assert (histo_ref == histo2).all()

def test_int32_histoPython():

  # Compute the histogram of a int32 random array

  # Generate random int32 array
  #input_array = numpy.ndarray((50, 70), 'int32')
  #random_int(input_array, -20,20)
  #bob.io.base.save(input_array,os.path.join('histo','input_int32.hdf5'))

  input_array = bob.io.base.load(datafile(os.path.join('histo','input_int32.hdf5'), __name__))

  histo2 = numpy.ndarray((41,), 'uint64')

  histo1 = histogram(input_array, -20, 20, 41)
  histogram_(input_array, histo2, -20, 20, 41)

  # Save computed data
  #bob.io.base.save(histo, os.path.join('histo','input_int32.histo.hdf5'))

  histo_ref = bob.io.base.load(datafile(os.path.join('histo','input_int32.histo.hdf5'), __name__))

  assert input_array.size == histo1.sum()
  assert input_array.size == histo2.sum()
  assert (histo_ref == histo1).all()
  assert (histo_ref == histo2).all()

def test_uint32_accumulate_histoPython():

  # Accumulate the histogram of a int32 random array

  input_array = bob.io.base.load(datafile(os.path.join('histo','input_int32.hdf5'), __name__))

  histo = histogram(input_array, -20, 20, 41)

  histogram_(input_array, histo, -20, 20, 41, True)

  histo_ref = bob.io.base.load(datafile(os.path.join('histo','input_int32.histo.hdf5'), __name__))

  assert input_array.size * 2 == histo.sum()
  assert (histo_ref * 2 == histo).all()

def test_uint16():

  # Simple test as described in ticket #101
  x = numpy.array([[-1., 1.],[-1., 1.]])
  res = histogram(x, -2, +2, 2)

  histo_ref = numpy.array([2, 2], 'uint64')
  assert (histo_ref == res).all()

def test_histogram_equalization():

  # Test that the histogram equalization function works as expected

  x = numpy.array(
    [[0, 1, 2, 3, 4],
     [0, 2, 4, 6, 8],
     [0, 3, 6, 9, 12],
     [0, 4, 8, 12, 16],
     [0, 5, 10, 15, 20]],
    dtype = numpy.uint8)
  y = numpy.ndarray((5,5), dtype = numpy.uint16)
  y_ref = numpy.array(
    [[0, 3276, 9830, 16383, 26214],
     [0, 9830, 26214, 36044, 42597],
     [0, 16383, 36044, 45874, 55704],
     [0, 26214, 42597, 55704, 62258],
     [0, 29490, 49151, 58981, 65535]],
    dtype = numpy.uint16)

  histogram_equalization(x,y)
  assert (y - y_ref == 0).all()

  y2 = histogram_equalization(x)
  y2_ref = numpy.array(
    [[0, 12, 38, 63, 102],
     [0, 38, 102, 140, 165],
     [0, 63, 140, 178, 216],
     [0, 102, 165, 216, 242],
     [0, 114, 191, 229, 255]],
    dtype = numpy.uint8)

  assert (y2 - y2_ref == 0).all()
