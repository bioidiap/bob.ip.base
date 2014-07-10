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
from bob.io.base.test_utils import datafile

import bob.ip.base

def random_int(array, min_value, max_value):
  for i in range(0,array.shape()[0]):
    for j in range(0,array.shape()[1]):
      array[i, j] = random.randint(min_value, max_value)

def random_float(array, min_value, max_value):
  for i in range(0,array.shape()[0]):
    for j in range(0,array.shape()[1]):
      array[i, j] = random.uniform(min_value, max_value)


def test_small():
  # Simple test as described in ticket #101
  x = numpy.array([[-1., 1.],[-1., 1.]])
  res = bob.ip.base.histogram(x, (-2, +2), 2)

  histo_ref = numpy.array([2, 2], 'uint64')
  assert (histo_ref == res).all()


def test_uint8_histoPython():

  # Compute the histogram of a uint8 image
  input_image = bob.io.base.load(datafile('image.hdf5', "bob.ip.base"))


  histo1 = bob.ip.base.histogram(input_image)
  histo2 = bob.ip.base.histogram(input_image, 256)
  histo3 = bob.ip.base.histogram(input_image, (0, 255), 256)

  histo4 = numpy.ndarray((256,), numpy.uint64)
  histo5 = numpy.ndarray((256,), numpy.uint64)

  bob.ip.base.histogram(input_image, histo4)
  bob.ip.base.histogram(input_image, (0, 255), histo5)

  # Save the computed data
  #bob.io.base.save(histo1, datafile('image_histo.hdf5', 'bob.ip.base', 'data/histo'))

  histo_ref = bob.io.base.load(datafile('image_histo.hdf5', 'bob.ip.base', 'data/histo'))

  assert input_image.size == histo1.sum()
  assert input_image.size == histo2.sum()
  assert input_image.size == histo3.sum()
  assert input_image.size == histo4.sum()
  assert input_image.size == histo5.sum()
  assert (histo_ref == histo1).all()
  assert (histo_ref == histo2).all()
  assert (histo_ref == histo3).all()
  assert (histo_ref == histo4).all()
  assert (histo_ref == histo5).all()

def test_uint16_histoPython():

  # Compute the histogram of a uint16 random array

  # Generate random uint16 array
  #input_array = numpy.ndarray((50, 70), 'uint16')
  #random_int(input_array, 0, 65535)
  #bob.io.base.save(input_array, os.path.join('histo','input_uint16.hdf5'))

  input_array = bob.io.base.load(datafile('input_uint16.hdf5', 'bob.ip.base', 'data/histo'))

  histo1 = bob.ip.base.histogram(input_array)
  histo2 = bob.ip.base.histogram(input_array, 65536)
  histo3 = bob.ip.base.histogram(input_array, (0, 65535), 65536)

  histo4 = numpy.ndarray((65536,), numpy.uint64)
  histo5 = numpy.ndarray((65536,), numpy.uint64)

  bob.ip.base.histogram(input_array, histo4)
  bob.ip.base.histogram(input_array, (0, 65535), histo5)

  # Save computed data
  #bob.io.base.save(histo1, os.path.join('histo','input_uint16.histo.hdf5'))

  histo_ref = bob.io.base.load(datafile('input_uint16.histo.hdf5', 'bob.ip.base', 'data/histo'))

  assert input_array.size == histo1.sum()
  assert input_array.size == histo2.sum()
  assert input_array.size == histo3.sum()
  assert input_array.size == histo4.sum()
  assert input_array.size == histo5.sum()
  assert (histo_ref == histo1).all()
  assert (histo_ref == histo2).all()
  assert (histo_ref == histo3).all()
  assert (histo_ref == histo4).all()
  assert (histo_ref == histo5).all()


def test_float_histoPython():
  # Compute the histogram of a float random array
  # Generate random float32 array
  #input_array = numpy.ndarray((50, 70), 'float32')
  #random_float(input_array, 0, 1)
  #bob.io.base.save(input_array, os.path.join('histo','input_float.hdf5'))

  input_array = bob.io.base.load(datafile('input_float.hdf5', 'bob.ip.base', 'data/histo'))
  histo2 = numpy.ndarray((10,), numpy.uint64)

  histo1 = bob.ip.base.histogram(input_array, (0, 1), 10)
  bob.ip.base.histogram(input_array, (0, 1), histo2)

  # Save computed data
  #bob.io.base.save(histo1,os.path.join('histo','input_float.histo.hdf5'))

  histo_ref = bob.io.base.load(datafile('input_float.histo.hdf5', 'bob.ip.base', 'data/histo'))

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

  input_array = bob.io.base.load(datafile('input_int32.hdf5', 'bob.ip.base', 'data/histo'))
  histo2 = numpy.ndarray((41,), numpy.uint64)

  histo1 = bob.ip.base.histogram(input_array, (-20, 20), 41)
  bob.ip.base.histogram(input_array, (-20, 20), histo2)

  # Save computed data
  #bob.io.base.save(histo, os.path.join('histo','input_int32.histo.hdf5'))
  histo_ref = bob.io.base.load(datafile('input_int32.histo.hdf5', 'bob.ip.base', 'data/histo'))

  assert input_array.size == histo1.sum()
  assert input_array.size == histo2.sum()
  assert (histo_ref == histo1).all()
  assert (histo_ref == histo2).all()


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

  bob.ip.base.histogram_equalization(x,y)
  assert (y - y_ref == 0).all()

  # in-place equalization
  bob.ip.base.histogram_equalization(x)
  y2_ref = numpy.array(
    [[0, 12, 38, 63, 102],
     [0, 38, 102, 140, 165],
     [0, 63, 140, 178, 216],
     [0, 102, 165, 216, 242],
     [0, 114, 191, 229, 255]],
    dtype = numpy.uint8)

  assert (x - y2_ref == 0).all()
