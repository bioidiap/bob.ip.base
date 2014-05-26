#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Apr 19 10:06:07 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests our HOG features extractor
"""

import numpy
import math

from . import HOG, GradientMaps, GradientMagnitudeType, \
    hog_compute_histogram, hog_compute_histogram_, BlockNorm, \
    normalize_block, normalize_block_

SRC_A = numpy.array([[0, 0, 4, 0, 0],  [0, 0, 4, 0, 0],  [0, 0, 4, 0, 0],
                     [0, 0, 4, 0, 0],  [0, 0, 4, 0, 0]],  dtype='float64')
MAG1_A = numpy.array([[0, 2, 0, 2, 0], [0, 2, 0, 2, 0], [0, 2, 0, 2, 0],
                      [0, 2, 0, 2, 0], [0, 2, 0, 2, 0]], dtype='float64')
MAG2_A = numpy.array([[0, 4, 0, 4, 0], [0, 4, 0, 4, 0], [0, 4, 0, 4, 0],
                      [0, 4, 0, 4, 0], [0, 4, 0, 4, 0]], dtype='float64')
SQ = math.sqrt(2)
MAGSQRT_A = numpy.array([[0, SQ, 0, SQ, 0], [0, SQ, 0, SQ, 0], [0, SQ, 0, SQ, 0],
                         [0, SQ, 0, SQ, 0], [0, SQ, 0, SQ, 0]], dtype='float64')
PI = math.pi
ORI_A = numpy.array([[0, 0, 0, PI, 0], [0, 0, 0, PI, 0], [0, 0, 0, PI, 0],
                     [0, 0, 0, PI, 0], [0, 0, 0, PI, 0]], dtype='float64')
HIST_A = numpy.array([20, 0, 0, 0, 0, 0, 0, 0], dtype='float64')

SRC_B = numpy.array([[0, 0, 0, 0, 0],  [0, 0, 0, 0, 0],  [4, 4, 4, 4, 4],
                     [0, 0, 0, 0, 0],  [0, 0, 0, 0, 0]],  dtype='float64')
MAG_B = numpy.array([[0, 0, 0, 0, 0],  [2, 2, 2, 2, 2],  [0, 0, 0, 0, 0],
                     [2, 2, 2, 2, 2],  [0, 0, 0, 0, 0]],  dtype='float64')
PIH = math.pi / 2.
ORI_B = numpy.array([[0, 0, 0, 0, 0],  [PIH, PIH, PIH, PIH, PIH],  [0, 0, 0, 0, 0],
                     [-PIH, -PIH, -PIH, -PIH, -PIH],  [0, 0, 0, 0, 0]],  dtype='float64')
HIST_B = numpy.array([0, 0, 0, 0, 20, 0, 0, 0], dtype='float64')
EPSILON = 1e-10

HIST_3D = numpy.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                       [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]], dtype='float64')
HIST_NORM_L1 = numpy.zeros(dtype='float64', shape=(20,))

IMG_8x8_A = numpy.array([ [0, 2, 0, 0, 0, 0, 2, 0],
                          [0, 2, 0, 0, 0, 0, 2, 0],
                          [0, 2, 0, 0, 0, 0, 2, 0],
                          [0, 2, 0, 0, 0, 0, 2, 0],
                          [0, 2, 0, 0, 0, 0, 2, 0],
                          [0, 2, 0, 0, 0, 0, 2, 0],
                          [0, 2, 0, 0, 0, 0, 2, 0],
                          [0, 2, 0, 0, 0, 0, 2, 0]], dtype='float64')
HIST_IMG_A = numpy.array([0.5, 0, 0, 0, 0, 0, 0, 0,
                          0.5, 0, 0, 0, 0, 0, 0, 0,
                          0.5, 0, 0, 0, 0, 0, 0, 0,
                          0.5, 0, 0, 0, 0, 0, 0, 0], dtype='float64')

def test_GradientMaps():
  #"""Test the Gradient maps computation"""

  # Declare reference arrays
  hgm = GradientMaps(5,5)
  mag = numpy.zeros(shape=(5,5), dtype='float64')
  ori = numpy.zeros(shape=(5,5), dtype='float64')

  # Magnitude
  hgm(SRC_A, mag, ori)
  assert numpy.allclose(mag, MAG1_A, EPSILON)
  assert numpy.allclose(ori, ORI_A, EPSILON)
  hgm.forward(SRC_A, mag, ori)
  assert numpy.allclose(mag, MAG1_A, EPSILON)
  assert numpy.allclose(ori, ORI_A, EPSILON)
  hgm.forward_(SRC_A, mag, ori)
  assert numpy.allclose(mag, MAG1_A, EPSILON)
  assert numpy.allclose(ori, ORI_A, EPSILON)
  (mag2, ori2) = hgm(SRC_A)
  assert numpy.allclose(mag2, MAG1_A, EPSILON)
  assert numpy.allclose(ori2, ORI_A, EPSILON)
  (mag2, ori2) = hgm.forward(SRC_A)
  assert numpy.allclose(mag2, MAG1_A, EPSILON)
  assert numpy.allclose(ori2, ORI_A, EPSILON)
  (mag2, ori2) = hgm.forward_(SRC_A)
  assert numpy.allclose(mag2, MAG1_A, EPSILON)
  assert numpy.allclose(ori2, ORI_A, EPSILON)

  # MagnitudeSquare
  hgm.magnitude_type = GradientMagnitudeType.MagnitudeSquare
  hgm.forward(SRC_A, mag, ori)
  assert numpy.allclose(mag, MAG2_A, EPSILON)
  assert numpy.allclose(ori, ORI_A, EPSILON)

  # SqrtMagnitude
  hgm.magnitude_type = GradientMagnitudeType.SqrtMagnitude
  hgm.forward(SRC_A, mag, ori)
  assert numpy.allclose(mag, MAGSQRT_A, EPSILON)
  assert numpy.allclose(ori, ORI_A, EPSILON)

  # SqrtMagnitude
  hgm.magnitude_type = GradientMagnitudeType.Magnitude
  hgm.forward(SRC_B, mag, ori)
  assert numpy.allclose(mag, MAG_B, EPSILON)
  assert numpy.allclose(ori, ORI_B, EPSILON)

  # Equal/Not equal operator
  hgm.magnitude_type = GradientMagnitudeType.Magnitude
  hgm2 = GradientMaps(5,5)
  assert hgm == hgm2
  assert (hgm != hgm2) is False
  hgm2.height = 6
  assert (hgm == hgm2) is False
  assert hgm != hgm2
  hgm2.height = 5
  assert hgm == hgm2
  assert (hgm != hgm2) is False
  hgm2.width = 6
  assert (hgm == hgm2) is False
  assert hgm != hgm2
  hgm2.width = 5
  assert hgm == hgm2
  assert (hgm != hgm2) is False
  hgm2.magnitude_type = GradientMagnitudeType.MagnitudeSquare
  assert (hgm == hgm2) is False
  assert hgm != hgm2
  hgm2.magnitude_type = GradientMagnitudeType.Magnitude
  assert hgm == hgm2
  assert (hgm != hgm2) is False

  # Resize
  hgm.resize(7,7)
  assert hgm.height == 7
  assert hgm.width == 7

  # Copy constructor
  hgm3 = GradientMaps(hgm)
  assert hgm == hgm3
  assert (hgm != hgm3) is False


def test_hogComputeCellHistogram():

  # Test the HOG computation for a given cell using hog_compute_cell()

  # Check with first input array
  hist = numpy.ndarray(shape=(8,), dtype='float64')
  hog_compute_histogram(MAG1_A, ORI_A, hist)
  assert numpy.allclose(hist, HIST_A, EPSILON)
  hog_compute_histogram_(MAG1_A, ORI_A, hist)
  assert numpy.allclose(hist, HIST_A, EPSILON)
  hist2 = hog_compute_histogram(MAG1_A, ORI_A, 8)
  assert numpy.allclose(hist2, HIST_A, EPSILON)
  hist2 = hog_compute_histogram_(MAG1_A, ORI_A, 8)
  assert numpy.allclose(hist, HIST_A, EPSILON)

  # Check with second input array
  hog_compute_histogram(MAG_B, ORI_B, hist)
  assert numpy.allclose(hist, HIST_B, EPSILON)
  hog_compute_histogram_(MAG_B, ORI_B, hist)
  assert numpy.allclose(hist, HIST_B, EPSILON)

def test_hogNormalizeBlock():

  # Test the block normalization using hog_normalize_block()

  # Vectorizes the 3D histogram into a 1D one
  HIST_1D = numpy.reshape(HIST_3D, (20,))
  # Declares 1D output histogram of size 20
  hist = numpy.ndarray(shape=(20,), dtype='float64')
  # No norm
  normalize_block(HIST_3D, hist, BlockNorm.Nonorm)
  assert numpy.allclose(hist, HIST_1D, EPSILON)
  normalize_block_(HIST_3D, hist, BlockNorm.Nonorm)
  assert numpy.allclose(hist, HIST_1D, EPSILON)
  # L2 Norm
  py_L2ref = HIST_1D / numpy.linalg.norm(HIST_1D)
  normalize_block(HIST_3D, hist)
  assert numpy.allclose(hist, py_L2ref, EPSILON)
  normalize_block_(HIST_3D, hist)
  assert numpy.allclose(hist, py_L2ref, EPSILON)
  hist2 = normalize_block(HIST_3D)
  assert numpy.allclose(hist2, py_L2ref, EPSILON)
  hist2 = normalize_block_(HIST_3D)
  assert numpy.allclose(hist2, py_L2ref, EPSILON)
  # L2Hys Norm
  py_L2Hysref = HIST_1D / numpy.linalg.norm(HIST_1D)
  py_L2Hysref = numpy.clip(py_L2Hysref, a_min=0, a_max=0.2)
  py_L2Hysref = py_L2Hysref / numpy.linalg.norm(py_L2Hysref)
  normalize_block(HIST_3D, hist, BlockNorm.L2Hys)
  assert numpy.allclose(hist, py_L2Hysref, EPSILON)
  normalize_block_(HIST_3D, hist, BlockNorm.L2Hys)
  assert numpy.allclose(hist, py_L2Hysref, EPSILON)
  # L1 Norm
  py_L1ref = HIST_1D / numpy.linalg.norm(HIST_1D, 1)
  normalize_block(HIST_3D, hist, BlockNorm.L1)
  assert numpy.allclose(hist, py_L1ref, EPSILON)
  normalize_block_(HIST_3D, hist, BlockNorm.L1)
  assert numpy.allclose(hist, py_L1ref, EPSILON)
  # L1 Norm sqrt
  py_L1sqrtref = numpy.sqrt(HIST_1D / numpy.linalg.norm(HIST_1D, 1))
  normalize_block(HIST_3D, hist, BlockNorm.L1sqrt)
  assert numpy.allclose(hist, py_L1sqrtref, EPSILON)
  normalize_block_(HIST_3D, hist, BlockNorm.L1sqrt)
  assert numpy.allclose(hist, py_L1sqrtref, EPSILON)

def test_HOG():

  # Test the HOG class which is used to perform the full feature extraction

  # HOG features extractor
  hog = HOG(8,12)
  # Check members
  assert hog.height == 8
  assert hog.width == 12
  assert hog.magnitude_type == GradientMagnitudeType.Magnitude
  assert hog.cell_dim == 8
  assert hog.full_orientation is False
  assert hog.cell_y == 4
  assert hog.cell_x == 4
  assert hog.cell_ov_y == 0
  assert hog.cell_ov_x == 0
  assert hog.block_y == 4
  assert hog.block_x == 4
  assert hog.block_ov_y == 0
  assert hog.block_ov_x == 0
  assert hog.block_norm == BlockNorm.L2
  assert hog.block_norm_eps == 1e-10
  assert hog.block_norm_threshold == 0.2

  # Resize
  hog.resize(12, 16)
  assert hog.height == 12
  assert hog.width == 16

  # Disable block normalization
  hog.disable_block_normalization()
  assert hog.block_y == 1
  assert hog.block_x == 1
  assert hog.block_ov_y == 0
  assert hog.block_ov_x == 0
  assert hog.block_norm == BlockNorm.Nonorm

  # Get the dimensionality of the output
  assert numpy.array_equal( hog.get_output_shape(), numpy.array([3,4,8]))
  hog.resize(16, 16)
  assert numpy.array_equal( hog.get_output_shape(), numpy.array([4,4,8]))
  hog.block_y = 4
  hog.block_x = 4
  hog.block_ov_y = 0
  hog.block_ov_x = 0
  assert numpy.array_equal( hog.get_output_shape(), numpy.array([1,1,128]))
  hog.cell_dim = 12
  hog.block_y = 2
  hog.block_x = 2
  hog.block_ov_y = 1
  hog.block_ov_x = 1
  assert numpy.array_equal( hog.get_output_shape(), numpy.array([3,3,48]))

  # Check descriptor computation
  hog.resize(8, 8)
  hog.cell_dim = 8
  hog.cell_y = 4
  hog.cell_x = 4
  hog.cell_ov_y = 0
  hog.cell_ov_x = 0
  hog.block_y = 2
  hog.block_x = 2
  hog.block_ov_y = 0
  hog.block_ov_x = 0
  hog.block_norm = BlockNorm.L2
  hist_3D = numpy.ndarray(dtype='float64', shape=(1,1,32))
  hog.forward(IMG_8x8_A, hist_3D)
  assert numpy.allclose( hist_3D, HIST_IMG_A, EPSILON)
  hog.forward(IMG_8x8_A.astype(numpy.uint8), hist_3D)
  assert numpy.allclose( hist_3D, HIST_IMG_A, EPSILON)
  hog.forward(IMG_8x8_A.astype(numpy.uint16), hist_3D)
  assert numpy.allclose( hist_3D, HIST_IMG_A, EPSILON)
  hist3 = hog.forward(IMG_8x8_A)
  assert numpy.allclose( hist3, HIST_IMG_A, EPSILON)
  hist3 = hog.forward(IMG_8x8_A.astype(numpy.uint8))
  assert numpy.allclose( hist3, HIST_IMG_A, EPSILON)
  hist3 = hog.forward(IMG_8x8_A.astype(numpy.uint16))
  assert numpy.allclose( hist3, HIST_IMG_A, EPSILON)

  # Check equal/not equal operators
  hog1 = HOG(8,8)
  hog2 = HOG(8,8)
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.width = 9
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.width = 8
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.height = 9
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.height = 8
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.magnitude_type = GradientMagnitudeType.SqrtMagnitude
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.magnitude_type = GradientMagnitudeType.Magnitude
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.cell_dim = 10
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.cell_dim = 8
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.full_orientation = True
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.full_orientation = False
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.cell_y = 6
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.cell_y = 4
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.cell_x = 6
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.cell_x = 4
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.cell_ov_y = 2
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.cell_ov_y = 0
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.cell_ov_x = 2
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.cell_ov_x = 0
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.block_y = 6
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.block_y = 4
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.block_x = 6
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.block_x = 4
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.block_ov_y = 2
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.block_ov_y = 0
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.block_ov_x = 2
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.block_ov_x = 0
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.block_norm = BlockNorm.L1
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.block_norm = BlockNorm.L2
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.block_norm_eps = 1e-6
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.block_norm_eps = 1e-10
  assert hog1 == hog2
  assert (hog1 != hog2) is False
  hog1.block_norm_threshold = 0.4
  assert (hog1 == hog2) is False
  assert hog1 != hog2
  hog1.block_norm_threshold = 0.2
  assert hog1 == hog2
  assert (hog1 != hog2) is False

  # Copy constructor
  hog2.resize(16,16)
  hog3 = HOG(hog2)
  assert hog3 == hog2
  assert (hog3 != hog2) is False
