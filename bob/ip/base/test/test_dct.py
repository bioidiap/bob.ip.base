#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Mar 3 18:53:00 2013 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test the DCTFeatures extractor
"""

import numpy
import bob.ip.base

A_org    = numpy.array(range(1,17), 'float64').reshape((4,4))
A_ans_0_3D  = numpy.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13, 14]], [[11, 12], [15, 16]]], 'float64')
A_ans_0_4D  = numpy.array([[[[1, 2], [5, 6]], [[3, 4], [7, 8]]], [[[9, 10], [13, 14]], [[11, 12], [15, 16]]]], 'float64')

src = numpy.array([[0, 1, 2, 3, 4, 5, 6, 7],
  [ 8,  9, 10, 11, 12, 13, 14, 15],
  [16, 17, 18, 19, 20, 21, 22, 23],
  [24, 25, 26, 27, 28, 29, 30, 31],
  [32, 33, 34, 35, 36, 37, 38, 39],
  [40, 41, 42, 43, 44, 45, 46, 47]], numpy.float64)

dst1 = numpy.array([32.9090, -3.8632, -22.6274, 0., 0., 0.], numpy.float64);
dst2 = numpy.array([46.7654, -3.8632, -22.6274, 0., 0., 0.], numpy.float64);
dst3 = numpy.array([116.0474, -3.8632, -22.6274, 0., 0., 0.], numpy.float64);
dst4 = numpy.array([129.9038, -3.8632, -22.6274, 0., 0., 0.], numpy.float64);
dst_mat = [dst1, dst2, dst3, dst4]

# Reference values from the former (Idiap internal) facereclib python scripts
srcB = numpy.array([[1.,3.,5.,2.], [5.,7.,3.,2.], [4.,7.,6.,1.], [1.,3.,5.,4.]], numpy.float64);
dstB_ff = numpy.array([[8., -2., -4.], [6., 2., 1.], [7.5, -2.5, 3.5], [8., 3., -1.]], numpy.float64);
dstB_tf = numpy.array([[-0.89442719, -1.78885438], [1.63299316, 0.81649658],
            [-1.15470054, 1.61658075], [1.60356745, -0.53452248]], numpy.float64);
dstB_ft = numpy.array([[0.76249285, -0.88259602, -1.41054884], [-1.67748427, 0.7787612, 0.40951418],
            [0.15249857, -1.09026568, 1.31954569], [0.76249285, 1.1941005, -0.31851103]], numpy.float64);
dstB_tt = numpy.array([[-0.89931199, -1.39685855], [1.00866019, 0.60685661],
            [-1.09579466, 1.22218284], [0.98644646, -0.4321809]], numpy.float64);


def test_extract():
  dct_op = bob.ip.base.DCTFeatures(6, (3, 4), (0, 0))
  # Pythonic way (2D)
  dst = dct_op(src)
  assert dst.shape[0] == len(dst_mat)
  for i in range(len(dst_mat)):
    assert numpy.allclose(dst[i,:], dst_mat[i], 1e-5, 1e-4)
  # extract (3D)
  dst = dct_op(src, flat=False)
  assert dst.shape[0]*dst.shape[1] == len(dst_mat)
  for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
      assert numpy.allclose(dst[i,j,:], dst_mat[i*dst.shape[1]+j], 1e-5, 1e-4)
  # extract (2D)
  dst_c = numpy.ndarray(shape=(4,6), dtype=numpy.float64)
  dct_op(src, dst_c)
  for i in range(len(dst_mat)):
    assert numpy.allclose(dst_c[i,:], dst_mat[i], 1e-5, 1e-4)
  # C-API way (3D)
  dst_c = numpy.ndarray(shape=(2,2,6), dtype=numpy.float64)
  dct_op(src, dst_c)
  for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
      assert numpy.allclose(dst_c[i,j,:], dst_mat[i*dst.shape[1]+j], 1e-5, 1e-4)

def test_extract_normalize():
  dct_op = bob.ip.base.DCTFeatures(3, (2, 2), (0, 0))
  dst = dct_op(srcB);
  assert numpy.allclose(dst, dstB_ff, 1e-5, 1e-8)

  dct_op.normalize_block = True
  dst = dct_op(srcB);
  assert numpy.allclose(dst, dstB_tf, 1e-5, 1e-8)

  dct_op.normalize_block = False
  dct_op.normalize_dct = True
  dst = dct_op(srcB);
  assert numpy.allclose(dst, dstB_ft, 1e-5, 1e-8)

  dct_op.normalize_block = True
  dst = dct_op(srcB);
  assert numpy.allclose(dst, dstB_tt, 1e-5, 1e-8)

def test_attributes():
  dct_op = bob.ip.base.DCTFeatures(3, (2, 2), (0, 0))
  dct_op2 = bob.ip.base.DCTFeatures(dct_op)
  assert dct_op == dct_op2
  assert (dct_op != dct_op2) == False
  dct_op.block_size = (3,3)
  assert dct_op.block_size == (3,3)
  assert (dct_op == dct_op2) == False
  assert dct_op != dct_op2

  dct_op.block_overlap = (1,1)
  assert dct_op.block_overlap == (1,1)
  dct_op.coefficients = 4
  assert dct_op.coefficients == 4
  dct_op.normalize_block = True
  assert dct_op.normalize_block == True
  dct_op.normalize_dct = True
  assert dct_op.normalize_dct == True
  dct_op.square_pattern = True
  assert dct_op.square_pattern == True
  dct_op.normalization_epsilon = 1e-15
  assert dct_op.normalization_epsilon <= 2e-15

def test04_output_shape():
  dct_op = bob.ip.base.DCTFeatures(6, (3, 4), (0, 0))
  assert dct_op.output_shape(src, flat=True) == (4,6)
  assert dct_op.output_shape(src, flat=False) == (2,2,6)
  
  
def test_3Doutput_and_2Doutput():
  numpy.random.seed(3)
  data = numpy.random.randn(10,10)

  #Comparing the regular output with the output per block WITHOUT DCT coefficient normalization
  o = bob.ip.base.DCTFeatures(5, (3,3),(1,1),True,False,False)
  assert (numpy.allclose(o(data, False).flatten(),o(data, True).flatten(),1e-10))

  #Comparing the regular output with the output per block WITH DCT coefficient normalization
  o = bob.ip.base.DCTFeatures(5,(3,3),(1,1),True,True,False)
  assert (numpy.allclose(o(data, False).flatten(),o(data, True).flatten(),1e-10))

  
  
