#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Sep 18 18:16:50 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the SIFT features extractor
"""

import os
import numpy
import nose.tools

import bob.sp
import bob.io.base
from bob.io.base.test_utils import datafile

import bob.ip.base

eps = 1e-4

def test_parametrization():
  # Parametrization tests
  op = bob.ip.base.SIFT((200, 250),3,4,-1,0.5,1.6,4.)
  nose.tools.eq_(op.size[0], 200)
  nose.tools.eq_(op.size[1], 250)
  nose.tools.eq_(op.scales, 3)
  nose.tools.eq_(op.octaves, 4)
  nose.tools.eq_(op.octave_min, -1)
  nose.tools.eq_(op.sigma_n, 0.5)
  nose.tools.eq_(op.sigma0, 1.6)
  nose.tools.eq_(op.kernel_radius_factor, 4.)
  op.size = (300, 350)
  op.octaves = 3
  op.scales = 4
  op.octave_min = 0
  op.sigma_n = 0.6
  op.sigma0 = 2.
  op.kernel_radius_factor = 3.
  nose.tools.eq_(op.size[0], 300)
  nose.tools.eq_(op.size[1], 350)
  nose.tools.eq_(op.octaves, 3)
  nose.tools.eq_(op.scales, 4)
  nose.tools.eq_(op.octave_min, 0)
  nose.tools.eq_(op.sigma_n, 0.6)
  nose.tools.eq_(op.sigma0, 2.)
  nose.tools.eq_(op.kernel_radius_factor, 3.)

def test_processing():
  # Processing tests
  A = bob.io.base.load(datafile("vlimg_ref.hdf5", 'bob.ip.base', 'data/sift'))
  No = 3
  Ns = 3
  sigma0 = 1.6
  sigma_n = 0.5
  cont_t = 0.03
  edge_t = 10.
  norm_t = 0.2
  f=4.
  op = bob.ip.base.SIFT(A.shape,Ns,No,0,sigma_n,sigma0,cont_t,edge_t,norm_t,f,bob.sp.BorderType.NearestNeighbour)
  kp=[bob.ip.base.GSSKeypoint(1.6,(326,270))]
  B = numpy.ndarray(op.output_shape(1), numpy.float64)
  op.compute_descriptor(A,kp,B)
  C=B[0]
  #bob.io.base.save(C, datafile(os.path.join("sift","vlimg_ref_cmp.hdf5"), __name__)) # Generated using initial bob version
  C_ref = bob.io.base.load(datafile("vlimg_ref_cmp.hdf5", 'bob.ip.base', 'data/sift'))
  assert numpy.allclose(C, C_ref, 1e-5, 1e-5)
  """
  Descriptor returned by vlfeat 0.9.14.
    Differences with our implementation are (but not limited to):
    - Single vs. double precision (with error propagation in the Gaussian pyramid)

    0          0         0          0.290434    65.2558   62.7004   62.6646    0.557657
    0.592095   0.145797  0          0.00843264 127.977    56.457     7.54261   0.352965
   97.3214     9.24475   0          0.0204793   50.0755   12.69      1.2646   20.525
   91.3951     8.68794   0.232415   0.688901     7.03954   6.8892    8.55246  41.1051
    0.0116815  0.342656  2.76365    0.350923     3.48516  29.5739  127.977    28.5115
    5.92045    4.61406   1.16143    0.00232113  45.9274   90.237   127.977    21.7975
  116.967     68.2782    0.278292   0.000890405 20.5523   23.5499    5.12068  14.6013
   63.4585    69.2397   18.4443    18.6347       7.60615   4.41878   5.29352  19.1335
    0.0283694 11.3307  127.977     16.1103       0.351831  0.762431 51.0464   13.5331
   10.6187    71.1094  127.977      6.76088      0.157741  3.84676  40.6852   23.2877
  127.977    115.818    43.3812     7.07351      0.242382  1.60356   2.59673   2.55512
   96.3921    39.6973    8.31371   16.4943      17.4623    1.30552   0.224244  1.14927
    7.40859   13.8157  127.977     25.6779       8.35931   9.28288   1.93504   1.90398
    6.50493   26.9885  127.977     32.5336      16.6373    8.03625   0.242855  0.791766
   44.7504    20.7554   35.8107    34.2561      26.2423   10.6024    2.14291  12.8046
   54.9029     2.88965   0.0166734  0.227938    18.4405    6.35371   3.85071  28.1302
  """

def test_comparison():
  # Comparisons tests
  op1 = bob.ip.base.SIFT((200,250),3,4,-1,0.5,1.6,4.)
  op1b = bob.ip.base.SIFT((200,250),3,4,-1,0.5,1.6,4.)
  op2 = bob.ip.base.SIFT((300,250),3,4,-1,0.5,1.6,4.)
  op3 = bob.ip.base.SIFT((200,350),3,4,-1,0.5,1.6,4.)
  op4 = bob.ip.base.SIFT((200,250),3,3,-1,0.5,1.6,4.)
  op5 = bob.ip.base.SIFT((200,250),4,4,-1,0.5,1.6,4.)
  op6 = bob.ip.base.SIFT((200,250),3,4,0,0.5,1.6,4.)
  op7 = bob.ip.base.SIFT((200,250),3,4,-1,0.75,1.6,4.)
  op8 = bob.ip.base.SIFT((200,250),3,4,-1,0.5,1.8,4.)
  op9 = bob.ip.base.SIFT((200,250),3,4,-1,0.5,1.6,3.)
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
