#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Sep 16 16:44:00 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the Gaussian Scale Space
"""

import os
import numpy
import nose.tools

import bob.io.base
from bob.io.base.test_utils import datafile

import bob.ip.base

eps = 1e-4

def test_parametrization():
  # Parametrization tests
  op = bob.ip.base.GaussianScaleSpace((200,250),3,4,-1,0.5,1.6,4.)
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
  A = bob.io.base.load(datafile("vlimg_ref.hdf5", "bob.ip.base", "data/sift"))
  No = 3
  Ns = 3
  sigma0 = 1.6
  sigma_n = 0.5
  f=4.
  op = bob.ip.base.GaussianScaleSpace(A.shape,Ns,No,0,sigma_n,sigma0,f)
  pyr = op(A)

  import math
  # Assumes that octave_min = 0
  dsigma0 = sigma0 * math.sqrt(1.-math.pow(2,-2./float(Ns)))
  Aa = A
  for o in range(No):
    for s in range(-1,Ns+2):
      # Get Gaussian for this scale
      g_pyr = op.get_gaussian(s+1)

      # Filtering step
      if s!=-1 or (o==0 and s==-1):
        # Compute scale and radius
        if(o==0 and s==-1):
          sa = sigma0 #* math.pow(2.,s/float(Ns))
          sb = sigma_n
          sigma = math.sqrt(sa*sa - sb*sb)
        else:
          sigma = dsigma0 * math.pow(2,s/float(Ns))
        radius = int(math.ceil(f*sigma))
        # Check values
        assert abs(sigma - g_pyr.sigma[0]) < eps
        assert abs(sigma - g_pyr.sigma[1]) < eps
        assert abs(radius - g_pyr.radius[0]) < eps
        assert abs(radius - g_pyr.radius[1]) < eps

        g = bob.ip.base.Gaussian((sigma, sigma), (radius, radius))
        B = g(Aa)
      # Downsampling step
      else:
        # Select image as by VLfeat (seems wrong to me)
        Aa = pyr[o-1][Ns,:,:]
        # Downsample using a trick to make sure that if the length is l=2p+1,
        # the new one is p and not p+1.
        B = Aa[:2*(int(Aa.shape[0]/2)):2,:2*(int(Aa.shape[1]/2)):2]

      # Compare image of the pyramids (Python implementation vs. C++)
      Bpyr = pyr[o][s+1,:,:]
      assert numpy.allclose(B, Bpyr, eps)
      Aa = B
      ##For saving/visualizing images
      #base_dir = '/home/user'
      #bob.io.base.save(Bpyr.astype('uint8'), os.path.join(base_dir, 'pyr_o'+str(o)+'_s'+str(s+1)+'.pgm'))

def test_comparison():
  # Comparisons tests
  op1 = bob.ip.base.GaussianScaleSpace((200,250),3,4,-1,0.5,1.6,4.)
  op1b = bob.ip.base.GaussianScaleSpace((200,250),3,4,-1,0.5,1.6,4.)
  op2 = bob.ip.base.GaussianScaleSpace((300,250),3,4,-1,0.5,1.6,4.)
  op3 = bob.ip.base.GaussianScaleSpace((200,350),3,4,-1,0.5,1.6,4.)
  op4 = bob.ip.base.GaussianScaleSpace((200,250),3,3,-1,0.5,1.6,4.)
  op5 = bob.ip.base.GaussianScaleSpace((200,250),4,4,-1,0.5,1.6,4.)
  op6 = bob.ip.base.GaussianScaleSpace((200,250),3,4,0,0.5,1.6,4.)
  op7 = bob.ip.base.GaussianScaleSpace((200,250),3,4,-1,0.75,1.6,4.)
  op8 = bob.ip.base.GaussianScaleSpace((200,250),3,4,-1,0.5,1.8,4.)
  op9 = bob.ip.base.GaussianScaleSpace((200,250),3,4,-1,0.5,1.6,3.)
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
