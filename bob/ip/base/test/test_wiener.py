#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the bob.ip.base.Wiener filter
"""

import os
import numpy
import tempfile
import nose.tools

import bob.sp
import bob.io.base

import bob.ip.base

def test_initialization():
  # Getters/Setters
  m = bob.ip.base.Wiener((5,4),0.5)
  nose.tools.eq_(m.size, (5,4))
  m.size = (5,6)
  nose.tools.eq_(m.size, (5,6))
  ps1 = 0.2 + numpy.fabs(numpy.random.randn(5,6))
  ps2 = 0.2 + numpy.fabs(numpy.random.randn(5,6))
  m.Ps = ps1
  assert numpy.allclose(m.Ps, ps1)
  m.Ps = ps2
  assert numpy.allclose(m.Ps, ps2)
  pn1 = 0.5
  m.Pn = pn1
  assert abs(m.Pn - pn1) < 1e-5
  var_thd = 1e-5
  m.variance_threshold = var_thd
  assert abs(m.variance_threshold - var_thd) < 1e-5

  # Comparison operators
  m2 = bob.ip.base.Wiener(m)
  assert m == m2
  assert (m != m2) is False
  m3 = bob.ip.base.Wiener(ps2, pn1)
  m3.variance_threshold = var_thd
  assert m == m3
  assert (m != m3 ) is False

  # Computation of the Wiener filter W
  w_py = 1 / (1. + m.Pn / m.Ps)
  assert numpy.allclose(m.w, w_py)


def test_load_save():
  m = bob.ip.base.Wiener((5,4),0.5)

  # Save and read from file
  filename = str(tempfile.mkstemp(".hdf5")[1])
  m.save(bob.io.base.HDF5File(filename, 'w'))
  m_loaded = bob.ip.base.Wiener(bob.io.base.HDF5File(filename))
  assert m == m_loaded
  assert (m != m_loaded) is False
  assert m.is_similar_to(m_loaded)
  # Make them different
  m_loaded.variance_threshold = 0.001
  assert (m == m_loaded) is False
  assert m != m_loaded

  # Clean-up
  os.unlink(filename)


def test_filter():
  ps = 0.2 + numpy.fabs(numpy.random.randn(5,6))
  pn = 0.5
  m = bob.ip.base.Wiener(ps,pn)

  # Python way
  sample = numpy.random.randn(5,6)
  sample_fft = bob.sp.fft(sample.astype(numpy.complex128))
  w = m.w
  sample_fft_filtered = sample_fft * m.w
  sample_filtered_py = numpy.absolute(bob.sp.ifft(sample_fft_filtered))

  # Bob c++ way
  sample_filtered0 = m.filter(sample)
  sample_filtered1 = m(sample)
  sample_filtered2 = numpy.zeros((5,6),numpy.float64)
  m.filter(sample, sample_filtered2)
  sample_filtered3 = numpy.zeros((5,6),numpy.float64)
  m.filter(sample, sample_filtered3)
  sample_filtered4 = numpy.zeros((5,6),numpy.float64)
  m(sample, sample_filtered4)
  assert numpy.allclose(sample_filtered0, sample_filtered_py)
  assert numpy.allclose(sample_filtered1, sample_filtered_py)
  assert numpy.allclose(sample_filtered2, sample_filtered_py)
  assert numpy.allclose(sample_filtered3, sample_filtered_py)
  assert numpy.allclose(sample_filtered4, sample_filtered_py)


def test_train():

  def train_wiener_ps(training_set):
    # Python implementation
    n_samples = training_set.shape[0]
    height = training_set.shape[1]
    width = training_set.shape[2]
    training_fftabs = numpy.zeros((n_samples, height, width), dtype=numpy.float64)

    for n in range(n_samples):
      sample = (training_set[n,:,:]).astype(numpy.complex128)
      training_fftabs[n,:,:] = numpy.absolute(bob.sp.fft(sample))

    mean = numpy.mean(training_fftabs, axis=0)

    for n in range(n_samples):
      training_fftabs[n,:,:] -= mean

    training_fftabs = training_fftabs * training_fftabs
    var_ps = numpy.mean(training_fftabs, axis=0)

    return var_ps


  n_samples = 20
  height = 5
  width = 6
  training_set = 0.2 + numpy.fabs(numpy.random.randn(n_samples, height, width))

  # Python implementation
  var_ps = train_wiener_ps(training_set)

  # Bob C++ implementation (variant 1) + comparison against python one
  m1 = bob.ip.base.Wiener(training_set)
  assert numpy.allclose(var_ps, m1.Ps)

  # Bob C++ implementation (variant 2) + comparison against python one
  m2 = bob.ip.base.Wiener(training_set, 0.)

  assert numpy.allclose(var_ps, m2.Ps)

