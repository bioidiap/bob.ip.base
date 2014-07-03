#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <manuel.guenther@idiap.ch>
# Thu Jul  3 14:31:48 CEST 2014
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests filtertering"""

import numpy
import nose.tools
import bob.ip.base
import bob.sp

import bob.io.base
import bob.io.base.test_utils
import bob.io.image

def test_median():
  # tests median filtering
  src = numpy.array([
      [1, 2, 3, 4, 5],
      [6, 7, 8, 9, 10],
      [11, 12, 13, 14, 15],
      [16, 17, 18, 19, 20]],
      dtype = numpy.uint16
  )
  ref = numpy.array([
    [7, 8, 9],
    [12, 13, 14]],
    dtype = numpy.uint16)

  dst = bob.ip.base.median(src, (1,1))
  assert numpy.allclose(ref, dst)

