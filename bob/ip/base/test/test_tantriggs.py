#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Aug 24 18:48:00 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the TanTriggs filter
"""

import nose.tools
from .. import TanTriggs, BorderType

eps = 1e-4

def test_parametrization():
  # Parametrization tests
  op = TanTriggs(0.2,1.,2.,2,10.,0.1)
  nose.tools.eq_(op.gamma, 0.2)
  nose.tools.eq_(op.sigma0, 1.)
  nose.tools.eq_(op.sigma1, 2.)
  nose.tools.eq_(op.radius, 2)
  nose.tools.eq_(op.threshold, 10.)
  nose.tools.eq_(op.alpha, 0.1)
  nose.tools.eq_(op.conv_border, BorderType.Mirror)
  op.gamma = 0.1
  op.sigma0 = 2.
  op.sigma1 = 3.
  op.radius = 3
  op.threshold = 8.
  op.alpha = 0.2
  op.conv_border = BorderType.Circular
  nose.tools.eq_(op.gamma, 0.1)
  nose.tools.eq_(op.sigma0, 2.)
  nose.tools.eq_(op.sigma1, 3.)
  nose.tools.eq_(op.radius, 3)
  nose.tools.eq_(op.threshold, 8.)
  nose.tools.eq_(op.alpha, 0.2)
  nose.tools.eq_(op.conv_border, BorderType.Circular)
  op.reset(0.2,1.,2.,2,10.,0.1,BorderType.Mirror)
  nose.tools.eq_(op.gamma, 0.2)
  nose.tools.eq_(op.sigma0, 1.)
  nose.tools.eq_(op.sigma1, 2.)
  nose.tools.eq_(op.radius, 2)
  nose.tools.eq_(op.threshold, 10.)
  nose.tools.eq_(op.alpha, 0.1)
  nose.tools.eq_(op.conv_border, BorderType.Mirror)

@nose.tools.nottest
def test_processing():

  # Processing tests
  # TODO (also performed in the C++ part)

  pass

def test_comparison():
  # Comparisons tests
  op1 = TanTriggs(0.2,1.,2.,2,10.,0.1)
  op1b = TanTriggs(0.2,1.,2.,2,10.,0.1)
  op2 = TanTriggs(0.2,1.,2.,2,10.,0.1, BorderType.Circular)
  op3 = TanTriggs(0.2,1.,2.,2,10.,0.2)
  op4 = TanTriggs(0.2,1.,2.,2,8.,0.1)
  op5 = TanTriggs(0.2,1.,2.,3,10.,0.1)
  op6 = TanTriggs(0.2,1.,3.,2,10.,0.1)
  op7 = TanTriggs(0.2,1.5,2.,2,10.,0.1)
  op8 = TanTriggs(0.1,1.,2.,2,10.,0.1)
  assert op1 == op1
  assert op1 == op1b
  assert (op1 == op2) is False
  assert (op1 == op3) is False
  assert (op1 == op4) is False
  assert (op1 == op5) is False
  assert (op1 == op6) is False
  assert (op1 == op7) is False
  assert (op1 == op8) is False
  assert (op1 != op1) is False
  assert (op1 != op1b) is False
  assert op1 != op2
  assert op1 != op3
  assert op1 != op4
  assert op1 != op5
  assert op1 != op6
  assert op1 != op7
  assert op1 != op8
