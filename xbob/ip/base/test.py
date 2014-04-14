#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Mon Apr 14 20:51:23 CEST 2014
#
# Copyright (C) 2014 Idiap Research Institute, Martigny, Switzerland

"""Test image processing base routines
"""

import os
import platform
import numpy
import colorsys
import pkg_resources
import nose.tools
import xbob.io

from . import *

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))


def test_nothing():
  pass

