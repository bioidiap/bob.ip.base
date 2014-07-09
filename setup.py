#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 30 Jan 08:45:49 2014 CET

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz', 'bob.io.base', 'bob.sp']))
import bob.extension.utils
from bob.blitz.extension import Extension

import bob.io.base
import bob.sp

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'bob', 'ip', 'base')

packages = ['bob-ip >= 1.2.2', 'bob-io >= 1.2.2', 'bob-sp >= 1.2.2', 'boost']
version = '2.0.0a0'


class vl:

  def __init__ (self, only_static=False, have_vlfeat = True):
    """
    Searches for libvl in stock locations. Allows user to override.

    If the user sets the environment variable BOB_PREFIX_PATH, that prefixes
    the standard path locations.

    Parameters:

    only_static, boolean
      A flag, that indicates if we intend to link against the static library only.
      This will trigger our library search to disconsider shared libraries when searching.
    """

    self.name = 'vlfeat'
    header = 'vl/sift.h'
    module = 'vl'

    # get include directory
    candidates = bob.extension.utils.find_header(header)
    if not candidates:
      raise RuntimeError("could not find %s's `%s' - have you installed %s on this machine?" % (self.name, header, self.name))
    directory = os.path.dirname(candidates[0])
    self.include_directory = os.path.normpath(directory)

    # find library
    prefix = os.path.dirname(os.path.dirname(self.include_directory))
    candidates = bob.extension.utils.find_library(module, prefixes=[prefix], only_static=only_static)
    if not candidates:
      raise RuntimeError("cannot find required %s binary module `%s' - make sure libsvm is installed on `%s'" % (self.name, module, prefix))

    # libraries
    self.libraries = []
    name, ext = os.path.splitext(os.path.basename(candidates[0]))
    if ext in ['.so', '.a', '.dylib', '.dll']:
      self.libraries.append(name[3:]) #strip 'lib' from the name
    else: #link against the whole thing
      self.libraries.append(':' + os.path.basename(candidates[0]))

    # library path
    self.library_directory = os.path.dirname(candidates[0])
    # macros
    if have_vlfeat:
      self.macros = [('HAVE_%s' % self.name.upper(), '1')]
    else:
      self.macros = []


vl_pkg = vl(have_vlfeat=True)

include_dirs = [package_dir, bob.io.base.get_include(), bob.sp.get_include(), vl_pkg.include_directory]


setup(

    name='bob.ip.base',
    version=version,
    description='Basic Image Processing Utilities for Bob',
    url='http://github.com/bioidiap/bob.ip.base',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'matplotlib',
      'bob.blitz',
      'bob.sp',
      'bob.io.base',
      'bob.io.image',
    ],

    namespace_packages=[
      "bob",
      "bob.ip",
      ],

    ext_modules = [
      Extension("bob.ip.base.version",
        [
          "bob/ip/base/version.cpp",
          ],
        version = version,
        packages = packages,
        ),
      Extension("bob.ip.base._old_library",
        [
          "bob/ip/base/old/GLCM.cc",
          "bob/ip/base/old/GLCMProp.cc",
          "bob/ip/base/old/shear.cc",

          # external requirements as boost::python bindings
          "bob/ip/base/old/blitz_numpy.cc",
          "bob/ip/base/old/ndarray_numpy.cc",
          "bob/ip/base/old/ndarray.cc",
          "bob/ip/base/old/tinyvector.cc",
          "bob/ip/base/old/extrapolate.cc",
          "bob/ip/base/old/conv.cc",

          "bob/ip/base/old/main.cc",
          ],
        packages = packages,
        boost_modules = ['python'],
        version = version,
        ),
      Extension("bob.ip.base._library",
        [
          # pure C++ code
          "bob/ip/base/cpp/GeomNorm.cpp",
          "bob/ip/base/cpp/FaceEyesNorm.cpp",
          "bob/ip/base/cpp/Affine.cpp",
          "bob/ip/base/cpp/LBP.cpp",
          "bob/ip/base/cpp/LBPTop.cpp",
          "bob/ip/base/cpp/DCTFeatures.cpp",
          "bob/ip/base/cpp/TanTriggs.cpp",
          "bob/ip/base/cpp/Gaussian.cpp",
          "bob/ip/base/cpp/MultiscaleRetinex.cpp",
          "bob/ip/base/cpp/WeightedGaussian.cpp",
          "bob/ip/base/cpp/SelfQuotientImage.cpp",
          "bob/ip/base/cpp/GaussianScaleSpace.cpp",
          "bob/ip/base/cpp/SIFT.cpp",
          "bob/ip/base/cpp/HOG.cpp",

          # Python bindings
          "bob/ip/base/auxiliary.cpp",
          "bob/ip/base/geom_norm.cpp",
          "bob/ip/base/face_eyes_norm.cpp",
          "bob/ip/base/affine.cpp",
          "bob/ip/base/lbp.cpp",
          "bob/ip/base/lbp_top.cpp",
          "bob/ip/base/dct_features.cpp",
          "bob/ip/base/tan_triggs.cpp",
          "bob/ip/base/gaussian.cpp",
          "bob/ip/base/multiscale_retinex.cpp",
          "bob/ip/base/weighted_gaussian.cpp",
          "bob/ip/base/self_quotient_image.cpp",
          "bob/ip/base/gaussian_scale_space.cpp",
          "bob/ip/base/sift.cpp",
          "bob/ip/base/vl_feat.cpp",
          "bob/ip/base/hog.cpp",
          "bob/ip/base/filter.cpp",
          "bob/ip/base/utils.cpp",
          "bob/ip/base/main.cpp",
          ],
        packages = packages,
        include_dirs = include_dirs,
        version = version,
        library_dirs = [vl_pkg.library_directory],
        libraries = vl_pkg.libraries,
        define_macros = vl_pkg.macros,
        ),
      ],

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

    )
