#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 30 Jan 08:45:49 2014 CET

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['xbob.blitz']))
from xbob.blitz.extension import Extension

packages = ['bob-ip >= 1.2.2', 'boost']
version = '2.0.0a0'

setup(

    name='xbob.ip.base',
    version=version,
    description='Basic Image Processing Utilities for Bob',
    url='http://github.com/bioidiap/xbob.ip.base',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'xbob.blitz',
      'xbob.io.base',
      'xbob.io.image',
    ],

    namespace_packages=[
      "xbob",
      "xbob.ip",
      ],

    ext_modules = [
      Extension("xbob.ip.base.version",
        [
          "xbob/ip/base/version.cpp",
          ],
        version = version,
        packages = packages,
        ),
      Extension("xbob.ip.base._old_library",
        [
          "xbob/ip/base/old/block.cc",
          "xbob/ip/base/old/crop_shift.cc",
          "xbob/ip/base/old/DCTFeatures.cc",
          "xbob/ip/base/old/extrapolate_mask.cc",
          "xbob/ip/base/old/FaceEyesNorm.cc",
          "xbob/ip/base/old/flipflop.cc",
          "xbob/ip/base/old/GaborWaveletTransform.cc",
          "xbob/ip/base/old/gamma_correction.cc",
          "xbob/ip/base/old/gaussian.cc",
          "xbob/ip/base/old/GaussianScaleSpace.cc",
          "xbob/ip/base/old/GeomNorm.cc",
          "xbob/ip/base/old/GLCM.cc",
          "xbob/ip/base/old/GLCMProp.cc",
          "xbob/ip/base/old/histo.cc",
          "xbob/ip/base/old/HOG.cc",
          "xbob/ip/base/old/integral.cc",
          "xbob/ip/base/old/LBP.cc",
          "xbob/ip/base/old/Median.cc",
          "xbob/ip/base/old/MultiscaleRetinex.cc",
          "xbob/ip/base/old/rotate.cc",
          "xbob/ip/base/old/scale.cc",
          "xbob/ip/base/old/SelfQuotientImage.cc",
          "xbob/ip/base/old/shear.cc",
          "xbob/ip/base/old/SIFT.cc",
          "xbob/ip/base/old/Sobel.cc",
          "xbob/ip/base/old/TanTriggs.cc",
          "xbob/ip/base/old/vldsift.cc",
          "xbob/ip/base/old/vlsift.cc",
          "xbob/ip/base/old/WeightedGaussian.cc",
          "xbob/ip/base/old/zigzag.cc",

          # external requirements as boost::python bindings
          "xbob/ip/base/old/blitz_numpy.cc",
          "xbob/ip/base/old/ndarray_numpy.cc",
          "xbob/ip/base/old/ndarray.cc",
          "xbob/ip/base/old/tinyvector.cc",
          "xbob/ip/base/old/extrapolate.cc",
          "xbob/ip/base/old/conv.cc",

          "xbob/ip/base/old/main.cc",
          ],
        packages = packages,
        boost_modules = ['python'],
        version = version,
        ),
      Extension("xbob.ip.base._library",
        [
          "xbob/ip/base/zigzag.cpp",
          "xbob/ip/base/utils.cpp",
          "xbob/ip/base/main.cpp",
          ],
        packages = packages,
        version = version,
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
