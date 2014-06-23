#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 30 Jan 08:45:49 2014 CET

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz']))
from bob.blitz.extension import Extension

packages = ['bob-ip >= 1.2.2', 'boost']
version = '2.0.0a0'

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
      'bob.blitz',
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
          "bob/ip/base/old/block.cc",
          "bob/ip/base/old/crop_shift.cc",
          "bob/ip/base/old/DCTFeatures.cc",
          "bob/ip/base/old/extrapolate_mask.cc",
          "bob/ip/base/old/FaceEyesNorm.cc",
          "bob/ip/base/old/flipflop.cc",
          "bob/ip/base/old/gamma_correction.cc",
          "bob/ip/base/old/gaussian.cc",
          "bob/ip/base/old/GaussianScaleSpace.cc",
          "bob/ip/base/old/GeomNorm.cc",
          "bob/ip/base/old/GLCM.cc",
          "bob/ip/base/old/GLCMProp.cc",
          "bob/ip/base/old/histo.cc",
          "bob/ip/base/old/HOG.cc",
          "bob/ip/base/old/integral.cc",
          "bob/ip/base/old/LBP.cc",
          "bob/ip/base/old/Median.cc",
          "bob/ip/base/old/MultiscaleRetinex.cc",
          "bob/ip/base/old/rotate.cc",
          "bob/ip/base/old/scale.cc",
          "bob/ip/base/old/SelfQuotientImage.cc",
          "bob/ip/base/old/shear.cc",
          "bob/ip/base/old/SIFT.cc",
          "bob/ip/base/old/Sobel.cc",
          "bob/ip/base/old/TanTriggs.cc",
          "bob/ip/base/old/vldsift.cc",
          "bob/ip/base/old/vlsift.cc",
          "bob/ip/base/old/WeightedGaussian.cc",
          "bob/ip/base/old/zigzag.cc",

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
          "bob/ip/base/zigzag.cpp",
          "bob/ip/base/utils.cpp",
          "bob/ip/base/main.cpp",
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
