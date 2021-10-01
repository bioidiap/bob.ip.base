#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 30 Jan 08:45:49 2014 CET

bob_packages = ['bob.core', 'bob.io.base', 'bob.sp', 'bob.math']

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.extension', 'bob.blitz'] + bob_packages))
import bob.extension.utils
from bob.blitz.extension import Extension, Library, build_ext

from bob.extension.utils import load_requirements
build_requires = load_requirements()

# Define package version
version = open("version.txt").read().rstrip()

import os
packages = ['boost']
boost_modules = ['system']



setup(

    name='bob.ip.base',
    version=version,
    description='Basic Image Processing Utilities for Bob',
    url='http://gitlab.idiap.ch/bob/bob.ip.base',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    setup_requires = build_requires,
    install_requires = build_requires,



    ext_modules = [
      Extension("bob.ip.base.version",
        [
          "bob/ip/base/version.cpp",
        ],
        bob_packages = bob_packages,
        packages = packages,
        boost_modules = boost_modules,
        version = version,
      ),

      Library("bob.ip.base.bob_ip_base",
        [
          "bob/ip/base/cpp/GeomNorm.cpp",
          "bob/ip/base/cpp/FaceEyesNorm.cpp",
          "bob/ip/base/cpp/Affine.cpp",
          "bob/ip/base/cpp/LBP.cpp",
          "bob/ip/base/cpp/LBPTop.cpp",
          "bob/ip/base/cpp/DCTFeatures.cpp",
          "bob/ip/base/cpp/TanTriggs.cpp",
          "bob/ip/base/cpp/Gaussian.cpp",
          "bob/ip/base/cpp/WeightedGaussian.cpp",          
          "bob/ip/base/cpp/HOG.cpp",
          "bob/ip/base/cpp/GLCM.cpp",
        ],
        packages = packages,
        boost_modules = boost_modules,
        bob_packages = bob_packages,
        version = version,
      ),

      Extension("bob.ip.base._library",
        [
          "bob/ip/base/auxiliary.cpp",
          "bob/ip/base/geom_norm.cpp",
          "bob/ip/base/face_eyes_norm.cpp",
          "bob/ip/base/affine.cpp",
          "bob/ip/base/lbp.cpp",
          "bob/ip/base/lbp_top.cpp",
          "bob/ip/base/dct_features.cpp",
          "bob/ip/base/tan_triggs.cpp",
          "bob/ip/base/gaussian.cpp",        
          "bob/ip/base/weighted_gaussian.cpp",
          "bob/ip/base/hog.cpp",
          "bob/ip/base/glcm.cpp",
          "bob/ip/base/filter.cpp",
          "bob/ip/base/main.cpp",
        ],
        packages = packages,
        boost_modules = boost_modules,
        bob_packages = bob_packages,
        version = version,
      ),
    ],

    cmdclass = {
      'build_ext': build_ext
    },

    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],

)
