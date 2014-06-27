.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
.. Wed Mar 14 12:31:35 2012 +0100
..
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

.. testsetup:: iptest

  import numpy
  import math
  import bob.io.base
  import bob.io.image
  import bob.ip.base
  from bob.io.base.test_utils import datafile

  image_path = datafile('image_r10.pgm', 'bob.ip.base', 'data/affine')
  image = bob.io.base.load(image_path)

  numpy.set_printoptions(precision=3, suppress=True)

========================
 Image Processing Guide
========================

Introduction
============

The basic operations on images are the affine image conversions like image
scaling, rotation, and cutting. For most of the operations, two ways of
executing the functions exist. The easier API simply returns the processed
image, but the second version accepts input and output objects (to allow memory
reuse).

Scaling images
~~~~~~~~~~~~~~

To compute a scaled version of the image, simply create the image at the
desired scale. For instance, in the example below an image is up-scaled by
first creating the image and then initializing the larger image:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> A = numpy.array( [ [1, 2, 3], [4, 5, 6] ], dtype = numpy.uint8 ) # A small image of size 2x3
  >>> print(A)
  [[1 2 3]
   [4 5 6]]
  >>> B = numpy.ndarray( (3, 5), dtype = numpy.float64 )               # A larger image of size 3x5

the scale function of |project| is then called to up-scale the image:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> bob.ip.base.scale(A, B)
  >>> print(B)
  [[ 1.   1.5  2.   2.5  3. ]
   [ 2.5  3.   3.5  4.   4.5]
   [ 4.   4.5  5.   5.5  6. ]]

which bi-linearly interpolates image A to image B. Of course, scaling factors
can be different in horizontal and vertical direction:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> C = numpy.ndarray( (2, 5), dtype = numpy.float64 )
  >>> bob.ip.base.scale(A, C)
  >>> print(C)
  [[ 1.   1.5  2.   2.5  3. ]
   [ 4.   4.5  5.   5.5  6. ]]


Rotating images
~~~~~~~~~~~~~~~

The rotation of an image is slightly more difficult since the resulting image
size has to be computed in advance. To facilitate this there is a function
:py:func:`bob.ip.base.get_rotated_output_shape` which can be used:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> A = numpy.array( [ [1, 2, 3], [4, 5, 6] ], dtype = numpy.uint8 ) # A small image of size 3x3
  >>> print(A)
  [[1 2 3]
   [4 5 6]]
  >>> rotated_shape = bob.ip.base.get_rotated_output_shape( A, 90 )
  >>> print(rotated_shape)
  (3, 2)

After the creation of the image in the desired size, the
:py:func:`bob.ip.base.rotate` function can be executed:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> A_rotated = numpy.ndarray( rotated_shape, dtype = numpy.float64 ) # A small image of rotated size
  >>> bob.ip.base.rotate(A, A_rotated, 90)      # execute the rotation
  >>> print(A_rotated)
  [[ 3.  6.]
   [ 2.  5.]
   [ 1.  4.]]


Complex image operations
========================

Complex image operations are usually wrapped up by classes. The usual work flow
is to first generate an object of the desired class, specifying parameters that
are independent on the images to operate, and to second use the class on
images. Usually, objects that perform image operations have the **__call__**
function overloaded, so that one simply can use it as if it were functions.
Below we provide some examples.

Image filtering
~~~~~~~~~~~~~~~

One simple example of image filtering is to apply a Gaussian blur filter to an
image. This can be easily done by first creating an object of the
:py:class:`bob.ip.base.Gaussian` class:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> filter = bob.ip.base.Gaussian( radius_y = 1, radius_x = 1, sigma_y = math.sqrt(0.3*0.5), sigma_x = math.sqrt(0.3*0.5))

Now, let's see what happens to a small test image:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> test_image = numpy.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]], dtype = numpy.float64)
  >>> filtered_image = numpy.ndarray(test_image.shape, dtype = numpy.float64)
  >>> filter(test_image, filtered_image)
  >>> print(filtered_image)
  [[ 0.936  0.063  0.002  0.063  0.936]
   [ 0.063  0.873  0.093  0.873  0.063]
   [ 0.002  0.093  0.876  0.093  0.002]
   [ 0.063  0.873  0.093  0.873  0.063]
   [ 0.936  0.063  0.002  0.063  0.936]]

The image of the cross has now been nicely smoothed.


Normalizing images according to eye positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For many biometric applications, for instance face recognition, the images are
geometrically normalized according to the eye positions.  In such a case, the
first thing to do is to create an object of the class defining the image
properties of the geometrically normalized image (that will be generated when
applying the object):

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> face_eyes_norm = bob.ip.base.FaceEyesNorm(eyes_distance = 65, crop_height = 128, crop_width = 128, crop_eyecenter_offset_h = 32, crop_eyecenter_offset_w = 63.5)

Now, we have set up our object to generate images of size (128, 128) that will
put the left eye at the pixel position (32, 31) and the right eye at the
position (32, 96). Afterwards, this object is used to geometrically normalize
the face, given the eye positions in the original face image.  Note that the
left eye usually has a higher x-coordinate than the right eye:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> face_image = bob.io.base.load( image_path )
  >>> cropped_image = numpy.ndarray( (128, 128), dtype = numpy.float64 )
  >>> face_eyes_norm( face_image, cropped_image, re_y = 67, re_x = 47, le_y = 62, le_x = 71)


Simple feature extraction
~~~~~~~~~~~~~~~~~~~~~~~~~

Some simple feature extraction functionality is also included in the
:py:mod:`bob.ip.base` module. Here is some simple example, how to extract
local binary patterns (LBP) with 8 neighbors from an image:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> lbp_extractor = bob.ip.base.LBP(8)

You can either get the LBP feature for a single point by specifying the
position:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> lbp_local = lbp_extractor ( cropped_image, 69, 62 )
  >>> # print the binary representation of the LBP
  >>> print(bin ( lbp_local ))
  0b11110000

or you can extract the LBP features for all pixels in the image. In this case
you need to get the required shape of the output image:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> lbp_output_image_shape = lbp_extractor.get_lbp_shape(cropped_image)
  >>> print(lbp_output_image_shape)
  (126, 126)
  >>> lbp_output_image = numpy.ndarray ( lbp_output_image_shape, dtype = numpy.uint16 )
  >>> lbp_extractor ( cropped_image,  lbp_output_image )
  bob.blitz.array((126,126),'uint16')
  >>> # print the binary representation of the pixel at the same location as above;
  >>> # note that the index is shifted by 1 since the lbp image is smaller than the original
  >>> print(bin ( lbp_output_image [ 68, 61 ] ))
  0b11110000

.. Place here your external references
.. include:: links.rst
