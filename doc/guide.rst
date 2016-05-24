.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
.. Wed Mar 14 12:31:35 2012 +0100
..
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

.. testsetup:: iptest

  import numpy
  import math
  import bob.io.base
  import bob.ip.base
  from bob.io.base.test_utils import datafile

  image_path = datafile('image_r10.hdf5', 'bob.ip.base', 'data/affine')
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

the :py:func:`bob.ip.base.scale` function of |project| is then called to up-scale the image:

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
:py:func:`bob.ip.base.rotated_output_shape` which can be used:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> A = numpy.array( [ [1, 2, 3], [4, 5, 6] ], dtype = numpy.uint8 ) # A small image of size 3x3
  >>> print(A)
  [[1 2 3]
   [4 5 6]]
  >>> rotated_shape = bob.ip.base.rotated_output_shape( A, 90 )
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

One simple example of image filtering is to apply a Gaussian blur filter to an image.
This can be easily done by first creating an object of the :py:class:`bob.ip.base.Gaussian` class:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> filter = bob.ip.base.Gaussian(sigma = (3., 3.), radius = (5, 5))

Now, let's see what happens to a small test image:

.. plot:: plot/gaussian.py
   :include-source: True

The image of the cross has now been nicely smoothed.

A second example uses Sobel filters to extract edges from an image.
Two types of Sobel filters exist: The vertical filter :math:`S_y` and the horizontal filter :math:`S_x`:

.. math::
   S_y = \left\lgroup\begin{array}{ccc} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{array}\right\rgroup \qquad
   S_x = \left\lgroup\begin{array}{ccc} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{array}\right\rgroup

Both filters can be applied at the same time using the :py:func:`bob.ip.base.sobel` function, where the result of :math:`S_y` will be put to the first layer and :math:`S_x` to the second layer.

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> image = numpy.zeros((21,21))
  >>> image[5:16, 5:16] = 1
  >>> sobel = bob.ip.base.sobel(image)
  >>> sobel.shape
  (2, 21, 21)

Interestingly, the vertical filter :math:`S_y` extracts horizontal edges, while the :math:`S_x` extracts vertical edges.
In fact, the vector :math:`(s_y, s_x)^T` contains the gradient information at a given location in the image.
To get the direction-independent strength of the edge at that point, simply compute the Euclidean length of the gradient.
To compute rotation-dependent results, use the rotation matrix on the gradient vector.

.. plot:: plot/sobel.py
   :include-source: True


Normalizing face images according to eye positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For many face biometrics applications, for instance face recognition, the images are
geometrically normalized according to the eye positions.  In such a case, the
first thing to do is to create an object of the :py:class:`bob.ip.base.FaceEyesNorm` class defining the image
properties of the geometrically normalized image (that will be generated when
applying the object):

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> face_eyes_norm = bob.ip.base.FaceEyesNorm(eyes_distance = 65, crop_size = (128, 128), eyes_center = (32, 63.5))

Now, we have set up our object to generate images of size (128, 128) that will
put the left eye at the pixel position (32, 31) and the right eye at the
position (32, 96). Afterwards, this object is used to geometrically normalize
the face, given the eye positions in the original face image.  Note that the
left eye usually has a higher x-coordinate than the right eye:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> face_image = bob.io.base.load( image_path )
  >>> cropped_image = numpy.ndarray( (128, 128), dtype = numpy.float64 )
  >>> face_eyes_norm( face_image, cropped_image, right_eye = (66, 47), left_eye = (62, 70) )

Now, let's have a look at the original and normalized face:

.. plot:: plot/face_eyes_norm.py
   :include-source: True


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

  >>> lbp_local = lbp_extractor ( cropped_image, (69, 62) )
  >>> # print the binary representation of the LBP
  >>> print(bin ( lbp_local ))
  0b1111000

or you can extract the LBP features for all pixels in the image. In this case
you need to get the required shape of the output image using the :py:class:`bob.ip.base.LBP` feature extractor:

.. doctest:: iptest
  :options: +NORMALIZE_WHITESPACE

  >>> lbp_output_image_shape = lbp_extractor.lbp_shape(cropped_image)
  >>> print(lbp_output_image_shape)
  (126, 126)
  >>> lbp_output_image = numpy.ndarray ( lbp_output_image_shape, dtype = numpy.uint16 )
  >>> lbp_extractor ( cropped_image,  lbp_output_image )
  >>> # print the binary representation of the pixel at the same location as above;
  >>> # note that the index is shifted by 1 since the lbp image is smaller than the original
  >>> print(bin ( lbp_output_image [ 68, 61 ] ))
  0b1111000


LBP-TOP extraction
~~~~~~~~~~~~~~~~~~

  LBP-TOP [Zhao2007]_ extraction for temporal texture analysis.

.. doctest:: lbptoptest
  :options: +NORMALIZE_WHITESPACE

  >>> import bob.ip.base
  >>> import numpy
  >>> numpy.random.seed(10)
  >>> #Defining the lbp operator for each plane
  >>> lbp_xy = bob.ip.base.LBP(8,1)
  >>> lbp_xt = bob.ip.base.LBP(8,1)
  >>> lbp_yt = bob.ip.base.LBP(8,1)
  >>> lbptop = bob.ip.base.LBPTop(lbp_xy, lbp_xt, lbp_yt)

Defining the test 3D image and creating the containers for the outputs in each plane

.. doctest:: lbptoptest
  :options: +NORMALIZE_WHITESPACE

  >>> img3d = (numpy.random.rand(3,5,5)*100).astype('uint16')
  >>> t = int(max(lbp_xt.radius, lbp_yt.radius))
  >>> w = int(img3d.shape[1] - lbp_xy.radii[0]*2)
  >>> h = int(img3d.shape[2] - lbp_xy.radii[1]*2)
  >>> output_xy = numpy.zeros((t,w,h),dtype='uint16')
  >>> output_xt = numpy.zeros((t,w,h),dtype='uint16')
  >>> output_yt = numpy.zeros((t,w,h),dtype='uint16')

Extracting the bins for each plane

.. doctest:: lbptoptest
  :options: +NORMALIZE_WHITESPACE

  >>> lbptop(img3d,output_xy, output_xt, output_yt)
  >>> print(output_xy)
  [[[ 89   0 235]
  [255  72 255]
  [ 40  95   2]]]
  >>> print(output_xt)
  [[[ 55   2 135]
  [223 130 119]
  [  0 253  64]]]
  >>> print(output_yt)
  [[[ 45   0 173]
  [247   1 255]
  [130 127  64]]]
  

.. Place here your external references
.. include:: links.rst
