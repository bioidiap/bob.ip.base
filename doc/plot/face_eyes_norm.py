import numpy
import math
import bob.io.base
import bob.ip.base
from bob.io.base.test_utils import datafile

# load a test image
face_image = bob.io.base.load(datafile('image_r10.hdf5', 'bob.ip.base', 'data/affine'))

# create FaceEyesNorm class
face_eyes_norm = bob.ip.base.FaceEyesNorm(eyes_distance = 65, crop_size = (128, 128), eyes_center = (32, 63.5))

# normalize image
normalized_image = face_eyes_norm( face_image, right_eye = (66, 47), left_eye = (62, 70) )

# plot results, including eye locations in original and normalized image
from matplotlib import pyplot
pyplot.figure(figsize=(8,4))
pyplot.subplot(121) ; pyplot.imshow(face_image, cmap='gray')       ; pyplot.plot([47, 70], [66, 62], 'rx', ms=10, mew=2); pyplot.axis('tight'); pyplot.title('Original Image')
pyplot.subplot(122) ; pyplot.imshow(normalized_image, cmap='gray') ; pyplot.plot([31, 96], [32, 32], 'rx', ms=10, mew=2); pyplot.axis('tight'); pyplot.title('Cropped Image')
pyplot.show()

