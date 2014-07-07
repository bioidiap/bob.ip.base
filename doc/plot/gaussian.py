import bob.ip.base
import numpy
import math

# create test image
image = numpy.zeros((21,21))
for i in range(21):
  image[i,i] = 255
  image[-i,i] = 255

# perform Gaussian filtering
gaussian = bob.ip.base.Gaussian(sigma = (3., 3.), radius = (5, 5))
smoothed = gaussian(image)

# plot results
from matplotlib import pyplot
pyplot.figure(figsize=(8,4))
pyplot.subplot(121) ; pyplot.imshow(image, cmap='gray')    ; pyplot.title('Image')
pyplot.subplot(122) ; pyplot.imshow(smoothed, cmap='gray') ; pyplot.title('Smoothed')
pyplot.show()

