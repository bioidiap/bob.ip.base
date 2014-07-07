import bob.ip.base
import numpy
import math

# create test image
image = numpy.zeros((21,21))
image[5:16, 5:16] = 1

# perform Sobel filtering
sobel = bob.ip.base.sobel(image)

# compute direction-independent and direction-dependent results
abs_sobel = numpy.sqrt(numpy.square(sobel[0]) + numpy.square(sobel[1]))
angle = 45.
rot_sobel = math.sin(angle*math.pi/180) * sobel[0] + math.cos(angle*math.pi/180) * sobel[1]

# plot results
from matplotlib import pyplot
pyplot.figure(figsize=(20,4))
pyplot.subplot(151) ; pyplot.imshow(image, cmap='gray')     ; pyplot.title('Image')
pyplot.subplot(152) ; pyplot.imshow(sobel[0], cmap='gray')  ; pyplot.title('Sobel - Y')
pyplot.subplot(153) ; pyplot.imshow(sobel[1], cmap='gray')  ; pyplot.title('Sobel - X')
pyplot.subplot(154) ; pyplot.imshow(abs_sobel, cmap='gray') ; pyplot.title('Sobel - Abs')
pyplot.subplot(155) ; pyplot.imshow(rot_sobel, cmap='gray') ; pyplot.title('Sobel - %3.0f$^\circ$'%angle)
pyplot.show()

