/**
 * @date Thu July 19 12:27:15 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to process images with a weighted
 *        Gaussian kernel (Used by the Self Quotient Image)
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <boost/format.hpp>
#include <stdexcept>
#include <bob.ip.base/IntegralImage.h>
#include <bob.ip.base/WeightedGaussian.h>


bob::ip::base::WeightedGaussian::WeightedGaussian(
  const size_t radius_y, const size_t radius_x,
  const double sigma_y, const double sigma_x,
  const bob::sp::Extrapolation::BorderType border_type
):
  m_radius_y(radius_y),
  m_radius_x(radius_x),
  m_sigma_y(sigma_y),
  m_sigma_x(sigma_x),
  m_conv_border(border_type)
{
  computeKernel();
}


bob::ip::base::WeightedGaussian::WeightedGaussian(const WeightedGaussian& other)
:
  m_radius_y(other.m_radius_y),
  m_radius_x(other.m_radius_x),
  m_sigma_y(other.m_sigma_y),
  m_sigma_x(other.m_sigma_x),
  m_conv_border(other.m_conv_border)
{
  computeKernel();
}


void bob::ip::base::WeightedGaussian::computeKernel()
{
  m_kernel.resize(2 * m_radius_y + 1, 2 * m_radius_x + 1);
  m_kernel_weighted.resize(2 * m_radius_y + 1, 2 * m_radius_x + 1);
  // Computes the kernel
  const double inv_sigma_y = 1.0 / (m_sigma_y*m_sigma_y);
  const double inv_sigma_x = 1.0 / (m_sigma_x*m_sigma_x);
  for (int i = -(int)m_radius_y; i <= (int)m_radius_y; ++i)
    for (int j = -(int)m_radius_x; j <= (int)m_radius_x; ++j)
      m_kernel(i + (int)m_radius_y, j + (int)m_radius_x) =
        exp( -0.5 * (inv_sigma_y * (i * i) + inv_sigma_x * (j * j)));
  // Normalizes the kernel
  m_kernel /= blitz::sum(m_kernel);
}

void bob::ip::base::WeightedGaussian::reset(
  const size_t radius_y, const size_t radius_x,
  const double sigma_y, const double sigma_x,
  const bob::sp::Extrapolation::BorderType border_type
){
  m_radius_y = radius_y;
  m_radius_x = radius_x;
  m_sigma_y = sigma_y;
  m_sigma_x = sigma_x;
  m_conv_border = border_type;
  computeKernel();
}

bob::ip::base::WeightedGaussian& bob::ip::base::WeightedGaussian::operator=(const bob::ip::base::WeightedGaussian& other)
{
  if (this != &other)
  {
    m_radius_y = other.m_radius_y;
    m_radius_x = other.m_radius_x;
    m_sigma_y = other.m_sigma_y;
    m_sigma_x = other.m_sigma_x;
    m_conv_border = other.m_conv_border;
    computeKernel();
  }
  return *this;
}

bool bob::ip::base::WeightedGaussian::operator==(const bob::ip::base::WeightedGaussian& b) const
{
  return (this->m_radius_y == b.m_radius_y && this->m_radius_x == b.m_radius_x &&
          this->m_sigma_y == b.m_sigma_y && this->m_sigma_x == b.m_sigma_x &&
          this->m_conv_border == b.m_conv_border);
}

bool bob::ip::base::WeightedGaussian::operator!=(const bob::ip::base::WeightedGaussian& b) const
{
  return !(this->operator==(b));
}

void bob::ip::base::WeightedGaussian::filter_(const blitz::Array<double,2>& src, blitz::Array<double,2>& dst)
{
  // Checks input
  bob::core::array::assertZeroBase(src);
  bob::core::array::assertZeroBase(dst);
  bob::core::array::assertSameShape(src, dst);
  if(src.extent(0)<m_kernel.extent(0)) {
    boost::format m("The convolutional kernel has the first dimension larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
    m % src.extent(0) % m_kernel.extent(0);
    throw std::runtime_error(m.str());
  }
  if(src.extent(1)<m_kernel.extent(1)) {
    boost::format m("The convolutional kernel has the second dimension larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
    m % src.extent(1) % m_kernel.extent(1);
    throw std::runtime_error(m.str());
  }

  // 1/ Extrapolation of src
  // Resize temporary extrapolated src array
  blitz::TinyVector<int,2> shape = src.shape();
  shape(0) += 2 * (int)m_radius_y;
  shape(1) += 2 * (int)m_radius_x;
  m_src_extra.resize(shape);

  // Extrapolate
  if(m_conv_border == bob::sp::Extrapolation::Zero)
    bob::sp::extrapolateZero(src, m_src_extra);
  else if(m_conv_border == bob::sp::Extrapolation::NearestNeighbour)
    bob::sp::extrapolateNearest(src, m_src_extra);
  else if(m_conv_border == bob::sp::Extrapolation::Circular)
    bob::sp::extrapolateCircular(src, m_src_extra);
  else
    bob::sp::extrapolateMirror(src, m_src_extra);

  // 2/ Integral image then mean values
  shape += 1;
  m_src_integral.resize(shape);
  bob::ip::base::integral(m_src_extra, m_src_integral, true);

  // 3/ Convolution
  double n_elem = m_kernel.numElements();
  for(int y=0; y<src.extent(0); ++y)
    for(int x=0; x<src.extent(1); ++x)
    {
      // Computes the threshold associated to the current location
      // Integral image is used to speed up the process
      blitz::Array<double,2> src_slice = m_src_extra(
        blitz::Range(y,y+2*(int)m_radius_y), blitz::Range(x,x+2*(int)m_radius_x));
      double threshold = (m_src_integral(y,x) +
          m_src_integral(y+2*(int)m_radius_y+1,x+2*(int)m_radius_x+1) -
          m_src_integral(y,x+2*(int)m_radius_x+1) -
          m_src_integral(y+2*(int)m_radius_y+1,x)
        ) / n_elem;
      // Computes the weighted Gaussian kernel at this location
      // a/ M1 is the set of pixels whose values are above the threshold
      if( blitz::sum(src_slice >= threshold) >= n_elem/2.)
        m_kernel_weighted = blitz::where(src_slice >= threshold, m_kernel, 0.);
      // b/ M1 is the set of pixels whose values are below the threshold
      else
        m_kernel_weighted = blitz::where(src_slice < threshold, m_kernel, 0.);
      // Normalizes the kernel
      m_kernel_weighted /= blitz::sum(m_kernel_weighted);
      // Convolves: This is indeed not a real convolution but a multiplication,
      // as it seems that the authors aim at exclusively using the M1 part
      dst(y,x) = blitz::sum(src_slice * m_kernel_weighted);
    }
}

