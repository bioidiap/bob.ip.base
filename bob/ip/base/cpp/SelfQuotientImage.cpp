/**
 * @date Thu Jul 19 11:52:08 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the Self Quotient Image algorithm
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.ip.base/SelfQuotientImage.h>

bob::ip::base::SelfQuotientImage::SelfQuotientImage(
    const size_t n_scales,
    const size_t size_min,
    const size_t size_step,
    const double sigma,
    const bob::sp::Extrapolation::BorderType border_type
):
  m_n_scales(n_scales),
  m_size_min(size_min),
  m_size_step(size_step),
  m_sigma(sigma),
  m_conv_border(border_type),
  m_wgaussians(new bob::ip::base::WeightedGaussian[m_n_scales])
{
  computeKernels();
}

bob::ip::base::SelfQuotientImage::SelfQuotientImage(const bob::ip::base::SelfQuotientImage& other)
:
  m_n_scales(other.m_n_scales),
  m_size_min(other.m_size_min),
  m_size_step(other.m_size_step),
  m_sigma(other.m_sigma),
  m_conv_border(other.m_conv_border),
  m_wgaussians(new bob::ip::base::WeightedGaussian[m_n_scales])
{
  computeKernels();
}


void bob::ip::base::SelfQuotientImage::computeKernels()
{
  for( size_t s=0; s<m_n_scales; ++s)
  {
    // size of the kernel
    size_t s_size = m_size_min + s * m_size_step;
    // sigma of the kernel
    double s_sigma = m_sigma * s_size / m_size_min;
    // Initialize the Gaussian
    m_wgaussians[s].reset(s_size, s_size, s_sigma, s_sigma, m_conv_border);
  }
}

void bob::ip::base::SelfQuotientImage::reset(
  const size_t n_scales,
  const size_t size_min,
  const size_t size_step,
  const double sigma,
  const bob::sp::Extrapolation::BorderType border_type
){
  m_n_scales = n_scales;
  m_wgaussians.reset(new bob::ip::base::WeightedGaussian[m_n_scales]);
  m_size_min = size_min;
  m_size_step = size_step;
  m_sigma = sigma;
  m_conv_border = border_type;
  computeKernels();
}

bob::ip::base::SelfQuotientImage& bob::ip::base::SelfQuotientImage::operator=(const bob::ip::base::SelfQuotientImage& other)
{
  if (this != &other)
  {
    m_n_scales = other.m_n_scales;
    m_wgaussians.reset(new bob::ip::base::WeightedGaussian[m_n_scales]);
    m_size_min = other.m_size_min;
    m_size_step = other.m_size_step;
    m_sigma = other.m_sigma;
    m_conv_border = other.m_conv_border;
    computeKernels();
  }
  return *this;
}

bool bob::ip::base::SelfQuotientImage::operator==(const bob::ip::base::SelfQuotientImage& b) const
{
  return (this->m_n_scales == b.m_n_scales && this->m_size_min== b.m_size_min &&
          this->m_size_step == b.m_size_step && this->m_sigma == b.m_sigma &&
          this->m_conv_border == b.m_conv_border);
}

bool bob::ip::base::SelfQuotientImage::operator!=(const bob::ip::base::SelfQuotientImage& b) const
{
  return !(this->operator==(b));
}

