/**
 * @date Mon May 2 10:01:08 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the MultiscaleRetinex algorithm
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.ip.base/MultiscaleRetinex.h>

bob::ip::base::MultiscaleRetinex::MultiscaleRetinex(
  const size_t n_scales,
  const int size_min,
  const int size_step,
  const double sigma,
  const bob::sp::Extrapolation::BorderType border_type
):
  m_n_scales(n_scales),
  m_size_min(size_min),
  m_size_step(size_step),
  m_sigma(sigma),
  m_conv_border(border_type),
  m_gaussians(new bob::ip::base::Gaussian[m_n_scales])
{
  computeKernels();
}


bob::ip::base::MultiscaleRetinex::MultiscaleRetinex(const MultiscaleRetinex& other)
:
  m_n_scales(other.m_n_scales),
  m_size_min(other.m_size_min),
  m_size_step(other.m_size_step),
  m_sigma(other.m_sigma),
  m_conv_border(other.m_conv_border),
  m_gaussians(new bob::ip::base::Gaussian[m_n_scales])
{
  computeKernels();
}


void bob::ip::base::MultiscaleRetinex::computeKernels()
{
  for( size_t s=0; s<m_n_scales; ++s)
  {
    // size of the kernel
    int s_size = m_size_min + s * m_size_step;
    // sigma of the kernel
    double s_sigma = m_sigma * s_size / m_size_min;
    // Initialize the Gaussian
    m_gaussians[s].reset(s_size, s_size, s_sigma, s_sigma, m_conv_border);
  }
}

void bob::ip::base::MultiscaleRetinex::reset(
  const size_t n_scales,
  const int size_min,
  const int size_step,
  const double sigma,
  const bob::sp::Extrapolation::BorderType border_type
){
  m_n_scales = n_scales;
  m_gaussians.reset(new bob::ip::base::Gaussian[m_n_scales]);
  m_size_min = size_min;
  m_size_step = size_step;
  m_sigma = sigma;
  m_conv_border = border_type;
  computeKernels();
}

bob::ip::base::MultiscaleRetinex& bob::ip::base::MultiscaleRetinex::operator=(const bob::ip::base::MultiscaleRetinex& other)
{
  if (this != &other)
  {
    m_n_scales = other.m_n_scales;
    m_gaussians.reset(new bob::ip::base::Gaussian[m_n_scales]);
    m_size_min = other.m_size_min;
    m_size_step = other.m_size_step;
    m_sigma = other.m_sigma;
    m_conv_border = other.m_conv_border;
    computeKernels();
  }
  return *this;
}

bool bob::ip::base::MultiscaleRetinex::operator==(const bob::ip::base::MultiscaleRetinex& b) const
{
  return (this->m_n_scales == b.m_n_scales && this->m_size_min== b.m_size_min &&
          this->m_size_step == b.m_size_step && this->m_sigma == b.m_sigma &&
          this->m_conv_border == b.m_conv_border);
}

bool bob::ip::base::MultiscaleRetinex::operator!=(const bob::ip::base::MultiscaleRetinex& b) const
{
  return !(this->operator==(b));
}

