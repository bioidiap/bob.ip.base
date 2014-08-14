/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date MThu Jun 26 09:33:10 CEST 2014
 *
 * This file defines a class for geometrically normalizing images
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.ip.base/GeomNorm.h>

////////////////////////////////////////////////////////////////////////////////////
/////////// GeomNorm class /////////////////////////////////////////////////////////
bob::ip::base::GeomNorm::GeomNorm(
    const double rotation_angle, const double scaling_factor,
    const blitz::TinyVector<int,2>& crop_size,
    const blitz::TinyVector<double,2>& crop_offset
):
  m_rotation_angle(rotation_angle),
  m_scaling_factor(scaling_factor),
  m_crop_size(crop_size),
  m_crop_offset(crop_offset)
{
}

bob::ip::base::GeomNorm::GeomNorm(const bob::ip::base::GeomNorm& other):
  m_rotation_angle(other.m_rotation_angle),
  m_scaling_factor(other.m_scaling_factor),
  m_crop_size(other.m_crop_size),
  m_crop_offset(other.m_crop_offset)
{
}

bob::ip::base::GeomNorm::~GeomNorm()
{
}

bob::ip::base::GeomNorm&
bob::ip::base::GeomNorm::operator=(const bob::ip::base::GeomNorm& other)
{
  if (this != &other)
  {
    m_rotation_angle = other.m_rotation_angle;
    m_scaling_factor = other.m_scaling_factor;
    m_crop_size = other.m_crop_size;
    m_crop_offset = other.m_crop_offset;
  }
  return *this;
}

bool
bob::ip::base::GeomNorm::operator==(const bob::ip::base::GeomNorm& b) const
{
  return m_rotation_angle == b.m_rotation_angle &&
         m_scaling_factor == b.m_scaling_factor &&
         m_crop_size[0] == b.m_crop_size[0] && m_crop_size[1] == b.m_crop_size[1] &&
         m_crop_offset[0] == b.m_crop_offset[0] && m_crop_offset[1] == b.m_crop_offset[1];
}

bool
bob::ip::base::GeomNorm::operator!=(const bob::ip::base::GeomNorm& b) const
{
  return !(this->operator==(b));
}

blitz::TinyVector<double,2>
bob::ip::base::GeomNorm::process(const blitz::TinyVector<double,2>& position, const blitz::TinyVector<double,2>& center) const
{
  // compute scale and angle parameters
  const double sin_angle = sin(m_rotation_angle * M_PI / 180.) * m_scaling_factor,
               cos_angle = cos(m_rotation_angle * M_PI / 180.) * m_scaling_factor;

  const double centered_y = position(0) - center[0],
               centered_x = position(1) - center[1];

  return blitz::TinyVector<double,2> (
    centered_y * cos_angle - centered_x * sin_angle + m_crop_offset[0],
    centered_y * sin_angle + centered_x * cos_angle + m_crop_offset[1]
  );

}

