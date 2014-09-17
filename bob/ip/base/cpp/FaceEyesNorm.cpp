/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Jun 30 19:58:28 CEST 2014
 *
 * This file defines a class for geometrically normalizing facial images based on the eyes
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.ip.base/FaceEyesNorm.h>

bob::ip::base::FaceEyesNorm::FaceEyesNorm(
    const blitz::TinyVector<int,2>& cropSize,
    const double eyesDistance,
    const blitz::TinyVector<double,2>& cropOffset
):
  m_eyesDistance(eyesDistance),
  m_eyesAngle(0.),
  m_geomNorm(new GeomNorm(0., 0., cropSize, cropOffset))
{
}

bob::ip::base::FaceEyesNorm::FaceEyesNorm(
    const blitz::TinyVector<int,2>& cropSize,
    const blitz::TinyVector<double,2>& rightEye,
    const blitz::TinyVector<double,2>& leftEye
):
  m_eyesAngle(0.)
{
  double dy = leftEye[0] - rightEye[0], dx = leftEye[1] - rightEye[1];
  m_eyesDistance = std::sqrt(dx * dx + dy * dy);
  m_eyesAngle = std::atan2(dy, dx) * 180. / M_PI;
  blitz::TinyVector<double,2> eyeCenter((leftEye[0] + rightEye[0]) / 2., (leftEye[1] + rightEye[1]) / 2.);
  m_geomNorm = boost::shared_ptr<GeomNorm>(new GeomNorm(0., 0., cropSize, eyeCenter));
}


bob::ip::base::FaceEyesNorm::FaceEyesNorm(const FaceEyesNorm& other)
:
  m_eyesDistance(other.m_eyesDistance),
  m_eyesAngle(other.m_eyesAngle),
  m_geomNorm(new GeomNorm(*other.m_geomNorm))
{
}

bob::ip::base::FaceEyesNorm& bob::ip::base::FaceEyesNorm::operator=(const bob::ip::base::FaceEyesNorm& other)
{
  if (this != &other)
  {
    m_eyesDistance = other.m_eyesDistance;
    m_eyesAngle = other.m_eyesAngle;
    m_geomNorm.reset(new GeomNorm(*other.m_geomNorm));
  }
  return *this;
}

bool bob::ip::base::FaceEyesNorm::operator==(const bob::ip::base::FaceEyesNorm& b) const
{
  return (m_eyesDistance == b.m_eyesDistance && m_eyesAngle == b.m_eyesAngle && *m_geomNorm == *b.m_geomNorm);
}

bool bob::ip::base::FaceEyesNorm::operator!=(const bob::ip::base::FaceEyesNorm& b) const
{
  return !(this->operator==(b));
}



