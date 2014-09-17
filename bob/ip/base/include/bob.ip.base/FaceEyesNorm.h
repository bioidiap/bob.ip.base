/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Jun 30 19:58:28 CEST 2014
 *
 * This file defines a class for geometrically normalizing facial images based on the eyes
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_FACE_EYES_NORM_H
#define BOB_IP_BASE_FACE_EYES_NORM_H

#include <boost/shared_ptr.hpp>
#include <bob.core/assert.h>
#include <bob.core/check.h>

#include <bob.ip.base/GeomNorm.h>

static inline double _sqr(double x){return x*x;}

namespace bob { namespace ip { namespace base {

  /**
   * @brief A class to perform a geometric normalization of a face based
   * on the eye center locations.
   */
  class FaceEyesNorm
  {
    public:

      /**
        * @brief Constructor
        */
      explicit FaceEyesNorm(
        const blitz::TinyVector<int,2>& cropSize,
        const double eyes_distance,
        const blitz::TinyVector<double,2>& cropOffset // usually center between the eyes
      );

      /**
        * @brief Constructor taking the requested two eye positions
        */
      explicit FaceEyesNorm(
        const blitz::TinyVector<int,2>& cropSize,
        const blitz::TinyVector<double,2>& rightEye,
        const blitz::TinyVector<double,2>& leftEye
      );

      /**
       * @brief Copy constructor
       */
      FaceEyesNorm(const FaceEyesNorm& other);

      /**
        * @brief Destructor
        */
      virtual ~FaceEyesNorm() {}

      /**
       * @brief Assignment operator
       */
      FaceEyesNorm& operator=(const FaceEyesNorm& other);

      /**
       * @brief Equal to
       */
      bool operator==(const FaceEyesNorm& b) const;
      /**
       * @brief Not equal to
       */
      bool operator!=(const FaceEyesNorm& b) const;

      /**
        * @brief Accessors
        */
      double getEyesDistance() const {return m_eyesDistance;}
      double getEyesAngle() const {return m_eyesAngle;}
      const blitz::TinyVector<int,2>& getCropSize() const {return m_geomNorm->getCropSize();}
      const blitz::TinyVector<double,2>& getCropOffset() const {return m_geomNorm->getCropOffset();}
      double getLastAngle() const {return m_geomNorm->getRotationAngle();}
      double getLastScale() const {return m_geomNorm->getScalingFactor();}
      const blitz::TinyVector<double,2>& getLastOffset() const {return m_lastCenter;}

      /**
        * @brief Mutators
        */
      void setEyesDistance(const double eyesDistance) {m_eyesDistance = eyesDistance;}
      void setEyesAngle(const double eyesAngle) {m_eyesAngle = eyesAngle;}
      void setCropSize(const blitz::TinyVector<int,2>& cropSize) {m_geomNorm->setCropSize(cropSize);}
      void setCropOffset(const blitz::TinyVector<double,2>& cropOffset) {m_geomNorm->setCropOffset(cropOffset);}

      /**
        * @brief Process a 2D face image by applying the geometric
        * normalization
        */
      template <typename T>
      void extract(
        const blitz::Array<T,2>& src,
        blitz::Array<double,2>& dst,
        const blitz::TinyVector<double,2>& rightEye,
        const blitz::TinyVector<double,2>& leftEye
      ) const;

      template <typename T>
      void extract(
        const blitz::Array<T,2>& src,
        const blitz::Array<bool,2>& srcMask,
        blitz::Array<double,2>& dst,
        blitz::Array<bool,2>& dstMask,
        const blitz::TinyVector<double,2>& rightEye,
        const blitz::TinyVector<double,2>& leftEye
      ) const;

      /**
       * @brief Getter function for the bob::ip::GeomNorm object that is doing the job.
       *
       * @warning The returned GeomNorm object is only valid *after a call to extract *
       *
       * @return  The GeomNorm object that is used to perform the transformation.
       */
      const boost::shared_ptr<GeomNorm> getGeomNorm() const{return m_geomNorm;}

    private:

      template <typename T, bool mask>
      void processNoCheck(
        const blitz::Array<T,2>& src,
        const blitz::Array<bool,2>& srcMask,
        blitz::Array<double,2>& dst,
        blitz::Array<bool,2>& dstMask,
        const blitz::TinyVector<double,2>& rightEye,
        const blitz::TinyVector<double,2>& leftEye
      ) const;

      /**
        * Attributes
        */
      double m_eyesDistance;
      double m_eyesAngle;
      mutable blitz::TinyVector<double,2> m_lastCenter;

      mutable boost::shared_ptr<GeomNorm> m_geomNorm;
  };

  template <typename T>
  inline void FaceEyesNorm::extract(
    const blitz::Array<T,2>& src,
    blitz::Array<double,2>& dst,
    const blitz::TinyVector<double,2>& rightEye,
    const blitz::TinyVector<double,2>& leftEye
  ) const
  {
    // Check input
    bob::core::array::assertZeroBase(src);

    // Check output
    bob::core::array::assertZeroBase(dst);
    bob::core::array::assertSameShape(dst, m_geomNorm->getCropSize());

    // Process
    blitz::Array<bool,2> srcMask, dstMask;
    processNoCheck<T,false>(src, srcMask, dst, dstMask, rightEye, leftEye);
  }

  template <typename T>
  inline void FaceEyesNorm::extract(
    const blitz::Array<T,2>& src,
    const blitz::Array<bool,2>& srcMask,
    blitz::Array<double,2>& dst,
    blitz::Array<bool,2>& dstMask,
    const blitz::TinyVector<double,2>& rightEye,
    const blitz::TinyVector<double,2>& leftEye
  ) const
  {
    // Check input
    bob::core::array::assertZeroBase(src);
    bob::core::array::assertZeroBase(srcMask);
    bob::core::array::assertSameShape(src,srcMask);

    // Check output
    bob::core::array::assertZeroBase(dst);
    bob::core::array::assertZeroBase(dstMask);
    bob::core::array::assertSameShape(dst,dstMask);
    bob::core::array::assertSameShape(dst, m_geomNorm->getCropSize());

    // Process
    processNoCheck<T,true>(src, srcMask, dst, dstMask, rightEye, leftEye);
  }

  template <typename T, bool mask>
  inline void FaceEyesNorm::processNoCheck(
    const blitz::Array<T,2>& src,
    const blitz::Array<bool,2>& srcMask,
    blitz::Array<double,2>& dst,
    blitz::Array<bool,2>& dstMask,
    const blitz::TinyVector<double,2>& rightEye,
    const blitz::TinyVector<double,2>& leftEye
  ) const
  {
    // Get angle to horizontal
    double dy = leftEye[0] - rightEye[0], dx = leftEye[1] - rightEye[1];
    double angle = std::atan2(dy, dx);
    m_geomNorm->setRotationAngle(angle * 180. / M_PI - m_eyesAngle);

    // Get scaling factor
    m_geomNorm->setScalingFactor(m_eyesDistance / sqrt(_sqr(leftEye[0]-rightEye[0]) + _sqr(leftEye[1]-rightEye[1])));

    // Get the center (of the eye centers segment)
    m_lastCenter = blitz::TinyVector<double,2>(
      (rightEye[0] + leftEye[0]) / 2.,
      (rightEye[1] + leftEye[1]) / 2.
    );

    // Perform the normalization
    if(mask)
      m_geomNorm->process(src, srcMask, dst, dstMask, m_lastCenter);
    else
      m_geomNorm->process(src, dst, m_lastCenter);
  }

} } } // namespaces

#endif /* BOB_IP_BASE_FACE_EYES_NORM_H */

