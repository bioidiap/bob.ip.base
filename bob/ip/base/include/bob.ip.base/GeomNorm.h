/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Thu Jun 26 09:33:10 CEST 2014
 *
 * This file defines a class for affine image transformations
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_GEOM_NORM_H
#define BOB_IP_BASE_GEOM_NORM_H

#include <boost/shared_ptr.hpp>
#include <bob.core/assert.h>
#include <bob.core/check.h>

#include <bob.ip.base/Affine.h>

namespace bob { namespace ip { namespace base {

  /**
   * @brief This file defines a class to perform geometric normalization of
   * an image. This means that the image is:
   *   1/ rotated with a given angle and a rotation center
   *   2/ rescaled according to a given scaling factor
   *   3/ cropped with respect to the point given and the additional
   *        cropping parameters (will be substracted to the provided
   *        reference point in the final coordinate system)
   */
  class GeomNorm
  {
    public:

      /**
        * @brief Constructor
        */
      GeomNorm(
        const double rotation_angle, const double scaling_factor,
        const blitz::TinyVector<int,2>& crop_size,
        const blitz::TinyVector<double,2>& cropp_offset
      );

      /**
       * @brief Copy constructor
       */
      GeomNorm(const GeomNorm& other);

      /**
        * @brief Destructor
        */
      virtual ~GeomNorm();

      /**
       * @brief Assignment operator
       */
      GeomNorm& operator=(const GeomNorm& other);

      /**
       * @brief Equal to
       */
      bool operator==(const GeomNorm& b) const;
      /**
       * @brief Not equal to
       */
      bool operator!=(const GeomNorm& b) const;

      /**
        * @brief Accessors
        */
      double getRotationAngle() const { return m_rotation_angle; }
      double getScalingFactor() const { return m_scaling_factor; }
      const blitz::TinyVector<int,2>& getCropSize() const { return m_crop_size; }
      const blitz::TinyVector<double,2>& getCropOffset() const { return m_crop_offset; }

      /**
        * @brief Mutators
        */
      void setRotationAngle(const double angle) {m_rotation_angle = angle;}
      void setScalingFactor(const double scaling_factor) {m_scaling_factor = scaling_factor;}
      void setCropSize(const blitz::TinyVector<int,2>& size) {m_crop_size = size;}
      void setCropOffset(const blitz::TinyVector<double,2>& offset) {m_crop_offset = offset;}

      /**
        * @brief Process a 2D blitz Array/Image by applying the geometric
        * normalization
        */
      template <typename T>
      void process(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst, const blitz::TinyVector<double,2>& center) const;
      template <typename T>
      void process(const blitz::Array<T,2>& src, const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, blitz::Array<bool,2>& dst_mask, const blitz::TinyVector<double,2>& center) const;

      /**
       * @brief Process a 3D blitz Array/Image by applying the geometric
       * normalization to each color plane
       */
      template <typename T>
      void process(const blitz::Array<T,3>& src, blitz::Array<double,3>& dst, const blitz::TinyVector<double,2>& center) const;
      template <typename T>
      void process(const blitz::Array<T,3>& src, const blitz::Array<bool,3>& src_mask, blitz::Array<double,3>& dst, blitz::Array<bool,3>& dst_mask, const blitz::TinyVector<double,2>& center) const;

      /**
       * @brief applies the geometric normalization to the given input position
       */
      blitz::TinyVector<double,2> process(const blitz::TinyVector<double,2>& position, const blitz::TinyVector<double,2>& center) const;

    private:
      /**
        * Attributes
        */
      double m_rotation_angle;
      double m_scaling_factor;
      blitz::TinyVector<int,2> m_crop_size;
      blitz::TinyVector<double,2> m_crop_offset;
  };

  template <typename T>
  void GeomNorm::process(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst, const blitz::TinyVector<double,2>& center) const
  {
    // Check input
    bob::core::array::assertZeroBase(src);

    // Check output
    bob::core::array::assertZeroBase(dst);
    bob::core::array::assertSameDimensionLength(dst.extent(0), m_crop_size[0]);
    bob::core::array::assertSameDimensionLength(dst.extent(1), m_crop_size[1]);

    // Process
    blitz::Array<bool,2> src_mask, dst_mask;
    bob::ip::base::transform<T,false>(src, src_mask, center, dst, dst_mask, m_crop_offset, blitz::TinyVector<double,2>(m_scaling_factor, m_scaling_factor), m_rotation_angle);
  }

  template <typename T>
  void GeomNorm::process(const blitz::Array<T,2>& src, const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, blitz::Array<bool,2>& dst_mask, const blitz::TinyVector<double,2>& center) const
  {
    // Check input
    bob::core::array::assertZeroBase(src);
    bob::core::array::assertZeroBase(src_mask);
    bob::core::array::assertSameShape(src,src_mask);

    // Check output
    bob::core::array::assertZeroBase(dst);
    bob::core::array::assertZeroBase(dst_mask);
    bob::core::array::assertSameShape(dst, dst_mask);
    bob::core::array::assertSameDimensionLength(dst.extent(0), m_crop_size[0]);
    bob::core::array::assertSameDimensionLength(dst.extent(1), m_crop_size[1]);

    // Process
    bob::ip::base::transform<T,true>(src, src_mask, center, dst, dst_mask, m_crop_offset, blitz::TinyVector<double,2>(m_scaling_factor, m_scaling_factor), m_rotation_angle);
  }

  template <typename T>
  void GeomNorm::process(const blitz::Array<T,3>& src, blitz::Array<double,3>& dst, const blitz::TinyVector<double,2>& center) const
  {
    for( int p=0; p<dst.extent(0); ++p) {
      const blitz::Array<T,2> src_slice =
        src( p, blitz::Range::all(), blitz::Range::all() );
      blitz::Array<double,2> dst_slice =
        dst( p, blitz::Range::all(), blitz::Range::all() );

      // Process one plane
      process(src_slice, dst_slice, center);
    }
  }

  template <typename T>
  void GeomNorm::process(const blitz::Array<T,3>& src, const blitz::Array<bool,3>& src_mask, blitz::Array<double,3>& dst, blitz::Array<bool,3>& dst_mask, const blitz::TinyVector<double,2>& center) const
  {
    for( int p=0; p<dst.extent(0); ++p) {
      const blitz::Array<T,2> src_slice = src( p, blitz::Range::all(), blitz::Range::all() );
      const blitz::Array<bool,2> src_mask_slice = src_mask( p, blitz::Range::all(), blitz::Range::all() );
      blitz::Array<double,2> dst_slice = dst( p, blitz::Range::all(), blitz::Range::all() );
      blitz::Array<bool,2> dst_mask_slice = dst_mask( p, blitz::Range::all(), blitz::Range::all() );

      // Process one plane
      process(src_slice, src_mask_slice, dst_slice, dst_mask_slice, center);
    }
  }

} } } // namespaces

#endif // BOB_IP_BASE_GEOM_NORM_H

