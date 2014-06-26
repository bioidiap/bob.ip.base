/**
 * @author Manuel Günther <manuel.guenther@idiap.ch>
 * @date Thu Jun 26 09:33:10 CEST 2014
 *
 * This file defines functions and classes for affine image transformations
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_GEOM_NORM_H
#define BOB_IP_BASE_GEOM_NORM_H

#include <boost/shared_ptr.hpp>
#include "bob/core/assert.h"
#include "bob/core/check.h"


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
      blitz::TinyVector<int,2> getCropSize() const { return m_crop_size; }
      blitz::TinyVector<double,2> getCropOffset() const { return m_crop_offset; }

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
        * @brief Process a 2D blitz Array/Image
        */
      template <typename T, bool mask>
      void processNoCheck(const blitz::Array<T,2>& src, const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, blitz::Array<bool,2>& dst_mask, const blitz::TinyVector<double,2>& center) const;

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
    processNoCheck<T,false>(src, src_mask, dst, dst_mask, center);
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
    processNoCheck<T,true>(src, src_mask, dst, dst_mask, center);
  }

  template <typename T, bool mask>
  void GeomNorm::processNoCheck(const blitz::Array<T,2>& source, const blitz::Array<bool,2>& source_mask, blitz::Array<double,2>& target, blitz::Array<bool,2>& target_mask, const blitz::TinyVector<double,2>& center) const
  {
    // This is the fastest version of the function that I can imagine...
    // It handles two different coordinate systems: original image and new image

    // transformation center in original image
    const double original_center_y = center[0],
                 original_center_x = center[1];
    // transformation center in new image:
    const double new_center_y = m_crop_offset[0],
                 new_center_x = m_crop_offset[1];

    // With these positions, we can define a mapping from the new image to the original image
    const double sin_angle = -sin(m_rotation_angle * M_PI / 180.),
                 cos_angle = cos(m_rotation_angle * M_PI / 180.);
    // we compute the distance in the source image, when going 1 pixel in the new image
    const double dx = cos_angle / m_scaling_factor,
                 dy = -sin_angle / m_scaling_factor;

    // Now, we iterate through the target image, and compute pixel positions in the source.
    // For this purpose, get the (0,0) position of the target image in source image coordinates:
    double origin_x = original_center_x - (cos_angle * new_center_x + sin_angle * new_center_y) / m_scaling_factor;
    double origin_y = original_center_y - (cos_angle * new_center_y - sin_angle * new_center_x) / m_scaling_factor;

    // some helpers for the interpolation
    int ox, oy;
    double mx, my;
    int h = source.extent(0)-1;
    int w = source.extent(1)-1;

    // Ok, so let's do it.
    for (int y = 0; y < m_crop_size[0]; ++y){
      // set the source image point to first point in row
      double source_x = origin_x, source_y = origin_y;
      // iterate over the row
      for (int x = 0; x < m_crop_size[1]; ++x){

        // We are at the desired pixel in the new image. Interpolate the old image's pixels:
        double& res = target(y,x) = 0.;

        // split each source x and y in integral and decimal digits
        ox = std::floor(source_x);
        oy = std::floor(source_y);
        mx = source_x - ox;
        my = source_y - oy;

        // add the four values bi-linearly interpolated
        if (mask){
          bool& new_mask = target_mask(y,x) = false;
          // upper left
          if (ox >= 0 && oy >= 0 && ox <= w && oy <= h && source_mask(oy,ox)){
            res += (1.-mx) * (1.-my) * source(oy,ox);
            new_mask = true;
          }
          // upper right
          if (ox >= -1 && oy >= 0 && ox < w && oy <= h && source_mask(oy,ox+1)){
            res += mx * (1.-my) * source(oy,ox+1);
            new_mask = true;
          }
          // lower left
          if (ox >= 0 && oy >= -1 && ox <= w && oy < h && source_mask(oy+1,ox)){
            res += (1.-mx) * my * source(oy+1,ox);
            new_mask = true;
          }
          // lower right
          if (ox >= -1 && oy >= -1 && ox < w && oy < h && source_mask(oy+1,ox+1)){
            res += mx * my * source(oy+1,ox+1);
            new_mask = true;
          }
        } else {
          // upper left
          if (ox >= 0 && oy >= 0 && ox <= w && oy <= h)
            res += (1.-mx) * (1.-my) * source(oy,ox);

          // upper right
          if (ox >= -1 && oy >= 0 && ox < w && oy <= h)
            res += mx * (1.-my) * source(oy,ox+1);

          // lower left
          if (ox >= 0 && oy >= -1 && ox <= w && oy < h)
            res += (1.-mx) * my * source(oy+1,ox);

          // lower right
          if (ox >= -1 && oy >= -1 && ox < w && oy < h)
            res += mx * my * source(oy+1,ox+1);
        }

        // done with this pixel...
        // go to the next source pixel in the row
        source_x += dx;
        source_y += dy;
      }
      // at the end of the row, we shift the origin to the next line
      origin_x -= dy;
      origin_y += dx;
    }
    // done!
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

