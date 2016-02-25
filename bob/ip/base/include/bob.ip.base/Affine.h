/**
 * @author Manuel Günther <manuel.guenther@idiap.ch>
 * @date Thu Jun 26 09:33:10 CEST 2014
 *
 * This file defines functions and classes for affine image transformations
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_AFFINE_H
#define BOB_IP_BASE_AFFINE_H

#include <boost/shared_ptr.hpp>
#include <bob.core/assert.h>
#include <bob.core/check.h>
#include <bob.core/logging.h>

#include <boost/random.hpp>
#include <bob.core/random.h>


namespace bob { namespace ip { namespace base {

  /** Implementation of the bi-linear interpolation of a source to a target image. */

  template <typename T, bool mask>
  void transform(
      const blitz::Array<T,2>& source,
      const blitz::Array<bool,2>& source_mask,
      const blitz::TinyVector<double,2>& source_center,
      blitz::Array<double,2>& target,
      blitz::Array<bool,2>& target_mask,
      const blitz::TinyVector<double,2>& target_center,
      const blitz::TinyVector<double,2>& scaling_factor,
      const double& rotation_angle
   ){
    // This is the fastest version of the function that I can imagine...
    // It handles two different coordinate systems: original image and new image

    // transformation center in original image
    const double original_center_y = source_center[0],
                 original_center_x = source_center[1];
    // transformation center in new image:
    const double new_center_y = target_center[0],
                 new_center_x = target_center[1];

    // With these positions, we can define a mapping from the new image to the original image
    const double sin_angle = -sin(rotation_angle * M_PI / 180.),
                 cos_angle = cos(rotation_angle * M_PI / 180.);

    // we compute the distance in the source image, when going 1 pixel in the new image
    const double col_dy = -sin_angle / scaling_factor[0],
                 col_dx = cos_angle / scaling_factor[1];
    const double row_dy = cos_angle / scaling_factor[0],
                 row_dx = sin_angle / scaling_factor[1];


    // Now, we iterate through the target image, and compute pixel positions in the source.
    // For this purpose, get the (0,0) position of the target image in source image coordinates:
    double origin_y = original_center_y - (new_center_y * cos_angle - new_center_x * sin_angle) / scaling_factor[0];
    double origin_x = original_center_x - (new_center_x * cos_angle + new_center_y * sin_angle) / scaling_factor[1];
    // WARNING: I am not sure, if this is correct, or if we rather need to do something like:
    //double origin_y = original_center_y - (new_center_y * cos_angle / scaling_factor[0] - new_center_x * sin_angle / scaling_factor[1]);
    //double origin_x = original_center_x - (new_center_x * cos_angle / scaling_factor[1] + new_center_y * sin_angle / scaling_factor[0]);
    // Note: as long a single scale is used, or scaling is done without rotation, it should be the same.
    //   (at least, the tests pass with both ways)

    // some helpers for the interpolation
    int ox, oy;
    double mx, my;
    int h = source.extent(0)-1;
    int w = source.extent(1)-1;

    int size_y = target.extent(0), size_x = target.extent(1);

    // Ok, so let's do it.
    for (int y = 0; y < size_y; ++y){
      // set the source image point to first point in row
      double source_x = origin_x, source_y = origin_y;
      // iterate over the row
      for (int x = 0; x < size_x; ++x){

        // We are at the desired pixel in the new image. Interpolate the old image's pixels:
        double& res = target(y,x) = 0.;

        // split each source x and y in integral and decimal digits
        ox = std::floor(source_x);
        oy = std::floor(source_y);
        mx = source_x - ox;
        my = source_y - oy;

        // add the four values bi-linearly interpolated
        if (mask){
          bool& new_mask = target_mask(y,x) = true;
          // upper left
          if (ox >= 0 && oy >= 0 && ox <= w && oy <= h && source_mask(oy,ox)){
            res += (1.-mx) * (1.-my) * source(oy,ox);
          } else if ((1.-mx) * (1.-my) > 0.){
            new_mask = false;
          }

          // upper right
          if (ox >= -1 && oy >= 0 && ox < w && oy <= h && source_mask(oy,ox+1)){
            res += mx * (1.-my) * source(oy,ox+1);
          } else if (mx * (1.-my) > 0.){
            new_mask = false;
          }
          // lower left
          if (ox >= 0 && oy >= -1 && ox <= w && oy < h && source_mask(oy+1,ox)){
            res += (1.-mx) * my * source(oy+1,ox);
          } else if ((1.-mx) * my > 0.){
            new_mask = false;
          }
          // lower right
          if (ox >= -1 && oy >= -1 && ox < w && oy < h && source_mask(oy+1,ox+1)){
            res += mx * my * source(oy+1,ox+1);
          } else if (mx * my > 0.){
            new_mask = false;
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
        source_y += col_dy;
        source_x += col_dx;
      }
      // at the end of the row, we shift the origin to the next line
      origin_y += row_dy;
      origin_x += row_dx;
    }
    // done!
  }


/************************************************************************
**************  Scaling functionality  **********************************
************************************************************************/


  /** helper function to compute the scale required by bob.ip.base.GeomNorm for the given image shapes */
  static inline blitz::TinyVector<double,2> _get_scale_factor(const blitz::TinyVector<int,2>& src_shape, const blitz::TinyVector<int,2>& dst_shape){
    double y_scale = (dst_shape[0]-1.) / (src_shape[0]-1.);
    double x_scale = (dst_shape[1]-1.) / (src_shape[1]-1.);
    return blitz::TinyVector<double,2>(y_scale, x_scale);
  }

  /**
   * @brief Function which rescales a 2D blitz::array/image of a given type.
   *   The first dimension is the height (y-axis), whereas the second
   *   one is the width (x-axis).
   * @param src The input blitz array
   * @param dst The output blitz array. The new array is resized according
   *   to the dimensions of this dst array.
   */
  template <typename T>
  void scale(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst){
    blitz::TinyVector<double,2> offset(0,0);
    blitz::Array<bool,2> src_mask, dst_mask;
    // .. apply scale with (0,0) as offset and 0 as rotation angle
    transform<T,false>(src, src_mask, offset, dst, dst_mask, offset, _get_scale_factor(src.shape(), dst.shape()), 0.);
  }

  /**
   * @brief Function which rescales a 2D blitz::array/image of a given type.
   *   The first dimension is the height (y-axis), whereas the second
   *   one is the width (x-axis).
   * @param src The input blitz array
   * @param src_mask The input blitz boolean mask array
   * @param dst The output blitz array. The new array is resized according
   *   to the dimensions of this dst array.
   * @param dst_mask The output blitz boolean mask array
   */
  template <typename T>
  void scale(const blitz::Array<T,2>& src, const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, blitz::Array<bool,2>& dst_mask){
    blitz::TinyVector<double,2> offset(0,0);
    // .. apply scale with (0,0) as offset and 0 as rotation angle
    transform<T,true>(src, src_mask, offset, dst, dst_mask, offset, _get_scale_factor(src.shape(), dst.shape()), 0.);
  }

  /**
   * @brief Function which rescales a 3D blitz::array/image of a given type.
   *   The first dimension is the number of color plane, the second is the
   * height (y-axis), whereas the third one is the width (x-axis).
   * @param src The input blitz array
   * @param dst The output blitz array. The new array is resized according
   *   to the dimensions of this dst array.
   */
  template <typename T>
  void scale(const blitz::Array<T,3>& src, blitz::Array<double,3>& dst)
  {
    // Check number of planes
    bob::core::array::assertSameDimensionLength(src.extent(0), dst.extent(0));
    for (int p = 0; p < dst.extent(0); ++p){
      const blitz::Array<T,2> src_slice = src(p, blitz::Range::all(), blitz::Range::all());
      blitz::Array<double,2> dst_slice =dst(p, blitz::Range::all(), blitz::Range::all());
      // Process one plane
      scale(src_slice, dst_slice);
    }
  }

  template <typename T>
  void scale(const blitz::Array<T,3>& src, const blitz::Array<bool,3>& src_mask, blitz::Array<double,3>& dst, blitz::Array<bool,3>& dst_mask)
  {
    // Check number of planes
    bob::core::array::assertSameDimensionLength(src.extent(0), dst.extent(0));
    bob::core::array::assertSameDimensionLength(src.extent(0), src_mask.extent(0));
    bob::core::array::assertSameDimensionLength(src_mask.extent(0), dst_mask.extent(0));
    for (int p = 0; p < dst.extent(0); ++p){
      const blitz::Array<T,2> src_slice = src(p, blitz::Range::all(), blitz::Range::all());
      const blitz::Array<bool,2> src_mask_slice = src_mask(p, blitz::Range::all(), blitz::Range::all());
      blitz::Array<double,2> dst_slice = dst(p, blitz::Range::all(), blitz::Range::all());
      blitz::Array<bool,2> dst_mask_slice = dst_mask(p, blitz::Range::all(), blitz::Range::all());
      // Process one plane
      scale(src_slice, src_mask_slice, dst_slice, dst_mask_slice);
    }
  }

  /**
   * @brief Function which returns the shape of an output blitz::array
   *   when rescaling an input image with the given scale factor.
   * @param src The input blitz array shape
   * @param scale_factor The scaling factor to apply
   * @return A blitz::TinyVector containing the shape of the rescaled image
   */
  template <int D>
  blitz::TinyVector<int, D> getScaledShape(const blitz::TinyVector<int, D> src_shape, const double scale_factor){
    blitz::TinyVector<int, D> dst_shape = src_shape;
    dst_shape(D-2) = floor(dst_shape(D-2) * scale_factor + 0.5);
    dst_shape(D-1) = floor(dst_shape(D-1) * scale_factor + 0.5);
    return dst_shape;
  }


/************************************************************************
**************  Rotating functionality  *********************************
************************************************************************/

  /**
   * @brief Function which rotates a 2D blitz::array/image of a given type with the given angle in degrees.
   *   The first dimension is the height (y-axis), whereas the second
   *   one is the width (x-axis).
   * @param src The input blitz array
   * @param dst The output blitz array
   * @param rotation_angle The angle in degrees to rotate the image with
   */
  template <typename T>
  void rotate(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst, const double rotation_angle){
    // rotation offset is the center of the image
    blitz::TinyVector<double,2> src_offset((src.extent(0)-1.)/2.,(src.extent(1)-1.)/2.);
    blitz::TinyVector<double,2> dst_offset((dst.extent(0)-1.)/2.,(dst.extent(1)-1.)/2.);
    blitz::Array<bool,2> src_mask, dst_mask;
    // .. apply scale with (0,0) as offset and 0 as rotation angle
    transform<T,false>(src, src_mask, src_offset, dst, dst_mask, dst_offset, blitz::TinyVector<double,2>(1., 1.), rotation_angle);
  }

  /**
   * @brief Function which rotates a 2D blitz::array/image of a given type with the given angle in degrees.
   *   The first dimension is the height (y-axis), whereas the second
   *   one is the width (x-axis).
   * @param src The input blitz array
   * @param src_mask The input blitz boolean mask array
   * @param dst The output blitz array
   * @param dst_mask The output blitz boolean mask array
   * @param rotation_angle The angle in degrees to rotate the image with
   */
  template <typename T>
  void rotate(const blitz::Array<T,2>& src, const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, blitz::Array<bool,2>& dst_mask, const double rotation_angle){
    // rotation offset is the center of the image
    blitz::TinyVector<double,2> src_offset((src.extent(0)-1.)/2.,(src.extent(1)-1.)/2.);
    blitz::TinyVector<double,2> dst_offset((dst.extent(0)-1.)/2.,(dst.extent(1)-1.)/2.);
    // .. apply scale with (0,0) as offset and 0 as rotation angle
    transform<T,true>(src, src_mask, src_offset, dst, dst_mask, dst_offset, blitz::TinyVector<double,2>(1., 1.), rotation_angle);
  }

  /**
   * @brief Function which rotates a 3D blitz::array/image of a given type.
   *   The first dimension is the number of color plane, the second is the
   * height (y-axis), whereas the third one is the width (x-axis).
   * @param src The input blitz array
   * @param dst The output blitz array
   * @param rotation_angle The angle in degrees to rotate the image with
   */
  template <typename T>
  void rotate(const blitz::Array<T,3>& src, blitz::Array<double,3>& dst, const double rotation_angle)
  {
    // Check number of planes
    bob::core::array::assertSameDimensionLength(src.extent(0), dst.extent(0));
    for (int p = 0; p < dst.extent(0); ++p){
      const blitz::Array<T,2> src_slice = src(p, blitz::Range::all(), blitz::Range::all());
      blitz::Array<double,2> dst_slice = dst(p, blitz::Range::all(), blitz::Range::all());
      // Process one plane
      rotate(src_slice, dst_slice, rotation_angle);
    }
  }

  template <typename T>
  void rotate(const blitz::Array<T,3>& src, const blitz::Array<bool,3>& src_mask, blitz::Array<double,3>& dst, blitz::Array<bool,3>& dst_mask, const double rotation_angle)
  {
    // Check number of planes
    bob::core::array::assertSameDimensionLength(src.extent(0), dst.extent(0));
    bob::core::array::assertSameDimensionLength(src.extent(0), src_mask.extent(0));
    bob::core::array::assertSameDimensionLength(src_mask.extent(0), dst_mask.extent(0));
    for (int p = 0; p < dst.extent(0); ++p){
      const blitz::Array<T,2> src_slice = src(p, blitz::Range::all(), blitz::Range::all());
      const blitz::Array<bool,2> src_mask_slice = src_mask(p, blitz::Range::all(), blitz::Range::all());
      blitz::Array<double,2> dst_slice = dst(p, blitz::Range::all(), blitz::Range::all());
      blitz::Array<bool,2> dst_mask_slice = dst_mask(p, blitz::Range::all(), blitz::Range::all());
      // Process one plane
      rotate(src_slice, src_mask_slice, dst_slice, dst_mask_slice, rotation_angle);
    }
  }

  /**
   * @brief Function which returns the shape of an output blitz::array
   *   when rotating an input image with the given rotation angle.
   * @param src_shape The input blitz array shape
   * @param rotation_angle The angle in degrees to rotate the image with
   * @return A blitz::TinyVector containing the shape of the rotated image
   */
  template <int D>
  blitz::TinyVector<int, D> getRotatedShape(const blitz::TinyVector<int, D> src_shape, const double rotation_angle){
    blitz::TinyVector<int, D> dst_shape = src_shape;
    // compute rotation shape
    double rad_angle = rotation_angle * M_PI / 180.;
    const double absCos = std::abs(cos(rad_angle));
    const double absSin = std::abs(sin(rad_angle));
    dst_shape(D-2) = floor(src_shape[D-2] * absCos + src_shape[D-1] * absSin + 0.5);
    dst_shape(D-1) = floor(src_shape[D-1] * absCos + src_shape[D-2] * absSin + 0.5);
    return dst_shape;
  }



/************************************************************************
**************  Other functionalities  **********************************
************************************************************************/

  /**
    * @brief Function which extracts a rectangle of maximal area from a
    *   2D mask of booleans (i.e. a 2D blitz array).
    * @warning The function assumes that the true values on the mask form
    *   a convex area.
    * @param mask The 2D input blitz array mask.
    * @result A blitz::TinyVector which contains in the following order:
    *   0/ The y-coordinate of the top left corner
    *   1/ The x-coordinate of the top left corner
    *   2/ The height of the rectangle
    *   3/ The width of the rectangle
    */
  const blitz::TinyVector<int,4> maxRectInMask(const blitz::Array<bool,2>& mask);


  /**
    * @brief Function which extracts an image with a nearest neighbour
    *   technique, a boolean mask being given.
    *   a/ The columns of the image are firstly extrapolated wrt. to the
    *   nearest neighbour on the same column.
    *   b/ The rows of the image are the extrapolate wrt. to the
    *   closest neighbour on the same row.
    *   The first dimension is the height (y-axis), whereas the second one
    *   is the width (x-axis).
    * @param src_mask The 2D input blitz array mask.
    * @param img The 2D input/output blitz array/image.
    * @warning The function assumes that the true values on the mask form
    *   a convex area.
    * @warning img is used as both an input and output, in order to provide
    *   high performance. A copy might be done by the user before calling
    *   the function if required.
    */
  template <typename T>
  void extrapolateMask( const blitz::Array<bool,2>& src_mask, blitz::Array<T,2>& img){
    // Check input and output size
    bob::core::array::assertSameShape(src_mask, img);
    bob::core::array::assertZeroBase(src_mask);
    bob::core::array::assertZeroBase(img);

    // TODO: check that the input mask is convex

    // Determine the "full of false" columns
    blitz::firstIndex i;
    blitz::secondIndex j;

    blitz::Array<bool,1> column_true(blitz::any(src_mask(j,i), j) );
    int true_min_index=blitz::first(column_true);
    int true_max_index=blitz::last(column_true);

    if (true_min_index < 0 || true_max_index < 0){
      throw std::runtime_error("The given mask is invalid as it contains only 'False' values.");
    }

    // Extrapolate the "non false" columns
    for(int jj=true_min_index; jj<=true_max_index; ++jj)
    {
      blitz::Array<bool,1> src_col( src_mask( blitz::Range::all(), jj) );
      int i_first = blitz::first(src_col);
      if( i_first>0)
      {
        blitz::Range r_first(0,i_first-1);
        img(r_first,jj) = img(i_first,jj);
      }

      int i_last=blitz::last(src_col);
      if( i_last+1<src_mask.extent(0))
      {
        blitz::Range r_last(i_last+1,src_mask.extent(0)-1);
        img(r_last,jj) = img(i_last,jj);
      }
    }

    // Extrapolate the rows
    if(true_min_index>0)
    {
      blitz::Range r_left(0,true_min_index-1);
      for(int i=0; i<src_mask.extent(0); ++i)
        img(i,r_left) = img(i,true_min_index);
    }
    if(true_max_index+1<src_mask.extent(1))
    {
      blitz::Range r_right(true_max_index+1,src_mask.extent(1)-1);
      for(int i=0; i<src_mask.extent(0); ++i)
        img(i,r_right) = img(i,true_max_index);
    }
  }


  /**
    * @brief Function which fills  unmasked pixel areas of an image with pixel values from the border of the masked part of the image
    *   by adding some random noise.
    * @param src_mask The 2D input blitz array mask.
    * @param img The 2D input/output blitz array/image.
    * @param rng The random number generatir to consider
    * @param random_factor The standard deviation of a normal distribution to multiply pixel values with
    * @param neighbors The (maximum) number of additional neighboring border values to choose from
    * @warning The function assumes that the true values on the mask form
    *   a convex area.
    * @warning img is used as both an input and output, in order to provide
    *   high performance. A copy might be done by the user before calling
    *   the function if required.
    */
  template <typename T>
  void extrapolateMaskRandom(const blitz::Array<bool,2>& mask, blitz::Array<T,2>& img, boost::mt19937& rng, double random_factor = 0.01, int neighbors = 5){
    // Check input and output size
    bob::core::array::assertSameShape(mask, img);

    // get the masked center
    int miny = mask.extent(0)-1, maxy = 0, minx = mask.extent(1)-1, maxx = 0;
    for (int y = 0; y < mask.extent(0); ++y)
      for (int x = 0; x < mask.extent(1); ++x)
        if (mask(y,x)){
          miny = std::min(miny, y);
          maxy = std::max(maxy, y);
          minx = std::min(minx, x);
          maxx = std::max(maxx, x);
    }

    int center_y = (miny + maxy)/2;
    int center_x = (minx + maxx)/2;

    if (!mask(center_y, center_x)) throw std::runtime_error("The center of the masked area is not masked. Is your mask convex?");

    blitz::Array<bool,2> filled_mask(mask.shape());
    filled_mask = mask;

    // the four directions to go (in this order):
    // right, down, left, up
    int directions_y[] = {0, 1, 0, -1};
    int directions_x[] = {1, 0, -1, 0};
    // the border values for the four directions
    int border[] = {img.extent(1), img.extent(0), 1, 1};
    bool at_border[4] = {false};

    // the current maxima
    int maxima_y[4], maxima_x[4];
    for (int i = 0; i < 4; ++i){
      maxima_y[i] = center_y + directions_y[i];
      maxima_x[i] = center_x + directions_x[i];
    }
    // the current index (i.e., direction) to go
    int current_index = 0;
    int current_dir_y = directions_y[current_index];
    int current_dir_x = directions_x[current_index];

    // we start from the center
    int current_pos_y = center_y;
    int current_pos_x = center_x;

    // go from the mask center in all directions, using a spiral
    while (!at_border[0] || !at_border[1] || !at_border[2] || !at_border[3]){
      // check that we haven't reached our limits yet
      if (current_dir_y * current_pos_y + current_dir_x * current_pos_x >= maxima_y[current_index] * current_dir_y + maxima_x[current_index] * current_dir_x){
        // increase the maxima
        maxima_y[current_index] += current_dir_y;
        maxima_x[current_index] += current_dir_x;
        // check if we are at the border
        if (current_pos_y * current_dir_y + current_pos_x * current_dir_x >= border[current_index]){
          at_border[current_index] = true;
        }
        // change direction
        current_index = (current_index + 1) % 4;
        current_dir_y = directions_y[current_index];
        current_dir_x = directions_x[current_index];
      }

      // check if we have to write a value
      if (current_pos_y >= 0 && current_pos_y < img.extent(0) && current_pos_x >= 0 && current_pos_x < img.extent(1) && !mask(current_pos_y, current_pos_x)){
        // fill with pixel from the inner part of the spiral
        int next_index = (current_index + 1) % 4;
        int next_dir_y = directions_y[next_index];
        int next_dir_x = directions_x[next_index];

        // .. get valid border pixel (e.g. that has been set before)
        int valid_y = current_pos_y + next_dir_y;
        int valid_x = current_pos_x + next_dir_x;
        while (valid_y * next_dir_y + valid_x * next_dir_x < border[next_index] && !filled_mask(valid_y, valid_x)){
          valid_y += next_dir_y;
          valid_x += next_dir_x;
        }

        // check if we have found some part that is not connected anywhere
        if (valid_y * next_dir_y + valid_x * next_dir_x >= border[next_index]){
          bob::core::warn << "Could not find valid pixel in direction (" << next_dir_y << ", " << next_dir_x << ") at pixel position (" << current_pos_y << ", " << current_pos_x << "); is your mask convex?";
        } else {
          T value = static_cast<T>(0);
          // choose one of the next pixels
          if (neighbors >= 1){
            std::vector<T> values;
            for (int c = -neighbors; c <= neighbors; ++c){
              int pos_y = valid_y + c * current_dir_y;
              int pos_x = valid_x + c * current_dir_x;
              if (pos_y >= 0 && pos_y < img.extent(0) && pos_x >= 0 && pos_x < img.extent(1) && filled_mask(pos_y, pos_x)){
                values.push_back(img(pos_y, pos_x));
              }
            }
            if (!values.size()){
              bob::core::warn << "Could not find valid pixel in range " << neighbors << " close to the border at pixel position (" << current_pos_y << ", " << current_pos_x << "); is your mask convex?";
            } else {
              // choose random value
              value = values[boost::uniform_int<int>(0, values.size()-1)(rng)];
            }
          } else { // neighbors == 1
            value = img(valid_y, valid_x);
          }
          if (random_factor){
            value = static_cast<T>(bob::core::random::normal_distribution<double>(1., random_factor)(rng) * value);
          }
          img(current_pos_y, current_pos_x) = value;
          filled_mask(current_pos_y, current_pos_x) = true;
        }
      } // write value

      // move one step towards the current direction
      current_pos_y += current_dir_y;
      current_pos_x += current_dir_x;

    } // while
  }


} } } // namespaces

#endif // BOB_IP_BASE_AFFINE_H
