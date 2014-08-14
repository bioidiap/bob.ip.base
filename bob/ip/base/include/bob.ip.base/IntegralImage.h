/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu Apr 28 20:09:16 2011 +0200
 *
 * @brief This file defines a function to compute the integral image of a 2D image
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_INTEGRAL_IMAGE_H
#define BOB_IP_BASE_INTEGRAL_IMAGE_H

#include <bob.core/assert.h>
#include <bob.core/array_index.h>

namespace bob { namespace ip { namespace base {
    /**
      * @brief Function which computes the integral image of a 2D
      *   blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning No check is performed wrt. the array dimensions.
      * @param src The input image
      * @param dst The output integral image
      */
    template<typename T, typename U>
    void integral_(const blitz::Array<T,2>& src, blitz::Array<U,2>& dst)
    {
      dst(0,0) = static_cast<U>(src(0,0));
      // Compute first row
      for(int x=1; x<src.extent(1); ++x)
        dst(0,x) = dst(0,x-1) + static_cast<U>(src(0,x));
      // Compute remaining part
      for(int y=1; y<src.extent(0); ++y)
      {
        dst(y,0) = dst(y-1,0) + static_cast<U>(src(y,0));
        U row_sum_cur = src(y,0);
        for(int x=1; x<src.extent(1); ++x)
        {
          row_sum_cur += static_cast<U>(src(y,x));
          dst(y,x) = dst(y-1,x) + row_sum_cur;
        }
      }
    }


  /**
    * @brief Function which computes the integral image of a 2D
    *   blitz::array/image of a given type.
    *   The first dimension is the height (y-axis), whereas the second
    *   one is the width (x-axis).
    * @warning It is the user responsability to select a suitable type
    *   for the destination array. Using a type with the same range of
    *   value might cause out of range problems.
    * @param src The input image
    * @param dst The output integral image
    * @param addZeroBorder This requires the dst array to be 1 pixel
    *   larger in each dimension. Besides, an extra zero pixel will be
    *   added at the beginning of each row and column
    */
  template<typename T, typename U>
  void integral(const blitz::Array<T,2>& src, blitz::Array<U,2>& dst, const bool addZeroBorder=false)
  {
    // Checks that the src/dst arrays have zero base indices
    bob::core::array::assertZeroBase(src);
    bob::core::array::assertZeroBase(dst);
    if(addZeroBorder)
    {
      blitz::TinyVector<int,2> shape = src.shape();
      shape += 1;
      bob::core::array::assertSameShape(dst,shape);
    }
    else
      bob::core::array::assertSameShape(src,dst);

    // Compute the integral image of the 2D array
    if(addZeroBorder)
    {
      for(int y=0; y<dst.extent(0); ++y)
        dst(y,0) = 0;
      for(int x=1; x<dst.extent(1); ++x)
        dst(0,x) = 0;
      blitz::Array<U,2> dst_c =
        dst(blitz::Range(1,src.extent(0)), blitz::Range(1,src.extent(1)));
      integral_(src, dst_c);
    }
    else
      integral_(src, dst);
  }


  /**
    * @brief Function which computes the integral image and the
    *   integral square image of a 2D blitz::array/image of a given type.
    * @warning No check is performed wrt. the array dimensions.
    * @param src The input image
    * @param dst The output integral image
    * @param sqr The output integral square image
    */
  template<typename T, typename U>
  void integral_(const blitz::Array<T,2>& src, blitz::Array<U,2>& dst, blitz::Array<U,2>& sqr)
  {
    const U v = static_cast<U>(src(0,0));
    dst(0,0) = v;
    sqr(0,0) = v*v;

    // Compute first row
    for(int x=1; x<src.extent(1); ++x){
      const U v = static_cast<U>(src(0,x));
      dst(0,x) = dst(0,x-1) + v;
      sqr(0,x) = sqr(0,x-1) + v*v;
    }

    // Compute remaining part
    for(int y=1; y<src.extent(0); ++y)
    {
      U row_sum_cur = static_cast<U>(src(y,0));
      U row_sum_sqr = row_sum_cur * row_sum_cur;

      dst(y,0) = dst(y-1,0) + row_sum_cur;
      sqr(y,0) = sqr(y-1,0) + row_sum_sqr;

      for(int x=1; x<src.extent(1); ++x)
      {
        const U v = static_cast<U>(src(y,x));
        row_sum_cur += v;
        row_sum_sqr += v*v;

        dst(y,x) = dst(y-1,x) + row_sum_cur;
        sqr(y,x) = sqr(y-1,x) + row_sum_sqr;
      }
    }
  }

  /**
    * @brief Function which computes the integral image and the
    *   integral square image of a 2D blitz::array/image of a given type.
    * @warning It is the user responsibility to select a suitable type
    *   for the destination array. Using a type with the same range of
    *   value might cause out of range problems.
    * @param src The input image
    * @param dst The output integral image
    * @param sqr The output integral square image
    * @param addZeroBorder This requires the dst and sqr arrays to be 1
    *   pixel larger in each dimension. Besides, an extra zero pixel will
    *   be added at the beginning of each row and column
    */
  template<typename T, typename U>
  void integral(const blitz::Array<T,2>& src, blitz::Array<U,2>& dst, blitz::Array<U,2>& sqr, const bool addZeroBorder=false)
  {
    // Checks that the src/dst arrays have zero base indices
    bob::core::array::assertZeroBase(src);
    bob::core::array::assertZeroBase(dst);
    bob::core::array::assertZeroBase(sqr);
    if(addZeroBorder)
    {
      blitz::TinyVector<int,2> shape = src.shape();
      shape += 1;
      bob::core::array::assertSameShape(dst,shape);
      bob::core::array::assertSameShape(sqr,shape);
    }
    else{
      bob::core::array::assertSameShape(src,dst);
      bob::core::array::assertSameShape(src,sqr);
    }
    // Compute the integral image of the 2D array
    if(addZeroBorder)
    {
      for(int y=0; y<dst.extent(0); ++y)
        dst(y,0) = sqr(y,0) = 0;
      for(int x=1; x<dst.extent(1); ++x)
        dst(0,x) = sqr(0,x) = 0;
      blitz::Array<U,2> dst_c =
        dst(blitz::Range(1,src.extent(0)), blitz::Range(1,src.extent(1)));
      blitz::Array<U,2> sqr_c =
        sqr(blitz::Range(1,src.extent(0)), blitz::Range(1,src.extent(1)));
      integral_(src, dst_c, sqr_c);
    }
    else
      integral_(src, dst, sqr);
  }

} } } // namespaces

#endif /* BOB_IP_BASE_INTEGRAL_IMAGE_H */
