/**
 * @date Thu Jul  3 12:37:19 CEST 2014
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 *
 * @brief This file provides a function to perform median filtering
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_MEDIAN_H
#define BOB_IP_BASE_MEDIAN_H

#include <vector>
#include <algorithm>
#include <bob.core/assert.h>
#include <bob.core/cast.h>

namespace bob { namespace ip { namespace base {

  template <typename T>
  void medianFilter(
    const blitz::Array<T,2>& src,
    blitz::Array<T,2>& dst,
    const blitz::TinyVector<int,2>& radius
  ){
    // Checks
    bob::core::array::assertZeroBase(src);
    bob::core::array::assertZeroBase(dst);
    blitz::TinyVector<int,2> dst_size(src.extent(0) - 2 * radius[0], src.extent(1) - 2 * radius[1]);
    bob::core::array::assertSameShape(dst, dst_size);

    // compute centeral pixel
    int center = (2*radius[0]+1)*(2*radius[1]+1)/2;
    // we only sort the first half of the sequence (this is all we need)
    std::vector<T> _temp(center+1);
    // iterate over the destination array
    for (int y = 0; y < dst_size[0]; ++y)
      for (int x = 0; x < dst_size[1]; ++x){
        // get a slice from the src array
        const blitz::Array<T,2> slice(src(blitz::Range(y, y + 2 * radius[0]), blitz::Range(x, x + 2 * radius[1])));
        // compute the median
        // we only sort the first half of the sequence
        std::partial_sort_copy(slice.begin(), slice.end(), _temp.begin(), _temp.end());
        // get the central element
        dst(y,x) = _temp[center];
    }
  }


  template <typename T>
  void medianFilter(
    const blitz::Array<T,3>& src,
    blitz::Array<T,3>& dst,
    const blitz::TinyVector<int,2>& radius
  ){
    // iterate over the color layers
    for (int p = 0; p < dst.extent(0); ++p){
      const blitz::Array<T,2> src_slice = src(p, blitz::Range::all(), blitz::Range::all());
      blitz::Array<T,2> dst_slice = dst(p, blitz::Range::all(), blitz::Range::all());

      // Apply median filter to the plane
      medianFilter(src_slice, dst_slice, radius);
    }
  }

} } } // namespaces

#endif // BOB_IP_BASE_MEDIAN_H

