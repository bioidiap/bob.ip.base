/**
 * @date Fri Apr 29 12:13:22 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to process images with the Sobel operator
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_SOBEL_H
#define BOB_IP_BASE_SOBEL_H

#include <stdexcept>
#include <boost/format.hpp>

#include <bob.core/assert.h>
#include <bob.core/cast.h>
#include <bob.sp/conv.h>
#include <bob.sp/extrapolate.h>

namespace bob { namespace ip { namespace base {

  // compute filtering with the given filter
  template <typename T>
  void _sobel(
    const blitz::Array<T,2>& src,
    const blitz::Array<T,2>& kernel,
    blitz::Array<T,2>& dst,
    bob::sp::Extrapolation::BorderType border_type
  ){
    if (border_type == bob::sp::Extrapolation::Zero)
      bob::sp::conv(src, kernel, dst);
    else {
      blitz::Array<T,2> tmp(bob::sp::getConvOutputSize(src, kernel, bob::sp::Conv::Full));
      if (border_type == bob::sp::Extrapolation::NearestNeighbour) bob::sp::extrapolateNearest(src, tmp);
      else if (border_type == bob::sp::Extrapolation::Circular) bob::sp::extrapolateCircular(src, tmp);
      else if (border_type == bob::sp::Extrapolation::Mirror) bob::sp::extrapolateMirror(src, tmp);
      else throw std::runtime_error("The given border type is (currently) not supported");
      bob::sp::conv(tmp, kernel, dst, bob::sp::Conv::Valid);
    }
  }

  /**
    * @brief Process a 2D blitz Array/Image by applying the Sobel operator
    *   The resulting 3D array will contain two planes:
    *     - The first one for the convolution with the y-kernel
    *     - The second one for the convolution with the x-kernel
    * @warning The selected type should be signed (e.g. int64_t or double)
    */
  template <typename T>
  void sobel(
    const blitz::Array<T,2>& src,
    blitz::Array<T,3>& dst,
    bob::sp::Extrapolation::BorderType border_type = bob::sp::Extrapolation::Mirror
  ){
    // Check that dst has two planes
    if (dst.extent(0) != 2) throw std::runtime_error((boost::format("destination array extent for the first dimension (0) is not 2, but %d") % dst.extent(0)).str());

    // Check that dst has zero bases
    bob::core::array::assertZeroBase(dst);

    // Define kernels for y and x
    blitz::Array<T,2> kernel_y(3,3), kernel_x(3,3);
    kernel_y = -1, -2, -1, 0, 0, 0, 1, 2, 1;
    kernel_x = -1, 0, 1, -2, 0, 2, -1, 0, 1;

    // Define slices for y and x
    blitz::Array<T,2> dst_y = dst(0, blitz::Range::all(), blitz::Range::all());
    blitz::Array<T,2> dst_x = dst(1, blitz::Range::all(), blitz::Range::all());

    // execute
    _sobel(src, kernel_y, dst_y, border_type);
    _sobel(src, kernel_x, dst_x, border_type);
  }


} } } // namespaces

#endif /* BOB_IP_BASE_SOBEL_H */

