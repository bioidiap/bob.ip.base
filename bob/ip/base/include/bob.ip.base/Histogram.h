/**
 * @author Manuel Günther <manuel.guenther@idiap.ch>
 * @date Thu Jun 26 09:33:10 CEST 2014
 *
 * This file defines functions and classes for affine image transformations
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_HISTOGRAM_H
#define BOB_IP_BASE_HISTOGRAM_H


#include <bob.core/assert.h>
#include <bob.io.base/array_type.h>

namespace bob { namespace ip { namespace base {

  /**
   * Compute an histogram of a 2D array.
   *
   * @warning This function only accepts arrays of int or float (int8, int16,
   *          int32, int64, uint8, uint16, uint32, float32, float64
   *          and float128)
   *          Any other type raises a std::runtime_error exception
   * @warning You must have @c min <= @c src(i,j) <= @c max, for every i and j
   * @warning If @c min >= @c max or @c nb_bins == 0, a
   *
   * @param src source 2D array
   * @param histo result of the function. This array must have @c nb_bins
   *              elements
   * @param min least possible value in @c src
   * @param max greatest possible value in @c src
   */
  template<typename T>
  void histogram(
    const blitz::Array<T, 2>& src,
    blitz::Array<uint64_t, 1>& histo
  ){
    bob::io::base::array::ElementType element_type = bob::io::base::array::getElementType<T>();

    // Check that the given type is supported
    switch (element_type) {
      case bob::io::base::array::t_uint8:
      case bob::io::base::array::t_uint16:
      case bob::io::base::array::t_uint32:
      case bob::io::base::array::t_uint64:
        // Valid type
        break;
      default:
        // Invalid type
        throw std::runtime_error((boost::format("data type `%s' cannot be histogrammed without specifying a range") % bob::io::base::array::stringize<T>()).str());
    }

    // empty the histogram
    histo = 0;

    unsigned max = histo.extent(0);
    for (auto it = src.begin(); it != src.end(); ++it){
      // perform a check that the pixel is in range
      if ((unsigned)*it >= max)
        throw std::runtime_error((boost::format("The pixel with value (%d) in the source image is higher than the number of bins (%d)") % (unsigned)*it % (unsigned)max).str());
      ++histo(*it);
    }
  }


  /**
   * Compute an histogram of a 2D array.
   *
   * @warning This function only accepts arrays of int or float (int8, int16,
   *          int32, int64, uint8, uint16, uint32, float32, float64
   *          and float128)
   *          Any other type raises a std::runtime_error exception
   * @warning You must have @c min <= @c src(i,j) <= @c max, for every i and j
   * @warning If @c min >= @c max or @c nb_bins == 0, a
   *
   * @param src source 2D array
   * @param histo result of the function. This array must have @c nb_bins
   *              elements
   * @param min least possible value in @c src
   * @param max greatest possible value in @c src
   */
  template<typename T>
  void histogram(
    const blitz::Array<T, 2>& src,
    blitz::Array<uint64_t, 1>& histo,
    T min,
    T max
  ){
    bob::io::base::array::ElementType element_type = bob::io::base::array::getElementType<T>();

    // Check that the given type is supported
    switch (element_type) {
      case bob::io::base::array::t_int8:
      case bob::io::base::array::t_int16:
      case bob::io::base::array::t_int32:
      case bob::io::base::array::t_int64:
      case bob::io::base::array::t_uint8:
      case bob::io::base::array::t_uint16:
      case bob::io::base::array::t_uint32:
      case bob::io::base::array::t_uint64:
      case bob::io::base::array::t_float32:
      case bob::io::base::array::t_float64:
      case bob::io::base::array::t_float128:
        // Valid type
        break;
      default:
        // Invalid type
        throw std::runtime_error((boost::format("data type `%s' cannot be histogrammed") % bob::io::base::array::stringize<T>()).str());
    }

    if (max <= min) {
      throw std::runtime_error((boost::format("the `max' value (%1%) should be larger than the `min' value (%2%)") % max % min).str());
    }

    // empty the histogram
    histo = 0;

    int nb_bins = histo.extent(0);

    // Handle the special case nb_bins == 1
    if (nb_bins == 1) {
      histo(0) += histo.size();
      return;
    }

    double width = max - min;
    double bin_size = width / static_cast<double>(nb_bins);

    for(int i = src.lbound(0); i <= src.ubound(0); i++) {
      for(int j = src.lbound(1); j <= src.ubound(1); j++) {
        T element = src(i, j);
        if (element < min || element > max)
          throw std::runtime_error((boost::format("The pixel with value (%1%) in the source image is not in the given range (%2%, %3%)") % element % min % max).str());
        // Convert a value into a bin index
        int index = static_cast<int>((element - min) / bin_size);
        index = std::min(index, nb_bins-1);
        ++(histo(index));
      }
    }
  }

  /**
   * Performs a histogram equalization of an image.
   *
   * @warning This function only accepts source arrays of int (uint8, uint16)
   *          and target arrays of type int (see above) or float(float32, float64).
   *          Any other type raises a std::runtime_error exception
   *
   * If the given target image is of integral type, the values will be spread out to fill the complete range of that type.
   * If the target is of type float, the values will be spread out to fill the range of the **source** type.
   *
   * @param src   the source 2D array of integral type
   * @param dst   the target 2D array
   */
  template<typename T1, typename T2>
  void histogramEqualize(
    const blitz::Array<T1, 2>& src,
    blitz::Array<T2, 2>& dst
  ){
    bob::io::base::array::ElementType element_type = bob::io::base::array::getElementType<T1>();
    // Check that the given type is supported
    // we here only support integral types lower than 64 bit
    switch (element_type) {
      case bob::io::base::array::t_uint8:
      case bob::io::base::array::t_uint16:
        // Valid type
        break;
      default:
        // Invalid type
        throw std::runtime_error((boost::format("data type `%s' cannot be histogrammed") % bob::io::base::array::stringize<T1>()).str());
    }

    // range of the desired type
    T2 dst_min, dst_max;

    element_type = bob::io::base::array::getElementType<T2>();

    // Check that the given type is supported
    // we here only support integral types lower than 64 bit
    switch (element_type) {
      case bob::io::base::array::t_uint8:
      case bob::io::base::array::t_uint16:
      case bob::io::base::array::t_uint32:
        // Valid type
        dst_min = std::numeric_limits<T2>::min();
        dst_max = std::numeric_limits<T2>::max();
        break;
      case bob::io::base::array::t_float32:
      case bob::io::base::array::t_float64:
        dst_min = static_cast<T2>(std::numeric_limits<T1>::min());
        dst_max = static_cast<T2>(std::numeric_limits<T1>::max());
        break;
      default:
        // Invalid type
          std::runtime_error((boost::format("data type `%s' cannot be the destination of histogram equalization") % bob::io::base::array::stringize<T2>()).str());
    }
    bob::core::array::assertSameShape(src, dst);

    // first, compute histogram of the image
    uint32_t bin_count = std::numeric_limits<T1>::max() + 1;
    blitz::Array<uint64_t,1> hist(bin_count);
    histogram(src, hist);

    // now, compute the cumulative histogram distribution function
    blitz::Array<double,1> cdf(bin_count);
    // -- we don't count the black pixels...
    double pixel_count = src.size() - hist(0);
    cdf(0) = 0.;
    for (uint32_t i = 1; i < bin_count; ++i){
      cdf(i) = cdf(i-1) + (double)hist(i) / pixel_count;
    }

    // fill the resulting image
    T2 dst_range = dst_max - dst_min;
    for (int y = src.lbound(0); y <= src.ubound(0); ++y)
      for (int x = src.lbound(1); x <= src.ubound(1); ++x){
        // here, the CDF is indexed by the current pixel value to get
        dst(y + dst.lbound(0), x + dst.lbound(1)) = static_cast<T2>(cdf(src(y,x)) * dst_range + dst_min);
    }

  }

} } } // namespaces

#endif // BOB_IP_BASE_HISTOGRAM_H
