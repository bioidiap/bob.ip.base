/**
 * @date Mon May 2 10:01:08 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the Multiscale Retinex algorithm as described in:
 *  "A Multiscale Retinex for bridging the gap between color images and the
 *   Human observation of scenes", D. Jobson, Z. Rahman and G. Woodell,
 *  in IEEE Transactions on Image Processing, vol. 6, n. 7, July 1997
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_MULTISCALE_RETINEX_H
#define BOB_IP_BASE_MULTISCALE_RETINEX_H

#include <bob.core/assert.h>
#include <bob.core/cast.h>
#include <bob.sp/extrapolate.h>
#include <boost/shared_array.hpp>

#include <bob.ip.base/Gaussian.h>

namespace bob { namespace ip { namespace base {

  /**
    * @brief This class allows to preprocess an image with the Multiscale
    * Retinex algorithm as described in:
    *  "A Multiscale Retinex for bridging the gap between color images and
    *   the Human observation of scenes", D. Jobson, Z. Rahman and
    *   G. Woodell,
    *  in IEEE Transactions on Image Processing, vol. 6, n. 7, July 1997
    */
  class MultiscaleRetinex
  {
    public:
      /**
       * @brief Creates an object to preprocess images with the Multiscale
       *  Retinex algorithm
       * @param n_scales The number of scales
       * @param size_min The size of the smallest convolution kernel
       * @param size_step The step size of the convolution kernels
       * @param sigma The standard deviation of the kernel for the smallest
       *  convolution kernel.
       * @param border_type The interpolation type for the convolution
       */
      MultiscaleRetinex(
          const size_t n_scales=1,
          const int size_min=1,
          const int size_step=1,
          const double sigma=5.,
          const bob::sp::Extrapolation::BorderType border_type = bob::sp::Extrapolation::Mirror
      );

      /**
       * @brief Copy constructor
       */
      MultiscaleRetinex(const MultiscaleRetinex& other);

      /**
       * @brief Destructor
       */
      virtual ~MultiscaleRetinex() {}

      /**
       * @brief Assignment operator
       */
      MultiscaleRetinex& operator=(const MultiscaleRetinex& other);

      /**
       * @brief Equal to
       */
      bool operator==(const MultiscaleRetinex& b) const;
      /**
       * @brief Not equal to
       */
      bool operator!=(const MultiscaleRetinex& b) const;

      /**
       * @brief Resets the parameters of the filter
       * @param n_scales The number of scales
       * @param size_min The size of the smallest convolution kernel
       * @param size_step The step size of the convolution kernels
       * @param sigma The variance of the kernal for the smallest
       *  convolution kernel.
       * @param border_type The interpolation type for the convolution
       */
      void reset(
        const size_t n_scales=1,
        const int size_min=1,
        const int size_step=1,
        const double sigma=2.,
        const bob::sp::Extrapolation::BorderType border_type = bob::sp::Extrapolation::Mirror
      );

      /**
       * @brief Getters
       */
      size_t getNScales() const { return m_n_scales; }
      int getSizeMin() const { return m_size_min; }
      int getSizeStep() const { return m_size_step; }
      double getSigma() const { return m_sigma; }
      bob::sp::Extrapolation::BorderType getConvBorder() const { return m_conv_border; }

      /**
       * @brief Setters
       */
      void setNScales(const size_t n_scales) { m_n_scales = n_scales; m_gaussians.reset(new bob::ip::base::Gaussian[m_n_scales]); computeKernels(); }
      void setSizeMin(const int size_min) { m_size_min = size_min; computeKernels(); }
      void setSizeStep(const int size_step) { m_size_step = size_step; computeKernels(); }
      void setSigma(const double sigma) { m_sigma = sigma; computeKernels(); }
      void setConvBorder(const bob::sp::Extrapolation::BorderType border_type) { m_conv_border = border_type; computeKernels(); }

      /**
       * @brief Process a 2D blitz Array/Image
       * @param src The 2D input blitz array
       * @param dst The 2D output blitz array
       */
      template <typename T>
      void process(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst){
        // Checks are postponed to the Gaussian operator() function.
        dst = 0.;
        if( m_tmp.extent(0) != src.extent(0) || m_tmp.extent(1) != src.extent(1))
          m_tmp.resize(src.extent(0), src.extent(1) );
        for(size_t s=0; s<m_n_scales; ++s) {
          m_gaussians[s].filter(src,m_tmp);
          dst += (blitz::log(src+1.) - blitz::log(m_tmp+1.));
        }
        dst /= (double)m_n_scales;
      }

      /**
       * @brief Process a 3D blitz Array/Image
       * @param src The 3D input blitz array
       * @param dst The 3D output blitz array
       */
      template <typename T>
      void process(const blitz::Array<T,3>& src, blitz::Array<double,3>& dst){
        for( int p=0; p<dst.extent(0); ++p) {
          const blitz::Array<T,2> src_slice = src( p, blitz::Range::all(), blitz::Range::all() );
          blitz::Array<double,2> dst_slice = dst( p, blitz::Range::all(), blitz::Range::all() );

          // Gaussian smooth plane
          process(src_slice, dst_slice);
        }
      }

    private:
      void computeKernels();

      /**
       * @brief Attributes
       */
      size_t m_n_scales;
      int m_size_min;
      int m_size_step;
      double m_sigma;
      bob::sp::Extrapolation::BorderType m_conv_border;

      boost::shared_array<bob::ip::base::Gaussian> m_gaussians;
      blitz::Array<double,2> m_tmp;
  };

} } } // namespaces

#endif /* BOB_IP_BASE_MULTISCALE_RETINEX_H */

