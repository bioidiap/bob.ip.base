/**
 * @date Thu July 19 11:44:08 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the Self Quotient Image algorithm as described in:
 *  "Face Recognition under Varying Lighting Conditions Using Self Quotient
 *   Image", H. Wang, S.Z. Li and Y. Wang,
 *  in Proceedings of the IEEE International Conference on Image Processing,
 *     October 2004, vol. 2, p. 1397-1400
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_SELF_QUOTIENT_IMAGE_H
#define BOB_IP_BASE_SELF_QUOTIENT_IMAGE_H

#include <bob.core/assert.h>
#include <bob.sp/extrapolate.h>
#include <boost/shared_array.hpp>

#include <bob.ip.base/WeightedGaussian.h>

namespace bob { namespace ip { namespace base {

  /**
    * @brief This class allows to preprocess an image with the Self Quotient
    * Image algorithm as described in:
    *  "Face Recognition under Varying Lighting Conditions Using Self
    *   Quotient Image", H. Wang, S.Z. Li and Y. Wang,
    *  in Proceedings of the IEEE International Conference on Image
    *     Processing, October 2004, vol. 2, p. 1397-1400
    */
  class SelfQuotientImage
  {
    public:
      /**
        * @brief Creates an object to preprocess images with the Self
        *  Quotient Image algorithm
        * @param n_scales The number of scales
        * @param size_min The size of the smallest convolution kernel
        * @param size_step The step increase in size for consecutive
        *                  convolution kernels
        * @param sigma The variance of the smallest convolution kernel
        * @param border_type The interpolation type for the convolution
        */
      SelfQuotientImage(
          const size_t n_scales=1,
          const size_t size_min=1,
          const size_t size_step=1,
          const double sigma=sqrt(2.),
          const bob::sp::Extrapolation::BorderType border_type = bob::sp::Extrapolation::Mirror
      );

      /**
        * @brief Copy constructor
        */
      SelfQuotientImage(const SelfQuotientImage& other);

      /**
        * @brief Destructor
        */
      virtual ~SelfQuotientImage() {}

      /**
        * @brief Assignment operator
        */
      SelfQuotientImage& operator=(const SelfQuotientImage& other);

      /**
        * @brief Equal to
        */
      bool operator==(const SelfQuotientImage& b) const;
      /**
        * @brief Not equal to
        */
      bool operator!=(const SelfQuotientImage& b) const;

      /**
        * @brief Resets the parameters of the filter
        * @param n_scales The number of scales
        * @param size_min The size of the smallest convolution kernel
        * @param size_step The step increase in size for consecutive
        *                  convolution kernels
        * @param sigma The variance of the smallest convolution kernel
        * @param border_type The interpolation type for the convolution
       */
      void reset(
        const size_t n_scales=1,
        const size_t size_min=1,
        const size_t size_step=1,
        const double sigma=sqrt(2.),
        const bob::sp::Extrapolation::BorderType border_type = bob::sp::Extrapolation::Mirror
      );

      /**
       * @brief Getters
       */
      size_t getNScales() const { return m_n_scales; }
      size_t getSizeMin() const { return m_size_min; }
      size_t getSizeStep() const { return m_size_step; }
      double getSigma() const { return m_sigma; }
      bob::sp::Extrapolation::BorderType getConvBorder() const { return m_conv_border; }

      /**
       * @brief Setters
       */
      void setNScales(const size_t n_scales){
        m_n_scales = n_scales;
        m_wgaussians.reset(new bob::ip::base::WeightedGaussian[m_n_scales]);
        computeKernels();
      }
      void setSizeMin(const size_t size_min) { m_size_min = size_min; computeKernels(); }
      void setSizeStep(const size_t size_step) { m_size_step = size_step; computeKernels(); }
      void setSigma(const double sigma) { m_sigma = sigma; computeKernels(); }
      void setConvBorder(const bob::sp::Extrapolation::BorderType border_type) { m_conv_border = border_type; computeKernels(); }

        /**
         * @brief Process a 2D blitz Array/Image
         * @param src The 2D input blitz array
         * @param dst The 2D output blitz array
         */
        template <typename T>
        void process(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst){
          // TODO: assert array elements > -1.
          // Checks are postponed to the Weighted Gaussian operator() function.
          dst = 0.;
          if( m_tmp.extent(0) != src.extent(0) || m_tmp.extent(1) != src.extent(1))
            m_tmp.resize(src.extent(0), src.extent(1) );
          for(size_t s=0; s<m_n_scales; ++s) {
            m_wgaussians[s].filter(src,m_tmp);
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
          // Check number of planes
          bob::core::array::assertSameDimensionLength(src.extent(0), dst.extent(0));

          for( int p=0; p<dst.extent(0); ++p) {
            const blitz::Array<T,2> src_slice =
              src( p, blitz::Range::all(), blitz::Range::all() );
            blitz::Array<double,2> dst_slice =
              dst( p, blitz::Range::all(), blitz::Range::all() );

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
        size_t m_size_min;
        size_t m_size_step;
        double m_sigma;
        bob::sp::Extrapolation::BorderType m_conv_border;

        boost::shared_array<bob::ip::base::WeightedGaussian> m_wgaussians;
        blitz::Array<double,2> m_tmp;
  };

} } } // namespaces

#endif /* BOB_IP_BASE_SELF_QUOTIENT_IMAGE_H */

