/**
 * @date Tue Jan 22 12:31:59 CET 2013
 * @author Ivana Chingovska <ivana.chingovska@idiap.ch>
 *
 * This file defines a function to compute the Grey Level Co-occurence Matrix (GLCM)
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_GLCM_H
#define BOB_IP_BASE_GLCM_H

#include <math.h>
#include <iostream>
#include <blitz/array.h>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <bob.core/assert.h>
#include <bob.core/array_copy.h>
#include <bob.core/cast.h>
#include <bob.sp/Quantization.h>

namespace bob { namespace ip { namespace base {

  /**
   * @brief This class allows to extract Grey-Level Co-occurence Matrix (GLCM). For more information, please refer to the
   * following article: "Textural Features for Image Classification", from R. M. Haralick, K. Shanmugam, I. Dinstein
   * in the IEEE Transactions on Systems, Man and Cybernetics, vol.SMC-3, No. 6, p. 610-621.
   *
   * A thorough tutorial about GLCM and the textural (so-called Haralick) properties that can be derived from it, can be found at:
   * http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
   *
   * List of references:
   * [1] R. M. Haralick, K. Shanmugam, I. Dinstein; "Textural Features for Image Classification",
   * in IEEE Transactions on Systems, Man and Cybernetics, vol.SMC-3, No. 6, p. 610-621.
   * [2] http://www.mathworks.ch/ch/help/images/ref/graycomatrix.html
   */
  template <typename T>
  class GLCM {

    public: //api

      /**
       * @brief Complete constructor
       */
      GLCM()
      :
        m_offset(1,2),
        m_symmetric(false),
        m_normalized(false),
        m_quantization()
      {
        m_offset = 1, 0; // this is the default offset
      }

      GLCM(const int num_levels)
      :
        m_offset(1,2),
        m_symmetric(false),
        m_normalized(false),
        m_quantization(bob::sp::quantization::UNIFORM, num_levels)
      {
        m_offset = 1, 0; // this is the default offset
      }

      GLCM(const int num_levels, const T min_level, const T max_level)
      :
        m_offset(1,2),
        m_symmetric(false),
        m_normalized(false),
        m_quantization(bob::sp::quantization::UNIFORM, num_levels, min_level, max_level)
      {
        m_offset = 1, 0; // this is the default offset
      }

      GLCM(const blitz::Array<T,1>& quant_thres)
      :
        m_offset(1,2),
        m_symmetric(false),
        m_normalized(false),
        m_quantization(quant_thres)
      {
        m_offset = 1, 0; // this is the default offset
      }

      /**
       * @brief Copy constructor
       */
      GLCM(const GLCM& other)
      :
        m_offset(bob::core::array::ccopy(other.m_offset)),
        m_symmetric(other.m_symmetric),
        m_normalized(other.m_normalized),
        m_quantization(other.m_quantization)
      {}

      /**
       * @brief Destructor
       */
      virtual ~GLCM() {}

      /**
       * @brief Assignment
       */
      GLCM& operator= (const GLCM& other){
        if(this != &other)
        {
          m_offset.reference(bob::core::array::ccopy(other.m_offset));
          m_symmetric = other.m_symmetric;
          m_normalized = other.m_normalized;
          m_quantization = other.m_quantization;
        }
        return *this;
      }

      /**
       * @brief Comparison
       */
      bool operator== (const GLCM& other){
        return
            bob::core::array::isEqual(m_offset, other.m_offset) &&
            m_symmetric == other.m_symmetric &&
            m_normalized == other.m_normalized;
// TODO:           && m_quantization == other.m_quantization;
      }

      /**
       * @brief Get the required shape of the GLCM output blitz array, before calling
       * the operator() method.
       */
      const blitz::TinyVector<int,3> getGLCMShape() const { return blitz::TinyVector<int,3>(m_quantization.getNumLevels(), m_quantization.getNumLevels(), m_offset.extent(0)); }

      /**
       * @brief Compute Gray-Level Co-occurences from a 2D blitz::Array, and save the resulting
       * GLCM matrix in the dst 3D blitz::Array.
       */
      void extract(const blitz::Array<T,2>& src, blitz::Array<double,3>& glcm) const {
        // check if the size of the output matrix is as expected
        blitz::TinyVector<int,3> shape(getGLCMShape());
        bob::core::array::assertSameShape(glcm, shape);

        glcm=0;
        blitz::Array<uint32_t,2> src_quant = m_quantization(src);
        for(int off_ind = 0; off_ind < m_offset.extent(0); ++off_ind) // loop over all the possible offsets
          // loop over each pixel of the image
          for(int y = 0; y < src_quant.extent(0); ++y)
            for(int x = 0; x < src_quant.extent(1); ++x){
              int i_level = (int)(src_quant(y,x)); // the grey level of the current pixel
              const int y1 = y + m_offset(off_ind, 1);
              const int x1 = x + m_offset(off_ind, 0);

              if(y1 >= 0 && y1 < src_quant.extent(0) && x1 >= 0 && x1 < src_quant.extent(1)){
                int j_level = (int)(src_quant(y1, x1));
                glcm(i_level, j_level, off_ind) += 1;
              }
        }

        if(m_symmetric){ // make the matrix symmetric
          blitz::Array<double,3> temp(glcm.copy());
          temp.transposeSelf(1,0,2);
          glcm += temp;
        }

        if (m_normalized){ // normalize the output image
          blitz::firstIndex i;
          blitz::secondIndex j;
          blitz::thirdIndex k;
          blitz::Array<double, 2> summations_temp(blitz::sum(glcm(i, k, j), k));
          blitz::Array<double, 1> summations(blitz::sum(summations_temp(j,i), j));
          glcm /= summations(k);
        }
      } // void process

      /**
       * @brief Accessors
       */

      const blitz::Array<int32_t,2>&  getOffset() const { return m_offset; }
      const int getMaxLevel() const { return m_quantization.getMaxLevel(); }
      const int getMinLevel() const { return m_quantization.getMinLevel(); }
      const int getNumLevels() const { return m_quantization.getNumLevels(); }
      const bool getSymmetric() const { return m_symmetric; }
      const bool getNormalized() const { return m_normalized; }
      const bob::sp::Quantization<T> getQuantization() const { return m_quantization; }
      const blitz::Array<T,1>&  getQuantizationTable() const{ return m_quantization.getThresholds(); }

      /**
       * @brief Mutators
       */

      void setOffset(const blitz::Array<int32_t, 2>& offset) { m_offset.reference(bob::core::array::ccopy(offset)); }
      void setSymmetric(const bool symmetric) { m_symmetric = symmetric; }
      void setNormalized(const bool normalized) { m_normalized = normalized; }

    protected:
      /**
       * @brief Attributes
       */
      blitz::Array<int32_t,2> m_offset;
      bool m_symmetric;
      bool m_normalized;
      bob::sp::Quantization<T> m_quantization;
   };


  /**
   * This class contains a number of texture properties of the Grey-Level Co-occurence Matrix (GLCM). The texture properties are selected from several publications:
   *
   * [1] R. M. Haralick, K. Shanmugam, I. Dinstein; "Textural Features for Image calssification",
   * in IEEE Transactions on Systems, Man and Cybernetics, vol.SMC-3, No. 6, p. 610-621.
   * [2] L. Soh and C. Tsatsoulis; Texture Analysis of SAR Sea Ice Imagery Using Gray Level Co-Occurrence Matrices, IEEE Transactions on Geoscience and Remote Sensing, vol. 37, no. 2, March 1999.
   * [3] D A. Clausi, An analysis of co-occurrence texture statistics as a function of grey level quantization, Can. J. Remote Sensing, vol. 28, no.1, pp. 45-62, 2002
   * [4] http://murphylab.web.cmu.edu/publications/boland/boland_node26.html
   * [5] http://www.mathworks.com/matlabcentral/fileexchange/22354-glcmfeatures4-m-vectorized-version-of-glcmfeatures1-m-with-code-changes
   * [6] http://www.mathworks.ch/ch/help/images/ref/graycoprops.html
   */
  class GLCMProp {

    public: //api

      /**
       * @brief Complete constructor
       */

      GLCMProp();

      /**
       * @brief Copy constructor
       */
      GLCMProp(const GLCMProp& other) {}

      /**
       * @brief Destructor
       */
      virtual ~GLCMProp();

      /**
       * @brief Assignment
       */
      GLCMProp& operator= (const GLCMProp& other) { return *this; }

      /**
      * @brief Get the shape of the output array for the property
      */
      const blitz::TinyVector<int,1> get_prop_shape(const blitz::Array<double,3>& glcm) const;

      /**
       * @brief Compute each of the single GLCM properties from a 3D blitz::Array which is the GLCM matrix
       *
       * The following method provides texture properties of the GLCM matrix. Here is a list of all the implemented features.
       * f1. angular second moment [1] / energy [6]
       * f2. energy [4]
       * f3. sum of squares (variance) [1]
       * f4. contrast [1] == contrast [6]
       * f5. correlation [1]
       * f6. inverse difference moment [1] = homogeneity [2], homop[5]
       * f7. sum average [1]
       * f8. sum variance [1]
       * f9. sum entropy [1]
       * f10. entropy [1]
       * f11. difference variance [4]
       * f12. difference entropy [1]
       * f13. dissimilarity [4]
       * f14. homogeneity [6]
       * f15. cluster prominence [2]
       * f16. cluster shade [2]
       * f17. maximum probability [2]
       * f18. information measure of correlation 1 [1]
       * f19. information measure of correlation 2 [1]
       * f20. inverse difference (INV) is homom [3]
       * f21. inverse difference normalized (INN) [3]
       * f22. inverse difference moment normalized [3]
       * f23. auto-correlation [2]
       * f24. correlation as in MATLAB Image Processing Toolbox method graycoprops() ([6])
       */

      void angular_second_moment(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void energy(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void variance(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void contrast(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void auto_correlation(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void correlation(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void correlation_m(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void inv_diff_mom(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void sum_avg(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void sum_var(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void sum_entropy(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void entropy(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void diff_var(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void diff_entropy(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void dissimilarity(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void homogeneity(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void cluster_prom(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void cluster_shade(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void max_prob(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void inf_meas_corr1(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void inf_meas_corr2(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void inv_diff(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void inv_diff_norm(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;
      void inv_diff_mom_norm(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const;

    protected:
      /**
       * @brief Normalizes the glcm matrix (by offset. The separate matrix for each offset is separately normalized))
       */
      const blitz::Array<double,3> normalize_glcm(const blitz::Array<double,3>& glcm) const;
  };

} } }

#endif /* BOB_IP_BASE_GLCM_H */


