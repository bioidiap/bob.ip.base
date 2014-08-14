/**
 * @date Thu Apr 7 19:52:29 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_DCT_FEATURES_H
#define BOB_IP_BASE_DCT_FEATURES_H

#include <bob.core/cast.h>
#include <bob.core/array_copy.h>
#include <bob.sp/DCT2D.h>
#include <list>
#include <limits>

#include <bob.ip.base/Block.h>

namespace bob { namespace ip { namespace base {

  /**
    * @brief This class can be used to extract DCT features. This algorithm
    *   is described in the following article:
    *   "Polynomial Features for Robust Face Authentication",
    *   from C. Sanderson and K. Paliwal, in the proceedings of the
    *   IEEE International Conference on Image Processing 2002.
    *   In addition, it support pre- and post-normalization (zero mean and
    *   unit variance, at the block level, or DCT coefficient level)
    */
  class DCTFeatures
  {
    public:

      /**
        * @brief Constructor: generates a DCTFeatures extractor
        * @param block_h height of the blocks
        * @param block_w width of the blocks
        * @param overlap_h overlap of the blocks along the y-axis
        * @param overlap_w overlap of the blocks along the x-axis
        * @param n_dct_coefs number of DCT coefficients to keep. The first
        *   coefficient will be removed if norm_block is enabled. In this case,
        *   (n_dct_coefs-1) coefficients will be returned. The number
        *   of coefficients should be a square integer if square_pattern is
        *   enabled.
        * @param norm_block Normalize the block to zero mean and
        *   unit variance before the DCT extraction. If the variance is zero
        *   (i.e. all the block values are the same), this will set all the
        *   block values to zero before the DCT extraction. If norm_block is
        *   set, the first DCT coefficient will always be zero and hence will
        *   not be returned.
        * @param norm_dct Normalize the DCT coefficients (across blocks) to
        *   zero mean and unit variance after the DCT extraction. If the
        *   variance is zero (i.e. a specific DCT coefficient is the same for
        *   all the blocks), this will set these coefficients to a zero
        *   constant.
        * @param square_pattern Tells whether a zigzag pattern or a square
        *   pattern is used when retaining the DCT coefficients. When enabled,
        *   the number of DCT coefficients should be a square integer.
        */
      DCTFeatures(
        const size_t n_dct_coefs,
        const size_t block_h, const size_t block_w,
        const size_t overlap_h, const size_t overlap_w,
        const bool norm_block=false,
        const bool norm_dct=false,
        const bool square_pattern=false
      );

      /**
        * @brief Copy constructor
        */
      DCTFeatures(const DCTFeatures& other);

      /**
        * @brief Destructor
        */
      virtual ~DCTFeatures() { }

      /**
        * @brief Assignment operator
        */
      DCTFeatures& operator=(const DCTFeatures& other);

      /**
        * @brief Equal to
        */
      bool operator==(const DCTFeatures& b) const;
      /**
        * @brief Not equal to
        */
      bool operator!=(const DCTFeatures& b) const;

      /**
        * @brief Getters
        */
      size_t getBlockH() const { return m_block_h; }
      size_t getBlockW() const { return m_block_w; }
      blitz::TinyVector<int,2> getBlockSize() const {return blitz::TinyVector<int,2>(m_block_h, m_block_w);}
      size_t getOverlapH() const { return m_overlap_h; }
      size_t getOverlapW() const { return m_overlap_w; }
      blitz::TinyVector<int,2> getBlockOverlap() const {return blitz::TinyVector<int,2>(m_overlap_h, m_overlap_w);}
      size_t getNDctCoefs() const { return m_n_dct_coefs; }
      bool getNormalizeBlock() const { return m_norm_block; }
      bool getNormalizeDct() const { return m_norm_dct; }
      bool getSquarePattern() const { return m_square_pattern; }
      double getNormEpsilon() const { return m_norm_epsilon; }

      /**
        * @brief Setters
        */
      void setBlockH(const size_t block_h) { m_block_h = block_h; m_dct2d.setHeight(block_h); resetCacheBlock(); }
      void setBlockW(const size_t block_w) { m_block_w = block_w; m_dct2d.setWidth(block_w); resetCacheBlock(); }
      void setBlockSize(const blitz::TinyVector<int,2>& size) {m_block_h = size[0]; m_block_w = size[1]; m_dct2d.setHeight(m_block_h); m_dct2d.setWidth(m_block_w); resetCacheBlock();}
      void setOverlapH(const size_t overlap_h) { m_overlap_h = overlap_h; }
      void setOverlapW(const size_t overlap_w) { m_overlap_w = overlap_w; }
      void setBlockOverlap(const blitz::TinyVector<int,2>& overlap) {m_overlap_h = overlap[0]; m_overlap_w = overlap[1];}
      void setNDctCoefs(const size_t n_dct_coefs) { m_n_dct_coefs = n_dct_coefs; setCheckSqrtNDctCoefs(); resetCacheDct(); }
      void setNormalizeBlock(const bool norm_block) { m_norm_block = norm_block; resetCacheDct(); }
      void setNormalizeDct(const bool norm_dct) { m_norm_dct = norm_dct; }
      void setSquarePattern(const bool square_pattern) { m_square_pattern = square_pattern; setCheckSqrtNDctCoefs(); }
      void setNormEpsilon(const double norm_epsilon) { m_norm_epsilon = norm_epsilon; }

      /**
        * @brief Process a 2D blitz Array/Image by extracting DCT features.
        * @param src The 2D input blitz array
        * @param dst The 2D output array. The first dimension is for the block
        *   index, whereas the second one is for the dct index. The number of
        *   expected blocks can be obtained using the get2DOutputShape() method.
        */
      template <typename T, int D>
      void extract(const blitz::Array<T,2>& src, blitz::Array<double,D>& dst) const{
        // Casts the input to double
        blitz::Array<double,2> src_d = bob::core::array::cast<double>(src);
        // Calls the specialized template function for double
        this->extract_(src_d, dst);
      }

      // specialized method for double
      template <int D>
      void extract(const blitz::Array<double,2>& src, blitz::Array<double,D>& dst) const{
        //
        this->extract_(src, dst);
      }

      /**
        * @brief Process a 2D blitz Array/Image by extracting DCT features.
        * @param src The 2D input blitz array
        * @param dst The 2D output array. The first dimension is for the block
        *   index, whereas the second one is for the dct index. The number of
        *   expected blocks can be obtained using the get2DOutputShape() method.
        */
      void extract_(const blitz::Array<double,2>& src, blitz::Array<double,2>& dst) const;

      /**
        * @brief Process a 2D blitz Array/Image by extracting DCT features.
        * @param src The 2D input blitz array
        * @param dst The 3D output array. The first two dimensions are for the
        *   block indices, whereas the second one is for the dct index. The
        *   number of expected blocks can be obtained using the
        *   get3DOutputShape() method.
        */
      void extract_(const blitz::Array<double,2>& src, blitz::Array<double,3>& dst) const;

      /**
        * @brief Function which returns the expected shape of the output
        *   array when applying the DCTFeatures extractor on a 2D
        *   blitz::array/image. The first dimension is the height (y-axis),
        *   whereas the second one is the width (x-axis).
        * @param src The input blitz array
        */
      const blitz::TinyVector<int,2> get2DOutputShape(const blitz::TinyVector<int, 2>& shape) const{
          const blitz::TinyVector<int,3> res = getBlock3DOutputShape(shape[0], shape[1], m_block_h, m_block_w, m_overlap_h, m_overlap_w);
          return blitz::TinyVector<int,2>(res(0), m_n_dct_coefs - (m_norm_block?1:0));
      }

      /**
        * @brief Function which returns the expected shape of the output
        *   array when applying the DCTFeatures extractor on a 2D
        *   blitz::array/image. The first dimension is the height (y-axis),
        *   whereas the second one is the width (x-axis).
        * @param src The input blitz array
        */
      const blitz::TinyVector<int,3> get3DOutputShape(const blitz::TinyVector<int, 2>& shape) const{
        const blitz::TinyVector<int,4> res = getBlock4DOutputShape(shape[0], shape[1], m_block_h, m_block_w, m_overlap_h, m_overlap_w);
        return blitz::TinyVector<int,3> (res(0), res(1), m_n_dct_coefs - (m_norm_block?1:0));
      }

    private:

      /**
        * Attributes
        */
      bob::sp::DCT2D m_dct2d;
      size_t m_block_h;
      size_t m_block_w;
      size_t m_overlap_h;
      size_t m_overlap_w;
      size_t m_n_dct_coefs;
      size_t m_sqrt_n_dct_coefs;
      bool m_norm_block;
      bool m_norm_dct;
      bool m_square_pattern;
      double m_norm_epsilon;

      void setCheckSqrtNDctCoefs();
      void normalizeBlock(const blitz::Array<double,2>& src) const;
      void extractRowDCTCoefs(blitz::Array<double,1>& coefs) const;

      /**
        * Working arrays/variables in cache
        */
      void resetCache() const;
      void resetCacheBlock() const;
      void resetCacheDct() const;

      mutable blitz::Array<double,2> m_cache_block1;
      mutable blitz::Array<double,2> m_cache_block2;
      mutable blitz::Array<double,1> m_cache_dct_full;
      mutable blitz::Array<double,1> m_cache_dct1;
      mutable blitz::Array<double,1> m_cache_dct2;
  };

} } } // namespaces

#endif /* BOB_IP_BASE_DCT_FEATURES_H */

