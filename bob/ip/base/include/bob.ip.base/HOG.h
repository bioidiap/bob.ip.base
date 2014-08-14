/**
 * @date Sun Apr 22 16:03:15 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @note: removed templates:
 * @date Wed Jul  9 14:30:18 CEST 2014
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 *
 * @brief Computes Histogram of Oriented Gradients (HOG) descriptors
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_CELL_BLOCK_DESCRIPTORS_H
#define BOB_IP_BASE_CELL_BLOCK_DESCRIPTORS_H

#include <bob.core/assert.h>
#include <bob.math/gradient.h>

#include <bob.ip.base/Block.h>

#include <boost/shared_ptr.hpp>

namespace bob { namespace ip { namespace base {

  /**
    * Vectorizes an array and multiply values by a constant factor
    */
  template <typename T>
  void _vectorizeMultArray(const blitz::Array<T,3> in,
    blitz::Array<T,1> out, const T factor=1)
  {
    int n_cells_y = in.extent(0);
    int n_cells_x = in.extent(1);
    int n_bins = in.extent(2);
    blitz::Range rall = blitz::Range::all();
    for(int cy=0; cy<n_cells_y; ++cy)
      for(int cx=0; cx<n_cells_x; ++cx)
      {
        blitz::Array<T,1> in_ = in(cy,cx,rall);
        blitz::Array<T,1> out_ = out(blitz::Range(
              (cy*n_cells_x+cx)*n_bins,(cy*n_cells_x+cx+1)*n_bins-1));
        out_ = in_ * factor;
      }
  }

  template <typename T>
  void _vectorizeMultArray(const blitz::Array<T,2> in,
    blitz::Array<T,1> out, const T factor=1)
  {
    int n_cells = in.extent(0);
    int n_bins = in.extent(1);
    blitz::Range rall = blitz::Range::all();
    for(int c=0; c<n_cells; ++c)
    {
      blitz::Array<T,1> in_ = in(c,rall);
      blitz::Array<T,1> out_ = out(blitz::Range(
            c*n_bins,(c+1)*n_bins-1));
      out_ = in_ * factor;
    }
  }

  template <typename T>
  void _vectorizeMultArray(const blitz::Array<T,1> in,
    blitz::Array<T,1> out, const T factor=1)
  {
    out = in * factor;
  }


  /**
    * @brief Norm used for normalizing the descriptor blocks
    * - L2: Euclidean norm
    * - L2Hys: L2 norm with clipping of high values
    * - L1: L1 norm (Manhattan distance)
    * - L1sqrt: Square root of the L1 norm
    * - Nonorm: no norm used
    * TODO: ZeroMean/UnitVariance normalization?
    */
  typedef enum {
    L2 = 0,
    L2Hys,
    L1,
    L1sqrt,
    Nonorm,
    BlockNorm_Count
  } BlockNorm;


  /**
    * @brief Function which normalizes a set of cells, and returns the
    *   corresponding 1D block descriptor.
    * @param descr The input descriptor (first two dimensions are for the
    *   spatial location of the cell, whereas the length of the last
    *   dimension corresponds to the dimensionality of the cell descriptor).
    * @param norm_descr The output 1D normalized block descriptor
    * @param block_norm The norm used by the procedure
    * @param eps The epsilon used for the block normalization
    *   (to avoid division by zero norm)
    * @param threshold The threshold used for the block normalization
    *   This is only used with the L2Hys norm, for the clipping of large
    *   values.
    */
  template <typename U, int D>
  void _normalizeBlock(
      const blitz::Array<U,D>& descr,
      blitz::Array<U,1>& norm_descr,
      const BlockNorm block_norm=L2,
      const double eps=1e-10,
      const double threshold=0.2
  ){
    // Checks input/output arrays
    int ndescr=1;
    for(int d=0; d<D; ++d) ndescr *= descr.extent(d);
    bob::core::array::assertSameDimensionLength(ndescr,    norm_descr.extent(0));

    // Use multiplication rather than inversion (should be faster)
    double sumInv;
    switch(block_norm)
    {
      case Nonorm:
        _vectorizeMultArray(descr, norm_descr);
        break;
      case L2Hys:
        // Normalizes to unit length (using L2)
        sumInv = 1. / sqrt(blitz::sum(blitz::pow2(blitz::abs(descr))) + eps*eps);
        _vectorizeMultArray(descr, norm_descr, sumInv);
        // Clips values above threshold
        norm_descr = blitz::where(blitz::abs(norm_descr) <= threshold, norm_descr, threshold);
        // Normalizes to unit length (using L2)
        sumInv = 1. / sqrt(blitz::sum(blitz::pow2(blitz::abs(norm_descr))) + eps*eps);
        norm_descr = norm_descr * sumInv;
        break;
      case L1:
        // Normalizes to unit length (using L1)
        sumInv = 1. / (blitz::sum(blitz::abs(descr)) + eps);
        _vectorizeMultArray(descr, norm_descr, sumInv);
        break;
      case L1sqrt:
        // Normalizes to unit length (using L1)
        sumInv = 1. / (blitz::sum(blitz::abs(descr)) + eps);
        _vectorizeMultArray(descr, norm_descr, sumInv);
        norm_descr = blitz::sqrt(norm_descr);
        break;
      case L2:
      default:
        // Normalizes to unit length (using L2)
        sumInv = 1. / sqrt(blitz::sum(blitz::pow2(blitz::abs(descr))) + eps*eps);
        _vectorizeMultArray(descr, norm_descr, sumInv);
        break;
    }
  }


  /**
    * @brief Abstract class to extract descriptors using a decomposition
    *   into cells (unormalized descriptors) and blocks (groups of cells
    *   used for normalization purpose)
    */
  class BlockCellDescriptors
  {
    public:
      /**
        * Constructor
        */
      BlockCellDescriptors(
        const size_t height,
        const size_t width,
        const size_t cell_dim=8,
        const size_t cell_y=4,
        const size_t cell_x=4,
        const size_t cell_ov_y=0,
        const size_t cell_ov_x=0,
        const size_t block_y=4,
        const size_t block_x=4,
        const size_t block_ov_y=0,
        const size_t block_ov_x=0
      );


      /**
        * @brief Copy constructor
        */
      BlockCellDescriptors(const BlockCellDescriptors& other);

      /**
        * Destructor
        */
      virtual ~BlockCellDescriptors() {}

      /**
        * @brief Assignment operator
        */
      BlockCellDescriptors& operator=(const BlockCellDescriptors& other);
      /**
        * @brief Equal to
        */
      bool operator==(const BlockCellDescriptors& b) const;

      /**
        * @brief Not equal to
        */
      bool operator!=(const BlockCellDescriptors& b) const { return !(this->operator==(b));}

      /**
        * Getters
        */
      size_t getHeight() const { return m_height; }
      size_t getWidth() const { return m_width; }
      size_t getCellDim() const { return m_cell_dim; }
      size_t getCellHeight() const { return m_cell_y; }
      size_t getCellWidth() const { return m_cell_x; }
      size_t getCellOverlapHeight() const { return m_cell_ov_y; }
      size_t getCellOverlapWidth() const { return m_cell_ov_x; }
      size_t getBlockHeight() const { return m_block_y; }
      size_t getBlockWidth() const { return m_block_x; }
      size_t getBlockOverlapHeight() const { return m_block_ov_y; }
      size_t getBlockOverlapWidth() const { return m_block_ov_x; }
      BlockNorm getBlockNorm() const { return m_block_norm; }
      double getBlockNormEps() const { return m_block_norm_eps; }
      double getBlockNormThreshold() const { return m_block_norm_threshold; }
      /**
        * Setters
        */
      void setHeight(const size_t height) { m_height = height; resizeCache(); }
      void setWidth(const size_t width) { m_width = width; resizeCache(); }
      void setSize(const size_t height, const size_t width){m_height = height; m_width = width; resizeCache();}
      void setCellDim(const size_t cell_dim) { m_cell_dim = cell_dim; resizeCellCache(); }
      void setCellHeight(const size_t cell_y) { m_cell_y = cell_y; resizeCellCache(); }
      void setCellWidth(const size_t cell_x) { m_cell_x = cell_x; resizeCellCache(); }
      void setCellSize(const size_t height, const size_t width){m_cell_y = height; m_cell_x = width; resizeCellCache();}
      void setCellOverlapHeight(const size_t cell_ov_y) { m_cell_ov_y = cell_ov_y; resizeCellCache(); }
      void setCellOverlapWidth(const size_t cell_ov_x) { m_cell_ov_x = cell_ov_x; resizeCellCache(); }
      void setCellOverlap(const size_t height, const size_t width){m_cell_ov_y = height; m_cell_ov_x = width; resizeCellCache();}
      void setBlockHeight(const size_t block_y) { m_block_y = block_y; resizeBlockCache(); }
      void setBlockWidth(const size_t block_x) { m_block_x = block_x; resizeBlockCache(); }
      void setBlockSize(const size_t height, const size_t width){m_block_y = height; m_block_x = width; resizeBlockCache();}
      void setBlockOverlapHeight(const size_t block_ov_y) { m_block_ov_y = block_ov_y; resizeBlockCache(); }
      void setBlockOverlapWidth(const size_t block_ov_x) { m_block_ov_x = block_ov_x; resizeBlockCache(); }
      void setBlockOverlap(const size_t height, const size_t width){m_block_ov_y = height; m_block_ov_x = width; resizeBlockCache();}
      void setBlockNorm(const BlockNorm block_norm) { m_block_norm = block_norm; }
      void setBlockNormEps(const double block_norm_eps) { m_block_norm_eps = block_norm_eps; }
      void setBlockNormThreshold(const double block_norm_threshold) { m_block_norm_threshold = block_norm_threshold; }

      /**
        * Disable block normalization. This is performed by setting
        * parameters such that the cells are not further processed, that is
        * block_y=1, block_x=1, block_ov_y=0, block_ov_x=0, and
        * block_norm=Nonorm.
        */
      void disableBlockNormalization();

      /**
        * Gets the descriptor output size given the current parameters and
        * size. (number of blocks along Y x number of block along X x number
        *       of bins)
        */
      const blitz::TinyVector<int,3> getOutputShape() const { return blitz::TinyVector<int,3>(m_nb_blocks_y, m_nb_blocks_x, m_block_y * m_block_x * m_cell_dim); }

      /**
        * Normalizes all the blocks, given the current state of the cell
        * descriptors
        */
      virtual void normalizeBlocks(blitz::Array<double,3>& output);

    protected:
      // Methods to resize arrays in cache
      virtual void resizeCache() { resizeCellCache(); }
      virtual void resizeCellCache();
      virtual void resizeBlockCache();

      // Input size
      size_t m_height;
      size_t m_width;
      // Cell-related variables
      size_t m_cell_dim;
      size_t m_cell_y;
      size_t m_cell_x;
      size_t m_cell_ov_y;
      size_t m_cell_ov_x;
      // Block-related variables (normalization)
      bool m_block_normalization;
      size_t m_block_y;
      size_t m_block_x;
      size_t m_block_ov_y;
      size_t m_block_ov_x;
      BlockNorm m_block_norm;
      double m_block_norm_eps;
      double m_block_norm_threshold;

      // Cache
      // Number of blocks along Y- and X- axes
      size_t m_nb_cells_y;
      size_t m_nb_cells_x;
      size_t m_nb_blocks_y;
      size_t m_nb_blocks_x;

      // Non-normalized descriptors computed at the cell level
      blitz::Array<double,3> m_cell_descriptor;
  };



  /**
    * Gradient 'magnitude' used
    * - Magnitude: L2 magnitude over X and Y
    * - MagnitudeSquare: Square of the L2 magnitude
    * - SqrtMagnitude: Square root of the L2 magnitude
    */
  typedef enum {
      Magnitude = 0,
      MagnitudeSquare,
      SqrtMagnitude,
      MagnitudeType_Count
  } GradientMagnitudeType;

  /**
    * @brief Class to extract gradient magnitude and orientation maps
    */
  class GradientMaps
  {
    public:
      /**
        * Constructor
        */
      GradientMaps(
          const size_t height,
          const size_t width,
          const GradientMagnitudeType mag_type=Magnitude
      );

      /**
        * Copy constructor
        */
      GradientMaps(const GradientMaps& other);
      /**
        * Destructor
        */
      virtual ~GradientMaps() {}

      /**
       * @brief Assignment operator
       */
      GradientMaps& operator=(const GradientMaps& other);
      /**
       * @brief Equal to
       */
      bool operator==(const GradientMaps& b) const;
      /**
       * @brief Not equal to
       */
      bool operator!=(const GradientMaps& b) const;

      /**
        * Sets the height
        */
      void setHeight(const size_t height);
      /**
        * Sets the width
        */
      void setWidth(const size_t width);
      /**
        * Resizes the cache
        */
      void setSize(const size_t height, const size_t width);
      /**
        * Sets the magnitude type to use
        */
      void setGradientMagnitudeType(const GradientMagnitudeType mag_type){ m_mag_type = mag_type; }
      /**
        * Returns the current height
        */
      size_t getHeight() const { return m_gy.extent(0); }
      /**
        * Returns the current width
        */
      size_t getWidth() const { return m_gy.extent(1); }
      /**
        * Returns the magnitude type used
        */
      GradientMagnitudeType getGradientMagnitudeType() const { return m_mag_type; }

      /**
        * Processes an input array
        */
      template <typename T>
      void process(
        const blitz::Array<T,2>& input,
        blitz::Array<double,2>& magnitude,
        blitz::Array<double,2>& orientation
      ){
        // Checks input/output arrays
        bob::core::array::assertSameShape(input, m_gy);
        bob::core::array::assertSameShape(magnitude, m_gy);
        bob::core::array::assertSameShape(orientation, m_gy);

        // Computes the gradient
        bob::math::gradient<T,double>(input, m_gy, m_gx);

        // Computes the magnitude map
        switch(m_mag_type)
        {
          case MagnitudeSquare:
            magnitude = blitz::pow2(m_gy) + blitz::pow2(m_gx);
            break;
          case SqrtMagnitude:
            magnitude = blitz::sqrt(blitz::sqrt(blitz::pow2(m_gy) + blitz::pow2(m_gx)));
            break;
          case Magnitude:
            magnitude = blitz::sqrt(blitz::pow2(m_gy) + blitz::pow2(m_gx));
            break;
          case MagnitudeType_Count:
            break;
        }
        // Computes the orientation map (range: [-PI,PI])
        orientation = blitz::atan2(m_gy, m_gx);
      }

    private:
      blitz::Array<double,2> m_gy;
      blitz::Array<double,2> m_gx;
      GradientMagnitudeType m_mag_type;
  };



  /**
    * @brief Abstract class to extract Gradient-based descriptors using a
    *   decomposition into cells (unormalized descriptors) and blocks
    *   (groups of cells used for normalization purpose)
    */
  class BlockCellGradientDescriptors: public BlockCellDescriptors
  {
    public:
      /**
        * Constructor
        */
      BlockCellGradientDescriptors(
        const size_t height,
        const size_t width,
        const size_t cell_dim=8,
        const size_t cell_y=4,
        const size_t cell_x=4,
        const size_t cell_ov_y=0,
        const size_t cell_ov_x=0,
        const size_t block_y=4,
        const size_t block_x=4,
        const size_t block_ov_y=0,
        const size_t block_ov_x=0
      );

      /**
        * Copy constructor
        */
      BlockCellGradientDescriptors(const BlockCellGradientDescriptors& b);

      /**
        * Destructor
        */
      virtual ~BlockCellGradientDescriptors() {}

      /**
       * @brief Assignment operator
       */
      BlockCellGradientDescriptors& operator=(const BlockCellGradientDescriptors& other);
      /**
       * @brief Equal to
       */
      bool operator==(const BlockCellGradientDescriptors& b) const;
      /**
       * @brief Not equal to
       */
      bool operator!=(const BlockCellGradientDescriptors& b) const;

      /**
        * Getters
        */
      GradientMagnitudeType getGradientMagnitudeType() const { return m_gradient_maps->getGradientMagnitudeType(); }
      /**
        * Setters
        */
      void setGradientMagnitudeType(const GradientMagnitudeType m) { m_gradient_maps->setGradientMagnitudeType(m); }

    protected:
      /**
        * Computes the gradient maps, and their decomposition into cells
        */
      template <typename T>
      void computeGradientMaps(const blitz::Array<T,2>& input){
        // Computes the Gradients maps (magnitude and orientation)
        m_gradient_maps->process(input, m_magnitude, m_orientation);

        // Performs the block decomposition on the Gradients maps
        bob::ip::base::block(m_magnitude, m_cell_magnitude, m_cell_y, m_cell_x, m_cell_ov_y, m_cell_ov_x);
        bob::ip::base::block(m_orientation, m_cell_orientation, m_cell_y, m_cell_x, m_cell_ov_y, m_cell_ov_x);
      }

      // Methods to resize arrays in cache
      virtual void resizeCache();
      virtual void resizeCellCache();

      // Gradient related
      boost::shared_ptr<GradientMaps> m_gradient_maps;
      // Gradient maps for magnitude and orientation
      blitz::Array<double,2> m_magnitude;
      blitz::Array<double,2> m_orientation;
      // Gradient maps decomposed into blocks
      blitz::Array<double,4> m_cell_magnitude;
      blitz::Array<double,4> m_cell_orientation;
  };



  /**
    * @brief Class to extract Histogram of Gradients (HOG) descriptors
    * This implementation relies on the following article,
    * "Histograms of Oriented Gradients for Human Detection",
    * N. Dalal, B. Triggs, in proceedings of the IEEE Conf. on Computer
    * Vision and Pattern Recognition, 2005.
    * Few remarks:
    *  1) Only single channel inputs (a.k.a. grayscale) are considered.
    *     Therefore, it does not take the maximum gradient over several
    *     channels as proposed in the above article.
    *  2) Gamma/Colour normalization is not part of the descriptor
    *     computation. However, this can easily be done (using this library)
    *     before extracting the descriptors.
    *  3) Gradients are computed using standard 1D centered gradient (except
    *     at the borders where the gradient is uncentered [-1 1]). This
    *     is the method which achieved best performance reported in the
    *     article.
    *     To avoid too many uncentered gradients to be used, the gradients
    *     are computed on the full image prior to the cell decomposition.
    *     This implies that extra-pixels at each boundary of the cell are
    *     contributing to the gradients, although these pixels are not
    *     located inside the cell.
    *  4) R-HOG blocks (rectangular) normalisation is supported, but
    *     not C-HOG blocks (circular).
    *  5) Due to the similarity with the SIFT descriptors, this can also be
    *     used to extract dense-SIFT features.
    *  6) The first bin of each histogram is always centered around 0. This
    *     implies that the 'orientations are in [0-e,180-e]' rather than
    *     [0,180], e being half the angle size of a bin (same with [0,360]).
    */
  class HOG: public BlockCellGradientDescriptors
  {
    public:
      /**
        * Constructor
        */
      HOG(
          const size_t height,
          const size_t width,
          const size_t cell_dim=8,
          const bool full_orientation=false,
          const size_t cell_y=4,
          const size_t cell_x=4,
          const size_t cell_ov_y=0,
          const size_t cell_ov_x=0,
          const size_t block_y=4,
          const size_t block_x=4,
          const size_t block_ov_y=0,
          const size_t block_ov_x=0
      );

      /**
        * Copy constructor
        */
      HOG(const HOG& other);

      /**
        * Destructor
        */
      virtual ~HOG() {}

      /**
        * @brief Assignment operator
        */
      HOG& operator=(const HOG& other);

      /**
        * @brief Equal to
        */
      bool operator==(const HOG& b) const;
      /**
        * @brief Not equal to
        */
      bool operator!=(const HOG& b) const;

      /**
        * Getters
        */
      bool getFullOrientation() const { return m_full_orientation; }
      /**
        * Setters
        */
      void setFullOrientation(const bool full_orientation) { m_full_orientation = full_orientation; }

      /**
        * @brief Function which computes an Histogram of Gradients for
        *   a given 'cell'. The inputs are the gradient magnitudes and the
        *   orientations for each pixel of the cell.
        *   The number of bins is given by the dimension of the output array.
        * @param mag The input blitz array with the gradient magnitudes
        * @param ori The input blitz array with the orientations
        * @param hist The output blitz array which will contain the histogram
        */
      void computeHistogram(
        const blitz::Array<double,2>& mag,
        const blitz::Array<double,2>& ori,
        blitz::Array<double,1>& hist
      ) const;

      /**
        * Processes an input array. This extracts HOG descriptors from the
        * input image. The output is 3D, the first two dimensions being the
        * y- and x- indices of the block, and the last one the index of the
        * bin (among the concatenated cell histograms for this block).
        */
      template <typename T>
      void extract(const blitz::Array<T,2>& input, blitz::Array<double,3>& output){
        // Checks input/output arrays
        const blitz::TinyVector<int,3> r = getOutputShape();
        bob::core::array::assertSameShape(output, r);

        computeGradientMaps(input);

        // Computes the histograms for each cell
        m_cell_descriptor = 0.;
        blitz::Range rall = blitz::Range::all();
        for(size_t cy=0; cy<m_nb_cells_y; ++cy)
          for(size_t cx=0; cx<m_nb_cells_x; ++cx){
            blitz::Array<double,1> hist = m_cell_descriptor(cy,cx,rall);
            blitz::Array<double,2> mag = m_cell_magnitude(cy,cx,rall,rall);
            blitz::Array<double,2> ori = m_cell_orientation(cy,cx,rall,rall);
            computeHistogram(mag, ori, hist);
        }

        normalizeBlocks(output);
      }

    protected:
      bool m_full_orientation;
  };

} } } // namespaces

#endif /* BOB_IP_BASE_BLOCK_CELL_DESCRIPTORS_H */

