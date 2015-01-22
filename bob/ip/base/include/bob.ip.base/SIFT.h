/**
 * @date Sun Sep 9 19:21:00 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_SIFT_H
#define BOB_IP_BASE_SIFT_H

#include <blitz/array.h>
// TODO: import into bob.ip.base
#include <bob.sp/conv.h>
#include <boost/shared_ptr.hpp>
#include <vector>

#include <bob.ip.base/GaussianScaleSpace.h>
#include <bob.ip.base/HOG.h>

#if HAVE_VLFEAT
#include <vl/generic.h>
#include <vl/sift.h>
#include <vl/dsift.h>
#endif // HAVE_VLFEAT

namespace bob { namespace ip { namespace base {

  /**
   * @brief This class can be used to extract SIFT descriptors
   * Reference: "Distinctive Image Features from Scale-Invariant Keypoints",
   * D. Lowe, International Journal of Computer Vision, 2004.
   */
  class SIFT
  {
    public:
      /**
       * @brief Constructor: generates a SIFT extractor
       */
      SIFT(
          const size_t height,
          const size_t width,
          const size_t n_intervals,
          const size_t n_octaves,
          const int octave_min,
          const double sigma_n=0.5,
          const double sigma0=1.6,
          const double contrast_thres=0.03,
          const double edge_thres=10.,
          const double norm_thres=0.2,
          const double kernel_radius_factor=4.,
          const bob::sp::Extrapolation::BorderType border_type = bob::sp::Extrapolation::Mirror
      );

      /**
       * @brief Copy constructor
       */
      SIFT(const SIFT& other);

      /**
       * @brief Destructor
       */
      virtual ~SIFT();

      /**
       * @brief Assignment operator
       */
      SIFT& operator=(const SIFT& other);

      /**
       * @brief Equal to
       */
      bool operator==(const SIFT& b) const;
      /**
       * @brief Not equal to
       */
      bool operator!=(const SIFT& b) const;

      /**
       * @brief Getters
       */
      size_t getHeight() const { return m_gss->getHeight(); }
      size_t getWidth() const { return m_gss->getWidth(); }
      size_t getNOctaves() const { return m_gss->getNOctaves(); }
      size_t getNIntervals() const { return m_gss->getNIntervals(); }
      int getOctaveMin() const { return m_gss->getOctaveMin(); }
      int getOctaveMax() const { return m_gss->getOctaveMax(); }
      double getSigmaN() const { return m_gss->getSigmaN(); }
      double getSigma0() const { return m_gss->getSigma0(); }
      double getKernelRadiusFactor() const { return m_gss->getKernelRadiusFactor(); }
      bob::sp::Extrapolation::BorderType getConvBorder() const { return m_gss->getConvBorder(); }
      double getContrastThreshold() const { return m_contrast_thres; }
      double getEdgeThreshold() const { return m_edge_thres; }
      double getNormThreshold() const { return m_norm_thres; }
      size_t getNBlocks() const { return m_descr_n_blocks; }
      size_t getNBins() const { return m_descr_n_bins; }
      double getGaussianWindowSize() const { return m_descr_gaussian_window_size; }
      double getMagnif() const { return m_descr_magnif; }
      double getNormEpsilon() const { return m_norm_eps; }

      /**
       * @brief Setters
       */
      void setHeight(const size_t height) { m_gss->setHeight(height); }
      void setWidth(const size_t width) { m_gss->setWidth(width); }
      void setNOctaves(const size_t n_octaves) { m_gss->setNOctaves(n_octaves); }
      void setNIntervals(const size_t n_intervals) { m_gss->setNIntervals(n_intervals); }
      void setOctaveMin(const int octave_min) { m_gss->setOctaveMin(octave_min); }
      void setSigmaN(const double sigma_n) { m_gss->setSigmaN(sigma_n); }
      void setSigma0(const double sigma0) { m_gss->setSigma0(sigma0); }
      void setKernelRadiusFactor(const double kernel_radius_factor) { m_gss->setKernelRadiusFactor(kernel_radius_factor); }
      void setConvBorder(const bob::sp::Extrapolation::BorderType border_type) { m_gss->setConvBorder(border_type); }
      void setContrastThreshold(const double threshold) { m_contrast_thres = threshold; }
      void setEdgeThreshold(const double threshold) { m_edge_thres = threshold; updateEdgeEffThreshold(); }
      void setNormThreshold(const double threshold) { m_norm_thres = threshold; }
      void setNBlocks(const size_t n_blocks) { m_descr_n_blocks = n_blocks; }
      void setNBins(const size_t n_bins) { m_descr_n_bins = n_bins; }
      void setGaussianWindowSize(const double size) { m_descr_gaussian_window_size = size; }
      void setMagnif(const double magnif) { m_descr_magnif = magnif; }
      void setNormEpsilon(const double norm_eps) { m_norm_eps = norm_eps; }

      /**
       * @brief  Automatically sets sigma0 to a value such that there is no
       * smoothing initially. sigma0 is then set such that the sigma value for
       * the first scale (index -1) of the octave octave_min is equal to
       * sigma_n*2^(-octave_min).
       */
      void setSigma0NoInitSmoothing() { m_gss->setSigma0NoInitSmoothing(); }

      /**
       * @brief Compute SIFT descriptors for the given keypoints
       * @param src The 2D input blitz array/image
       * @param keypoints The keypoints
       * @param dst The descriptor for the keypoints
       */
      template <typename T>
      void computeDescriptor(
        const blitz::Array<T,2>& src,
        const std::vector<boost::shared_ptr<bob::ip::base::GSSKeypoint> >& keypoints,
        blitz::Array<double,4>& dst
      ){
        // Computes the Gaussian pyramid
        computeGaussianPyramid(src);
        // Computes the Difference of Gaussians pyramid
        computeDog();
        // Computes the Gradient of the Gaussians pyramid
        computeGradient();
        // Computes the descriptors for the given keypoints
        computeDescriptor(keypoints, dst);
      }

      /**
       * @brief Get the shape of a descriptor for a given keypoint (y,x,orientation)
       */
      const blitz::TinyVector<int,3> getDescriptorShape() const;

    private:
      /**
       * @brief Resets the cache
       */
      void resetCache();

      /**
       * @brief Recomputes the value effectively used in the edge-like rejection
       * from the curvature/edge threshold
       */
      void updateEdgeEffThreshold() { m_edge_eff_thres = (m_edge_thres+1.)*(m_edge_thres+1.)/m_edge_thres; }

      /**
       * @brief Get the size of Gaussian filtered images for a given octave
       */
      const blitz::TinyVector<int,3> getGaussianOutputShape(const int octave) const;

      /**
       * @brief Computes the Gaussian pyramid
       */
      template <typename T>
      void computeGaussianPyramid(const blitz::Array<T,2>& src){
        // Computes the Gaussian pyramid
        m_gss->process(src, m_gss_pyr);
      }
      /**
       * @brief Computes the Difference of Gaussians pyramid
       * @warning assumes that the Gaussian pyramid has already been computed
       */
      void computeDog();

      /**
       * @brief Computes gradients from the Gaussian pyramid
       */
      void computeGradient();

      /**
       * @brief Compute SIFT descriptors for the given keypoints
       * @param keypoints The keypoints
       * @param dst The descriptor for the keypoints
       * @warning Assume that the Gaussian scale-space is already in cache
       */
      void computeDescriptor(const std::vector<boost::shared_ptr<bob::ip::base::GSSKeypoint> >& keypoints, blitz::Array<double,4>& dst) const;
      /**
       * @brief Compute SIFT descriptor for a given keypoint
       */
      void computeDescriptor(const bob::ip::base::GSSKeypoint& keypoint, const bob::ip::base::GSSKeypointInfo& keypoint_i, blitz::Array<double,3>& dst) const;
      void computeDescriptor(const bob::ip::base::GSSKeypoint& keypoint, blitz::Array<double,3>& dst) const;
      /**
       * @brief Compute SIFT keypoint additional information, from a regular
       * SIFT keypoint
       */
      void computeKeypointInfo(const bob::ip::base::GSSKeypoint& keypoint, bob::ip::base::GSSKeypointInfo& keypoint_info) const;


      /**
       * Attributes
       */
      boost::shared_ptr<bob::ip::base::GaussianScaleSpace> m_gss;
      double m_contrast_thres; //< Threshold for low-contrast keypoint rejection
      double m_edge_thres; //< Threshold (for the ratio of principal curvatures) for edge-like keypoint rejection
      double m_edge_eff_thres; //< Effective threshold for edge-like keypoint rejection.
      double m_norm_thres; //< Threshold used to clip high values during the descriptor normalization step
      // This is equal to the (r+1)^2/r, r being the regular edge threshold.

      size_t m_descr_n_blocks;
      size_t m_descr_n_bins;
      double m_descr_gaussian_window_size;
      double m_descr_magnif;
      double m_norm_eps;

      /**
       * Cache
       */
      std::vector<blitz::Array<double,3> > m_gss_pyr;
      std::vector<blitz::Array<double,3> > m_dog_pyr;
      std::vector<blitz::Array<double,3> > m_gss_pyr_grad_mag;
      std::vector<blitz::Array<double,3> > m_gss_pyr_grad_or;
      std::vector<boost::shared_ptr<bob::ip::base::GradientMaps> > m_gradient_maps;
  };


#if HAVE_VLFEAT

  class VLSIFT
  {
    public:
      /**
        * @brief Constructor
        */
      VLSIFT(
        const size_t height,
        const size_t width,
        const size_t n_intervals,
        const size_t n_octaves,
        const int octave_min,
        const double peak_thres=0.03,
        const double edge_thres=10.,
        const double magnif=3.
      );

      /**
        * @brief Copy constructor
        */
      VLSIFT(const VLSIFT& other);

      /**
        * @brief Destructor
        */
      virtual ~VLSIFT();

      /**
        * @brief Assignment operator
        */
      VLSIFT& operator=(const VLSIFT& other);

      /**
        * @brief Equal to
        */
      bool operator==(const VLSIFT& b) const;
      /**
        * @brief Not equal to
        */
      bool operator!=(const VLSIFT& b) const;

      /**
        * @brief Getters
        */
      size_t getHeight() const { return m_height; }
      size_t getWidth() const { return m_width; }
      size_t getNIntervals() const { return m_n_intervals; }
      size_t getNOctaves() const { return m_n_octaves; }
      int getOctaveMin() const { return m_octave_min; }
      int getOctaveMax() const { return m_octave_min+(int)m_n_octaves-1; }
      double getPeakThres() const { return m_peak_thres; }
      double getEdgeThres() const { return m_edge_thres; }
      double getMagnif() const { return m_magnif; }

      /**
        * @brief Setters
        */
      void setHeight(const size_t height) { m_height = height; cleanup(); allocateAndSet(); }
      void setWidth(const size_t width) { m_width = width; cleanup(); allocateAndSet(); }
      void setSize(const blitz::TinyVector<int,2>& size) { m_width = size[0]; m_height = size[1]; cleanup(); allocateAndSet(); }
      void setNIntervals(const size_t n_intervals) { m_n_intervals = n_intervals; cleanupFilter(); allocateFilterAndSet(); }
      void setNOctaves(const size_t n_octaves) { m_n_octaves = n_octaves; cleanupFilter(); allocateFilterAndSet(); }
      void setOctaveMin(const int octave_min) { m_octave_min = octave_min; cleanupFilter(); allocateFilterAndSet(); }
      void setPeakThres(const double peak_thres) { m_peak_thres = peak_thres; vl_sift_set_peak_thresh(m_filt, m_peak_thres); }
      void setEdgeThres(const double edge_thres) { m_edge_thres = edge_thres; vl_sift_set_edge_thresh(m_filt, m_edge_thres); }
      void setMagnif(const double magnif) { m_magnif = magnif; vl_sift_set_magnif(m_filt, m_magnif); }

      /**
        * @brief Extract SIFT features from a 2D blitz::Array, and save
        *   the resulting features in the dst vector of 1D blitz::Arrays.
        */
      void extract(
        const blitz::Array<uint8_t,2>& src,
        std::vector<blitz::Array<double,1> >& dst
      );
      /**
        * @brief Extract SIFT features from a 2D blitz::Array, at the
        *   keypoints specified by the 2D blitz::Array (Each row of length 3
        *   or 4 corresponds to a keypoint: y,x,sigma,[orientation]). The
        *   the resulting features are saved in the dst vector of
        *   1D blitz::Arrays.
        */
      void extract(
        const blitz::Array<uint8_t,2>& src,
        const blitz::Array<double,2>& keypoints,
        std::vector<blitz::Array<double,1> >& dst
      );


    protected:
      /**
        * @brief Allocation methods
        */
      void allocateBuffers();
      void allocateFilter();
      void allocate();
      /**
        * @brief Resets the properties of the VLfeat filter object
        */
      void setFilterProperties();
      /**
        * @brief Reallocate and resets the properties of the VLfeat filter
        * object
        */
      void allocateFilterAndSet();
      /**
        * @brief Reallocate and resets the properties of the VLfeat objects
        */
      void allocateAndSet();

      /**
        * @brief Deallocation methods
        */
      void cleanupBuffers();
      void cleanupFilter();
      void cleanup();

      /**
        * @brief Attributes
        */
      size_t m_height;
      size_t m_width;
      size_t m_n_intervals;
      size_t m_n_octaves;
      int m_octave_min; // might be negative
      double m_peak_thres;
      double m_edge_thres;
      double m_magnif;

      VlSiftFilt *m_filt;
      vl_uint8 *m_data;
      vl_sift_pix *m_fdata;
  };


    /**
    * @brief This class allows the computation of Dense SIFT features.
    *   The computation is done using the VLFeat library
    *   For more information, please refer to the following article:
    *     "Distinctive Image Features from Scale-Invariant Keypoints",
    *     from D.G. Lowe,
    *     International Journal of Computer Vision, 60, 2, pp. 91-110, 2004
    */
  class VLDSIFT
  {
    public:
      /**
        * @brief Constructor
        * @param height Input/image height
        * @param width Input/image width
        * @param step The x- and y-step for generating the grid of keypoins
        * @param block_size The x and y- size of a unit block
        */
      VLDSIFT(
        const blitz::TinyVector<int,2>& size,
        const blitz::TinyVector<int,2>& step=blitz::TinyVector<int,2>(5,5),
        const blitz::TinyVector<int,2>& block_size=blitz::TinyVector<int,2>(5,5)
      );

      /**
        * @brief Copy constructor
        */
      VLDSIFT(const VLDSIFT& other);

      /**
        * @brief Destructor
        */
      virtual ~VLDSIFT();

      /**
        * @brief Assignment operator
        */
      VLDSIFT& operator=(const VLDSIFT& other);

      /**
        * @brief Equal to
        */
      bool operator==(const VLDSIFT& b) const;
      /**
        * @brief Not equal to
        */
      bool operator!=(const VLDSIFT& b) const;

      /**
        * @brief Getters
        */
      size_t getHeight() const { return m_height; }
      size_t getWidth() const { return m_width; }
      blitz::TinyVector<int,2> getSize() const { return blitz::TinyVector<int,2>(m_height, m_width);}
      size_t getStepY() const { return m_step_y; }
      size_t getStepX() const { return m_step_x; }
      blitz::TinyVector<int,2> getStep() const { return blitz::TinyVector<int,2>(m_step_y, m_step_x);}
      size_t getBlockSizeY() const { return m_block_size_y; }
      size_t getBlockSizeX() const { return m_block_size_x; }
      blitz::TinyVector<int,2> getBlockSize() const { return blitz::TinyVector<int,2>(m_block_size_y, m_block_size_x);}
      bool getUseFlatWindow() const { return m_use_flat_window; }
      double getWindowSize() const { return m_window_size; }

      /**
        * @brief Setters
        */
      void setHeight(const size_t height) { m_height = height; cleanup(); allocateAndSet(); }
      void setWidth(const size_t width) { m_width = width; cleanup(); allocateAndSet(); }
      void setSize(const blitz::TinyVector<int,2>& size) { m_height = size[0]; m_width = size[1]; cleanup(); allocateAndSet(); }
      void setStepY(const size_t step_y) { m_step_y = step_y; vl_dsift_set_steps(m_filt, m_step_x, m_step_y); }
      void setStepX(const size_t step_x) { m_step_x = step_x; vl_dsift_set_steps(m_filt, m_step_x, m_step_y); }
      void setStep(const blitz::TinyVector<int,2>& step) {m_step_y = step[0]; m_step_x = step[1]; vl_dsift_set_steps(m_filt, m_step_x, m_step_y); }
      void setBlockSizeY(const size_t block_size_y);
      void setBlockSizeX(const size_t block_size_x);
      void setBlockSize(const blitz::TinyVector<int,2>& step);
      void setUseFlatWindow(const bool use) { m_use_flat_window = use; vl_dsift_set_flat_window(m_filt, use); }
      void setWindowSize(const double size) { m_window_size = size; vl_dsift_set_window_size(m_filt, size); }

      /**
        * @brief Extract Dense SIFT features from a 2D blitz::Array, and save
        *   the resulting features in the dst 2D blitz::Arrays.
        * @warning The src and dst arrays should have the correct size
        *   (for dst the expected size is (getNKeypoints(), getDescriptorSize())
        *   An exception is thrown otherwise.
        */
      void extract(const blitz::Array<float,2>& src, blitz::Array<float,2>& dst);

      /**
        * @brief Returns the number of keypoints given the current parameters
        * when processing an image of the expected size.
        */
      size_t getNKeypoints() const { return vl_dsift_get_keypoint_num(m_filt); }

      /**
        * @brief Returns the current size of a descriptor for a given keypoint
        * given the current parameters.
        * (number of bins = n_blocks_along_X x n_blocks_along_Y x n_hist_bins
        */
      size_t getDescriptorSize() const { return vl_dsift_get_descriptor_size(m_filt); }

    protected:
      /**
        * @brief Allocation methods
        */
      void allocate();
      /**
        * @brief Resets the properties of the VLfeat filter object
        */
      void setFilterProperties();
      /**
        * @brief Allocate and initialize the properties
        */
      void allocateAndInit();
      /**
        * @brief Reallocate and resets the properties of the VLfeat objects
        */
      void allocateAndSet();

      /**
        * @brief Deallocation method
        */
      void cleanup();

      /**
        * @brief Attributes
        */
      size_t m_height;
      size_t m_width;
      size_t m_step_y;
      size_t m_step_x;
      size_t m_block_size_y;
      size_t m_block_size_x;
      bool m_use_flat_window;
      double m_window_size;
      VlDsiftFilter *m_filt;
  };


#endif // HAVE_VLFEAT

} } } // namespaces

#endif /* BOB_IP_BASE_SIFT_H */
