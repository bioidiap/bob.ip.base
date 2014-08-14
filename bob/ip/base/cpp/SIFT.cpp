/**
 * @date Sun Sep 9 19:22:00 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.core/assert.h>
#include <algorithm>

#include <bob.ip.base/SIFT.h>

bob::ip::base::SIFT::SIFT(
  const size_t height,
  const size_t width,
  const size_t n_intervals,
  const size_t n_octaves,
  const int octave_min,
  const double sigma_n,
  const double sigma0,
  const double contrast_thres,
  const double edge_thres,
  const double norm_thres,
  const double kernel_radius_factor,
  const bob::sp::Extrapolation::BorderType border_type
):
  m_gss(new bob::ip::base::GaussianScaleSpace(height, width, n_intervals, n_octaves, octave_min, sigma_n, sigma0, kernel_radius_factor, border_type)),
  m_contrast_thres(contrast_thres),
  m_edge_thres(edge_thres),
  m_norm_thres(norm_thres),
  m_descr_n_blocks(4),
  m_descr_n_bins(8),
  m_descr_gaussian_window_size(m_descr_n_blocks/2.),
  m_descr_magnif(3.),
  m_norm_eps(1e-10)
{
  updateEdgeEffThreshold();
  resetCache();
}

bob::ip::base::SIFT::SIFT(const bob::ip::base::SIFT& other)
:
  m_gss(new bob::ip::base::GaussianScaleSpace(*(other.m_gss))),
  m_contrast_thres(other.m_contrast_thres),
  m_edge_thres(other.m_edge_thres), m_norm_thres(other.m_norm_thres),
  m_descr_n_blocks(other.m_descr_n_blocks),
  m_descr_n_bins(other.m_descr_n_bins),
  m_descr_gaussian_window_size(other.m_descr_gaussian_window_size),
  m_descr_magnif(other.m_descr_magnif), m_norm_eps(other.m_norm_eps)
{
  updateEdgeEffThreshold();
  resetCache();
  // Update cache content
  for (size_t i=0; i<m_gss_pyr.size(); ++i)
  {
    m_gss_pyr[i] = other.m_gss_pyr[i];
    m_dog_pyr[i] = other.m_dog_pyr[i];
    m_gss_pyr_grad_mag[i] = other.m_gss_pyr_grad_mag[i];
    m_gss_pyr_grad_or[i] = other.m_gss_pyr_grad_or[i];
  }
}

bob::ip::base::SIFT::~SIFT()
{
}

bob::ip::base::SIFT& bob::ip::base::SIFT::operator=(const bob::ip::base::SIFT& other)
{
  if (this != &other)
  {
    m_gss.reset(new bob::ip::base::GaussianScaleSpace(*(other.m_gss)));
    m_contrast_thres = other.m_contrast_thres;
    m_edge_thres = other.m_edge_thres;
    m_descr_n_blocks = other.m_descr_n_blocks;
    m_descr_n_bins = other.m_descr_n_bins;
    m_descr_gaussian_window_size = other.m_descr_gaussian_window_size;
    m_descr_magnif = other.m_descr_magnif;
    m_norm_eps = other.m_norm_eps;
    updateEdgeEffThreshold();
    m_norm_thres = other.m_norm_thres;
    resetCache();
    // Update cache content
    for (size_t i=0; i<m_gss_pyr.size(); ++i)
    {
      m_gss_pyr[i] = other.m_gss_pyr[i];
      m_dog_pyr[i] = other.m_dog_pyr[i];
      m_gss_pyr_grad_mag[i] = other.m_gss_pyr_grad_mag[i];
      m_gss_pyr_grad_or[i] = other.m_gss_pyr_grad_or[i];
    }
  }
  return *this;
}

bool bob::ip::base::SIFT::operator==(const bob::ip::base::SIFT& b) const
{
  if (*(this->m_gss) != *(b.m_gss) ||
        this->m_contrast_thres != b.m_contrast_thres ||
        this->m_edge_thres != b.m_edge_thres ||
        this->m_edge_eff_thres != b.m_edge_eff_thres ||
        this->m_norm_thres != b.m_norm_thres ||
        this->m_descr_n_blocks != b.m_descr_n_blocks ||
        this->m_descr_n_bins != b.m_descr_n_bins ||
        this->m_descr_gaussian_window_size != b.m_descr_gaussian_window_size ||
        this->m_descr_magnif != b.m_descr_magnif ||
        this->m_norm_thres != b.m_norm_thres)
    return false;

 if (this->m_gss_pyr.size() != b.m_gss_pyr.size() ||
     this->m_dog_pyr.size() != b.m_dog_pyr.size() ||
     this->m_gss_pyr_grad_mag.size() != b.m_gss_pyr_grad_mag.size() ||
     this->m_gss_pyr_grad_or.size() != b.m_gss_pyr_grad_or.size() ||
     this->m_gradient_maps.size() != b.m_gradient_maps.size())
    return false;

  for (size_t i=0; i<m_gss_pyr.size(); ++i)
    if (!bob::core::array::isEqual(this->m_gss_pyr[i], b.m_gss_pyr[i]))
      return false;

  for (size_t i=0; i<m_dog_pyr.size(); ++i)
    if (!bob::core::array::isEqual(this->m_dog_pyr[i], b.m_dog_pyr[i]))
      return false;

  for (size_t i=0; i<m_gss_pyr_grad_mag.size(); ++i)
    if (!bob::core::array::isEqual(this->m_gss_pyr_grad_mag[i], b.m_gss_pyr_grad_mag[i]))
      return false;

  for (size_t i=0; i<m_gss_pyr_grad_or.size(); ++i)
    if (!bob::core::array::isEqual(this->m_gss_pyr_grad_or[i], b.m_gss_pyr_grad_or[i]))
      return false;

  for (size_t i=0; i<m_gradient_maps.size(); ++i)
    if (*(this->m_gradient_maps[i]) != *(b.m_gradient_maps[i]))
      return false;

  return true;
}

bool bob::ip::base::SIFT::operator!=(const bob::ip::base::SIFT& b) const
{
  return !(this->operator==(b));
}

const blitz::TinyVector<int,3> bob::ip::base::SIFT::getDescriptorShape() const
{
  return blitz::TinyVector<int,3>(m_descr_n_blocks, m_descr_n_blocks, m_descr_n_bins);
}


void bob::ip::base::SIFT::resetCache()
{
  m_gss->allocateOutputPyramid(m_gss_pyr);
  m_dog_pyr.clear();
  m_gss_pyr_grad_mag.clear();
  m_gss_pyr_grad_or.clear();
  for (size_t i=0; i<m_gss_pyr.size(); ++i)
  {
    m_dog_pyr.push_back(blitz::Array<double,3>(m_gss_pyr[i].extent(0)-1,
      m_gss_pyr[i].extent(1), m_gss_pyr[i].extent(2)));
    m_gss_pyr_grad_mag.push_back(blitz::Array<double,3>(m_gss_pyr[i].extent(0)-3,
      m_gss_pyr[i].extent(1), m_gss_pyr[i].extent(2)));
    m_gss_pyr_grad_or.push_back(blitz::Array<double,3>(m_gss_pyr[i].extent(0)-3,
      m_gss_pyr[i].extent(1), m_gss_pyr[i].extent(2)));
    m_gradient_maps.push_back(boost::shared_ptr<bob::ip::base::GradientMaps>(new
      bob::ip::base::GradientMaps(m_gss_pyr[i].extent(1), m_gss_pyr[i].extent(2))));
    m_gss_pyr[i] = 0.;
    m_dog_pyr[i] = 0.;
    m_gss_pyr_grad_mag[i] = 0.;
    m_gss_pyr_grad_or[i] = 0.;
  }
}

const blitz::TinyVector<int,3> bob::ip::base::SIFT::getGaussianOutputShape(const int octave) const
{
  return m_gss->getOutputShape(octave);
}

void bob::ip::base::SIFT::computeDog()
{
  // Computes the Difference of Gaussians pyramid
  blitz::Range rall = blitz::Range::all();
  for (size_t o=0; o<m_gss_pyr.size(); ++o)
    for (size_t s=0; s<(size_t)(m_gss_pyr[o].extent(0)-1); ++s)
    {
      blitz::Array<double,2> dst_os = m_dog_pyr[o](s, rall, rall);
      blitz::Array<double,2> src1 = m_gss_pyr[o](s, rall, rall);
      blitz::Array<double,2> src2 = m_gss_pyr[o](s+1, rall, rall);
      dst_os = src2 - src1;
    }
}

void bob::ip::base::SIFT::computeGradient()
{
  blitz::Range rall = blitz::Range::all();
  for (size_t i=0; i<m_gss_pyr.size(); ++i)
  {
    blitz::Array<double,3>& gss = m_gss_pyr[i];
    blitz::Array<double,3>& gmag = m_gss_pyr_grad_mag[i];
    blitz::Array<double,3>& gor = m_gss_pyr_grad_or[i];
    boost::shared_ptr<bob::ip::base::GradientMaps> gmap = m_gradient_maps[i];
    for (int s=0; s<gmag.extent(0); ++s)
    {
      blitz::Array<double,2> gss_s = gss(s+1, rall, rall);
      blitz::Array<double,2> gmag_s = gmag(s, rall, rall);
      blitz::Array<double,2> gor_s = gor(s, rall, rall);
      gmap->process(gss_s, gmag_s, gor_s);
    }
  }
}

void bob::ip::base::SIFT::computeDescriptor(const std::vector<boost::shared_ptr<bob::ip::base::GSSKeypoint> >& keypoints, blitz::Array<double,4>& dst) const
{
  blitz::Range rall = blitz::Range::all();
  for (size_t k=0; k<keypoints.size(); ++k)
  {
    blitz::Array<double,3> dst_k = dst(k, rall, rall, rall);
    computeDescriptor(*(keypoints[k]), dst_k);
  }
}

void bob::ip::base::SIFT::computeDescriptor(const bob::ip::base::GSSKeypoint& keypoint, blitz::Array<double,3>& dst) const
{
  // Extracts more detailed information about the keypoint (octave and scale)
  bob::ip::base::GSSKeypointInfo keypoint_info;
  computeKeypointInfo(keypoint, keypoint_info);
  computeDescriptor(keypoint, keypoint_info, dst);
}

void bob::ip::base::SIFT::computeDescriptor(const bob::ip::base::GSSKeypoint& keypoint, const bob::ip::base::GSSKeypointInfo& keypoint_info, blitz::Array<double,3>& dst) const
{
  // Check output dimensionality
  const blitz::TinyVector<int,3> shape = getDescriptorShape();
  bob::core::array::assertSameShape(dst, shape);

  // Get gradient
  blitz::Range rall = blitz::Range::all();
  // Index scale has a -1, as the gradients are not computed for scale -1, Ns and Ns+1
  // but the provided index is the one, for which scale -1 corresponds to keypoint_info.s=0.
  blitz::Array<double,2> gmag = m_gss_pyr_grad_mag[keypoint_info.o](keypoint_info.s-1,rall,rall);
  blitz::Array<double,2> gor = m_gss_pyr_grad_or[keypoint_info.o](keypoint_info.s-1,rall,rall);

  // Dimensions of the image at the octave associated with the keypoint
  const int H = gmag.extent(0);
  const int W = gmag.extent(1);

  // Coordinates and sigma wrt. to the image size at the octave associated with the keypoint
  const double factor = pow(2., m_gss->getOctaveMin()+(double)keypoint_info.o);
  const double sigma = keypoint.sigma / factor;
  const double yc = keypoint.y / factor;
  const double xc = keypoint.x / factor;

  // Cosine and sine of the keypoint orientation
  const double cosk = cos(keypoint.orientation);
  const double sink = sin(keypoint.orientation);

  // Each local spatial histogram has an extension hist_width = MAGNIF*sigma
  // pixels. Furthermore, the concatenated histogram has a spatial support of
  // hist_width * DESCR_NBLOCKS pixels. Because of the interpolation, 1 extra
  // pixel might be used, leading to hist_width * (DESCR_NBLOCKS+1). Finally,
  // this square support might be arbitrarily rotated, leading to an effective
  // support of sqrt(2) * hist_width * (DESCR_NBLOCKS+1).
  const double hist_width = m_descr_magnif * sigma;
  const int descr_radius = (int)floor(sqrt(2)*hist_width*(m_descr_n_blocks+1)/2. + 0.5);
  const double window_factor = 0.5 / (m_descr_gaussian_window_size*m_descr_gaussian_window_size);
  static const double two_pi = 2.*M_PI;

  // Determines boundaries to make sure that we remain on the image while
  // computing the descriptor
  const int yci = (int)floor(yc+0.5);
  const int xci = (int)floor(xc+0.5);

  const int dymin = std::max(-descr_radius,1-yci);
  const int dymax = std::min(descr_radius,H-2-yci);
  const int dxmin = std::max(-descr_radius,1-xci);
  const int dxmax = std::min(descr_radius,W-2-xci);

  // Loop over the pixels
  // Initializes descriptor to zero
  dst = 0.;
  for (int dyi=dymin; dyi<=dymax; ++dyi)
    for (int dxi=dxmin; dxi<=dxmax; ++dxi)
    {
      // Current integer indices
      int yi = yci + dyi;
      int xi = xci + dxi;
      // Values of the current gradient (magnitude and orientation)
      double mag = gmag(yi,xi);
      double ori = gor(yi,xi);
      // Angle between keypoint orientation and gradient orientation
      double theta = fmod(ori-keypoint.orientation, two_pi);
      if (theta < 0.) theta += two_pi;
      if (theta >= two_pi) theta -= two_pi;

      // Current floating point offset wrt. descriptor center
      double dy = yi - yc;
      double dx = xi - xc;

      // Normalized offset wrt. the keypoint orientation, offset and scale
      double ny = (-sink*dx + cosk*dy) / hist_width;
      double nx = ( cosk*dx + sink*dy) / hist_width;
      double no = (theta / two_pi) * m_descr_n_bins;

      // Gaussian weight for the current pixel
      double window_value = exp(-(nx*nx+ny*ny)*window_factor);

      // Indices of the first bin used in the interpolation
      // Substract -0.5 before flooring such as the weight rbiny=0.5 when
      // we are between the two centered pixels (assuming that DESCR_NBLOCKS
      // is {equal to 4/even}), for which ny=0.
      // (ny=0-> rbiny=0.5 -> (final) biny = DESCR_NBLOCKS/2-1 (which
      // corresponds to the left centered pixel)
      int biny = (int)floor(ny-0.5);
      int binx = (int)floor(nx-0.5);
      int bino = (int)floor(no);
      double rbiny = ny - (biny + 0.5);
      double rbinx = nx - (binx + 0.5);
      double rbino = no - bino;
      // Make indices start at 0
      biny += m_descr_n_blocks/2;
      binx += m_descr_n_blocks/2;

      for (int dbiny=0; dbiny<2; ++dbiny)
      {
        int biny_ = biny+dbiny;
        if (biny_ >= 0 && biny_ < (int)m_descr_n_blocks)
        {
          double wy = ( dbiny==0 ? fabs(1.-rbiny) : fabs(rbiny) );
          for (int dbinx=0; dbinx<2; ++dbinx)
          {
            int binx_ = binx+dbinx;
            if (binx_ >= 0 && binx_ < (int)m_descr_n_blocks)
            {
              double wx = ( dbinx==0 ? fabs(1.-rbinx) : fabs(rbinx) );
              for (int dbino=0; dbino<2; ++dbino)
              {
                double wo = ( dbino==0 ? fabs(1.-rbino) : fabs(rbino) );
                dst(biny_, binx_, (bino+dbino) % (int)m_descr_n_bins) += window_value * mag * wy * wx * wo;
              }
            }
          }
        }
      }
    }

  // Normalization
  double norm = sqrt(blitz::sum(blitz::pow2(dst))) + m_norm_eps;
  dst /= norm;
  // Clip values above norm threshold
  dst = blitz::where(dst > m_norm_thres, m_norm_thres, dst);
  // Renormalize
  norm = sqrt(blitz::sum(blitz::pow2(dst))) + m_norm_eps;
  dst /= norm;
}

void bob::ip::base::SIFT::computeKeypointInfo(const bob::ip::base::GSSKeypoint& keypoint, bob::ip::base::GSSKeypointInfo& keypoint_i) const
{
  const int No = (int)getNOctaves();
  const int Ns = (int)getNIntervals();
  const int& omin = getOctaveMin();

  // sigma_{o,s} = sigma0 * 2^{o+s/N_SCALES}, where
  //   o is the octave index, and s the scale index
  // Define phi = log2(sigma_{o,s} / sigma0) = o+s/N_SCALES
  const double phi = log(keypoint.sigma / getSigma0()) / log(2.);

  // A. Octave index
  // Use -0.5/NIntervals term in order to center around scales of indices [1,S]
  // TODO: check if +0.5/Ns or 0!
  int o = (int)floor(phi + 0.5/Ns);
  // Check boundaries
  if (o < omin) o = omin; // min
  if (o > omin+No-1) o = omin+No-1; // max
  keypoint_i.o = o-omin;

  // B. Scale index
  // Adds 1 after the flooring for the conversion of the scale location into
  // a scale index (first scale is located at -1 in the GSS pyramid)
  size_t s = (int)floor(Ns*(phi-o) + 0.5) + 1;
  if (s < 1) s = 1; // min
  if (s > (size_t)Ns) s = Ns; // max
  keypoint_i.s = s;

  // C. (y,x) coordinates
  const double factor = pow(2.,o);
  keypoint_i.iy = (int)floor(keypoint.y/factor + 0.5);
  keypoint_i.ix = (int)floor(keypoint.x/factor + 0.5);
}


#if HAVE_VLFEAT
#include <vl/pgm.h>
#include <bob.core/array_copy.h>
/// VLSIFT
bob::ip::base::VLSIFT::VLSIFT(const size_t height, const size_t width,
    const size_t n_intervals, const size_t n_octaves, const int octave_min,
    const double peak_thres, const double edge_thres, const double magnif):
  m_height(height), m_width(width), m_n_intervals(n_intervals),
  m_n_octaves(n_octaves), m_octave_min(octave_min),
  m_peak_thres(peak_thres), m_edge_thres(edge_thres), m_magnif(magnif)
{
  // Allocates buffers and filter, and set filter properties
  allocateAndSet();
}

bob::ip::base::VLSIFT::VLSIFT(const VLSIFT& other):
  m_height(other.m_height), m_width(other.m_width),
  m_n_intervals(other.m_n_intervals), m_n_octaves(other.m_n_octaves),
  m_octave_min(other.m_octave_min), m_peak_thres(other.m_peak_thres),
  m_edge_thres(other.m_edge_thres), m_magnif(other.m_magnif)
{
  // Allocates buffers and filter, and set filter properties
  allocateAndSet();
}

bob::ip::base::VLSIFT& bob::ip::base::VLSIFT::operator=(const bob::ip::base::VLSIFT& other)
{
  if (this != &other)
  {
    m_height = other.m_height;
    m_width = other.m_width;
    m_n_intervals = other.m_n_intervals;
    m_n_octaves = other.m_n_octaves;
    m_octave_min = other.m_octave_min;
    m_peak_thres = other.m_peak_thres;
    m_edge_thres = other.m_edge_thres;
    m_magnif = other.m_magnif;

    // Allocates buffers and filter, and set filter properties
    allocateAndSet();
  }
  return *this;
}

bool bob::ip::base::VLSIFT::operator==(const bob::ip::base::VLSIFT& b) const
{
  return (this->m_height == b.m_height && this->m_width == b.m_width &&
          this->m_n_intervals == b.m_n_intervals &&
          this->m_n_octaves == b.m_n_octaves &&
          this->m_octave_min == b.m_octave_min &&
          this->m_peak_thres == b.m_peak_thres &&
          this->m_edge_thres == b.m_edge_thres &&
          this->m_magnif == b.m_magnif);
}

bool bob::ip::base::VLSIFT::operator!=(const bob::ip::base::VLSIFT& b) const
{
  return !(this->operator==(b));
}

void bob::ip::base::VLSIFT::extract(const blitz::Array<uint8_t,2>& src,
  std::vector<blitz::Array<double,1> >& dst)
{
  // Clears the vector
  dst.clear();
  vl_bool err=VL_ERR_OK;

  // Copies data
  for(unsigned int q=0; q<(unsigned)(m_width * m_height); ++q)
    m_data[q] = src((int)(q/m_width), (int)(q%m_width));
  // Converts data type
  for(unsigned int q=0; q<(unsigned)(m_width * m_height); ++q)
    m_fdata[q] = m_data[q];

  // Processes each octave
  int i=0;
  bool first=true;
  while(1)
  {
    VlSiftKeypoint const *keys = 0;
    int nkeys;

    // Calculates the GSS for the next octave
    if(first)
    {
      first = false;
      err = vl_sift_process_first_octave(m_filt, m_fdata);
    }
    else
      err = vl_sift_process_next_octave(m_filt);

    if(err)
    {
      err = VL_ERR_OK;
      break;
    }

    // Runs the detector
    vl_sift_detect(m_filt);
    keys = vl_sift_get_keypoints(m_filt);
    nkeys = vl_sift_get_nkeypoints(m_filt);
    i = 0;

    // Loops over the keypoint
    for(; i < nkeys ; ++i) {
      double angles[4];
      int nangles;
      VlSiftKeypoint const *k;

      // Obtains keypoint orientations
      k = keys + i;
      nangles = vl_sift_calc_keypoint_orientations(m_filt, angles, k);

      // For each orientation
      for(unsigned int q=0; q<(unsigned)nangles; ++q) {
        blitz::Array<double,1> res(128+4);
        vl_sift_pix descr[128];

        // Computes the descriptor
        vl_sift_calc_keypoint_descriptor(m_filt, descr, k, angles[q]);

        int l;
        res(0) = k->x;
        res(1) = k->y;
        res(2) = k->sigma;
        res(3) = angles[q];
        for(l=0; l<128; ++l)
          res(4+l) = 512. * descr[l];

        // Adds it to the vector
        dst.push_back(res);
      }
    }
  }

}

void bob::ip::base::VLSIFT::extract(const blitz::Array<uint8_t,2>& src,
  const blitz::Array<double,2>& keypoints,
  std::vector<blitz::Array<double,1> >& dst)
{
  if(keypoints.extent(1) != 3 && keypoints.extent(1) != 4) {
    boost::format m("extent for dimension 1 of keypoints is %d where it should be either 3 or 4");
    m % keypoints.extent(1);
    throw std::runtime_error(m.str());
  }

  // Clears the vector
  dst.clear();
  vl_bool err=VL_ERR_OK;

  // Copies data
  for(unsigned int q=0; q<(unsigned)(m_width * m_height); ++q)
    m_data[q] = src((int)(q/m_width), (int)(q%m_width));
  // Converts data type
  for(unsigned int q=0; q<(unsigned)(m_width * m_height); ++q)
    m_fdata[q] = m_data[q];

  // Processes each octave
  bool first=true;
  while(1)
  {
    // Calculates the GSS for the next octave
    if(first)
    {
      first = false;
      err = vl_sift_process_first_octave(m_filt, m_fdata);
    }
    else
      err = vl_sift_process_next_octave(m_filt);

    if(err)
    {
      err = VL_ERR_OK;
      break;
    }

    // Loops over the keypoint
    for(int i=0; i<keypoints.extent(0); ++i) {
      double angles[4];
      int nangles;
      VlSiftKeypoint ik;
      VlSiftKeypoint const *k;

      // Obtain keypoint orientations
      vl_sift_keypoint_init(m_filt, &ik,
        keypoints(i,1), keypoints(i,0), keypoints(i,2)); // x, y, sigma

      if(ik.o != vl_sift_get_octave_index(m_filt))
        continue; // Not current scale/octave

      k = &ik ;

      // Compute orientations if required
      if(keypoints.extent(1) == 4)
      {
        angles[0] = keypoints(i,3);
        nangles = 1;
      }
      else
        // TODO: No way to know if several keypoints are generated from one location
        nangles = vl_sift_calc_keypoint_orientations(m_filt, angles, k);

      // For each orientation
      for(unsigned int q=0; q<(unsigned)nangles; ++q) {
        blitz::Array<double,1> res(128+4);
        vl_sift_pix descr[128];

        // Computes the descriptor
        vl_sift_calc_keypoint_descriptor(m_filt, descr, k, angles[q]);

        int l;
        res(0) = k->x;
        res(1) = k->y;
        res(2) = k->sigma;
        res(3) = angles[q];
        for(l=0; l<128; ++l)
          res(4+l) = 512. * descr[l];

        // Adds it to the vector
        dst.push_back(res);
      }
    }
  }
}


void bob::ip::base::VLSIFT::allocateBuffers()
{
  const size_t npixels = m_height * m_width;
  // Allocates buffers
  m_data  = (vl_uint8*)malloc(npixels * sizeof(vl_uint8));
  m_fdata = (vl_sift_pix*)malloc(npixels * sizeof(vl_sift_pix));
  // TODO: deals with allocation error?
}

void bob::ip::base::VLSIFT::allocateFilter()
{
  // Generates the filter
  m_filt = vl_sift_new(m_width, m_height, m_n_octaves, m_n_intervals, m_octave_min);
  // TODO: deals with allocation error?
}

void bob::ip::base::VLSIFT::allocate()
{
  allocateBuffers();
  allocateFilter();
}

void bob::ip::base::VLSIFT::setFilterProperties()
{
  // Set filter properties
  vl_sift_set_edge_thresh(m_filt, m_edge_thres);
  vl_sift_set_peak_thresh(m_filt, m_peak_thres);
  vl_sift_set_magnif(m_filt, m_magnif);
}

void bob::ip::base::VLSIFT::allocateFilterAndSet()
{
  allocateFilter();
  setFilterProperties();
}

void bob::ip::base::VLSIFT::allocateAndSet()
{
  allocateBuffers();
  allocateFilterAndSet();
}

void bob::ip::base::VLSIFT::cleanupBuffers()
{
  // Releases image data
  free(m_fdata);
  m_fdata = 0;
  free(m_data);
  m_data = 0;
}

void bob::ip::base::VLSIFT::cleanupFilter()
{
  // Releases filter
  vl_sift_delete(m_filt);
  m_filt = 0;
}

void bob::ip::base::VLSIFT::cleanup()
{
  cleanupBuffers();
  cleanupFilter();
}

bob::ip::base::VLSIFT::~VLSIFT()
{
  cleanup();
}



/// VLDSIFT
bob::ip::base::VLDSIFT::VLDSIFT(
  const blitz::TinyVector<int,2>& size,
  const blitz::TinyVector<int,2>& step,
  const blitz::TinyVector<int,2>& block_size
):
  m_height(size[0]), m_width(size[1]), m_step_y(step[0]), m_step_x(step[1]),
  m_block_size_y(block_size[0]), m_block_size_x(block_size[1])
{
  allocateAndInit();
}


bob::ip::base::VLDSIFT::VLDSIFT(const VLDSIFT& other):
  m_height(other.m_height), m_width(other.m_width),
  m_step_y(other.m_step_y), m_step_x(other.m_step_x),
  m_block_size_y(other.m_block_size_y),
  m_block_size_x(other.m_block_size_x),
  m_use_flat_window(other.m_use_flat_window),
  m_window_size(other.m_window_size)
{
  allocateAndSet();
}

bob::ip::base::VLDSIFT::~VLDSIFT()
{
  cleanup();
}

bob::ip::base::VLDSIFT& bob::ip::base::VLDSIFT::operator=(const bob::ip::base::VLDSIFT& other)
{
  if (this != &other)
  {
    m_height = other.m_height;
    m_width = other.m_width;
    m_step_y = other.m_step_y;
    m_step_x = other.m_step_x;
    m_block_size_y = other.m_block_size_y;
    m_block_size_x = other.m_block_size_x;
    m_use_flat_window = other.m_use_flat_window;
    m_window_size = other.m_window_size;

    // Allocates filter, and set filter properties
    allocateAndSet();
  }
  return *this;
}

bool bob::ip::base::VLDSIFT::operator==(const bob::ip::base::VLDSIFT& b) const
{
  return (this->m_height == b.m_height && this->m_width == b.m_width &&
          this->m_step_y == b.m_step_y && this->m_step_x == b.m_step_x &&
          this->m_block_size_y == b.m_block_size_y &&
          this->m_block_size_x == b.m_block_size_x &&
          this->m_use_flat_window == b.m_use_flat_window &&
          this->m_window_size == b.m_window_size);
}

bool bob::ip::base::VLDSIFT::operator!=(const bob::ip::base::VLDSIFT& b) const
{
  return !(this->operator==(b));
}

void bob::ip::base::VLDSIFT::setBlockSizeY(const size_t block_size_y)
{
  m_block_size_y = block_size_y;
  VlDsiftDescriptorGeometry geom = *vl_dsift_get_geometry(m_filt);
  geom.binSizeY = (int)m_block_size_y;
  geom.binSizeX = (int)m_block_size_x;
  vl_dsift_set_geometry(m_filt, &geom) ;
}

void bob::ip::base::VLDSIFT::setBlockSizeX(const size_t block_size_x)
{
  m_block_size_x = block_size_x;
  VlDsiftDescriptorGeometry geom = *vl_dsift_get_geometry(m_filt);
  geom.binSizeY = (int)m_block_size_y;
  geom.binSizeX = (int)m_block_size_x;
  vl_dsift_set_geometry(m_filt, &geom) ;
}

void bob::ip::base::VLDSIFT::setBlockSize(const blitz::TinyVector<int,2>& block_size)
{
  m_block_size_y = block_size[0];
  m_block_size_x = block_size[1];
  VlDsiftDescriptorGeometry geom = *vl_dsift_get_geometry(m_filt);
  geom.binSizeY = (int)m_block_size_y;
  geom.binSizeX = (int)m_block_size_x;
  vl_dsift_set_geometry(m_filt, &geom) ;
}

void bob::ip::base::VLDSIFT::extract(const blitz::Array<float,2>& src,
  blitz::Array<float,2>& dst)
{
  // Check parameters size size
  bob::core::array::assertSameDimensionLength(src.extent(0), m_height);
  bob::core::array::assertSameDimensionLength(src.extent(1), m_width);
  int num_frames = vl_dsift_get_keypoint_num(m_filt);
  int descr_size = vl_dsift_get_descriptor_size(m_filt);
  bob::core::array::assertSameDimensionLength(dst.extent(0), num_frames);
  bob::core::array::assertSameDimensionLength(dst.extent(1), descr_size);

  // Get C-style pointer to src data, making a copy if required
  const float* data;
  blitz::Array<float,2> x;
  if(bob::core::array::isCZeroBaseContiguous(src))
    data = src.data();
  else
  {
    x.reference(bob::core::array::ccopy(src));
    data = x.data();
  }

  // Computes features
  vl_dsift_process(m_filt, data);

  // Move output back to destination array
  float const *descrs = vl_dsift_get_descriptors(m_filt);
  if(bob::core::array::isCZeroBaseContiguous(dst))
    // fast copy
    std::memcpy(dst.data(), descrs, num_frames*descr_size*sizeof(float));
  else
  {
    // Iterate (slow...)
    for(int f=0; f<num_frames; ++f)
      for(int b=0; b<descr_size; ++b)
      {
        dst(f,b) = *descrs;
        ++descrs;
      }
  }
}

void bob::ip::base::VLDSIFT::allocate()
{
  // Generates the filter
  m_filt = vl_dsift_new_basic((int)m_width, (int)m_height, (int)m_step_y,
            (int)m_block_size_y);
}

void bob::ip::base::VLDSIFT::allocateAndInit()
{
  allocate();
  m_use_flat_window = vl_dsift_get_flat_window(m_filt);
  m_window_size = vl_dsift_get_window_size(m_filt);
}

void bob::ip::base::VLDSIFT::setFilterProperties()
{
  // Set filter properties
  vl_dsift_set_steps(m_filt, (int)m_step_x, (int)m_step_y);
  vl_dsift_set_flat_window(m_filt, m_use_flat_window);
  vl_dsift_set_window_size(m_filt, m_window_size);
  // Set block size
  VlDsiftDescriptorGeometry geom = *vl_dsift_get_geometry(m_filt);
  geom.binSizeY = (int)m_block_size_y;
  geom.binSizeX = (int)m_block_size_x;
  vl_dsift_set_geometry(m_filt, &geom) ;
}

void bob::ip::base::VLDSIFT::allocateAndSet()
{
  allocate();
  setFilterProperties();
}

void bob::ip::base::VLDSIFT::cleanup()
{
  // Releases filter
  vl_dsift_delete(m_filt);
  m_filt = 0;
}



#endif // HAVE_VLFEAT



