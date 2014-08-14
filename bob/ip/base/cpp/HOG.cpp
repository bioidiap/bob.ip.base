/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Tue Jul  8 14:37:19 CEST 2014
 *
 * @brief Computes Histogram of Oriented Gradients (HOG) descriptors
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.ip.base/HOG.h>

bob::ip::base::BlockCellDescriptors::BlockCellDescriptors(
  const size_t height,
  const size_t width,
  const size_t cell_dim,
  const size_t cell_y,
  const size_t cell_x,
  const size_t cell_ov_y,
  const size_t cell_ov_x,
  const size_t block_y,
  const size_t block_x,
  const size_t block_ov_y,
  const size_t block_ov_x
):
  m_height(height),
  m_width(width),
  m_cell_dim(cell_dim),
  m_cell_y(cell_y),
  m_cell_x(cell_x),
  m_cell_ov_y(cell_ov_y),
  m_cell_ov_x(cell_ov_x),
  m_block_y(block_y),
  m_block_x(block_x),
  m_block_ov_y(block_ov_y),
  m_block_ov_x(block_ov_x),
  m_block_norm(L2),
  m_block_norm_eps(1e-10),
  m_block_norm_threshold(0.2)
{
  resizeCellCache();
}

bob::ip::base::BlockCellDescriptors::BlockCellDescriptors(const bob::ip::base::BlockCellDescriptors& other)
:
  m_height(other.m_height),
  m_width(other.m_width),
  m_cell_dim(other.m_cell_dim),
  m_cell_y(other.m_cell_y),
  m_cell_x(other.m_cell_x),
  m_cell_ov_y(other.m_cell_ov_y),
  m_cell_ov_x(other.m_cell_ov_x),
  m_block_y(other.m_block_y),
  m_block_x(other.m_block_x),
  m_block_ov_y(other.m_block_ov_y),
  m_block_ov_x(other.m_block_ov_x),
  m_block_norm(other.m_block_norm),
  m_block_norm_eps(other.m_block_norm_eps),
  m_block_norm_threshold(other.m_block_norm_threshold)
{
  resizeCache();
}


bob::ip::base::BlockCellDescriptors& bob::ip::base::BlockCellDescriptors::operator=(const bob::ip::base::BlockCellDescriptors& other){
  if (this != &other){
    m_height = other.m_height;
    m_width = other.m_width;
    m_cell_dim = other.m_cell_dim;
    m_cell_y = other.m_cell_y;
    m_cell_x = other.m_cell_x;
    m_cell_ov_y = other.m_cell_ov_y;
    m_cell_ov_x = other.m_cell_ov_x;
    m_block_y = other.m_block_y;
    m_block_x = other.m_block_x;
    m_block_ov_y = other.m_block_ov_y;
    m_block_ov_x = other.m_block_ov_x;
    m_block_norm = other.m_block_norm;
    m_block_norm_eps = other.m_block_norm_eps;
    m_block_norm_threshold = other.m_block_norm_threshold;
    resizeCache();
  }
  return *this;
}

bool bob::ip::base::BlockCellDescriptors::operator==(const bob::ip::base::BlockCellDescriptors& b) const{
  return (m_height == b.m_height &&
          m_width == b.m_width &&
          m_cell_dim == b.m_cell_dim &&
          m_cell_y == b.m_cell_y &&
          m_cell_x == b.m_cell_x &&
          m_cell_ov_y == b.m_cell_ov_y &&
          m_cell_ov_x == b.m_cell_ov_x &&
          m_block_y == b.m_block_y &&
          m_block_x == b.m_block_x &&
          m_block_ov_y == b.m_block_ov_y &&
          m_block_ov_x == b.m_block_ov_x &&
          m_block_norm == b.m_block_norm &&
          m_block_norm_eps == b.m_block_norm_eps &&
          m_block_norm_threshold == b.m_block_norm_threshold);
}

void bob::ip::base::BlockCellDescriptors::BlockCellDescriptors::resizeCellCache()
{
  // Resizes the cell-related arrays
  const blitz::TinyVector<int,4> nb_cells = getBlock4DOutputShape(
      m_height, m_width, m_cell_y, m_cell_x, m_cell_ov_y, m_cell_ov_x);
  m_cell_descriptor.resize(nb_cells(0), nb_cells(1), m_cell_dim);

  // Updates the class members
  m_nb_cells_y = nb_cells(0);
  m_nb_cells_x = nb_cells(1);

  // Number of blocks should be updated
  resizeBlockCache();
}

void bob::ip::base::BlockCellDescriptors::resizeBlockCache()
{
  // Determines the number of blocks per row and column
  blitz::TinyVector<int,4> nb_blocks = getBlock4DOutputShape(
    m_nb_cells_y, m_nb_cells_x, m_block_y, m_block_x, m_block_ov_y,
    m_block_ov_x);

  // Updates the class members
  m_nb_blocks_y = nb_blocks(0);
  m_nb_blocks_x = nb_blocks(1);
}

void bob::ip::base::BlockCellDescriptors::disableBlockNormalization()
{
  m_block_y = 1;
  m_block_x = 1;
  m_block_ov_y = 0;
  m_block_ov_x = 0;
  m_block_norm = Nonorm;
  resizeBlockCache();
}

void bob::ip::base::BlockCellDescriptors::normalizeBlocks(blitz::Array<double,3>& output)
{
  blitz::Range rall = blitz::Range::all();
  // Normalizes by block
  for(size_t by=0; by<m_nb_blocks_y; ++by)
    for(size_t bx=0; bx<m_nb_blocks_x; ++bx)
    {
      blitz::Range ry(by,by+m_block_y-1);
      blitz::Range rx(bx,bx+m_block_x-1);
      blitz::Array<double,3> cells_block = m_cell_descriptor(ry,rx,rall);
      blitz::Array<double,1> block = output(by,bx,rall);
      _normalizeBlock(cells_block, block, m_block_norm, m_block_norm_eps, m_block_norm_threshold);
    }
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

bob::ip::base::BlockCellGradientDescriptors::BlockCellGradientDescriptors(
    const size_t height,
    const size_t width,
    const size_t cell_dim,
    const size_t cell_y,
    const size_t cell_x,
    const size_t cell_ov_y,
    const size_t cell_ov_x,
    const size_t block_y,
    const size_t block_x,
    const size_t block_ov_y,
    const size_t block_ov_x
):
  BlockCellDescriptors(height, width, cell_dim, cell_y, cell_x, cell_ov_y, cell_ov_x, block_y, block_x, block_ov_y, block_ov_x),
  m_gradient_maps(new GradientMaps(height, width))
{
  resizeCache();
}

bob::ip::base::BlockCellGradientDescriptors::BlockCellGradientDescriptors(const bob::ip::base::BlockCellGradientDescriptors& b)
:
  BlockCellDescriptors(b),
  m_gradient_maps(new GradientMaps(b.m_height, b.m_width, b.getGradientMagnitudeType()))
{
  resizeCache();
}

bob::ip::base::BlockCellGradientDescriptors& bob::ip::base::BlockCellGradientDescriptors::operator=(const bob::ip::base::BlockCellGradientDescriptors& other)
{
  if(this != &other)
  {
    BlockCellDescriptors::operator=(other);
    m_gradient_maps.reset(new GradientMaps(other.m_height, other.m_width, other.getGradientMagnitudeType()));
    resizeCache();
  }
  return *this;
}

bool bob::ip::base::BlockCellGradientDescriptors::operator==(const bob::ip::base::BlockCellGradientDescriptors& b) const
{
  return (BlockCellDescriptors::operator==(b) &&
          *(this->m_gradient_maps) == *(b.m_gradient_maps));
}

bool bob::ip::base::BlockCellGradientDescriptors::operator!=(const BlockCellGradientDescriptors& b) const
{
  return !(this->operator==(b));
}

void bob::ip::base::BlockCellGradientDescriptors::resizeCache()
{
  // Resizes BlockCellDescriptors first
  BlockCellDescriptors::resizeCache();
  // Resizes everything else
  m_gradient_maps->setSize(m_height, m_width);
  m_magnitude.resize(m_height, m_width);
  m_orientation.resize(m_height, m_width);
}

void bob::ip::base::BlockCellGradientDescriptors::resizeCellCache()
{
  // Resizes BlockCellDescriptors first
  BlockCellDescriptors::resizeCellCache();
  // Resizes everything else
  m_cell_magnitude.resize(m_nb_cells_y, m_nb_cells_x, m_cell_y, m_cell_x);
  m_cell_orientation.resize(m_nb_cells_y, m_nb_cells_x, m_cell_y, m_cell_x);
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

bob::ip::base::GradientMaps::GradientMaps(
    const size_t height,
    const size_t width,
    const GradientMagnitudeType mag_type
):
  m_gy(height, width),
  m_gx(height, width),
  m_mag_type(mag_type)
{
}

bob::ip::base::GradientMaps::GradientMaps(const bob::ip::base::GradientMaps& other)
:
  m_gy(other.m_gy.extent(0), other.m_gy.extent(1)),
  m_gx(other.m_gx.extent(0), other.m_gx.extent(1)),
  m_mag_type(other.m_mag_type)
{
}

bob::ip::base::GradientMaps& bob::ip::base::GradientMaps::operator=(const bob::ip::base::GradientMaps& other)
{
  if (this != &other)
  {
    m_gy.resize(other.m_gy.extent(0), other.m_gy.extent(1));
    m_gx.resize(other.m_gx.extent(0), other.m_gx.extent(1));
    m_mag_type = other.m_mag_type;
  }
  return *this;
}

bool bob::ip::base::GradientMaps::operator==(const bob::ip::base::GradientMaps& b) const
{
  return (this->m_gy.extent(0) == b.m_gy.extent(0) &&
          this->m_gy.extent(1) == b.m_gy.extent(1) &&
          this->m_gx.extent(0) == b.m_gx.extent(0) &&
          this->m_gx.extent(1) == b.m_gx.extent(1) &&
          this->m_mag_type == b.m_mag_type);
}

bool bob::ip::base::GradientMaps::operator!=(const bob::ip::base::GradientMaps& b) const
{
  return !(this->operator==(b));
}

void bob::ip::base::GradientMaps::setSize(const size_t height, const size_t width)
{
  m_gy.resize(height,width);
  m_gx.resize(height,width);
}

void bob::ip::base::GradientMaps::setHeight(const size_t height)
{
  m_gy.resize((int)height,m_gy.extent(1));
  m_gx.resize((int)height,m_gx.extent(1));
}

void bob::ip::base::GradientMaps::setWidth(const size_t width)
{
  m_gy.resize(m_gy.extent(0),(int)width);
  m_gx.resize(m_gx.extent(0),(int)width);
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

bob::ip::base::HOG::HOG(
    const size_t height,
    const size_t width,
    const size_t cell_dim,
    const bool full_orientation,
    const size_t cell_y,
    const size_t cell_x,
    const size_t cell_ov_y,
    const size_t cell_ov_x,
    const size_t block_y,
    const size_t block_x,
    const size_t block_ov_y,
    const size_t block_ov_x
):
  BlockCellGradientDescriptors(height, width, cell_dim, cell_y, cell_x, cell_ov_y, cell_ov_x, block_y, block_x, block_ov_y, block_ov_x),
  m_full_orientation(full_orientation)
{
}

bob::ip::base::HOG::HOG(const bob::ip::base::HOG& other)
:
  BlockCellGradientDescriptors(other),
  m_full_orientation(other.m_full_orientation)
{
}

bob::ip::base::HOG& bob::ip::base::HOG::operator=(const bob::ip::base::HOG& other)
{
  if(this != &other)
  {
    BlockCellGradientDescriptors::operator=(other);
    m_full_orientation = other.m_full_orientation;
  }
  return *this;
}

bool bob::ip::base::HOG::operator==(const bob::ip::base::HOG& b) const
{
  return (BlockCellGradientDescriptors::operator==(b) &&
          this->m_full_orientation == b.m_full_orientation);
}

bool bob::ip::base::HOG::operator!=(const bob::ip::base::HOG& b) const
{
  return !(this->operator==(b));
}

void bob::ip::base::HOG::computeHistogram(
  const blitz::Array<double,2>& mag,
  const blitz::Array<double,2>& ori,
  blitz::Array<double,1>& hist
) const
{
  // Checks input arrays
  bob::core::array::assertSameShape(mag, ori);

  static const double range_orientation = (m_full_orientation ? 2*M_PI : M_PI);
  bob::core::array::assertSameShape(hist, blitz::TinyVector<int,1>(m_cell_dim));

  // Initializes output to zero
  hist = 0.;

  for(int i=0; i<mag.extent(0); ++i)
    for(int j=0; j<mag.extent(1); ++j)
    {
      double energy = mag(i,j);
      double orientation = ori(i,j);

      // Computes "real" value of the closest bin
      double bin = orientation / range_orientation * m_cell_dim;
      // Computes the value of the "inferior" bin
      // ("superior" bin corresponds to the one after the inferior bin)
      int bin_index1 = floor(bin);
      // Computes the weight for the "inferior" bin
      double weight = 1.-(bin-bin_index1);

      // Computes integer indices in the range [0,nb_bins-1]
      bin_index1 = bin_index1 % m_cell_dim;
      // Additional check, because bin can be negative (hence bin_index1 as well, as an integer remainder)
      if(bin_index1<0) bin_index1+=m_cell_dim;
      // bin_index1 and nb_bins are positive. Thus, bin_index2 (integer remainder) as well!
      int bin_index2 = (bin_index1+1) % m_cell_dim;

      // Updates the histogram (bilinearly)
      hist(bin_index1) += weight * energy;
      hist(bin_index2) += (1. - weight) * energy;
    }
}

