/**
 * @date Tue Apr 5 12:38:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to perform a decomposition by block.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_BLOCK_H
#define BOB_IP_BASE_BLOCK_H

#include <bob.core/assert.h>

namespace bob { namespace ip { namespace base {

  /**
    * @brief Function which performs a block decomposition of a 2D
    *   blitz::array/image of a given type, and put the results in a
    *   3D blitz::array/image
    */
  template<typename T>
  inline void _blockNoCheck(
    const blitz::Array<T,2>& src,
    blitz::Array<T,3>& dst,
    const int block_h, const int block_w,
    const int overlap_h, const int overlap_w
  ){
    // Determine the number of block per row and column
    const int size_ov_h = block_h - overlap_h;
    const int size_ov_w = block_w - overlap_w;
    const int n_blocks_h = (src.extent(0) - (int)overlap_h) / size_ov_h;
    const int n_blocks_w = (src.extent(1) - (int)overlap_w) / size_ov_w;

    // Perform the block decomposition
    for (int h = 0; h < n_blocks_h; ++h)
      for (int w = 0; w < n_blocks_w; ++w)
        dst(h * n_blocks_w + w, blitz::Range::all(), blitz::Range::all()) = src(blitz::Range(h * size_ov_h, h * size_ov_h + block_h - 1), blitz::Range(w * size_ov_w, w * size_ov_w + block_w - 1));
  }


  /**
    * @brief Function which performs a block decomposition of a 2D
    *   blitz::array/image of a given type
    */
  template<typename T>
  inline void _blockNoCheck(
    const blitz::Array<T,2>& src,
    blitz::Array<T,4>& dst,
    const size_t block_h, const size_t block_w,
    const size_t overlap_h,const size_t overlap_w
  ){
    // Determine the number of block per row and column
    const int size_ov_h = block_h - overlap_h;
    const int size_ov_w = block_w - overlap_w;
    const int n_blocks_h = (src.extent(0)-(int)overlap_h) / size_ov_h;
    const int n_blocks_w = (src.extent(1)-(int)overlap_w) / size_ov_w;

    // Perform the block decomposition
    blitz::Array<bool,2> src_mask, dst_mask;
    for (int h = 0; h < n_blocks_h; ++h)
      for (int w = 0; w < n_blocks_w; ++w)
        dst(h, w, blitz::Range::all(), blitz::Range::all()) = src(blitz::Range(h * size_ov_h, h * size_ov_h + block_h - 1), blitz::Range(w * size_ov_w, w * size_ov_w + block_w - 1));
  }

  /**
    * @brief Function which checks the given parameters for a block
    *   decomposition of a 2D blitz::array/image.
    */
  inline void _blockCheckInput(
    const size_t height,
    const size_t width,
    const size_t block_h,
    const size_t block_w,
    const size_t overlap_h,
    const size_t overlap_w
  ){
    // Check parameters and throw exception if required
    if (block_h < 1 || block_h > height) throw std::runtime_error((boost::format("setting `block_h' to %lu is outside the expected range [1, %lu]") % block_h % height).str());
    if (block_w < 1 || block_w > width) throw std::runtime_error((boost::format("setting `block_w' to %lu is outside the expected range [1, %lu]") % block_w % width).str());
    if (overlap_h >= block_h) throw std::runtime_error((boost::format("setting `overlap_h' to %lu is outside the expected range [0, %lu]") % overlap_h % (block_h-1)).str());
    if (overlap_w >= block_w) throw std::runtime_error((boost::format("setting `overlap_w' to %lu is outside the expected range [0, %lu]") % overlap_w % (block_w-1)).str());
  }


  /**
    * @brief Function which returns the expected shape of the output
    *   3D blitz array when applying a decomposition by block of a
    *   2D blitz::array/image of a given size.
    *   Dimensions are returned in a 3D TinyVector as
    *   (N_blocks,block_h,block_w)
    * @param height  The height of the input array
    * @param width   The width of the input array
    * @param block_h The desired height of the blocks.
    * @param block_w The desired width of the blocks.
    * @param overlap_h The overlap between each block along the y axis.
    * @param overlap_w The overlap between each block along the x axis.
    */
  inline const blitz::TinyVector<int,3> getBlock3DOutputShape(
    const size_t height, const size_t width,
    const size_t block_h, const size_t block_w,
    const size_t overlap_h, const size_t overlap_w
  ){
    // Determine the number of block per row and column
    const int size_ov_h = block_h - overlap_h;
    const int size_ov_w = block_w - overlap_w;
    const int n_blocks_h = (int)(height-overlap_h) / size_ov_h;
    const int n_blocks_w = (int)(width-overlap_w) / size_ov_w;

    // Return the shape of the output
    return blitz::TinyVector<int,3> (n_blocks_h * n_blocks_w, block_h, block_w);
  }

  /**
    * @brief Function which returns the expected shape of the output
    *   4D blitz array when applying a decomposition by block of a
    *   2D blitz::array/image of a given size.
    *   Dimensions are returned in a 4D TinyVector as
    *   (N_blocks_y,N_blocks_x,block_h,block_w)
    * @param height  The height of the input array
    * @param width   The width of the input array
    * @param block_h The desired height of the blocks.
    * @param block_w The desired width of the blocks.
    * @param overlap_h The overlap between each block along the y axis.
    * @param overlap_w The overlap between each block along the x axis.
    */
  inline const blitz::TinyVector<int,4> getBlock4DOutputShape(
    const size_t height, const size_t width,
    const size_t block_h, const size_t block_w,
    const size_t overlap_h, const size_t overlap_w
  ){
    // Determine the number of block per row and column
    const int size_ov_h = block_h - overlap_h;
    const int size_ov_w = block_w - overlap_w;
    const int n_blocks_h = (int)(height-overlap_h) / size_ov_h;
    const int n_blocks_w = (int)(width-overlap_w) / size_ov_w;

    // Return the shape of the output
    return blitz::TinyVector<int,4> (n_blocks_h, n_blocks_w, block_h, block_w);
  }

  /**
    * @brief Function which performs a decomposition by block of a 2D
    *   blitz::array/image of a given type.
    *   The first dimension is the height (y-axis), whereas the second
    *   one is the width (x-axis).
    * @param src The input blitz array
    * @param dst The output 3D blitz arrays. The first coordinate is for the
    *   block index, and the second two are coordinates inside the blocks.
    * @param block_h The desired height of the blocks.
    * @param block_w The desired width of the blocks.
    * @param overlap_h The overlap between each block along the y axis.
    * @param overlap_w The overlap between each block along the x axis.
    */
  template<typename T>
  inline void block(
    const blitz::Array<T,2>& src,
    blitz::Array<T,3>& dst,
    const size_t block_h, const size_t block_w,
    const size_t overlap_h, const size_t overlap_w
  ){
    // Check input
    _blockCheckInput(src.extent(0), src.extent(1), block_h, block_w, overlap_h, overlap_w);
    auto shape = getBlock3DOutputShape(src.extent(0), src.extent(1), block_h, block_w, overlap_h, overlap_w);
    bob::core::array::assertSameShape(dst, shape);
    // Crop the 2D array
    _blockNoCheck(src, dst, block_h, block_w, overlap_h, overlap_w);
  }

  /**
    * @brief Function which performs a decomposition by block of a 2D
    *   blitz::array/image of a given type.
    *   The first dimension is the height (y-axis), whereas the second
    *   one is the width (x-axis).
    * @param src The input blitz array
    * @param dst The output 4D blitz arrays. The first coordinates are for
    *   y- and x-block indices, and the last two are coordinates inside the
    *   blocks.
    * @param block_h The desired height of the blocks.
    * @param block_w The desired width of the blocks.
    * @param overlap_h The overlap between each block along the y axis.
    * @param overlap_w The overlap between each block along the x axis.
    */
  template<typename T>
  inline void block(
    const blitz::Array<T,2>& src,
    blitz::Array<T,4>& dst,
    const size_t block_h, const size_t block_w,
    const size_t overlap_h, const size_t overlap_w
  ){
    // Check input
    _blockCheckInput(src.extent(0), src.extent(1), block_h, block_w, overlap_h, overlap_w);
    blitz::TinyVector<int,4> shape = getBlock4DOutputShape(src.extent(0), src.extent(1), block_h, block_w, overlap_h, overlap_w);
    bob::core::array::assertSameShape(dst, shape);
    // Crop the 2D array
    _blockNoCheck(src, dst, block_h, block_w, overlap_h, overlap_w);
  }


  /**
    * @brief Function which performs a decomposition by block of a 2D
    *   blitz::array/image of a given type.
    *   The first dimension is the height (y-axis), whereas the second
    *   one is the width (x-axis).
    * @warning The returned blocks will refer to the same data as the in
    *   input 2D blitz array.
    * @param src The input blitz array
    * @param dst The STL container of 2D block blitz arrays. The STL
    *   container requires to support the push_back method, such as
    *   a STL vector or list.
    * @param block_w The desired width of the blocks.
    * @param block_h The desired height of the blocks.
    * @param overlap_w The overlap between each block along the x axis.
    * @param overlap_h The overlap between each block along the y axis.
    */
  template<typename T>
  std::vector<blitz::Array<T,2>> blockReference(
    const blitz::Array<T,2>& src,
    const size_t block_h, const size_t block_w,
    const size_t overlap_h, const size_t overlap_w
  ){
    // Check input
    _blockCheckInput(src.extent(0), src.extent(1), block_h, block_w, overlap_h, overlap_w);

    // Determine the number of block per row and column
    const int size_ov_h = block_h - overlap_h;
    const int size_ov_w = block_w - overlap_w;
    const int n_blocks_h = (src.extent(0) - (int)overlap_h) / size_ov_h;
    const int n_blocks_w = (src.extent(1) - (int)overlap_w) / size_ov_w;

    // create list of blocks
    std::vector<blitz::Array<T,2>> blocks(n_blocks_h * n_blocks_w);

    // Perform the block decomposition
    for (int h = 0, i = 0; h < n_blocks_h; ++h)
      for (int w = 0; w < n_blocks_w; ++w, ++i)
        blocks[i].reference(src(blitz::Range(h * size_ov_h, h * size_ov_h + block_h - 1), blitz::Range(w * size_ov_w, w * size_ov_w + block_w - 1)));
    return blocks;
  }

} } } // namespaces

#endif /* BOB_IP_BASE_BLOCK_H */
