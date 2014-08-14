/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Wed Jun 25 19:10:30 CEST 2014
 *
 * @brief This file defines a function to compute local binary pattern histogram sequences.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_LBPHS_H
#define BOB_IP_BASE_LBPHS_H

#include <bob.core/assert.h>
#include <bob.core/array_index.h>

#include <bob.ip.base/LBP.h>
#include <bob.ip.base/Block.h>
#include <bob.ip.base/Histogram.h>

namespace bob { namespace ip { namespace base {

  /**
    * @brief Process a 2D blitz Array/Image by extracting LBPHS features.
    * @param src The 2D input blitz array
    * @param dst A container (with a push_back method such as an STL list)
    *   of 1D uint32_t blitz arrays.
    */
  template <typename T>
  void lbphs(
    const blitz::Array<T,2>& src,
    const LBP& lbp,
    const blitz::TinyVector<int,2>& block_size,
    const blitz::TinyVector<int,2>& block_overlap,
    blitz::Array<uint64_t,2> dst)
  {
    // extract LBP features
    blitz::Array<uint16_t, 2> lbp_image(lbp.getLBPShape(src.shape()));
    lbp.extract_(src, lbp_image);

    // get all the blocks
    auto blocks = blockReference(lbp_image, block_size[0], block_size[1], block_overlap[0], block_overlap[1]);

    if (dst.extent(0) != (int)blocks.size() || dst.extent(1) != (int)lbp.getMaxLabel()){
      throw std::runtime_error((boost::format("The given output image needs to be of size (%d, %d), but has shape (%d, %d)") % blocks.size() % lbp.getMaxLabel() % dst.extent(0) % dst.extent(1)).str());
    }

    // compute an lbp histogram for each block
    int i = 0;
    for (auto it = blocks.begin(); it != blocks.end(); ++it, ++i)
    {
      // Compute the LBP histogram
      blitz::Array<uint64_t, 1> block_histogram(dst(i, blitz::Range::all()));
      bob::ip::base::histogram<uint16_t>(*it, block_histogram, 0, lbp.getMaxLabel()-1);
    }
  }

} } } // namespaces

#endif /* BOB_IP_BASE_LBPHS_H */
