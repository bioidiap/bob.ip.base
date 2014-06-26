/**
 * @author Manuel Günther <manuel.guenther@idiap.ch>
 * @date Thu Jun 26 09:33:10 CEST 2014
 *
 * This file defines functions and classes for affine image transformations
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_AFFINE_H
#define BOB_IP_BASE_AFFINE_H

#include <boost/shared_ptr.hpp>
#include "bob/core/assert.h"
#include "bob/core/check.h"


namespace bob { namespace ip { namespace base {

    /**
      * @brief Function which extracts a rectangle of maximal area from a
      *   2D mask of booleans (i.e. a 2D blitz array).
      * @warning The function assumes that the true values on the mask form
      *   a convex area.
      * @param mask The 2D input blitz array mask.
      * @result A blitz::TinyVector which contains in the following order:
      *   0/ The y-coordinate of the top left corner
      *   1/ The x-coordinate of the top left corner
      *   2/ The height of the rectangle
      *   3/ The width of the rectangle
      */
    const blitz::TinyVector<int,4> maxRectInMask(const blitz::Array<bool,2>& mask);

} } } // namespaces

#endif // BOB_IP_BASE_AFFINE_H

