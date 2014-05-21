/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue Sep 27 23:26:46 2011 +0200
 *
 * @brief Binds extrapolation to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "ndarray.h"
#include <bob/sp/extrapolate.h>

using namespace boost::python;

void bind_sp_extrapolate()
{

  enum_<bob::sp::Extrapolation::BorderType>("BorderType")
    .value("Zero", bob::sp::Extrapolation::Zero)
    .value("Constant", bob::sp::Extrapolation::Constant)
    .value("NearestNeighbour", bob::sp::Extrapolation::NearestNeighbour)
    .value("Circular", bob::sp::Extrapolation::Circular)
    .value("Mirror", bob::sp::Extrapolation::Mirror)
    ;

}
