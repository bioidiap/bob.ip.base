/**
 * @author André Anjos <andre.anjos@idiap.ch>
 * @date Tue Jan 18 17:07:26 2011 +0100
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/config.h"
#include "ndarray.h"

/** extra bindings required for compatibility **/
void bind_core_tinyvector();
void bind_core_ndarray_numpy();
void bind_core_bz_numpy();
void bind_sp_extrapolate();
void bind_sp_convolution();

void bind_ip_shear();
void bind_ip_hog();
void bind_ip_glcm_uint8();
void bind_ip_glcm_uint16();
void bind_ip_glcmprop();

BOOST_PYTHON_MODULE(_old_library) {

  boost::python::docstring_options docopt(true, true, false);

  bob::python::setup_python("old-style bob image processing classes and sub-classes");

  /** extra bindings required for compatibility **/
  bind_core_tinyvector();
  bind_core_ndarray_numpy();
  bind_core_bz_numpy();
  bind_sp_extrapolate();
  bind_sp_convolution();

  bind_ip_shear();
  bind_ip_hog();
  bind_ip_glcm_uint8();
  bind_ip_glcm_uint16();
  bind_ip_glcmprop();

}
