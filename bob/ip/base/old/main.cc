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

void bind_ip_block();
void bind_ip_crop_shift();
void bind_ip_extrapolate_mask();
void bind_ip_flipflop();
void bind_ip_gamma_correction();
void bind_ip_integral();
void bind_ip_scale();
void bind_ip_shear();
void bind_ip_zigzag();
void bind_ip_rotate();
void bind_ip_dctfeatures();
void bind_ip_geomnorm();
void bind_ip_faceeyesnorm();
void bind_ip_tantriggs();
void bind_ip_histogram();
void bind_ip_lbp();
void bind_ip_gaussian();
void bind_ip_gaussian_scale_space();
void bind_ip_wgaussian();
void bind_ip_msr();
void bind_ip_sqi();
void bind_ip_median();
void bind_ip_sobel();
void bind_ip_hog();
void bind_ip_glcm_uint8();
void bind_ip_glcm_uint16();
void bind_ip_glcmprop();
void bind_ip_sift();

#if WITH_VLFEAT
void bind_ip_vlsift();
void bind_ip_vldsift();
#endif

BOOST_PYTHON_MODULE(_old_library) {

  boost::python::docstring_options docopt(true, true, false);

  bob::python::setup_python("old-style bob image processing classes and sub-classes");

  /** extra bindings required for compatibility **/
  bind_core_tinyvector();
  bind_core_ndarray_numpy();
  bind_core_bz_numpy();
  bind_sp_extrapolate();
  bind_sp_convolution();

  bind_ip_block();
  bind_ip_crop_shift();
  bind_ip_extrapolate_mask();
  bind_ip_flipflop();
  bind_ip_gamma_correction();
  bind_ip_integral();
  bind_ip_scale();
  bind_ip_shear();
  bind_ip_zigzag();
  bind_ip_rotate();
  bind_ip_dctfeatures();
  bind_ip_geomnorm();
  bind_ip_faceeyesnorm();
  bind_ip_tantriggs();
  bind_ip_histogram();
  bind_ip_lbp();
  bind_ip_gaussian();
  bind_ip_gaussian_scale_space();
  bind_ip_wgaussian();
  bind_ip_msr();
  bind_ip_sqi();
  bind_ip_median();
  bind_ip_sobel();
  bind_ip_hog();
  bind_ip_glcm_uint8();
  bind_ip_glcm_uint16();
  bind_ip_glcmprop();
  bind_ip_sift();

#if WITH_VLFEAT
  bind_ip_vlsift();
  bind_ip_vldsift();
#endif

}
