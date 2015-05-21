/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri  1 Nov 07:10:59 2013
 *
 * @brief General directives for all modules in bob.io
 */

#ifndef BOB_IP_BASE_CONFIG_H
#define BOB_IP_BASE_CONFIG_H

/* Macros that define versions and important names */
#define BOB_IP_BASE_API_VERSION 0x0200

#ifdef BOB_IMPORT_VERSION

  /***************************************
  * Here we define some functions that should be used to build version dictionaries in the version.cpp file
  * There will be a compiler warning, when these functions are not used, so use them!
  ***************************************/

  #include <Python.h>
  #include <boost/preprocessor/stringize.hpp>

  /**
   * VLFeat version
   */
  #if HAVE_VLFEAT
  #include <vl/generic.h>
  static PyObject* vlfeat_version() {
    return Py_BuildValue("s", VL_VERSION_STRING);
  }
  #endif // HAVE_VLFEAT

  /**
   * bob.ip.base c/c++ api version
   */
  static PyObject* bob_ip_base_version() {
    return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_IP_BASE_API_VERSION));
  }

#endif // BOB_IMPORT_VERSION

#endif /* BOB_IP_BASE_CONFIG_H */
