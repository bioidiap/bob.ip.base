/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Mon Apr 14 20:45:21 CEST 2014
 *
 * @brief Bindings to bob::ip routines
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include "main.h"

static PyMethodDef module_methods[] = {
  {
    s_scale.name(),
    (PyCFunction)PyBobIpBase_scale,
    METH_VARARGS|METH_KEYWORDS,
    s_scale.doc()
  },
  {
    s_scaledOutputShape.name(),
    (PyCFunction)PyBobIpBase_scaledOutputShape,
    METH_VARARGS|METH_KEYWORDS,
    s_scaledOutputShape.doc()
  },
  {
    s_rotate.name(),
    (PyCFunction)PyBobIpBase_rotate,
    METH_VARARGS|METH_KEYWORDS,
    s_rotate.doc()
  },
  {
    s_rotatedOutputShape.name(),
    (PyCFunction)PyBobIpBase_rotatedOutputShape,
    METH_VARARGS|METH_KEYWORDS,
    s_rotatedOutputShape.doc()
  },
  {
    s_maxRectInMask.name(),
    (PyCFunction)PyBobIpBase_maxRectInMask,
    METH_VARARGS|METH_KEYWORDS,
    s_maxRectInMask.doc()
  },
  {
    s_extrapolateMask.name(),
    (PyCFunction)PyBobIpBase_extrapolateMask,
    METH_VARARGS|METH_KEYWORDS,
    s_extrapolateMask.doc()
  },
  {
    s_block.name(),
    (PyCFunction)PyBobIpBase_block,
    METH_VARARGS|METH_KEYWORDS,
    s_block.doc()
  },
  {
    s_blockOutputShape.name(),
    (PyCFunction)PyBobIpBase_blockOutputShape,
    METH_VARARGS|METH_KEYWORDS,
    s_blockOutputShape.doc()
  },
  {
    s_lbphs.name(),
    (PyCFunction)PyBobIpBase_lbphs,
    METH_VARARGS|METH_KEYWORDS,
    s_lbphs.doc()
  },
  {
    s_lbphsOutputShape.name(),
    (PyCFunction)PyBobIpBase_lbphsOutputShape,
    METH_VARARGS|METH_KEYWORDS,
    s_lbphsOutputShape.doc()
  },
  {
    s_integral.name(),
    (PyCFunction)PyBobIpBase_integral,
    METH_VARARGS|METH_KEYWORDS,
    s_integral.doc()
  },
  {
    s_histogram.name(),
    (PyCFunction)PyBobIpBase_histogram,
    METH_VARARGS|METH_KEYWORDS,
    s_histogram.doc()
  },
  {
    s_histogramEqualization.name(),
    (PyCFunction)PyBobIpBase_histogramEqualization,
    METH_VARARGS|METH_KEYWORDS,
    s_histogramEqualization.doc()
  },
  {
    s_gammaCorrection.name(),
    (PyCFunction)PyBobIpBase_gammaCorrection,
    METH_VARARGS|METH_KEYWORDS,
    s_gammaCorrection.doc()
  },
  {
    s_zigzag.name(),
    (PyCFunction)PyBobIpBase_zigzag,
    METH_VARARGS|METH_KEYWORDS,
    s_zigzag.doc()
  },
  {
    s_median.name(),
    (PyCFunction)PyBobIpBase_median,
    METH_VARARGS|METH_KEYWORDS,
    s_median.doc()
  },
  {
    s_sobel.name(),
    (PyCFunction)PyBobIpBase_sobel,
    METH_VARARGS|METH_KEYWORDS,
    s_sobel.doc()
  },
  {0}  // Sentinel
};


PyDoc_STRVAR(module_docstr, "Bob Image Processing Base Routines");

int PyBobIpBase_APIVersion = BOB_IP_BASE_API_VERSION;


#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  if (PyModule_AddStringConstant(module, "__version__", BOB_EXT_MODULE_VERSION) < 0) return 0;
  if (!init_BobIpBaseGeomNorm(module)) return 0;
  if (!init_BobIpBaseFaceEyesNorm(module)) return 0;
  if (!init_BobIpBaseLBP(module)) return 0;
  if (!init_BobIpBaseLBPTop(module)) return 0;
  if (!init_BobIpBaseDCTFeatures(module)) return 0;
  if (!init_BobIpBaseTanTriggs(module)) return 0;
  if (!init_BobIpBaseGaussian(module)) return 0;
  if (!init_BobIpBaseMultiscaleRetinex(module)) return 0;
  if (!init_BobIpBaseWeightedGaussian(module)) return 0;
  if (!init_BobIpBaseSelfQuotientImage(module)) return 0;
  if (!init_BobIpBaseGaussianScaleSpace(module)) return 0;
  if (!init_BobIpBaseSIFT(module)) return 0;
  if (!init_BobIpBaseHOG(module)) return 0;
  if (!init_BobIpBaseGLCM(module)) return 0;
  if (!init_BobIpBaseWiener(module)) return 0;

#if HAVE_VLFEAT
  if (!init_BobIpBaseVLFEAT(module)) return 0;
#endif // HAVE_VLFEAT


  static void* PyBobIpBase_API[PyBobIpBase_API_pointers];

  /* exhaustive list of C APIs */

  /**************
   * Versioning *
   **************/

  PyBobIpBase_API[PyBobIpBase_APIVersion_NUM] = (void *)&PyBobIpBase_APIVersion;

  /********************************
   * Bindings for bob.ip.base.LBP *
   ********************************/

  PyBobIpBase_API[PyBobIpBaseLBP_Type_NUM] = (void *)&PyBobIpBaseLBP_Type;
  PyBobIpBase_API[PyBobIpBaseLBP_Check_NUM] = (void *)&PyBobIpBaseLBP_Check;
  PyBobIpBase_API[PyBobIpBaseLBP_Converter_NUM] = (void *)&PyBobIpBaseLBP_Converter;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyBobIpBase_API,
      BOB_EXT_MODULE_PREFIX "." BOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyBobIpBase_API, 0);

#endif

  if (!c_api_object) return 0;

  if (PyModule_AddObject(module, "_C_API", c_api_object) < 0) return 0;


  /* imports bob.ip.base's C-API dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_random() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;
  if (import_bob_sp() < 0) return 0;

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
