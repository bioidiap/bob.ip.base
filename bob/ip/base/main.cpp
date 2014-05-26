/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Mon Apr 14 20:45:21 CEST 2014
 *
 * @brief Bindings to bob::ip routines
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>

extern PyObject* PyBobIpBase_zigzag(PyObject*, PyObject*, PyObject*);
static bob::extension::FunctionDoc s_zigzag = bob::extension::FunctionDoc(
    "zigzag",

    "Extracts a 1D array using a zigzag pattern from a 2D array",

    "This function extracts a 1D array using a zigzag pattern from a 2D array. "
    "If bottom_first is set to True, the second element of the pattern "
    "is taken at the bottom of the upper left element, otherwise it is "
    "taken at the right of the upper left element. "
    "\n"
    "The input is expected to be a 2D dimensional array. "
    "The output is expected to be a 1D dimensional array. "
    "\n"
    " This method only supports arrays of the following data types:\n"
    "\n"
    " * :py:class:`numpy.uint8`\n"
    " * :py:class:`numpy.uint16`\n"
    " * :py:class:`numpy.float64` (or the native python ``float``)\n"
    " \n"
    " To create an object with a scalar type that will be accepted by this "
    " method, use a construction like the following:\n"
    " \n"
    " .. code-block:: python\n"
    " \n"
    " >> import numpy\n"
    " >> input_righttype = input_wrongtype.astype(numpy.float64)"
    )

    .add_prototype("src, dst, bf")
    .add_parameter("src", "array_like (uint8|uint16|float64, 2D)", "The source matrix.")
    .add_parameter("dst", "array_like (uint8|uint16|float64, 1D)", "The destination matrix.")
    .add_parameter("right_first", "scalar (bool)", "Tells whether the zigzag pattern start to move to the right or not")
;


static PyMethodDef module_methods[] = {
    {
      s_zigzag.name(),
      (PyCFunction)PyBobIpBase_zigzag,
      METH_VARARGS|METH_KEYWORDS,
      s_zigzag.doc()
    },
    {0}  // Sentinel
};


PyDoc_STRVAR(module_docstr, "Bob Image Processing Base Routines");

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
  PyObject* m = PyModule_Create(&module_definition);
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m); ///< protects against early returns

  if (PyModule_AddStringConstant(m, "__version__", BOB_EXT_MODULE_VERSION) < 0)
    return 0;

  /* imports bob.blitz C-API + dependencies */
  if (import_bob_blitz() < 0) return 0;

  Py_INCREF(m);
  return m;

}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
