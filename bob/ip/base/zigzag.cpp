/**
* @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
* @date Mon Apr 14 21:09:43 CEST 2014
*
* @brief Binds zigzag to python
*
* Copyright (C) 2014 Idiap Research Institute, Martigny, Switzerland
*/

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob/ip/zigzag.h>


template <typename T> PyObject* inner_zigzag(PyBlitzArrayObject* src,
    PyBlitzArrayObject* dst,
    PyObject* bf) {

  //converts value into a proper scalar
  bool c_bf = false;
  if (bf) {
    c_bf = PyBlitzArrayCxx_AsCScalar<bool>(bf);
    if (PyErr_Occurred()) return 0;
  }

  try {
    bob::ip::zigzag(*PyBlitzArrayCxx_AsBlitz<T,2>(src),
        *PyBlitzArrayCxx_AsBlitz<T,1>(dst), c_bf);
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "%s", e.what());
    return 0;
  }

  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "caught unknown exception while calling C++ bob::spp::extrapolate");
    return 0;
  }

  Py_RETURN_NONE;

}

PyObject* PyBobIpBase_zigzag(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "src",
    "dst",
    "right_first",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* src = 0;
  PyBlitzArrayObject* dst = 0;
  PyObject* bf = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|O",
        kwlist,
        &PyBlitzArray_Converter, &src,
        &PyBlitzArray_OutputConverter, &dst,
        &bf)) return 0;
  auto src_ = make_safe(src);
  auto dst_ = make_safe(dst);

  if (src->type_num != dst->type_num) {
    PyErr_Format(PyExc_TypeError, "source and destination arrays must have the same data types (src: `%s' != dst: `%s')",
        PyBlitzArray_TypenumAsString(src->type_num),
        PyBlitzArray_TypenumAsString(dst->type_num));
    return 0;
  }

  if (src->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "source array must have 2 dimensions types (src_given: `%" PY_FORMAT_SIZE_T "d' != src_expected: 2d')", src->ndim);
    return 0;
  }

  if (dst->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "destination array must have 1 dimension type (dst_given: `%" PY_FORMAT_SIZE_T "d' != dst_expected: 1d')", dst->ndim);
    return 0;
  }

  switch (src->type_num) {
      return inner_zigzag<uint8_t>(src, dst, bf);
    case NPY_UINT16:
      return inner_zigzag<uint16_t>(src, dst, bf);
    case NPY_FLOAT64:
      return inner_zigzag<double>(src, dst, bf);
    default:
      PyErr_Format(PyExc_TypeError, "zigzag from `%s' (%d) is not supported", PyBlitzArray_TypenumAsString(src->type_num), src->type_num);
  }

  return 0;

}


