/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Wed Jun 25 18:28:03 CEST 2014
 *
 * @brief Binds auxiliary functions of bob::ip::base class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


#include "main.h"
#include <bob.ip.base/IntegralImage.h>
#include <bob.ip.base/LBPHS.h>
#include <bob.ip.base/Histogram.h>
#include <bob.ip.base/ZigZag.h>

static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns false if object is NULL */


bob::extension::FunctionDoc s_histogram = bob::extension::FunctionDoc(
  "histogram",
  "Computes an histogram of the given input image",
  "This function computes a histogram of the given input image, in several ways.\n\n"
  "* (version 1 and 2, only valid for uint8 and uint16 types -- and uint32 and uint64 when ``bin_count`` is specified or ``hist`` is given as parameter): For each pixel value of the ``src`` image, a histogram bin is computed, using a fast implementation. "
  "The number of bins can be limited, and there will be a check that the source image pixels are actually in the desired range ``(0, bin_count-1)``\n\n"
  "* (version 3 and 4, valid for many data types): The histogram is computed by defining regular bins between the provided minimum and maximum values."
)
.add_prototype("src, [bin_count]", "hist")
.add_prototype("src, hist")
.add_prototype("src, min_max, bin_count", "hist")
.add_prototype("src, min_max, hist")
.add_parameter("src", "array_like (2D)", "The source image to compute the histogram for")
.add_parameter("hist", "array_like (1D, uint64)", "The histogram with the desired number of bins; the histogram will be cleaned before running the extraction")
.add_parameter("min_max", "(scalar, scalar)", "The minimum value and the maximum value in the source image")
.add_parameter("bin_count", "int", "[default: 256 or 65536] The number of bins in the histogram to create, defaults to the maximum number of values")
.add_return("hist", "array_like(2D, uint64)", "The histogram with the desired number of bins, which is filled with the histogrammed source data")
;

template <typename T, char C> bool inner_histogram(PyBlitzArrayObject* src, PyBlitzArrayObject* hist, PyObject* min_max) {
  std::string format = (boost::format("%1%%1%") % C).str();
  T min, max;
  if (!PyArg_ParseTuple(min_max, format.c_str(), &min, &max)) {
    return false;
  }
  bob::ip::base::histogram(*PyBlitzArrayCxx_AsBlitz<T, 2>(src), *PyBlitzArrayCxx_AsBlitz<uint64_t, 1>(hist), min, max);
  return true;
}

template <typename T, char C> void inner_histogram(PyBlitzArrayObject* src, PyBlitzArrayObject* hist){
  bob::ip::base::histogram(*PyBlitzArrayCxx_AsBlitz<T, 2>(src), *PyBlitzArrayCxx_AsBlitz<uint64_t, 1>(hist));
}

PyObject* PyBobIpBase_histogram(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist1 = s_histogram.kwlist(0);
  char** kwlist2 = s_histogram.kwlist(1);
  char** kwlist3 = s_histogram.kwlist(2);
  char** kwlist4 = s_histogram.kwlist(3);

  PyBlitzArrayObject* src = 0,* hist = 0;
  PyObject* min_max = 0;
  int bins = 0;

  auto src_ = make_xsafe(src), hist_ = make_xsafe(hist);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch (nargs){
    case 1:{
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|i", kwlist1, &PyBlitzArray_Converter, &src, &bins)) return 0;
      // get the number of bins
      src_ = make_safe(src);
      break;
    }
    case 2:{
      PyObject* k = Py_BuildValue("s", kwlist1[1]);
      auto k_ = make_safe(k);
      if ((args && PyTuple_Size(args) == 2 && PyInt_Check(PyTuple_GET_ITEM(args,1))) || (kwargs && PyDict_Contains(kwargs, k))){
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|i", kwlist1, &PyBlitzArray_Converter, &src, &bins)) return 0;
      } else {
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&", kwlist2, &PyBlitzArray_Converter, &src, &PyBlitzArray_OutputConverter, &hist)) return 0;
      }
      src_ = make_safe(src);
      hist_ = make_xsafe(hist);
      break;
    }
    case 3:{
      // get values for min and max
      PyObject* k = Py_BuildValue("s", kwlist3[2]);
      auto k_ = make_safe(k);
      if ((args && PyTuple_Size(args) == 3 && PyInt_Check(PyTuple_GET_ITEM(args,2))) || (kwargs && PyDict_Contains(kwargs, k))){
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&Oi", kwlist3, &PyBlitzArray_Converter, &src, &min_max, &bins)) return 0;
      } else {
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&OO&", kwlist4, &PyBlitzArray_Converter, &src, &min_max, &PyBlitzArray_OutputConverter, &hist)) return 0;
      }
      src_ = make_safe(src);
      hist_ = make_xsafe(hist);
      break;
    }
    default:
      PyErr_Format(PyExc_ValueError, "'histogram' called with an unsupported number of arguments");
      return 0;
  }

  // check input size
  if (src->ndim != 2){
    PyErr_Format(PyExc_TypeError, "'histogram' : The input image must be 2D.");
    return 0;
  }

  // allocate the output, if needed
  bool return_out = false;
  if (!hist){
    return_out = true;
    // first, get the number of bins
    if (!bins){
      if (src->type_num == NPY_UINT8) bins = std::numeric_limits<uint8_t>::max() + 1;
      else if (src->type_num == NPY_UINT16) bins = std::numeric_limits<uint16_t>::max() + 1;
      else {
        PyErr_Format(PyExc_TypeError, "'histogram' : The given input data type %s is not supported, when no bin count is specified.", PyBlitzArray_TypenumAsString(src->type_num));
        return 0;
      }
    }
    Py_ssize_t n[] = {bins};
    hist = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_UINT64, 1, n));
    hist_ = make_safe(hist);
  } else {
    if (hist->type_num != NPY_UINT64){
      PyErr_Format(PyExc_TypeError, "'histogram' : The given hist data type %s is not supported, only uint64 is allowed.", PyBlitzArray_TypenumAsString(src->type_num));
      return 0;
    }
  }

  // now, get the histogram running
  bool res = true;
  if (min_max){
    switch (src->type_num){
      case NPY_UINT8:    res = inner_histogram<uint8_t, 'B'>(src, hist, min_max); break;
      case NPY_UINT16:   res = inner_histogram<uint16_t, 'H'>(src, hist, min_max); break;
      case NPY_UINT32:   res = inner_histogram<uint32_t, 'I'>(src, hist, min_max); break;
      case NPY_UINT64:   res = inner_histogram<uint64_t, 'K'>(src, hist, min_max); break;
      case NPY_INT8:     res = inner_histogram<int8_t, 'b'>(src, hist, min_max); break;
      case NPY_INT16:    res = inner_histogram<int16_t, 'h'>(src, hist, min_max); break;
      case NPY_INT32:    res = inner_histogram<int32_t, 'i'>(src, hist, min_max); break;
      case NPY_INT64:    res = inner_histogram<int64_t, 'L'>(src, hist, min_max); break;
      case NPY_FLOAT32:  res = inner_histogram<float, 'f'>(src, hist, min_max); break;
      case NPY_FLOAT64:  res = inner_histogram<double, 'd'>(src, hist, min_max); break;
      default:
        PyErr_Format(PyExc_TypeError, "'histogram' : The given input data type %s is not supported.", PyBlitzArray_TypenumAsString(src->type_num));
        return 0;
    }
  } else {
    switch (src->type_num){
      case NPY_UINT8:    inner_histogram<uint8_t, 'B'>(src, hist); break;
      case NPY_UINT16:   inner_histogram<uint16_t, 'H'>(src, hist); break;
      case NPY_UINT32:   inner_histogram<uint32_t, 'I'>(src, hist); break;
      case NPY_UINT64:   inner_histogram<uint64_t, 'K'>(src, hist); break;
      default:
        PyErr_Format(PyExc_TypeError, "'histogram' : The given input data type %s is not supported.", PyBlitzArray_TypenumAsString(src->type_num));
        return 0;
    }
  }
  if (!res) return 0;

  // return the histogram, if wanted
  if (return_out){
    return PyBlitzArray_AsNumpyArray(hist, 0);
  } else {
    Py_RETURN_NONE;
  }

  BOB_CATCH_FUNCTION("in histogram", 0)
}


bob::extension::FunctionDoc s_histogramEqualization = bob::extension::FunctionDoc(
  "histogram_equalization",
  "Performs a histogram equalization of a given 2D image",
  "The first version computes the normalization **in-place** (in opposition to the old implementation, which returned a equalized image), while the second version fills the given ``dst`` array and leaves the input untouched."
)
.add_prototype("src")
.add_prototype("src, dst")
.add_parameter("src", "array_like (2D, uint8 or uint16)", "The source image to compute the histogram for")
.add_parameter("dst", "array_like (2D, uint8, uint16, uint32 or float)", "The histogram-equalized image to write; if not specified, the equalization is computed **in-place**.")
;

template <typename T1, typename T2> PyObject* inner_histogramEq(PyBlitzArrayObject* src, PyBlitzArrayObject* dst) {
  bob::ip::base::histogramEqualize(*PyBlitzArrayCxx_AsBlitz<T1, 2>(src), *PyBlitzArrayCxx_AsBlitz<T2, 2>(dst));
  Py_RETURN_NONE;
}

PyObject* PyBobIpBase_histogramEqualization(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist1 = s_histogramEqualization.kwlist(0);
  char** kwlist2 = s_histogramEqualization.kwlist(1);

  PyBlitzArrayObject* src = 0,* dst = 0;

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch (nargs){
    case 1:{
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist1, &PyBlitzArray_OutputConverter, &src)) return 0;
      break;
    }
    case 2:{
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&", kwlist2, &PyBlitzArray_Converter, &src, &PyBlitzArray_OutputConverter, &dst)) return 0;
      break;
    }
    default:
      PyErr_Format(PyExc_ValueError, "'histogram_equalization' called with an unsupported number of arguments");
      return 0;
  }

  auto src_ = make_safe(src), dst_ = make_xsafe(dst);

  // perform some checks
  if (src->ndim != 2 || (dst && dst->ndim != 2)){
    PyErr_Format(PyExc_ValueError, "'histogram_equalization' can be performed on 2D arrays only");
    return 0;
  }

  // in-place
  switch (src->type_num){
    case NPY_UINT8:{
      if (!dst) return inner_histogramEq<uint8_t, uint8_t>(src, src);
      switch (dst->type_num){
        case NPY_UINT8:  return inner_histogramEq<uint8_t, uint8_t>(src, dst);
        case NPY_UINT16: return inner_histogramEq<uint8_t, uint16_t>(src, dst);
        case NPY_UINT32: return inner_histogramEq<uint8_t, uint32_t>(src, dst);
        case NPY_FLOAT32: return inner_histogramEq<uint8_t, float>(src, dst);
        case NPY_FLOAT64: return inner_histogramEq<uint8_t, double>(src, dst);
        default:
          PyErr_Format(PyExc_ValueError, "'histogram_equalization' can be performed to uint8, uint16, uint32, float32 or float64 arrays, but not to %s", PyBlitzArray_TypenumAsString(dst->type_num));
          return 0;
      }
    }
    case NPY_UINT16:{
      if (!dst) return inner_histogramEq<uint16_t, uint16_t>(src, src);
      switch (dst->type_num){
        case NPY_UINT8:  return inner_histogramEq<uint16_t, uint8_t>(src, dst);
        case NPY_UINT16: return inner_histogramEq<uint16_t, uint16_t>(src, dst);
        case NPY_UINT32: return inner_histogramEq<uint16_t, uint32_t>(src, dst);
        case NPY_FLOAT32: return inner_histogramEq<uint16_t, float>(src, dst);
        case NPY_FLOAT64: return inner_histogramEq<uint16_t, double>(src, dst);
        default:
          PyErr_Format(PyExc_ValueError, "'histogram_equalization' can be performed to uint8, uint16, uint32, float32 or float64 arrays, but not to %s", PyBlitzArray_TypenumAsString(dst->type_num));
          return 0;
      }
    }
    default:
      PyErr_Format(PyExc_ValueError, "'histogram_equalization' can be performed on uint8 or uint16 images, but not on %s",  PyBlitzArray_TypenumAsString(src->type_num));
  }

  return 0;
  BOB_CATCH_FUNCTION("in histogram_equalization", 0)
}


bob::extension::FunctionDoc s_gammaCorrection = bob::extension::FunctionDoc(
  "gamma_correction",
  "Performs a power-law gamma correction of a given 2D image",
  ".. todo:: Explain gamma correction in more detail"
)
.add_prototype("src, gamma, [dst]", "dst")
.add_parameter("src", "array_like (2D)", "The source image to compute the histogram for")
.add_parameter("gamma", "float", "The gamma value to apply")
.add_parameter("dst", "array_like (2D, float)", "The gamma-corrected image to write; if not specified, it will be created in the desired size")
.add_return("dst", "array_like (2D, float)", "The gamma-corrected image; the same as the ``dst`` parameter, if specified")
;

template <typename T> PyObject* inner_gammaCorrection(PyBlitzArrayObject* src, PyBlitzArrayObject* dst, double gamma) {
  bob::ip::base::gammaCorrection(*PyBlitzArrayCxx_AsBlitz<T, 2>(src), *PyBlitzArrayCxx_AsBlitz<double, 2>(dst), gamma);
  return PyBlitzArray_AsNumpyArray(dst, 0);
}

PyObject* PyBobIpBase_gammaCorrection(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = s_gammaCorrection.kwlist();

  PyBlitzArrayObject* src = 0,* dst = 0;
  double gamma;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&d|O&", kwlist, &PyBlitzArray_Converter, &src, &gamma, &PyBlitzArray_OutputConverter, &dst)) return 0;
  auto src_ = make_safe(src), dst_ = make_xsafe(dst);

  // perform some checks
  if (src->ndim != 2 || (dst && dst->ndim != 2)){
    PyErr_Format(PyExc_ValueError, "'gamma_correction' can be performed on 2D arrays only");
    return 0;
  }

  if (dst){
    if (dst->ndim != 2 || dst->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "'gamma_correction': ``dst`` must be a 2D array of type float");
      return 0;
    }
  } else {
    dst = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, 2, src->shape);
    dst_ = make_safe(dst);
  }

  switch (src->type_num){
    case NPY_UINT8:  return inner_gammaCorrection<uint8_t>(src, dst, gamma);
    case NPY_UINT16: return inner_gammaCorrection<uint16_t>(src, dst, gamma);
    case NPY_FLOAT64: return inner_gammaCorrection<double>(src, dst, gamma);
    default:
      PyErr_Format(PyExc_ValueError, "'gamma_correction' of %s arrays is currently not supported, only uint8, uint16 or float64 arrays are", PyBlitzArray_TypenumAsString(dst->type_num));
      return 0;
  }
  BOB_CATCH_FUNCTION("in gamma_correction", 0)
}


bob::extension::FunctionDoc s_zigzag = bob::extension::FunctionDoc(
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
    "This method only supports arrays of the following data types:\n"
    "\n"
    " * `numpy.uint8`\n"
    " * `numpy.uint16`\n"
    " * `numpy.float64` (or the native python ``float``)\n"
    " \n"
    " To create an object with a scalar type that will be accepted by this "
    " method, use a construction like the following:\n"
    " \n"
    " .. code-block:: python\n"
    " \n"
    "   >> import numpy\n"
    "   >> input_righttype = input_wrongtype.astype(numpy.float64)"
    )

    .add_prototype("src, dst, right_first")
    .add_parameter("src", "array_like (uint8|uint16|float64, 2D)", "The source matrix.")
    .add_parameter("dst", "array_like (uint8|uint16|float64, 1D)", "The destination matrix.")
    .add_parameter("right_first", "scalar (bool)", "Tells whether the zigzag pattern start to move to the right or not")
;

template <typename T> PyObject* inner_zigzag(PyBlitzArrayObject* src, PyBlitzArrayObject* dst, PyObject* bf) {
  //converts value into a proper scalar
  bool c_bf = false;
  if (bf) {
    c_bf = PyBlitzArrayCxx_AsCScalar<bool>(bf);
    if (PyErr_Occurred()) return 0;
  }
  bob::ip::base::zigzag(*PyBlitzArrayCxx_AsBlitz<T,2>(src), *PyBlitzArrayCxx_AsBlitz<T,1>(dst), c_bf);
  Py_RETURN_NONE;
}

PyObject* PyBobIpBase_zigzag(PyObject*, PyObject* args, PyObject* kwds) {
  BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_zigzag.kwlist();

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

  BOB_CATCH_FUNCTION("in zigzag", 0)
}


bob::extension::FunctionDoc s_integral = bob::extension::FunctionDoc(
  "integral",
  "Computes an integral image for the given input image",
  "It is the responsibility of the user to select an appropriate type for the numpy array ``dst`` (and ``sqr``), which will contain the integral image. "
  "By default, ``src`` and ``dst`` should have the same size. "
  "When the ``sqr`` matrix is given as well, it will be filled with the squared integral image (useful to compute variances of pixels).\n\n"
  ".. note::\n\n  The ``sqr`` image is expected to have the same data type as the ``dst`` image.\n\n"
  "If ``add_zero_border`` is set to ``True``, ``dst`` (and ``sqr``) should be one pixel larger than ``src`` in each dimension. "
  "In this case, an extra zero pixel will be added at the beginning of each row and column."

)
.add_prototype("src, dst, [sqr], [add_zero_border]")
.add_parameter("src", "array_like (2D)", "The source image")
.add_parameter("dst", "array_like (2D)", "The resulting integral image")
.add_parameter("sqr", "array_like (2D)", "The resulting squared integral image with the same data type as ``dst``")
.add_parameter("add_zero_border", "bool", "If enabled, an extra zero pixel will be added at the beginning of each row and column")
;

template <typename T1, typename T2>
static inline PyObject* integral_inner(PyBlitzArrayObject* src, PyBlitzArrayObject* dst, PyBlitzArrayObject* sqr, bool add_zero_border) {
  if (sqr)
    bob::ip::base::integral(*PyBlitzArrayCxx_AsBlitz<T1,2>(src), *PyBlitzArrayCxx_AsBlitz<T2,2>(dst), *PyBlitzArrayCxx_AsBlitz<T2,2>(sqr), add_zero_border);
  else
    bob::ip::base::integral(*PyBlitzArrayCxx_AsBlitz<T1,2>(src), *PyBlitzArrayCxx_AsBlitz<T2,2>(dst), add_zero_border);
  Py_RETURN_NONE;
}

template <typename T1>
static inline PyObject* integral_middle(PyBlitzArrayObject* src, PyBlitzArrayObject* dst, PyBlitzArrayObject* sqr, bool b) {
  switch (dst->type_num){
    case NPY_INT8: return integral_inner<T1,int8_t>(src, dst, sqr, b);
    case NPY_INT16: return integral_inner<T1,int16_t>(src, dst, sqr, b);
    case NPY_INT32: return integral_inner<T1,int32_t>(src, dst, sqr, b);
    case NPY_INT64: return integral_inner<T1,int64_t>(src, dst, sqr, b);
    case NPY_UINT8: return integral_inner<T1,uint8_t>(src, dst, sqr, b);
    case NPY_UINT16: return integral_inner<T1,uint16_t>(src, dst, sqr, b);
    case NPY_UINT32: return integral_inner<T1,uint32_t>(src, dst, sqr, b);
    case NPY_UINT64: return integral_inner<T1,uint64_t>(src, dst, sqr, b);
    case NPY_FLOAT32: return integral_inner<T1,float>(src, dst, sqr, b);
    case NPY_FLOAT64: return integral_inner<T1,double>(src, dst, sqr, b);
    default:
      PyErr_Format(PyExc_TypeError, "integral does not work on 'dst' images of type %s", PyBlitzArray_TypenumAsString(dst->type_num));
  }
  return 0;
}

PyObject* PyBobIpBase_integral(PyObject*, PyObject* args, PyObject* kwds) {
  BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_integral.kwlist();

  PyBlitzArrayObject* src = 0,* dst = 0,* sqr = 0;
  PyObject* azb = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|O&O!", kwlist, &PyBlitzArray_Converter, &src, &PyBlitzArray_OutputConverter, &dst, &PyBlitzArray_OutputConverter, &sqr, &PyBool_Type, &azb)) return 0;
  auto src_ = make_safe(src), dst_ = make_safe(dst), sqr_ = make_xsafe(sqr);
  bool b = azb && PyObject_IsTrue(azb);

  if (src->ndim != 2 || dst->ndim != 2 || (sqr && sqr->ndim != 2)) {
    PyErr_Format(PyExc_TypeError, "integral images can only be computed from and to 2D arrays");
    return 0;
  }

  if (sqr && dst->type_num != sqr->type_num) {
    PyErr_Format(PyExc_TypeError, "'dst' and 'sqr' arrays must have the same data types (dst: `%s' != sqr: `%s')", PyBlitzArray_TypenumAsString(dst->type_num), PyBlitzArray_TypenumAsString(sqr->type_num));
    return 0;
  }

  switch (src->type_num){
    case NPY_INT8: return integral_middle<int8_t>(src, dst, sqr, b);
    case NPY_INT16: return integral_middle<int16_t>(src, dst, sqr, b);
    case NPY_INT32: return integral_middle<int32_t>(src, dst, sqr, b);
    case NPY_INT64: return integral_middle<int64_t>(src, dst, sqr, b);
    case NPY_UINT8: return integral_middle<uint8_t>(src, dst, sqr, b);
    case NPY_UINT16: return integral_middle<uint16_t>(src, dst, sqr, b);
    case NPY_UINT32: return integral_middle<uint32_t>(src, dst, sqr, b);
    case NPY_UINT64: return integral_middle<uint64_t>(src, dst, sqr, b);
    case NPY_FLOAT32: return integral_middle<float>(src, dst, sqr, b);
    case NPY_FLOAT64: return integral_middle<double>(src, dst, sqr, b);
    default:
      PyErr_Format(PyExc_TypeError, "integral does not work on 'src' images of type %s", PyBlitzArray_TypenumAsString(src->type_num));
  }
  return 0;

  BOB_CATCH_FUNCTION("in integral", 0)
}

bob::extension::FunctionDoc s_block = bob::extension::FunctionDoc(
  "block",
  "Performs a block decomposition of a 2D array/image",
  "If given, the output 3D or 4D destination array should be allocated and of the correct size, see :py:func:`bob.ip.base.block_output_shape`.",
  "Blocks are extracted such that they fit into the given image. "
  "The blocks can be split into either a 3D array of shape ``(block_index, block_height, block_width)``, or into a 4D array of shape ``(block_index_y, block_index_x, block_height, block_width)``. "
  "To toggle between both ways, select the ``flat`` parameter accordingly."
)
.add_prototype("input, block_size, [block_overlap], [output], [flat]", "output")
.add_parameter("input", "array_like (2D)", "The source image to decompose into blocks")
.add_parameter("block_size", "(int, int)", "The size of the blocks in which the image is decomposed")
.add_parameter("block_overlap", "(int, int)", "[default: ``(0, 0)``] The overlap of the blocks")
.add_parameter("output", "array_like(3D or 4D)", "[default: ``None``] If given, the resulting blocks will be saved into this parameter; must be initialized in the correct size (see :py:func:`block_output_shape`)")
.add_parameter("flat", "bool", "[default: ``False``] If ``output`` is not specified, the ``flat`` parameter is used to decide whether 3D (``flat = True``) or 4D (``flat = False``) output is generated")
.add_return("output", "array_like(3D or 4D)", "The resulting blocks that the image is decomposed into; the same array as the ``output`` parameter, when given.")
;

// helper function to compute the output shape
static inline blitz::TinyVector<int,3> block_shape3(PyBlitzArrayObject* input, blitz::TinyVector<int,2> block_size, blitz::TinyVector<int,2> block_overlap){
  return bob::ip::base::getBlock3DOutputShape(input->shape[0], input->shape[1], block_size[0], block_size[1], block_overlap[0], block_overlap[1]);
}
static inline blitz::TinyVector<int,4> block_shape4(PyBlitzArrayObject* input, blitz::TinyVector<int,2> block_size, blitz::TinyVector<int,2> block_overlap){
  return bob::ip::base::getBlock4DOutputShape(input->shape[0], input->shape[1], block_size[0], block_size[1], block_overlap[0], block_overlap[1]);
}

template <typename T, int D>
static inline void block_inner(PyBlitzArrayObject* input, blitz::TinyVector<int,2> block_size, blitz::TinyVector<int,2> block_overlap, PyBlitzArrayObject* output){
  bob::ip::base::block(*PyBlitzArrayCxx_AsBlitz<T,2>(input), *PyBlitzArrayCxx_AsBlitz<T,D>(output), block_size[0], block_size[1], block_overlap[0], block_overlap[1]);
}

PyObject* PyBobIpBase_block(PyObject*, PyObject* args, PyObject* kwds) {
  BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_block.kwlist();

  PyBlitzArrayObject* input = 0,* output = 0;
  blitz::TinyVector<int,2> size, overlap(0,0);
  PyObject* flat_ = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&(ii)|(ii)O&O!", kwlist, &PyBlitzArray_Converter, &input, &size[0], &size[1], &overlap[0], &overlap[1], &PyBlitzArray_OutputConverter, &output, &PyBool_Type, &flat_)) return 0;

  auto input_ = make_safe(input), output_ = make_xsafe(output);
  bool flat = f(flat_);

  if (input->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "blocks can only be extracted from and to 2D arrays");
    return 0;
  }
  bool return_out = false;
  if (output){
    if (output->type_num != input->type_num){
      PyErr_Format(PyExc_TypeError, "``input`` and ``output`` must have the same data type");
      return 0;
    }
    if (output->ndim != 3 && output->ndim != 4){
      PyErr_Format(PyExc_TypeError, "``output`` must have either three or four dimensions, not %" PY_FORMAT_SIZE_T "d", output->ndim);
      return 0;
    }
    flat = output->ndim == 3;
  } else {
    return_out = true;
    // generate output in the desired shape
    if (flat){
      auto res = block_shape3(input, size, overlap);
      Py_ssize_t osize[] = {res[0], res[1], res[2]};
      output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(input->type_num, 3, osize);
    } else {
      auto res = block_shape4(input, size, overlap);
      Py_ssize_t osize[] = {res[0], res[1], res[2], res[3]};
      output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(input->type_num, 4, osize);
    }
    output_ = make_safe(output);
  }

  switch (input->type_num){
    case NPY_UINT8:   if (flat) block_inner<uint8_t,3>(input, size, overlap, output);  else block_inner<uint8_t,4>(input, size, overlap, output); break;
    case NPY_UINT16:  if (flat) block_inner<uint16_t,3>(input, size, overlap, output); else block_inner<uint16_t,4>(input, size, overlap, output); break;
    case NPY_FLOAT64: if (flat) block_inner<double,3>(input, size, overlap, output);   else block_inner<double,4>(input, size, overlap, output); break;
    default:
      PyErr_Format(PyExc_TypeError, "block does not work on 'input' images of type %s", PyBlitzArray_TypenumAsString(input->type_num));
  }

  if (return_out){
    return PyBlitzArray_AsNumpyArray(output, 0);
  } else
    Py_RETURN_NONE;

  BOB_CATCH_FUNCTION("in block", 0)
}


bob::extension::FunctionDoc s_blockOutputShape = bob::extension::FunctionDoc(
  "block_output_shape",
  "Returns the shape of the output image that is required to compute the :py:func:`bob.ip.base.block` function",
  0
)
.add_prototype("input, block_size, [block_overlap], [flat]", "shape")
.add_parameter("input", "array_like (2D)", "The source image to decompose into blocks")
.add_parameter("block_size", "(int, int)", "The size of the blocks in which the image is decomposed")
.add_parameter("block_overlap", "(int, int)", "[default: ``(0, 0)``] The overlap of the blocks")
.add_parameter("flat", "bool", "[default: ``False``] The ``flat`` parameter is used to decide whether 3D (``flat = True``) or 4D (``flat = False``) output is generated")
.add_return("shape", "(int, int, int) or (int, int, int, int)", "The shape of the blocks.")
;

PyObject* PyBobIpBase_blockOutputShape(PyObject*, PyObject* args, PyObject* kwds) {
  BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_blockOutputShape.kwlist();

  PyBlitzArrayObject* input = 0;
  blitz::TinyVector<int,2> size, overlap(0,0);
  PyObject* flat_ = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&(ii)|(ii)O!", kwlist, &PyBlitzArray_Converter, &input, &size[0], &size[1], &overlap[0], &overlap[1], &PyBool_Type, &flat_)) return 0;

  auto input_ = make_safe(input);

  if (input->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "block shape can only be computed from and to 2D arrays");
    return 0;
  }

  if (f(flat_)){
    auto shape = block_shape3(input, size, overlap);
    return Py_BuildValue("(iii)", shape[0], shape[1], shape[2]);
  } else {
    auto shape = block_shape4(input, size, overlap);
    return Py_BuildValue("(iiii)", shape[0], shape[1], shape[2], shape[3]);
  }

  BOB_CATCH_FUNCTION("in block_output_shape", 0)
}



bob::extension::FunctionDoc s_lbphs = bob::extension::FunctionDoc(
  "lbphs",
  "Computes an local binary pattern histogram sequences from the given image",
  ".. warning:: This is a re-implementation of the old bob.ip.LBPHSFeatures class, but with a different handling of blocks. "
  "Before, the blocks where extracted from the image, and LBP's were extracted in the blocks. "
  "Hence, in each block, the border pixels where not taken into account, and the histogram contained far less elements. "
  "Now, the LBP's are extracted first, and then the image is split into blocks.\n\n"
  "This function computes the LBP features for the whole image, using the given :py:class:`bob.ip.base.LBP` instance. "
  "Afterwards, the resulting image is split into several blocks with the given block size and overlap, and local LBH histograms are extracted from each region.\n\n"
  ".. note::\n\n  To get the required output shape, you can use :py:func:`lbphs_output_shape` function."
)
.add_prototype("input, lbp, block_size, [block_overlap], [output]", "output")
.add_parameter("input", "array_like (2D)", "The source image to compute the LBPHS for")
.add_parameter("lbp", ":py:class:`bob.ip.base.LBP`", "The LBP class to be used for feature extraction")
.add_parameter("block_size", "(int, int)", "The size of the blocks in which the LBP histograms are split")
.add_parameter("block_overlap", "(int, int)", "[default: ``(0, 0)``] The overlap of the blocks in which the LBP histograms are split")
.add_parameter("output", "array_like(2D, uint64)", "If given, the resulting LBPHS features will be written to this array; must have the size #output-blocks, #LBP-labels (see :py:func:`lbphs_output_shape`)")
.add_return("output", "array_like(2D, uint64)", "The resulting LBPHS features of the size #output-blocks, #LBP-labels; the same array as the ``output`` parameter, when given.")
;

// helper function to compute the output shape
static inline blitz::TinyVector<int,2> lbphs_shape(PyBlitzArrayObject* input, PyBobIpBaseLBPObject* lbp, blitz::TinyVector<int,2> block_size, blitz::TinyVector<int,2> block_overlap){
  auto res = lbp->cxx->getLBPShape(blitz::TinyVector<int,2>(input->shape[0], input->shape[1]));
  return blitz::TinyVector<int,2>(bob::ip::base::getBlock3DOutputShape(res[0], res[1], block_size[0], block_size[1], block_overlap[0], block_overlap[1])[0], lbp->cxx->getMaxLabel());
}

template <typename T>
static inline PyObject* lbphs_inner(PyBlitzArrayObject* input, PyBobIpBaseLBPObject* lbp, blitz::TinyVector<int,2> block_size, blitz::TinyVector<int,2> block_overlap, PyBlitzArrayObject* output){
  bob::ip::base::lbphs(*PyBlitzArrayCxx_AsBlitz<T,2>(input), *lbp->cxx, block_size, block_overlap, *PyBlitzArrayCxx_AsBlitz<uint64_t,2>(output));
  return PyBlitzArray_AsNumpyArray(output, 0);
}

PyObject* PyBobIpBase_lbphs(PyObject*, PyObject* args, PyObject* kwds) {
  BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_lbphs.kwlist();

  PyBlitzArrayObject* input = 0,* output = 0;
  PyBobIpBaseLBPObject* lbp;
  blitz::TinyVector<int,2> size, overlap(0,0);

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O!(ii)|(ii)O&", kwlist, &PyBlitzArray_Converter, &input, &PyBobIpBaseLBP_Type, &lbp, &size[0], &size[1], &overlap[0], &overlap[1], &PyBlitzArray_OutputConverter, &output)) return 0;

  auto input_ = make_safe(input), output_ = make_xsafe(output);

  if (input->ndim != 2 || (output && output->ndim != 2)) {
    PyErr_Format(PyExc_TypeError, "lbphs images can only be computed from and to 2D arrays");
    return 0;
  }
  if (output && output->type_num != NPY_UINT64){
    PyErr_Format(PyExc_TypeError, "lbphs datatype must be uint64");
  }
  if (!output){
    // generate output in the desired shape
    auto res = lbphs_shape(input, lbp, size, overlap);
    Py_ssize_t osize[] = {res[0], res[1]};
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_UINT64, 2, osize);
    output_ = make_safe(output);
  }

  switch (input->type_num){
    case NPY_UINT8: return lbphs_inner<uint8_t>(input, lbp, size, overlap, output);
    case NPY_UINT16: return lbphs_inner<uint16_t>(input, lbp, size, overlap, output);
    case NPY_FLOAT64: return lbphs_inner<double>(input, lbp, size, overlap, output);
    default:
      PyErr_Format(PyExc_TypeError, "lbphs does not work on 'input' images of type %s", PyBlitzArray_TypenumAsString(input->type_num));
  }
  return 0;

  BOB_CATCH_FUNCTION("in lbphs", 0)
}


bob::extension::FunctionDoc s_lbphsOutputShape = bob::extension::FunctionDoc(
  "lbphs_output_shape",
  "Returns the shape of the output image that is required to compute the :py:func:`bob.ip.base.lbphs` function",
  0
)
.add_prototype("input, lbp, block_size, [block_overlap]", "shape")
.add_parameter("input", "array_like (2D)", "The source image to compute the LBPHS for")
.add_parameter("lbp", ":py:class:`bob.ip.base.LBP`", "The LBP class to be used for feature extraction")
.add_parameter("block_size", "(int, int)", "The size of the blocks in which the LBP histograms are split")
.add_parameter("block_overlap", "(int, int)", "[default: ``(0, 0)``] The overlap of the blocks in which the LBP histograms are split")
.add_return("shape", "(int, int)", "The shape of the LBP histogram sequences, which is ``(#blocks, #labels)``.")
;

PyObject* PyBobIpBase_lbphsOutputShape(PyObject*, PyObject* args, PyObject* kwds) {
  BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_lbphsOutputShape.kwlist();

  PyBlitzArrayObject* input = 0;
  PyBobIpBaseLBPObject* lbp;
  blitz::TinyVector<int,2> size, overlap(0,0);

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O!(ii)|(ii)", kwlist, &PyBlitzArray_Converter, &input, &PyBobIpBaseLBP_Type, &lbp, &size[0], &size[1], &overlap[0], &overlap[1])) return 0;

  auto input_ = make_safe(input);

  if (input->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "lbphs images can only be computed from and to 2D arrays");
    return 0;
  }

  auto shape = lbphs_shape(input, lbp, size, overlap);

  return Py_BuildValue("(ii)", shape[0], shape[1]);

  BOB_CATCH_FUNCTION("in lbphs_output_shape", 0)
}
