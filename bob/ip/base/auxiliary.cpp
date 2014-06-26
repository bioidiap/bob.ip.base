/**
 * @author Manuel Guenther <manuel.guenthr@idiap.ch>
 * @date Wed Jun 25 18:28:03 CEST 2014
 *
 * @brief Binds auxiliary functions of bob::ip::base class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


#include "main.h"
#include <bob/ip/zigzag.h>
#include "cpp/IntegralImage.h"
#include "cpp/LBPHS.h"

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
  bob::ip::zigzag(*PyBlitzArrayCxx_AsBlitz<T,2>(src), *PyBlitzArrayCxx_AsBlitz<T,1>(dst), c_bf);
  Py_RETURN_NONE;
}

PyObject* PyBobIpBase_zigzag(PyObject*, PyObject* args, PyObject* kwds) {
  TRY
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

  CATCH_("in zigzag", 0)
}


bob::extension::FunctionDoc s_integral = bob::extension::FunctionDoc(
  "integral",
  "Computes an integral image for the given input image",
  "It is the responsibility of the user to select an appropriate type for the numpy array ``dst`` (and ``sqr``), which will contain the integral image. "
  "By default, ``src`` and ``dst`` should have the same size. "
  "When the ``sqr`` matrix is given as well, it will be filled with the squared integral image (useful to compute variances of pixels).\n\n"
  ".. note:: The ``sqr`` image is expected to have the same data type as the ``dst`` image.\n\n"
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
  TRY
  /* Parses input arguments in a single shot */
  static char* kwlist[] = {c("src"), c("dst"), c("sqr"), c("add_zero_border"), NULL};

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

  CATCH_("in integral", 0)
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
  ".. note:: To get the required output shape, you can use :py:func:`lbphs_output_shape` function."
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
  return blitz::TinyVector<int,2>(bob::ip::getBlock3DOutputShape(res[0], res[1], block_size[0], block_size[1], block_overlap[0], block_overlap[1])[0], lbp->cxx->getMaxLabel());
}

template <typename T>
static inline PyObject* lbphs_inner(PyBlitzArrayObject* input, PyBobIpBaseLBPObject* lbp, blitz::TinyVector<int,2> block_size, blitz::TinyVector<int,2> block_overlap, PyBlitzArrayObject* output){
  bob::ip::base::lbphs(*PyBlitzArrayCxx_AsBlitz<T,2>(input), *lbp->cxx, block_size, block_overlap, *PyBlitzArrayCxx_AsBlitz<uint64_t,2>(output));
  Py_INCREF(output);
  return Py_BuildValue("O", output);
}

PyObject* PyBobIpBase_lbphs(PyObject*, PyObject* args, PyObject* kwds) {
  TRY
  /* Parses input arguments in a single shot */
  static char* kwlist[] = {c("input"), c("lbp"), c("block_size"), c("block_overlap"), c("output"), NULL};

  PyBlitzArrayObject* input = 0,* output = 0;
  PyBobIpBaseLBPObject* lbp;
  blitz::TinyVector<int,2> size, overlap(0,0);

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O!(ii)|(ii)O&", kwlist, &PyBlitzArray_Converter, &input, &PyBobIpBaseLBPType, &lbp, &size[0], &size[1], &overlap[0], &overlap[1], &PyBlitzArray_OutputConverter, &output)) return 0;

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

  CATCH_("in lbphs", 0)
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
  TRY
  /* Parses input arguments in a single shot */
  static char* kwlist[] = {c("input"), c("lbp"), c("block_size"), c("block_overlap"), NULL};

  PyBlitzArrayObject* input = 0;
  PyBobIpBaseLBPObject* lbp;
  blitz::TinyVector<int,2> size, overlap(0,0);

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O!(ii)|(ii)", kwlist, &PyBlitzArray_Converter, &input, &PyBobIpBaseLBPType, &lbp, &size[0], &size[1], &overlap[0], &overlap[1])) return 0;

  auto input_ = make_safe(input);

  if (input->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "lbphs images can only be computed from and to 2D arrays");
    return 0;
  }

  auto shape = lbphs_shape(input, lbp, size, overlap);

  return Py_BuildValue("(ii)", shape[0], shape[1]);

  CATCH_("in lbphs_output_shape", 0)
}



