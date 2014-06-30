/**
 * @author Manuel Guenther <manuel.guenthr@idiap.ch>
 * @date Wed Jun 25 18:28:03 CEST 2014
 *
 * @brief Binds auxiliary functions of bob::ip::base class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


#include "main.h"
#include "cpp/Affine.h"

bob::extension::FunctionDoc s_scale = bob::extension::FunctionDoc(
  "scale",
  "Scales an image.",
  "This function scales an image using bi-linear interpolation. "
  "It supports 2D and 3D input array/image (NumPy array) of type numpy.uint8, numpy.uint16 and numpy.float64. "
  "Basically, this function can be called in three different ways:\n\n"
  "1. Given a source image and a scale factor, the scaled image is returned in the size :py:func:`bob.ip.base.get_scaled_output_shape`\n\n"
  "2. Given source and destination image, the source image is scaled such that it fits into the destination image.\n\n"
  "3. Same as 2., but additionally boolean masks will be read and filled with according values.\n\n"
  ".. note:: For 2. and 3., scale factors are computed for both directions independently. "
  "Factually, this means that the image **might be** stretched in either direction, i.e., the aspect ratio is **not** identical for the horizontal and vertical direction. "
  "Even for 1. this might apply, e.g., when ``src.shape * scaling_factor`` does not result in integral values."
)
.add_prototype("src, scaling_factor", "dst")
.add_prototype("src, dst")
.add_prototype("src, src_mask, dst, dst_mask")
.add_parameter("src", "array_like (2D or 3D)", "The input image (gray or colored) that should be scaled")
.add_parameter("dst", "array_like (2D or 3D, float)", "The resulting scaled gray or color image")
.add_parameter("src_mask", "array_like (bool, 2D or 3D)", "An input mask of valid pixels before geometric normalization, must be of same size as ``src``")
.add_parameter("dst_mask", "array_like (bool, 2D or 3D)", "The output mask of valid pixels after geometric normalization, must be of same size as ``dst``")
.add_parameter("scaling_factor", "float", "the scaling factor that should be applied to the image")
.add_return("dst", "array_like (2D, float)", "The resulting scaled image")
;

template <typename T, int D>
static void scale_inner(PyBlitzArrayObject* input, PyBlitzArrayObject* input_mask, PyBlitzArrayObject* output, PyBlitzArrayObject* output_mask) {
  if (input_mask && output_mask){
    bob::ip::base::scale<T>(*PyBlitzArrayCxx_AsBlitz<T,D>(input), *PyBlitzArrayCxx_AsBlitz<bool,D>(input_mask), *PyBlitzArrayCxx_AsBlitz<double,D>(output), *PyBlitzArrayCxx_AsBlitz<bool,D>(output_mask));
  } else {
    bob::ip::base::scale<T>(*PyBlitzArrayCxx_AsBlitz<T,D>(input), *PyBlitzArrayCxx_AsBlitz<double,D>(output));
  }
}

PyObject* PyBobIpBase_scale(PyObject*, PyObject* args, PyObject* kwargs) {
  TRY
  /* Parses input arguments in a single shot */
  static char* kwlist1[] = {c("src"), c("scaling_factor"), NULL};
  static char* kwlist2[] = {c("src"), c("dst"), NULL};
  static char* kwlist3[] = {c("src"), c("src_mask"), c("dst"), c("dst_mask"), NULL};

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyBlitzArrayObject* src,* src_mask = 0,* dst = 0,* dst_mask = 0;
  double scale_factor = std::numeric_limits<float>::quiet_NaN();
  if (nargs == 4){
    // with masks
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&O&O&", kwlist3, &PyBlitzArray_Converter, &src, &PyBlitzArray_Converter, &src_mask, &PyBlitzArray_OutputConverter, &dst, &PyBlitzArray_OutputConverter, &dst_mask)) return 0;
  }
  else if (nargs == 2){
    PyObject* k = Py_BuildValue("s", kwlist1[1]);
    auto k_ = make_safe(k);
    // check second parameter
    if ((args && PyTuple_Size(args) == 2 && (PyInt_Check(PyTuple_GET_ITEM(args,1)) || PyFloat_Check(PyTuple_GET_ITEM(args,1)))) || (kwargs && PyDict_Contains(kwargs, k))){
      // with scale
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&d", kwlist1, &PyBlitzArray_Converter, &src, &scale_factor)) return 0;
    } else {
      // with input and output
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&", kwlist2, &PyBlitzArray_Converter, &src, &PyBlitzArray_OutputConverter, &dst)) return 0;
    }
  }
  else{
    PyErr_Format(PyExc_ValueError, "scale was called with an unknown number of arguments");
    return 0;
  }

  auto src_ = make_safe(src), src_mask_ = make_xsafe(src_mask), dst_ = make_xsafe(dst), dst_mask_ = make_xsafe(dst_mask);

  if (src->ndim != 2 && src->ndim != 3){
    PyErr_Format(PyExc_TypeError, "only 2D and 3D images can be scaled");
    return 0;
  }

  if (dst){
    // check that data type is correct and dimensions fit
    if (dst->ndim != src->ndim){
      PyErr_Format(PyExc_TypeError, "scale: the src and dst array must have the same number of dimensions");
      return 0;
    }
    if (dst->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "scale: the dst array must be of type float64");
      return 0;
    }
  } else {
    assert (!isnan(scale_factor));
    // create output in the same dimensions as input
    switch (src->ndim){
      case 2:{
        blitz::TinyVector<int,2> orig_shape(src->shape[0], src->shape[1]);
        auto new_shape = bob::ip::base::getScaledShape(orig_shape, scale_factor);
        Py_ssize_t n[] = {new_shape[0], new_shape[1]};
        dst = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_FLOAT64, 2, n));
        break;
      }
      case 3:{
        blitz::TinyVector<int,3> orig_shape(src->shape[0], src->shape[1], src->shape[2]);
        auto new_shape = bob::ip::base::getScaledShape(orig_shape, scale_factor);
        Py_ssize_t n[] = {new_shape[0], new_shape[1], new_shape[2]};
        dst = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_FLOAT64, 3, n));
        break;
      }
      default:
        PyErr_Format(PyExc_TypeError, "only 2D and 3D images can be scaled");
        return 0;
    }
    dst_ = make_safe(dst);
  }

  if (src_mask && dst_mask && ((src_mask->ndim != src->ndim || src_mask->type_num != NPY_BOOL) || dst_mask->ndim != dst->ndim || dst_mask->type_num != NPY_BOOL)) {
    PyErr_Format(PyExc_TypeError, "scale: the masks must be of boolean type and have the same dimensions as src or dst images.");
    return 0;
  }

  switch (src->type_num){
    case NPY_UINT8:   (src->ndim == 2 ? scale_inner<uint8_t,2> : scale_inner<uint8_t,3>) (src, src_mask, dst, dst_mask); break;
    case NPY_UINT16:  (src->ndim == 2 ? scale_inner<uint16_t,2> : scale_inner<uint16_t,3>) (src, src_mask, dst, dst_mask); break;
    case NPY_FLOAT64: (src->ndim == 2 ? scale_inner<double,2> : scale_inner<double,3>) (src, src_mask, dst, dst_mask); break;
    default:
      PyErr_Format(PyExc_TypeError, "scale: src arrays of type %s are currently not supported", PyBlitzArray_TypenumAsString(src->type_num));
      return 0;
  }

  if (!isnan(scale_factor)){
    Py_INCREF(dst);
    return PyBlitzArray_AsNumpyArray(dst,0);
  }

  Py_RETURN_NONE;
  CATCH_("scale", 0)
}


bob::extension::FunctionDoc s_getScaledOutputShape = bob::extension::FunctionDoc(
  "get_scaled_output_shape",
  "This function returns the shape of the scaled image for the given image and scale",
  "The function tries its best to compute an integral-valued shape given the shape of the input image and the given scale factor. "
  "Nevertheless, for non-round scale factors this might not work out perfectly.",
  true
)
.add_prototype("src, scaling_factor", "scaled_shape")
//.add_prototype("shape, scale", "scaled_shape")
.add_parameter("src", "array_like (2D,3D)", "The src image which which should be scaled")
.add_parameter("scaling_factor", "float", "The scaling factor to scale the src image with")
//.add_parameter("shape", "(int, int) or (int, int, int)", "The shape of the input image which which should be scaled")
.add_return("scaled_shape", "(int, int) or (int, int, int)", "The shape of the scaled ``dst`` image required in a call to :py:func:`bob.ip.base.scale`")
;

PyObject* PyBobIpBase_getScaledOutputShape(PyObject*, PyObject* args, PyObject* kwargs) {
  TRY

  static char* kwlist1[] = {c("src"), c("scaling_factor"), 0};
// TODO: implement from shape
//  static char* kwlist2[] = {c("shape"), c("scale"), 0};

  PyBlitzArrayObject* image = 0;
  double scaling_factor;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&d", kwlist1, &PyBlitzArray_Converter, &image, &scaling_factor)) return 0;

  auto image_ = make_safe(image);
  switch (image->ndim){
    case 2:{
      blitz::TinyVector<int,2> src_shape(image->shape[0], image->shape[1]);
      auto scaled_shape = bob::ip::base::getScaledShape(src_shape, scaling_factor);
      return Py_BuildValue("(ii)", scaled_shape[0], scaled_shape[1]);
    }
    case 3:{
      blitz::TinyVector<int,3> src_shape(image->shape[0], image->shape[1], image->shape[2]);
      auto scaled_shape = bob::ip::base::getScaledShape(src_shape, scaling_factor);
      return Py_BuildValue("(iii)", scaled_shape[0], scaled_shape[1], scaled_shape[2]);
    }
    default:
      PyErr_Format(PyExc_TypeError, "'get_scaled_output_shape' only accepts 2D or 3D arrays (not %" PY_FORMAT_SIZE_T "dD arrays)", image->ndim);
  }

  return 0;
  CATCH_("cannot get scaled output shape", 0)
}


bob::extension::FunctionDoc s_maxRectInMask = bob::extension::FunctionDoc(
  "max_rect_in_mask",
  "Given a 2D mask (a 2D blitz array of booleans), compute the maximum rectangle which only contains true values.",
  "The resulting rectangle contains the coordinates in the following order:\n\n"
  "0. The y-coordinate of the top left corner\n\n"
  "1. The x-coordinate of the top left corner\n\n"
  "2. The height of the rectangle\n\n"
  "3. The width of the rectangle"
)
.add_prototype("mask", "rect")
.add_parameter("mask", "array_like (2D, bool)", "The mask of boolean values, e.g., as a result of :py:func:`bob.ip.base.GeomNorm.process`")
.add_return("rect", "(int, int, int, int)", "The resulting rectangle: (top, left, height, width)")
;

PyObject* PyBobIpBase_maxRectInMask(PyObject*, PyObject* args, PyObject* kwds) {
  TRY
  /* Parses input arguments in a single shot */
  static char* kwlist[] = {c("mask"), NULL};

  PyBlitzArrayObject* mask = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBlitzArray_Converter, &mask)) return 0;

  auto mask_ = make_safe(mask);

  if (mask->ndim != 2 || mask->type_num != NPY_BOOL) {
    PyErr_Format(PyExc_TypeError, "max_rect_in_mask: the mask must be 2D and of boolean type");
    return 0;
  }

  auto rect = bob::ip::base::maxRectInMask(*PyBlitzArrayCxx_AsBlitz<bool, 2>(mask));

  return Py_BuildValue("(iiii)", rect[0], rect[1], rect[2], rect[3]);

  CATCH_("in max_rect_in_mask", 0)
}

