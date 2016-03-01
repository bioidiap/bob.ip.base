/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Wed Jun 25 18:28:03 CEST 2014
 *
 * @brief Binds auxiliary functions of bob::ip::base class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


#include "main.h"

bob::extension::FunctionDoc s_scale = bob::extension::FunctionDoc(
  "scale",
  "Scales an image.",
  "This function scales an image using bi-linear interpolation. "
  "It supports 2D and 3D input array/image (NumPy array) of type numpy.uint8, numpy.uint16 and numpy.float64. "
  "Basically, this function can be called in three different ways:\n\n"
  "1. Given a source image and a scale factor, the scaled image is returned in the size :py:func:`bob.ip.base.scaled_output_shape`\n\n"
  "2. Given source and destination image, the source image is scaled such that it fits into the destination image.\n\n"
  "3. Same as 2., but additionally boolean masks will be read and filled with according values.\n\n"
  ".. note::\n\n  For 2. and 3., scale factors are computed for both directions independently. "
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
.add_parameter("scaling_factor", "float", "the scaling factor that should be applied to the image; can be negative, but cannot be ``0.``")
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
  BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist1 = s_scale.kwlist(0);
  char** kwlist2 = s_scale.kwlist(1);
  char** kwlist3 = s_scale.kwlist(2);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyBlitzArrayObject* src,* src_mask = 0,* dst = 0,* dst_mask = 0;
  double scale_factor = 0;
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
      if (!scale_factor){
        PyErr_SetString(PyExc_ValueError, "scaling with a scale factor of 0. is not supported.");
        return 0;
      }
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
    case NPY_UINT8:   if (src->ndim == 2) scale_inner<uint8_t,2>(src, src_mask, dst, dst_mask);  else scale_inner<uint8_t,3>(src, src_mask, dst, dst_mask); break;
    case NPY_UINT16:  if (src->ndim == 2) scale_inner<uint16_t,2>(src, src_mask, dst, dst_mask); else scale_inner<uint16_t,3>(src, src_mask, dst, dst_mask); break;
    case NPY_FLOAT64: if (src->ndim == 2) scale_inner<double,2>(src, src_mask, dst, dst_mask);   else scale_inner<double,3>(src, src_mask, dst, dst_mask); break;
    default:
      PyErr_Format(PyExc_TypeError, "scale: src arrays of type %s are currently not supported", PyBlitzArray_TypenumAsString(src->type_num));
      return 0;
  }

  if (scale_factor){
    return PyBlitzArray_AsNumpyArray(dst,0);
  }

  Py_RETURN_NONE;
  BOB_CATCH_FUNCTION("scale", 0)
}


bob::extension::FunctionDoc s_scaledOutputShape = bob::extension::FunctionDoc(
  "scaled_output_shape",
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

PyObject* PyBobIpBase_scaledOutputShape(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = s_scaledOutputShape.kwlist();
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
      PyErr_Format(PyExc_TypeError, "'scaled_output_shape' only accepts 2D or 3D arrays (not %" PY_FORMAT_SIZE_T "dD arrays)", image->ndim);
  }

  return 0;
  BOB_CATCH_FUNCTION("cannot get scaled output shape", 0)
}


bob::extension::FunctionDoc s_rotate = bob::extension::FunctionDoc(
  "rotate",
  "Rotates an image.",
  "This function rotates an image using bi-linear interpolation. "
  "It supports 2D and 3D input array/image (NumPy array) of type numpy.uint8, numpy.uint16 and numpy.float64. "
  "Basically, this function can be called in three different ways:\n\n"
  "1. Given a source image and a rotation angle, the rotated image is returned in the size :py:func:`bob.ip.base.rotated_output_shape`\n\n"
  "2. Given source and destination image and the rotation angle, the source image is rotated and filled into the destination image.\n\n"
  "3. Same as 2., but additionally boolean masks will be read and filled with according values.\n\n"
  ".. note::\n\n  Since the implementation uses a different interpolation style than before, results might *slightly* differ."
)
.add_prototype("src, rotation_angle", "dst")
.add_prototype("src, dst, rotation_angle")
.add_prototype("src, src_mask, dst, dst_mask, rotation_angle")
.add_parameter("src", "array_like (2D or 3D)", "The input image (gray or colored) that should be rotated")
.add_parameter("dst", "array_like (2D or 3D, float)", "The resulting scaled gray or color image, should be in size :py:func:`bob.ip.base.rotated_output_shape`")
.add_parameter("src_mask", "array_like (bool, 2D or 3D)", "An input mask of valid pixels before geometric normalization, must be of same size as ``src``")
.add_parameter("dst_mask", "array_like (bool, 2D or 3D)", "The output mask of valid pixels after geometric normalization, must be of same size as ``dst``")
.add_parameter("rotation_angle", "float", "the rotation angle that should be applied to the image")
.add_return("dst", "array_like (2D, float)", "The resulting rotated image")
;

template <typename T, int D>
static void rotate_inner(PyBlitzArrayObject* input, PyBlitzArrayObject* input_mask, PyBlitzArrayObject* output, PyBlitzArrayObject* output_mask, double angle) {
  if (input_mask && output_mask){
    bob::ip::base::rotate<T>(*PyBlitzArrayCxx_AsBlitz<T,D>(input), *PyBlitzArrayCxx_AsBlitz<bool,D>(input_mask), *PyBlitzArrayCxx_AsBlitz<double,D>(output), *PyBlitzArrayCxx_AsBlitz<bool,D>(output_mask), angle);
  } else {
    bob::ip::base::rotate<T>(*PyBlitzArrayCxx_AsBlitz<T,D>(input), *PyBlitzArrayCxx_AsBlitz<double,D>(output), angle);
  }
}

PyObject* PyBobIpBase_rotate(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist1 = s_rotate.kwlist(0);
  char** kwlist2 = s_rotate.kwlist(1);
  char** kwlist3 = s_rotate.kwlist(2);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyBlitzArrayObject* src,* src_mask = 0,* dst = 0,* dst_mask = 0;
  double angle = 0.;
  switch (nargs){
    case 2: // src and angle; create the destination afterwards
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&d", kwlist1, &PyBlitzArray_Converter, &src, &angle)) return 0;
      break;
    case 3: // src, dst and angle
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&d", kwlist2, &PyBlitzArray_Converter, &src, &PyBlitzArray_OutputConverter, &dst, &angle)) return 0;
      break;
    case 5: // src, dst and angle
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&O&O&d", kwlist3, &PyBlitzArray_Converter, &src, &PyBlitzArray_Converter, &src_mask, &PyBlitzArray_OutputConverter, &dst, &PyBlitzArray_OutputConverter, &dst_mask, &angle)) return 0;
      break;
    default:
      PyErr_Format(PyExc_ValueError, "rotate was called with a wrong number of arguments");
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
      PyErr_Format(PyExc_TypeError, "rotate: the src and dst array must have the same number of dimensions");
      return 0;
    }
    if (dst->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "rotate: the dst array must be of type float64");
      return 0;
    }
  } else {
    // create output in the same dimensions as input
    switch (src->ndim){
      case 2:{
        blitz::TinyVector<int,2> orig_shape(src->shape[0], src->shape[1]);
        auto new_shape = bob::ip::base::getRotatedShape(orig_shape, angle);
        Py_ssize_t n[] = {new_shape[0], new_shape[1]};
        dst = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_FLOAT64, 2, n));
        break;
      }
      case 3:{
        blitz::TinyVector<int,3> orig_shape(src->shape[0], src->shape[1], src->shape[2]);
        auto new_shape = bob::ip::base::getRotatedShape(orig_shape, angle);
        Py_ssize_t n[] = {new_shape[0], new_shape[1], new_shape[2]};
        dst = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_FLOAT64, 3, n));
        break;
      }
      default:
        PyErr_Format(PyExc_TypeError, "only 2D and 3D images can be rotated");
        return 0;
    }
    dst_ = make_safe(dst);
  }

  if (src_mask && dst_mask && ((src_mask->ndim != src->ndim || src_mask->type_num != NPY_BOOL) || dst_mask->ndim != dst->ndim || dst_mask->type_num != NPY_BOOL)) {
    PyErr_Format(PyExc_TypeError, "rotate: the masks must be of boolean type and have the same dimensions as src or dst images.");
    return 0;
  }

  switch (src->type_num){
    case NPY_UINT8:   if (src->ndim == 2) rotate_inner<uint8_t,2>(src, src_mask, dst, dst_mask, angle);  else rotate_inner<uint8_t,3>(src, src_mask, dst, dst_mask, angle); break;
    case NPY_UINT16:  if (src->ndim == 2) rotate_inner<uint16_t,2>(src, src_mask, dst, dst_mask, angle); else rotate_inner<uint16_t,3>(src, src_mask, dst, dst_mask, angle); break;
    case NPY_FLOAT64: if (src->ndim == 2) rotate_inner<double,2>(src, src_mask, dst, dst_mask, angle);   else rotate_inner<double,3>(src, src_mask, dst, dst_mask, angle); break;
    default:
      PyErr_Format(PyExc_TypeError, "rotate: src arrays of type %s are currently not supported", PyBlitzArray_TypenumAsString(src->type_num));
      return 0;
  }

  if (nargs == 2){
    return PyBlitzArray_AsNumpyArray(dst,0);
  }

  Py_RETURN_NONE;
  BOB_CATCH_FUNCTION("rotate", 0)
}


bob::extension::FunctionDoc s_rotatedOutputShape = bob::extension::FunctionDoc(
  "rotated_output_shape",
  "This function returns the shape of the rotated image for the given image and angle",
  0,
  true
)
.add_prototype("src, angle", "rotated_shape")
//.add_prototype("shape, scale", "scaled_shape")
.add_parameter("src", "array_like (2D,3D)", "The src image which which should be scaled")
.add_parameter("angle", "float", "The rotation angle **in degrees** to rotate the src image with")
//.add_parameter("shape", "(int, int) or (int, int, int)", "The shape of the input image which which should be scaled")
.add_return("rotated_shape", "(int, int) or (int, int, int)", "The shape of the rotated ``dst`` image required in a call to :py:func:`bob.ip.base.rotate`")
;

PyObject* PyBobIpBase_rotatedOutputShape(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = s_rotatedOutputShape.kwlist();
// TODO: implement from shape
//  static char* kwlist2[] = {c("shape"), c("scale"), 0};

  PyBlitzArrayObject* image = 0;
  double angle;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&d", kwlist1, &PyBlitzArray_Converter, &image, &angle)) return 0;

  auto image_ = make_safe(image);
  switch (image->ndim){
    case 2:{
      blitz::TinyVector<int,2> src_shape(image->shape[0], image->shape[1]);
      auto scaled_shape = bob::ip::base::getRotatedShape(src_shape, angle);
      return Py_BuildValue("(ii)", scaled_shape[0], scaled_shape[1]);
    }
    case 3:{
      blitz::TinyVector<int,3> src_shape(image->shape[0], image->shape[1], image->shape[2]);
      auto scaled_shape = bob::ip::base::getRotatedShape(src_shape, angle);
      return Py_BuildValue("(iii)", scaled_shape[0], scaled_shape[1], scaled_shape[2]);
    }
    default:
      PyErr_Format(PyExc_TypeError, "'rotated_output_shape' only accepts 2D or 3D arrays (not %" PY_FORMAT_SIZE_T "dD arrays)", image->ndim);
  }

  return 0;
  BOB_CATCH_FUNCTION("cannot get rotated output shape", 0)
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

PyObject* PyBobIpBase_maxRectInMask(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_maxRectInMask.kwlist();

  PyBlitzArrayObject* mask = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &mask)) return 0;

  auto mask_ = make_safe(mask);

  if (mask->ndim != 2 || mask->type_num != NPY_BOOL) {
    PyErr_Format(PyExc_TypeError, "max_rect_in_mask: the mask must be 2D and of boolean type");
    return 0;
  }

  auto rect = bob::ip::base::maxRectInMask(*PyBlitzArrayCxx_AsBlitz<bool, 2>(mask));

  return Py_BuildValue("(iiii)", rect[0], rect[1], rect[2], rect[3]);

  BOB_CATCH_FUNCTION("in max_rect_in_mask", 0)
}


bob::extension::FunctionDoc s_extrapolateMask = bob::extension::FunctionDoc(
  "extrapolate_mask",
  "Extrapolate a 2D array/image, taking a boolean mask into account",
  "The ``img`` argument is used both as an input and an output. "
  "Only values where the mask is set to false are extrapolated. "
  "The regions, where the mask is set to True is expected to be convex.\n\n"
  "This function can be called in two ways:\n\n"
  "The first way is by giving only the mask and the image. "
  "Then a nearest neighbor technique is used as:\n\n"
  "1. The columns of the image are firstly extrapolated wrt. to the nearest neighbour on the same column.\n"
  "2. The rows of the image are the extrapolate wrt. to the closest neighbour on the same row.\n\n"
  "The second way, the mask is interpolated by adding random values to the border pixels. "
  "The image is scanned in a spiral way, starting at the center of the masked area. "
  "When a pixel of the unmasked area is reached:\n\n"
  "1. The next pixel of the masked area is searched perpendicular to the current spiral direction\n"
  "2. From that pixel, ``neigbors`` pixels are extratced from the image in both sides of the current spiral direction, and a random value os choosen\n"
  "3. A normal distributed random value with mean 1 and standard deviation ``random_sigma`` is added to the pixel value\n"
  "4. The pixel value is set to the image at the current position\n\n"
  "Any action considering a random number will use the given ``rng`` to create random numbers.\n\n"
  ".. note::\n\n  For the second variant, images of type ``float`` are preferred."
)
.add_prototype("mask, img")
.add_prototype("mask, img, random_sigma, [neighbors], [rng]")
.add_parameter("mask", "array_like (2D, bool)", "The mask which has the valid pixel set to ``True`` and the invalid pixel set to ``False``")
.add_parameter("img", "array_like (2D, bool)", "The image that will be filled; must have the same shape as ``mask``")
.add_parameter("random_sigma", "float", "The standard deviation of the random factor to multiply thevalid pixel value from the border with; must be greater than or equal to 0")
.add_parameter("neighbors", "int", "[Default: 5] The number of neighbors of valid border pixels to choose one from; set ``neighbors=0`` to disable random selection")
.add_parameter("rng", ":py:class:`bob.core.random.mt19937`", "[Default: rng initialized with the system time] The random number generator to consider")
;

PyObject* PyBobIpBase_extrapolateMask(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_extrapolateMask.kwlist(1);

  PyBlitzArrayObject* mask,* img;
  double sigma = -1.;
  int neighbors = 5;
  PyBoostMt19937Object* rng = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|diO&", kwlist, &PyBlitzArray_Converter, &mask, &PyBlitzArray_OutputConverter, &img, &sigma, &neighbors, &PyBoostMt19937_Converter, &rng)) return 0;

  auto mask_ = make_safe(mask), img_ = make_safe(img);
  auto rng_ = make_xsafe(rng);

  if (!rng){
    rng = reinterpret_cast<PyBoostMt19937Object*>(PyBoostMt19937_SimpleNew());
    rng_ = make_safe(rng);
  }

  if (mask->ndim != 2 || mask->type_num != NPY_BOOL) {
    PyErr_Format(PyExc_TypeError, "extrapolate_mask: the mask must be 2D and of boolean type");
    return 0;
  }
  if (img->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "extrapolate_mask: the img must be 2D");
    return 0;
  }

  if (sigma < 0){
    // first variant
    switch (img->type_num){
      case NPY_UINT8:   bob::ip::base::extrapolateMask(*PyBlitzArrayCxx_AsBlitz<bool, 2>(mask), *PyBlitzArrayCxx_AsBlitz<uint8_t, 2>(img)); break;
      case NPY_UINT16:  bob::ip::base::extrapolateMask(*PyBlitzArrayCxx_AsBlitz<bool, 2>(mask), *PyBlitzArrayCxx_AsBlitz<uint16_t, 2>(img)); break;
      case NPY_FLOAT64: bob::ip::base::extrapolateMask(*PyBlitzArrayCxx_AsBlitz<bool, 2>(mask), *PyBlitzArrayCxx_AsBlitz<double, 2>(img)); break;
      default:
        PyErr_Format(PyExc_TypeError, "extrapolate_mask: img arrays of type %s are currently not supported", PyBlitzArray_TypenumAsString(img->type_num));
        return 0;
    }
  } else {
    // second variant
    switch (img->type_num){
      case NPY_UINT8:   bob::ip::base::extrapolateMaskRandom(*PyBlitzArrayCxx_AsBlitz<bool, 2>(mask), *PyBlitzArrayCxx_AsBlitz<uint8_t, 2>(img), *rng->rng, sigma, neighbors); break;
      case NPY_UINT16:  bob::ip::base::extrapolateMaskRandom(*PyBlitzArrayCxx_AsBlitz<bool, 2>(mask), *PyBlitzArrayCxx_AsBlitz<uint16_t, 2>(img), *rng->rng, sigma, neighbors); break;
      case NPY_FLOAT64: bob::ip::base::extrapolateMaskRandom(*PyBlitzArrayCxx_AsBlitz<bool, 2>(mask), *PyBlitzArrayCxx_AsBlitz<double, 2>(img), *rng->rng, sigma, neighbors); break;
      default:
        PyErr_Format(PyExc_TypeError, "extrapolate_mask: img arrays of type %s are currently not supported", PyBlitzArray_TypenumAsString(img->type_num));
        return 0;
    }
  }

  Py_RETURN_NONE;

  BOB_CATCH_FUNCTION("in extrapolate_mask", 0)
}
