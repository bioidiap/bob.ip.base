/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Thu Jul  3 13:30:38 CEST 2014
 *
 * @brief Binds image filter functions of bob::ip::base class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


#include "main.h"
#include "cpp/Median.h"

static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns false if object is NULL */


bob::extension::FunctionDoc s_median = bob::extension::FunctionDoc(
  "median",
  "Performs a median filtering of the input image with the given radius",
  "This function performs a median filtering of the given ``src`` image with the given radius and writes the result to the given ``dst`` image. "
  "Both gray-level and color images are supported, and the input and output datatype must be identical.\n\n"
  "Median filtering iterates with a mask of size ``(2*radius[0]+1, 2*radius[1]+1)`` over the input image. "
  "For each input region, the pixels under the mask are sorted and the median value (the middle element of the sorted list) is written into the ``dst`` image. "
  "Therefore, the ``dst`` is smaller than the ``src`` image, i.e., by ``2*radius`` pixels."
)
.add_prototype("src, radius, [dst]", "dst")
.add_parameter("src", "array_like (2D or 3D)", "The source image to filter, might be a gray level image or a color image")
.add_parameter("radius", "(int, int)", "The radius of the median filter; the final filter will have the size ``(2*radius[0]+1, 2*radius[1]+1)``")
.add_parameter("dst", "array_like (2D or 3D)", "The median-filtered image to write; need to be of size ``src.shape - 2*radius``; if not specified, it will be created")
.add_return("dst", "array_like (2D or 3D)", "The median-filtered image; the same as the ``dst`` parameter, if specified")
;

template <typename T, int D> PyObject* inner_median(PyBlitzArrayObject* src, PyBlitzArrayObject* dst, const blitz::TinyVector<int,2>& radius) {
  bob::ip::base::medianFilter(*PyBlitzArrayCxx_AsBlitz<T, D>(src), *PyBlitzArrayCxx_AsBlitz<T, D>(dst), radius);
  Py_INCREF(dst);
  return PyBlitzArray_AsNumpyArray(dst, 0);
}

PyObject* PyBobIpBase_median(PyObject*, PyObject* args, PyObject* kwargs) {
  TRY

  static char* kwlist[] = {c("src"), c("radius"), c("dst"), NULL};

  PyBlitzArrayObject* src,* dst = 0;
  blitz::TinyVector<int,2> radius;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&(ii)|O&", kwlist, &PyBlitzArray_Converter, &src, &radius[0], &radius[1], &PyBlitzArray_OutputConverter, &dst)) return 0;

  auto src_ = make_safe(src), dst_ = make_xsafe(dst);

  // allocate the output, if needed
  if (!dst){
    if (src->ndim == 2){
      Py_ssize_t n[] = {src->shape[0] - 2*radius[0], src->shape[1] - 2*radius[1]};
      dst = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(src->type_num, 2, n));
    } else if (src->ndim == 3){
      Py_ssize_t n[] = {src->shape[0], src->shape[1] - 2*radius[0], src->shape[2] - 2*radius[1]};
      dst = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(src->type_num, 3, n));
    } else {
      PyErr_Format(PyExc_TypeError, "'median' : only 2D or 3D arrays are supported.");
      return 0;
    }
    dst_ = make_safe(dst);
  } else {
    if (dst->type_num != src->type_num || dst->ndim != src->ndim){
      PyErr_Format(PyExc_TypeError, "'median' : 'src' and 'dst' images must have the same type and number of dimensions, but %s != %s or %d != %d.", PyBlitzArray_TypenumAsString(src->type_num), PyBlitzArray_TypenumAsString(dst->type_num), (int)src->ndim, (int)dst->ndim);
      return 0;
    }
  }

  // compute the median
  switch (src->type_num){
    case NPY_UINT8:   if (src->ndim == 2) return inner_median<uint8_t,2>(src, dst, radius);  else return inner_median<uint8_t,3>(src, dst, radius);
    case NPY_UINT16:  if (src->ndim == 2) return inner_median<uint16_t,2>(src, dst, radius); else return inner_median<uint16_t,3>(src, dst, radius);
    case NPY_FLOAT16: if (src->ndim == 2) return inner_median<double,2>(src, dst, radius);   else return inner_median<double,3>(src, dst, radius);
    default:
      PyErr_Format(PyExc_ValueError, "'median' of %s arrays is currently not supported, only uint8, uint16 or float64 arrays are", PyBlitzArray_TypenumAsString(src->type_num));
      return 0;
  }

  CATCH_("in median", 0)
}


