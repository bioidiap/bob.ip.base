/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Tue Jun 24 14:03:17 CEST 2014
 *
 * @brief Binds the GeomNorm class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto GeomNorm_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".GeomNorm",
  "Objects of this class, after configuration, can perform a geometric normalization of images",
  "The geometric normalization is a combination of rotation, scaling and cropping an image."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a GeomNorm object with the given scale, angle, size of the new image and transformation offset in the new image",
    "When the GeomNorm is applied to an image, it is rotated and scaled such that it **visually** rotated counter-clock-wise (mathematically positive) with the given angle, i.e., to mimic the behavior of ImageMagick. "
    "Since the origin in the image is in the top-left corner, this means that the rotation is **actually** clock-wise (mathematically negative). "
    "This **also applies** for the second version of the landmarks, which will be rotated mathematically negative as well, to keep it consistent with the image.\n\n"
    ".. warning:: The behavior of the landmark rotation has changed from Bob version 1.x, where the landmarks were mistakenly rotated mathematically positive.",
    true
  )
  .add_prototype("rotation_angle, scaling_factor, crop_size, crop_offset", "")
  .add_prototype("other", "")
  .add_parameter("rotation_angle", "float", "The rotation angle **in degrees** that should be applied")
  .add_parameter("scaling_factor", "float", "The scale factor to apply")
  .add_parameter("crop_size", "(int, int)", "The resolution of the processed images")
  .add_parameter("crop_offset", "(float, float)", "The transformation offset in the processed images")
  .add_parameter("other", ":py:class:`GeomNorm`", "Another GeomNorm object to copy")
);


static int PyBobIpBaseGeomNorm_init(PyBobIpBaseGeomNormObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = GeomNorm_doc.kwlist(0);
  char** kwlist2 = GeomNorm_doc.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if (!nargs){
    // at least one argument is required
    GeomNorm_doc.print_usage();
    PyErr_Format(PyExc_TypeError, "`%s' constructor requires at least one parameter", Py_TYPE(self)->tp_name);
    return -1;
  } // nargs == 0

  if (nargs == 1){
    // copy constructor
    PyBobIpBaseGeomNormObject* geomNorm;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseGeomNorm_Type, &geomNorm)){
      GeomNorm_doc.print_usage();
      return -1;
    }
    self->cxx.reset(new bob::ip::base::GeomNorm(*geomNorm->cxx));
    return 0;
  } // nargs == 1

  double scale, angle;
  blitz::TinyVector<int,2> size;
  blitz::TinyVector<double,2> offset;
  // more than one parameter; check the second one
  if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "dd(ii)(dd)", kwlist1, &angle, &scale, &size[0], &size[1], &offset[0], &offset[1]))){
    GeomNorm_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::ip::base::GeomNorm(angle, scale, size, offset));
  return 0;

  BOB_CATCH_MEMBER("cannot create GeomNorm object", -1)
}

static void PyBobIpBaseGeomNorm_delete(PyBobIpBaseGeomNormObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseGeomNorm_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseGeomNorm_Type));
}

static PyObject* PyBobIpBaseGeomNorm_RichCompare(PyBobIpBaseGeomNormObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseGeomNorm_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseGeomNormObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare GeomNorm objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto angle = bob::extension::VariableDoc(
  "rotation_angle",
  "float",
  "The rotation angle, with read and write access"
);
PyObject* PyBobIpBaseGeomNorm_getAngle(PyBobIpBaseGeomNormObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getRotationAngle());
  BOB_CATCH_MEMBER("rotation_angle could not be read", 0)
}
int PyBobIpBaseGeomNorm_setAngle(PyBobIpBaseGeomNormObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setRotationAngle(d);
  return 0;
  BOB_CATCH_MEMBER("rotation_angle could not be set", -1)
}

static auto scale = bob::extension::VariableDoc(
  "scaling_factor",
  "float",
  "The scale factor, with read and write access"
);
PyObject* PyBobIpBaseGeomNorm_getScale(PyBobIpBaseGeomNormObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getScalingFactor());
  BOB_CATCH_MEMBER("scaling_factor could not be read", 0)
}
int PyBobIpBaseGeomNorm_setScale(PyBobIpBaseGeomNormObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setScalingFactor(d);
  return 0;
  BOB_CATCH_MEMBER("scaling_factor could not be set", -1)
}

static auto cropSize = bob::extension::VariableDoc(
  "crop_size",
  "(int, int)",
  "The size of the processed image, with read and write access"
);
PyObject* PyBobIpBaseGeomNorm_getCropSize(PyBobIpBaseGeomNormObject* self, void*){
  BOB_TRY
  auto r = self->cxx->getCropSize();
  return Py_BuildValue("(ii)", r[0], r[1]);
  BOB_CATCH_MEMBER("crop_size could not be read", 0)
}
int PyBobIpBaseGeomNorm_setCropSize(PyBobIpBaseGeomNormObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<double,2> r;
  if (!PyArg_ParseTuple(value, "dd", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, cropSize.name());
    return -1;
  }
  self->cxx->setCropSize(r);
  return 0;
  BOB_CATCH_MEMBER("crop_size could not be set", -1)
}

static auto cropOffset = bob::extension::VariableDoc(
  "crop_offset",
  "(float, float)",
  "The transformation center in the processed image, with read and write access"
);
PyObject* PyBobIpBaseGeomNorm_getCropOffset(PyBobIpBaseGeomNormObject* self, void*){
  BOB_TRY
  auto r = self->cxx->getCropOffset();
  return Py_BuildValue("(dd)", r[0], r[1]);
  BOB_CATCH_MEMBER("crop_offset could not be read", 0)
}
int PyBobIpBaseGeomNorm_setCropOffset(PyBobIpBaseGeomNormObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<double,2> r;
  if (!PyArg_ParseTuple(value, "dd", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, cropOffset.name());
    return -1;
  }
  self->cxx->setCropOffset(r);
  return 0;
  BOB_CATCH_MEMBER("crop_offset could not be set", -1)
}

static PyGetSetDef PyBobIpBaseGeomNorm_getseters[] = {
    {
      angle.name(),
      (getter)PyBobIpBaseGeomNorm_getAngle,
      (setter)PyBobIpBaseGeomNorm_setAngle,
      angle.doc(),
      0
    },
    {
      scale.name(),
      (getter)PyBobIpBaseGeomNorm_getScale,
      (setter)PyBobIpBaseGeomNorm_setScale,
      scale.doc(),
      0
    },
    {
      cropSize.name(),
      (getter)PyBobIpBaseGeomNorm_getCropSize,
      (setter)PyBobIpBaseGeomNorm_setCropSize,
      cropSize.doc(),
      0
    },
    {
      cropOffset.name(),
      (getter)PyBobIpBaseGeomNorm_getCropOffset,
      (setter)PyBobIpBaseGeomNorm_setCropOffset,
      cropOffset.doc(),
      0
    },
    {0}  /* Sentinel */
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto process = bob::extension::FunctionDoc(
  "process",
  "This function geometrically normalizes an image or a position in the image",
  "The function rotates and scales the given image, or a position in image coordinates, such that the result is **visually** rotated and scaled with the :py:attr:`rotation_angle` and :py:attr:`scaling_factor`.\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("input, output, center")
.add_prototype("input, input_mask, output, output_mask, center")
.add_prototype("position, center", "transformed")
.add_parameter("input", "array_like (2D or 3D)", "The input image to which GeomNorm should be applied")
.add_parameter("output", "array_like (2D or 3D, float)", "The output image, which must be of size :py:attr:`crop_size`")
.add_parameter("center", "(float, float)", "The transformation center in the given image; this will be placed to :py:attr:`crop_offset` in the output image")
.add_parameter("input_mask", "array_like (bool, 2D or 3D)", "An input mask of valid pixels before geometric normalization, must be of same size as ``input``")
.add_parameter("output_mask", "array_like (bool, 2D or 3D)", "The output mask of valid pixels after geometric normalization, must be of same size as ``output``")
.add_parameter("position", "(float, float)", "A position in input image space that will be transformed to output image space (might be outside of the crop area)")
.add_return("transformed", "uint16", "The resulting GeomNorm code at the given position in the image")
;

template <typename T>
static PyObject* process_inner(PyBobIpBaseGeomNormObject* self, PyBlitzArrayObject* input, PyBlitzArrayObject* input_mask, PyBlitzArrayObject* output, PyBlitzArrayObject* output_mask, const blitz::TinyVector<double,2>& offset){
  if (input_mask && output_mask){
    self->cxx->process(*PyBlitzArrayCxx_AsBlitz<T,2>(input), *PyBlitzArrayCxx_AsBlitz<bool,2>(input_mask), *PyBlitzArrayCxx_AsBlitz<double,2>(output), *PyBlitzArrayCxx_AsBlitz<bool,2>(output_mask), offset);
  } else {
    self->cxx->process(*PyBlitzArrayCxx_AsBlitz<T,2>(input), *PyBlitzArrayCxx_AsBlitz<double,2>(output), offset);
  }
  Py_RETURN_NONE;
}

static PyObject* PyBobIpBaseGeomNorm_process(PyBobIpBaseGeomNormObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist1 = process.kwlist(0);
  char** kwlist2 = process.kwlist(1);
  char** kwlist3 = process.kwlist(2);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyBlitzArrayObject* input = 0,* input_mask = 0,* output = 0,* output_mask = 0;
  blitz::TinyVector<double,2> center(0,0), position(0,0);

  switch (nargs){
    case 2:{
      // with position
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(dd)(dd)", kwlist3, &position[0], &position[1], &center[0], &center[1])){
        process.print_usage();
        return 0;
      }
      auto transformed = self->cxx->process(position, center);
      // return here
      return Py_BuildValue("(dd)", transformed[0], transformed[1]);
    }
    case 3:{
      // with input and output array
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&(dd)", kwlist1, &PyBlitzArray_Converter, &input, &PyBlitzArray_OutputConverter, &output, &center[0], &center[1])){
        process.print_usage();
        return 0;
      }
      break;
    }
    case 5:{
      // with mask
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&O&O&(dd)", kwlist2, &PyBlitzArray_Converter, &input, &PyBlitzArray_Converter, &input_mask, &PyBlitzArray_OutputConverter, &output, &PyBlitzArray_OutputConverter, &output_mask, &center[0], &center[1])){
        process.print_usage();
        return 0;
      }
      break;
    }
    default:{
      process.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' process called with wrong number of parameters", Py_TYPE(self)->tp_name);
      return 0;
    }
  } // switch

  auto input_ = make_safe(input), output_ = make_safe(output);
  auto input_mask_ = make_xsafe(input_mask), output_mask_ = make_xsafe(output_mask);

  // perform checks on input and output image
  if (input->ndim != 2 && input->ndim != 3){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D or 3D arrays", Py_TYPE(self)->tp_name);
    process.print_usage();
    return 0;
  }
  if (output->ndim != input->ndim){
    PyErr_Format(PyExc_TypeError, "`%s' processes only input and output arrays with the same number of dimensions", Py_TYPE(self)->tp_name);
    process.print_usage();
    return 0;
  }
  if (output->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' processes only output arrays of type float", Py_TYPE(self)->tp_name);
    process.print_usage();
    return 0;
  }

  if (input_mask && output_mask){
    if (input_mask->ndim != input->ndim || output_mask->ndim != output->ndim){
      PyErr_Format(PyExc_TypeError, "`%s' masks must have the same shape as the input matrix", Py_TYPE(self)->tp_name);
      process.print_usage();
      return 0;
    }
    if (input_mask->type_num != NPY_BOOL || output_mask->type_num != NPY_BOOL){
      PyErr_Format(PyExc_TypeError, "`%s' masks must be of boolean type", Py_TYPE(self)->tp_name);
      process.print_usage();
      return 0;
    }
  }

  // finally, process the data
  switch (input->type_num){
    case NPY_UINT8:   return process_inner<uint8_t>(self, input, input_mask, output, output_mask, center);
    case NPY_UINT16:  return process_inner<uint16_t>(self, input, input_mask, output, output_mask, center);
    case NPY_FLOAT64: return process_inner<double>(self, input, input_mask, output, output_mask, center);
    default:
      PyErr_Format(PyExc_TypeError, "`%s' input array of type %s are currently not supported", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(input->type_num));
      process.print_usage();
      return 0;
  }

  BOB_CATCH_MEMBER("cannot process image", 0)
}

static PyMethodDef PyBobIpBaseGeomNorm_methods[] = {
  {
    process.name(),
    (PyCFunction)PyBobIpBaseGeomNorm_process,
    METH_VARARGS|METH_KEYWORDS,
    process.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the GeomNorm type struct; will be initialized later
PyTypeObject PyBobIpBaseGeomNorm_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseGeomNorm(PyObject* module)
{
  // initialize the type struct
  PyBobIpBaseGeomNorm_Type.tp_name = GeomNorm_doc.name();
  PyBobIpBaseGeomNorm_Type.tp_basicsize = sizeof(PyBobIpBaseGeomNormObject);
  PyBobIpBaseGeomNorm_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseGeomNorm_Type.tp_doc = GeomNorm_doc.doc();

  // set the functions
  PyBobIpBaseGeomNorm_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseGeomNorm_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseGeomNorm_init);
  PyBobIpBaseGeomNorm_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseGeomNorm_delete);
  PyBobIpBaseGeomNorm_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseGeomNorm_RichCompare);
  PyBobIpBaseGeomNorm_Type.tp_methods = PyBobIpBaseGeomNorm_methods;
  PyBobIpBaseGeomNorm_Type.tp_getset = PyBobIpBaseGeomNorm_getseters;
  PyBobIpBaseGeomNorm_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseGeomNorm_process);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseGeomNorm_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseGeomNorm_Type);
  return PyModule_AddObject(module, "GeomNorm", (PyObject*)&PyBobIpBaseGeomNorm_Type) >= 0;
}
