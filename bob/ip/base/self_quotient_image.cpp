/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Fri Jul  4 09:55:18 CEST 2014
 *
 * @brief Binds the SelfQuotientImage class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto SelfQuotientImage_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".SelfQuotientImage",
  "This class allows after configuration to apply the Self Quotient Image algorithm to images",
  "Details of the Self Quotient Image algorithm is described in [Wang2004]_."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates an object to preprocess images with the Self Quotient Image algorithm",
    ".. todo:: explain SelfQuotientImage constructor\n\n"
    ".. warning:: Compared to the last Bob version, here the sigma parameter is the **standard deviation** and not the variance. "
    "This includes that the :py:class:`WeightedGaussian` pyramid is **different**, see https://gitlab.idiap.ch/bob/bob.ip.base/issues/1.",
    true
  )
  .add_prototype("[scales], [size_min], [size_step], [sigma], [border]","")
  .add_prototype("sqi", "")
  .add_parameter("scales", "int", "[default: 1] The number of scales (:py:class:`bob.ip.base.WeightedGaussian`)")
  .add_parameter("size_min", "int", "[default: 1] The radius of the kernel of the smallest :py:class:`bob.ip.base.WeightedGaussian`")
  .add_parameter("size_step", "int", "[default: 1] The step used to set the kernel size of other weighted Gaussians: ``size_s = 2 * (size_min + s * size_step) + 1``")
  .add_parameter("sigma", "double", "[default: ``math.sqrt(2.)``] The standard deviation of the kernel of the smallest weighted Gaussian; other sigmas: ``sigma_s = sigma * (size_min + s * size_step) / size_min``")
  .add_parameter("border", ":py:class:`bob.sp.BorderType`", "[default: ``bob.sp.BorderType.Mirror``] The extrapolation method used by the convolution at the border")
  .add_parameter("sqi", ":py:class:`bob.ip.base.SelfQuotientImage`", "The ``SelfQuotientImage`` object to use for copy-construction")
);

static int PyBobIpBaseSelfQuotientImage_init(PyBobIpBaseSelfQuotientImageObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = SelfQuotientImage_doc.kwlist(0);
  char** kwlist2 = SelfQuotientImage_doc.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseSelfQuotientImage_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseSelfQuotientImageObject* sqi;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseSelfQuotientImage_Type, &sqi)) return -1;

    self->cxx.reset(new bob::ip::base::SelfQuotientImage(*sqi->cxx));
    return 0;
  }

  int scales = 1, size_min = 1, size_step = 1;
  double sigma = sqrt(2.);
  bob::sp::Extrapolation::BorderType border = bob::sp::Extrapolation::Mirror;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiidO&", kwlist1, &scales, &size_min, &size_step, &sigma, &PyBobSpExtrapolationBorder_Converter, &border)){
    SelfQuotientImage_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::ip::base::SelfQuotientImage(scales, size_min, size_step, sigma, border));
  return 0;

  BOB_CATCH_MEMBER("cannot create SelfQuotientImage", -1)
}

static void PyBobIpBaseSelfQuotientImage_delete(PyBobIpBaseSelfQuotientImageObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseSelfQuotientImage_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseSelfQuotientImage_Type));
}

static PyObject* PyBobIpBaseSelfQuotientImage_RichCompare(PyBobIpBaseSelfQuotientImageObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseSelfQuotientImage_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseSelfQuotientImageObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare SelfQuotientImage objects", 0)
}

/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto scales = bob::extension::VariableDoc(
  "scales",
  "int",
  "The number of scales (Weighted Gaussian); with read and write access"
);
PyObject* PyBobIpBaseSelfQuotientImage_getScales(PyBobIpBaseSelfQuotientImageObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNScales());
  BOB_CATCH_MEMBER("scales could not be read", 0)
}
int PyBobIpBaseSelfQuotientImage_setScales(PyBobIpBaseSelfQuotientImageObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, scales.name());
    return -1;
  }
  self->cxx->setNScales(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("scales could not be set", -1)
}


static auto sizeMin = bob::extension::VariableDoc(
  "size_min",
  "int",
  "The radius (size=2*radius+1) of the kernel of the smallest weighted Gaussian; with read and write access"
);
PyObject* PyBobIpBaseSelfQuotientImage_getSizeMin(PyBobIpBaseSelfQuotientImageObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getSizeMin());
  BOB_CATCH_MEMBER("size_min could not be read", 0)
}
int PyBobIpBaseSelfQuotientImage_setSizeMin(PyBobIpBaseSelfQuotientImageObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, sizeMin.name());
    return -1;
  }
  self->cxx->setSizeMin(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("size_min could not be set", -1)
}

static auto sizeStep = bob::extension::VariableDoc(
  "size_step",
  "int",
  "The step used to set the kernel size of other Weighted Gaussians (size_s=2*(size_min+s*size_step)+1); with read and write access"
);
PyObject* PyBobIpBaseSelfQuotientImage_getSizeStep(PyBobIpBaseSelfQuotientImageObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getSizeStep());
  BOB_CATCH_MEMBER("size_step could not be read", 0)
}
int PyBobIpBaseSelfQuotientImage_setSizeStep(PyBobIpBaseSelfQuotientImageObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, sizeStep.name());
    return -1;
  }
  self->cxx->setSizeStep(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("size_step could not be set", -1)
}


static auto sigma = bob::extension::VariableDoc(
  "sigma",
  "float",
  "The standard deviation of the kernel of the smallest weighted Gaussian (sigma_s = sigma * (size_min+s*size_step)/size_min); with read and write access"
);
PyObject* PyBobIpBaseSelfQuotientImage_getSigma(PyBobIpBaseSelfQuotientImageObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getSigma());
  BOB_CATCH_MEMBER("sigma could not be read", 0)
}
int PyBobIpBaseSelfQuotientImage_setSigma(PyBobIpBaseSelfQuotientImageObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setSigma(d);
  return 0;
  BOB_CATCH_MEMBER("sigma could not be set", -1)
}

static auto border = bob::extension::VariableDoc(
  "border",
  ":py:class:`bob.sp.BorderType`",
  "The extrapolation method used by the convolution at the border; with read and write access"
);
PyObject* PyBobIpBaseSelfQuotientImage_getBorder(PyBobIpBaseSelfQuotientImageObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getConvBorder());
  BOB_CATCH_MEMBER("border could not be read", 0)
}
int PyBobIpBaseSelfQuotientImage_setBorder(PyBobIpBaseSelfQuotientImageObject* self, PyObject* value, void*){
  BOB_TRY
  bob::sp::Extrapolation::BorderType b;
  if (!PyBobSpExtrapolationBorder_Converter(value, &b)) return -1;
  self->cxx->setConvBorder(b);
  return 0;
  BOB_CATCH_MEMBER("border could not be set", -1)
}

static PyGetSetDef PyBobIpBaseSelfQuotientImage_getseters[] = {
    {
      scales.name(),
      (getter)PyBobIpBaseSelfQuotientImage_getScales,
      (setter)PyBobIpBaseSelfQuotientImage_setScales,
      scales.doc(),
      0
    },
    {
      sizeMin.name(),
      (getter)PyBobIpBaseSelfQuotientImage_getSizeMin,
      (setter)PyBobIpBaseSelfQuotientImage_setSizeMin,
      sizeMin.doc(),
      0
    },
    {
      sizeStep.name(),
      (getter)PyBobIpBaseSelfQuotientImage_getSizeStep,
      (setter)PyBobIpBaseSelfQuotientImage_setSizeStep,
      sizeStep.doc(),
      0
    },
    {
      sigma.name(),
      (getter)PyBobIpBaseSelfQuotientImage_getSigma,
      (setter)PyBobIpBaseSelfQuotientImage_setSigma,
      sigma.doc(),
      0
    },
    {
      border.name(),
      (getter)PyBobIpBaseSelfQuotientImage_getBorder,
      (setter)PyBobIpBaseSelfQuotientImage_setBorder,
      border.doc(),
      0
    },
    {0}  /* Sentinel */
};



/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto process = bob::extension::FunctionDoc(
  "process",
  "Applies the Self Quotient Image algorithm to an image (2D/grayscale or 3D/color) of type uint8, uint16 or double",
  "If given, the ``dst`` array should have the type float and the same size as the ``src`` array.\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like (2D)", "The input image which should be processed")
.add_parameter("dst", "array_like (2D, float)", "[default: ``None``] If given, the output will be saved into this image; must be of the same shape as ``src``")
.add_return("dst", "array_like (2D, float)", "The resulting output image, which is the same as ``dst`` (if given)")
;

template <typename T, int D>
static PyObject* process_inner(PyBobIpBaseSelfQuotientImageObject* self, PyBlitzArrayObject* input, PyBlitzArrayObject* output){
  self->cxx->process(*PyBlitzArrayCxx_AsBlitz<T,D>(input), *PyBlitzArrayCxx_AsBlitz<double,D>(output));
  return PyBlitzArray_AsNumpyArray(output, 0);
}

static PyObject* PyBobIpBaseSelfQuotientImage_process(PyBobIpBaseSelfQuotientImageObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = process.kwlist();

  PyBlitzArrayObject* src,* dst = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O&", kwlist, &PyBlitzArray_Converter, &src, &PyBlitzArray_OutputConverter, &dst)) return 0;

  auto src_ = make_safe(src), dst_ = make_xsafe(dst);

  // perform checks on input and output image
  if (src->ndim != 2 && src->ndim != 3){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D or 3D arrays", Py_TYPE(self)->tp_name);
    process.print_usage();
    return 0;
  }

  if (dst){
    if (dst->ndim != src->ndim){
      PyErr_Format(PyExc_TypeError, "`%s' 'src' and 'dst' shape has to be identical", Py_TYPE(self)->tp_name);
      process.print_usage();
      return 0;
    }
    if (dst->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "`%s' only processes to arrays of type float", Py_TYPE(self)->tp_name);
      process.print_usage();
      return 0;
    }
  } else {
    // create output in desired shape
    dst = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, src->ndim, src->shape);
    dst_ = make_safe(dst);
  }

  // finally, extract the features
  switch (src->type_num){
    case NPY_UINT8:   if (src->ndim == 2) return process_inner<uint8_t,2>(self, src, dst);  else return process_inner<uint8_t,3>(self, src, dst);
    case NPY_UINT16:  if (src->ndim == 2) return process_inner<uint16_t,2>(self, src, dst); else return process_inner<uint16_t,3>(self, src, dst);
    case NPY_FLOAT64: if (src->ndim == 2) return process_inner<double,2>(self, src, dst);   else return process_inner<double,3>(self, src, dst);
    default:
      process.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' processes only images of types uint8, uint16 or float, and not from %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(src->type_num));
      return 0;
  }

  BOB_CATCH_MEMBER("cannot perform Self Quotient Image processing in image", 0)
}


static PyMethodDef PyBobIpBaseSelfQuotientImage_methods[] = {
  {
    process.name(),
    (PyCFunction)PyBobIpBaseSelfQuotientImage_process,
    METH_VARARGS|METH_KEYWORDS,
    process.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the SelfQuotientImage type struct; will be initialized later
PyTypeObject PyBobIpBaseSelfQuotientImage_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseSelfQuotientImage(PyObject* module)
{
  // initialize the type struct
  PyBobIpBaseSelfQuotientImage_Type.tp_name = SelfQuotientImage_doc.name();
  PyBobIpBaseSelfQuotientImage_Type.tp_basicsize = sizeof(PyBobIpBaseSelfQuotientImageObject);
  PyBobIpBaseSelfQuotientImage_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseSelfQuotientImage_Type.tp_doc = SelfQuotientImage_doc.doc();

  // set the functions
  PyBobIpBaseSelfQuotientImage_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseSelfQuotientImage_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseSelfQuotientImage_init);
  PyBobIpBaseSelfQuotientImage_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseSelfQuotientImage_delete);
  PyBobIpBaseSelfQuotientImage_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseSelfQuotientImage_RichCompare);
  PyBobIpBaseSelfQuotientImage_Type.tp_methods = PyBobIpBaseSelfQuotientImage_methods;
  PyBobIpBaseSelfQuotientImage_Type.tp_getset = PyBobIpBaseSelfQuotientImage_getseters;
  PyBobIpBaseSelfQuotientImage_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseSelfQuotientImage_process);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseSelfQuotientImage_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseSelfQuotientImage_Type);
  return PyModule_AddObject(module, "SelfQuotientImage", (PyObject*)&PyBobIpBaseSelfQuotientImage_Type) >= 0;
}
