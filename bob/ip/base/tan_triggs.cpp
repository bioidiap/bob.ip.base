/**
 * @author Manuel Guenther <manuel.guenthr@idiap.ch>
 * @date Wed Jul  2 14:38:18 CEST 2014
 *
 * @brief Binds the TanTriggs class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

static inline bool t(PyObject* o){return o == 0 || PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns true if object is NULL */
static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns false if object is NULL */

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto TanTriggs_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".TanTriggs",
  "Objects of this class, after configuration, can preprocess images",
  "It does this using the method described by Tan and Triggs in the paper [TanTriggs2007]_."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new Tan and Triggs filter",
    ".. todo:: Explain TanTriggs constructor in more detail.",
    true
  )
  .add_prototype("[gamma], [sigma0], [sigma1], [radius], [threshold], [alpha], [border]","")
  .add_prototype("tan_triggs", "")
  .add_parameter("gamma", "float", "[default: ``0.2``] The value of gamma for the gamma correction")
  .add_parameter("sigma0", "float", "[default: ``1.``] The standard deviation of the inner Gaussian")
  .add_parameter("sigma1", "float", "[default: ``2.``] The standard deviation of the outer Gaussian")
  .add_parameter("radius", "int", "[default: ``2``] The radius of the Difference of Gaussians filter along both axes (size of the kernel=2*radius+1)")
  .add_parameter("threshold", "float", "[default: ``10.``] The threshold used for the contrast equalization")
  .add_parameter("alpha", "float", "[default: ``0.1``] The alpha value used for the contrast equalization")
  .add_parameter("border", ":py:class:`bob.sp.BorderType`", "[default: ``bob.sp.BorderType.Mirror``] The extrapolation method used by the convolution at the border")
  .add_parameter("tan_triggs", ":py:class:`bob.ip.base.TanTriggs`", "The TanTriggs object to use for copy-construction")
);


static int PyBobIpBaseTanTriggs_init(PyBobIpBaseTanTriggsObject* self, PyObject* args, PyObject* kwargs) {
  TRY

  char* kwlist1[] = {c("gamma"), c("sigma0"), c("sigma1"), c("radius"), c("threshold"), c("alpha"), c("border"), NULL};
  char* kwlist2[] = {c("tan_triggs"), NULL};

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseTanTriggs_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseTanTriggsObject* tt;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseTanTriggsType, &tt)) return -1;

    self->cxx.reset(new bob::ip::base::TanTriggs(*tt->cxx));
    return 0;
  }

  double gamma = 0.2, sigma0 = 1., sigma1 = 2., threshold = 10., alpha = 0.1;
  int radius = 2;
  bob::sp::Extrapolation::BorderType border = bob::sp::Extrapolation::Mirror;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|dddiddO&", kwlist1, &gamma, &sigma0, &sigma1, &radius, &threshold, &alpha, &PyBobSpExtrapolationBorder_Converter, &border)){
    TanTriggs_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::ip::base::TanTriggs(gamma, sigma0, sigma1, radius, threshold, alpha, border));
  return 0;

  CATCH("cannot create TanTriggs", -1)
}

static void PyBobIpBaseTanTriggs_delete(PyBobIpBaseTanTriggsObject* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseTanTriggs_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseTanTriggsType));
}

static PyObject* PyBobIpBaseTanTriggs_RichCompare(PyBobIpBaseTanTriggsObject* self, PyObject* other, int op) {
  TRY

  if (!PyBobIpBaseTanTriggs_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseTanTriggsObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  CATCH("cannot compare TanTriggs objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto gamma_ = bob::extension::VariableDoc(
  "gamma",
  "float",
  "The value of gamma for the gamma correction, with read and write access"
);
PyObject* PyBobIpBaseTanTriggs_getGamma(PyBobIpBaseTanTriggsObject* self, void*){
  TRY
  return Py_BuildValue("d", self->cxx->getGamma());
  CATCH("gamma could not be read", 0)
}
int PyBobIpBaseTanTriggs_setGamma(PyBobIpBaseTanTriggsObject* self, PyObject* value, void*){
  TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setGamma(d);
  return 0;
  CATCH("gamma could not be set", -1)
}

static auto sigma0 = bob::extension::VariableDoc(
  "sigma0",
  "float",
  "The standard deviation of the inner Gaussian, with read and write access"
);
PyObject* PyBobIpBaseTanTriggs_getSigma0(PyBobIpBaseTanTriggsObject* self, void*){
  TRY
  return Py_BuildValue("d", self->cxx->getSigma0());
  CATCH("sigma0 could not be read", 0)
}
int PyBobIpBaseTanTriggs_setSigma0(PyBobIpBaseTanTriggsObject* self, PyObject* value, void*){
  TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setSigma0(d);
  return 0;
  CATCH("sigma0 could not be set", -1)
}

static auto sigma1 = bob::extension::VariableDoc(
  "sigma1",
  "float",
  "The standard deviation of the inner Gaussian, with read and write access"
);
PyObject* PyBobIpBaseTanTriggs_getSigma1(PyBobIpBaseTanTriggsObject* self, void*){
  TRY
  return Py_BuildValue("d", self->cxx->getSigma1());
  CATCH("sigma0 could not be read", 0)
}
int PyBobIpBaseTanTriggs_setSigma1(PyBobIpBaseTanTriggsObject* self, PyObject* value, void*){
  TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setSigma1(d);
  return 0;
  CATCH("sigma1 could not be set", -1)
}

static auto radius = bob::extension::VariableDoc(
  "radius",
  "int",
  "The radius of the Difference of Gaussians filter along both axes (size of the kernel=2*radius+1)"
);
PyObject* PyBobIpBaseTanTriggs_getRadius(PyBobIpBaseTanTriggsObject* self, void*){
  TRY
  return Py_BuildValue("i", self->cxx->getRadius());
  CATCH("radius could not be read", 0)
}
int PyBobIpBaseTanTriggs_setRadius(PyBobIpBaseTanTriggsObject* self, PyObject* value, void*){
  TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, radius.name());
    return -1;
  }
  self->cxx->setRadius(PyInt_AS_LONG(value));
  return 0;
  CATCH("radius could not be set", -1)
}

static auto threshold = bob::extension::VariableDoc(
  "threshold",
  "float",
  "The threshold used for the contrast equalization, with read and write access"
);
PyObject* PyBobIpBaseTanTriggs_getThreshold(PyBobIpBaseTanTriggsObject* self, void*){
  TRY
  return Py_BuildValue("d", self->cxx->getThreshold());
  CATCH("threshold could not be read", 0)
}
int PyBobIpBaseTanTriggs_setThreshold(PyBobIpBaseTanTriggsObject* self, PyObject* value, void*){
  TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setThreshold(d);
  return 0;
  CATCH("threshold could not be set", -1)
}

static auto alpha = bob::extension::VariableDoc(
  "alpha",
  "float",
  "The alpha value used for the contrast equalization, with read and write access"
);
PyObject* PyBobIpBaseTanTriggs_getAlpha(PyBobIpBaseTanTriggsObject* self, void*){
  TRY
  return Py_BuildValue("d", self->cxx->getAlpha());
  CATCH("alpha could not be read", 0)
}
int PyBobIpBaseTanTriggs_setAlpha(PyBobIpBaseTanTriggsObject* self, PyObject* value, void*){
  TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setAlpha(d);
  return 0;
  CATCH("alpha could not be set", -1)
}

static auto border = bob::extension::VariableDoc(
  "border",
  ":py:class:`bob.sp.BorderType`",
  "The extrapolation method used by the convolution at the border, with read and write access"
);
PyObject* PyBobIpBaseTanTriggs_getBorder(PyBobIpBaseTanTriggsObject* self, void*){
  TRY
  return Py_BuildValue("i", self->cxx->getConvBorder());
  CATCH("border could not be read", 0)
}
int PyBobIpBaseTanTriggs_setBorder(PyBobIpBaseTanTriggsObject* self, PyObject* value, void*){
  TRY
  bob::sp::Extrapolation::BorderType b;
  if (!PyBobSpExtrapolationBorder_Converter(value, &b)) return -1;
  self->cxx->setConvBorder(b);
  return 0;
  CATCH("border could not be set", -1)
}

static auto kernel = bob::extension::VariableDoc(
  "kernel",
  "array_like (2D, float)",
  "The values of the DoG filter; read only access"
);
PyObject* PyBobIpBaseTanTriggs_getKernel(PyBobIpBaseTanTriggsObject* self, void*){
  TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getKernel());
  CATCH("kernel could not be read", 0)
}

static PyGetSetDef PyBobIpBaseTanTriggs_getseters[] = {
    {
      gamma_.name(),
      (getter)PyBobIpBaseTanTriggs_getGamma,
      (setter)PyBobIpBaseTanTriggs_setGamma,
      gamma_.doc(),
      0
    },
    {
      sigma0.name(),
      (getter)PyBobIpBaseTanTriggs_getSigma0,
      (setter)PyBobIpBaseTanTriggs_setSigma0,
      sigma0.doc(),
      0
    },
    {
      sigma1.name(),
      (getter)PyBobIpBaseTanTriggs_getSigma1,
      (setter)PyBobIpBaseTanTriggs_setSigma1,
      sigma1.doc(),
      0
    },
    {
      radius.name(),
      (getter)PyBobIpBaseTanTriggs_getRadius,
      (setter)PyBobIpBaseTanTriggs_setRadius,
      radius.doc(),
      0
    },
    {
      threshold.name(),
      (getter)PyBobIpBaseTanTriggs_getThreshold,
      (setter)PyBobIpBaseTanTriggs_setThreshold,
      threshold.doc(),
      0
    },
    {
      alpha.name(),
      (getter)PyBobIpBaseTanTriggs_getAlpha,
      (setter)PyBobIpBaseTanTriggs_setAlpha,
      alpha.doc(),
      0
    },
    {
      border.name(),
      (getter)PyBobIpBaseTanTriggs_getBorder,
      (setter)PyBobIpBaseTanTriggs_setBorder,
      border.doc(),
      0
    },
    {
      kernel.name(),
      (getter)PyBobIpBaseTanTriggs_getKernel,
      0,
      kernel.doc(),
      0
    },
    {0}  /* Sentinel */
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto process = bob::extension::FunctionDoc(
  "process",
  "Preprocesses a 2D/grayscale image using the algorithm from Tan and Triggs.",
  "The input array is a 2D array/grayscale image. "
  "The destination array, if given, should be a 2D array of type float64 and allocated in the same size as the input. "
  "If the destination array is not given, it is generated in the required size.\n\n"
  ".. note:: The :py:func:`__call__` function is an alias for this method.",
  true
)
.add_prototype("input, [output]", "output")
.add_parameter("input", "array_like (2D)", "The input image which should be normalized")
.add_parameter("output", "array_like (2D, float)", "[default: ``None``] If given, the output will be saved into this image; must be of the same shape as ``input``")
.add_return("output", "array_like (2D, float)", "The resulting output image, which is the same as ``output`` (if given)")
;

template <typename T>
static PyObject* process_inner(PyBobIpBaseTanTriggsObject* self, PyBlitzArrayObject* input, PyBlitzArrayObject* output){
  self->cxx->process(*PyBlitzArrayCxx_AsBlitz<T,2>(input), *PyBlitzArrayCxx_AsBlitz<double,2>(output));
  Py_INCREF(output);
  return PyBlitzArray_AsNumpyArray(output, 0);
}

static PyObject* PyBobIpBaseTanTriggs_process(PyBobIpBaseTanTriggsObject* self, PyObject* args, PyObject* kwargs) {
  TRY
  static char* kwlist[] = {c("input"), c("output"), 0};

  PyBlitzArrayObject* input,* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O&", kwlist, &PyBlitzArray_Converter, &input, &PyBlitzArray_OutputConverter, &output)) {
    process.print_usage();
    return 0;
  }

  auto input_ = make_safe(input), output_ = make_xsafe(output);

  // perform checks on input and output image
  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays", Py_TYPE(self)->tp_name);
    process.print_usage();
    return 0;
  }

  if (output){
    if (output->ndim != 2 || output->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "`%s' only processes to 2D arrays of type float", Py_TYPE(self)->tp_name);
      process.print_usage();
      return 0;
    }
  } else {
    // create output in desired shape
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, 2, input->shape);
    output_ = make_safe(output);
  }

  // finally, extract the features
  switch (input->type_num){
    case NPY_UINT8:   return process_inner<uint8_t>(self, input, output);
    case NPY_UINT16:  return process_inner<uint16_t>(self, input, output);
    case NPY_FLOAT64: return process_inner<double>(self, input, output);
    default:
      process.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' processes only images of types uint8, uint16 or float, and not from %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(input->type_num));
      return 0;
  }

  CATCH("cannot perform TanTriggs preprocessing in image", 0)
}


static PyMethodDef PyBobIpBaseTanTriggs_methods[] = {
  {
    process.name(),
    (PyCFunction)PyBobIpBaseTanTriggs_process,
    METH_VARARGS|METH_KEYWORDS,
    process.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the TanTriggs type struct; will be initialized later
PyTypeObject PyBobIpBaseTanTriggsType = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseTanTriggs(PyObject* module)
{
  // initialize the Gabor wavelet type struct
  PyBobIpBaseTanTriggsType.tp_name = TanTriggs_doc.name();
  PyBobIpBaseTanTriggsType.tp_basicsize = sizeof(PyBobIpBaseTanTriggsObject);
  PyBobIpBaseTanTriggsType.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseTanTriggsType.tp_doc = TanTriggs_doc.doc();

  // set the functions
  PyBobIpBaseTanTriggsType.tp_new = PyType_GenericNew;
  PyBobIpBaseTanTriggsType.tp_init = reinterpret_cast<initproc>(PyBobIpBaseTanTriggs_init);
  PyBobIpBaseTanTriggsType.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseTanTriggs_delete);
  PyBobIpBaseTanTriggsType.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseTanTriggs_RichCompare);
  PyBobIpBaseTanTriggsType.tp_methods = PyBobIpBaseTanTriggs_methods;
  PyBobIpBaseTanTriggsType.tp_getset = PyBobIpBaseTanTriggs_getseters;
  PyBobIpBaseTanTriggsType.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseTanTriggs_process);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseTanTriggsType) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseTanTriggsType);
  return PyModule_AddObject(module, "TanTriggs", (PyObject*)&PyBobIpBaseTanTriggsType) >= 0;
}
