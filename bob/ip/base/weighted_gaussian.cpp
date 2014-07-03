/**
 * @author Manuel Guenther <manuel.guenthr@idiap.ch>
 * @date Thu Jul  3 17:59:11 CEST 2014
 *
 * @brief Binds the WeightedGaussian class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto WeightedGaussian_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".WeightedGaussian",
  "This class performs weighted gaussian smoothing (anisotropic filtering)",
  "In particular, it is used by the Self Quotient Image (SQI) algorithm :py:class:`bob.ip.base.SelfQuotientImage`."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new weighted Gaussian filter",
    0,
    true
  )
  .add_prototype("variance, [radius], [border]","")
  .add_prototype("weighted_gaussian", "")
  .add_parameter("variance", "(double, double)", "The variance (i.e., the **square** standard deviation) of the WeightedGaussian along the y- and x-axes in pixels")
  .add_parameter("radius", "(int, int)", "[default: (-1, -1) -> ``3*sqrt(variance)`` ] The radius of the Gaussian in both directions -- the size of the kernel is ``2*radius+1``")
  .add_parameter("border", ":py:class:`bob.sp.BorderType`", "[default: ``bob.sp.BorderType.Mirror``] The extrapolation method used by the convolution at the border")
  .add_parameter("weighted_gaussian", ":py:class:`bob.ip.base.WeightedGaussian`", "The weighted Gaussian object to use for copy-construction")
);

static int PyBobIpBaseWeightedGaussian_init(PyBobIpBaseWeightedGaussianObject* self, PyObject* args, PyObject* kwargs) {
  TRY

  char* kwlist1[] = {c("variance"), c("radius"), c("border"), NULL};
  char* kwlist2[] = {c("gaussian"), NULL};

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseWeightedGaussian_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseWeightedGaussianObject* gaussian;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseWeightedGaussianType, &gaussian)) return -1;

    self->cxx.reset(new bob::ip::base::WeightedGaussian(*gaussian->cxx));
    return 0;
  }

  blitz::TinyVector<double,2> variance;
  blitz::TinyVector<int,2> radius (-1, -1);
  bob::sp::Extrapolation::BorderType border = bob::sp::Extrapolation::Mirror;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(dd)|(ii)O&", kwlist1, &variance[0], &variance[1], &radius[0], &radius[1], &PyBobSpExtrapolationBorder_Converter, &border)){
    WeightedGaussian_doc.print_usage();
    return -1;
  }
  // set the radius
  for (int i = 0; i < 2; ++i) if (radius[i] < 0) radius[i] = std::max(int(std::sqrt(variance[i]) * 3 + 0.5), 1);

  self->cxx.reset(new bob::ip::base::WeightedGaussian(radius[0], radius[1], variance[0], variance[1], border));
  return 0;

  CATCH("cannot create WeightedGaussian", -1)
}

static void PyBobIpBaseWeightedGaussian_delete(PyBobIpBaseWeightedGaussianObject* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseWeightedGaussian_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseWeightedGaussianType));
}

static PyObject* PyBobIpBaseWeightedGaussian_RichCompare(PyBobIpBaseWeightedGaussianObject* self, PyObject* other, int op) {
  TRY

  if (!PyBobIpBaseWeightedGaussian_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseWeightedGaussianObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  CATCH("cannot compare WeightedGaussian objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto sigma2 = bob::extension::VariableDoc(
  "variance",
  "(float, float)",
  "The variance of the weighted Gaussian along the y- and x-axes; with read and write access",
  ".. note:: The :py:attr:`radius` of the kernel is **not** reset by setting the ``variance`` value."
);
PyObject* PyBobIpBaseWeightedGaussian_getSigma2(PyBobIpBaseWeightedGaussianObject* self, void*){
  TRY
  return Py_BuildValue("(dd)", self->cxx->getSigma2Y(), self->cxx->getSigma2X());
  CATCH("sigma could not be read", 0)
}
int PyBobIpBaseWeightedGaussian_setSigma2(PyBobIpBaseWeightedGaussianObject* self, PyObject* value, void*){
  TRY
  blitz::TinyVector<double,2> r;
  if (!PyArg_ParseTuple(value, "dd", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, sigma2.name());
    return -1;
  }
  self->cxx->setSigma2(r);
  return 0;
  CATCH("variance could not be set", -1)
}

static auto radius = bob::extension::VariableDoc(
  "radius",
  "(int, int)",
  "The radius of the WeightedGaussian along the y- and x-axes (size of the kernel=2*radius+1); with read and write access",
  "When setting the radius to a negative value, it will be automatically computed as ``3*sqrt``:py:attr:`variance`."
);
PyObject* PyBobIpBaseWeightedGaussian_getRadius(PyBobIpBaseWeightedGaussianObject* self, void*){
  TRY
  return Py_BuildValue("(ii)", self->cxx->getRadiusY(), self->cxx->getRadiusX());
  CATCH("radius could not be read", 0)
}
int PyBobIpBaseWeightedGaussian_setRadius(PyBobIpBaseWeightedGaussianObject* self, PyObject* value, void*){
  TRY
  blitz::TinyVector<int,2> r;
  if (!PyArg_ParseTuple(value, "ii", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two integers", Py_TYPE(self)->tp_name, radius.name());
    return -1;
  }
  if (r[0] < 0) r[0] = std::max(int(std::sqrt(self->cxx->getSigma2Y()) * 3 + 0.5), 1);
  if (r[1] < 0) r[1] = std::max(int(std::sqrt(self->cxx->getSigma2X()) * 3 + 0.5), 1);
  self->cxx->setRadius(r);
  return 0;
  CATCH("radius could not be set", -1)
}

static auto border = bob::extension::VariableDoc(
  "border",
  ":py:class:`bob.sp.BorderType`",
  "The extrapolation method used by the convolution at the border, with read and write access"
);
PyObject* PyBobIpBaseWeightedGaussian_getBorder(PyBobIpBaseWeightedGaussianObject* self, void*){
  TRY
  return Py_BuildValue("i", self->cxx->getConvBorder());
  CATCH("border could not be read", 0)
}
int PyBobIpBaseWeightedGaussian_setBorder(PyBobIpBaseWeightedGaussianObject* self, PyObject* value, void*){
  TRY
  bob::sp::Extrapolation::BorderType b;
  if (!PyBobSpExtrapolationBorder_Converter(value, &b)) return -1;
  self->cxx->setConvBorder(b);
  return 0;
  CATCH("border could not be set", -1)
}

static PyGetSetDef PyBobIpBaseWeightedGaussian_getseters[] = {
    {
      sigma2.name(),
      (getter)PyBobIpBaseWeightedGaussian_getSigma2,
      (setter)PyBobIpBaseWeightedGaussian_setSigma2,
      sigma2.doc(),
      0
    },
    {
      radius.name(),
      (getter)PyBobIpBaseWeightedGaussian_getRadius,
      (setter)PyBobIpBaseWeightedGaussian_setRadius,
      radius.doc(),
      0
    },
    {
      border.name(),
      (getter)PyBobIpBaseWeightedGaussian_getBorder,
      (setter)PyBobIpBaseWeightedGaussian_setBorder,
      border.doc(),
      0
    },
    {0}  /* Sentinel */
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto filter = bob::extension::FunctionDoc(
  "filter",
  "Smooths an image (2D/grayscale or 3D/color)",
  "If given, the dst array should have the expected type (numpy.float64) and the same size as the src array.\n\n"
  ".. note:: The :py:func:`__call__` function is an alias for this method.",
  true
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like (2D)", "The input image which should be smoothed")
.add_parameter("dst", "array_like (2D, float)", "[default: ``None``] If given, the output will be saved into this image; must be of the same shape as ``src``")
.add_return("dst", "array_like (2D, float)", "The resulting output image, which is the same as ``dst`` (if given)")
;

template <typename T, int D>
static PyObject* filter_inner(PyBobIpBaseWeightedGaussianObject* self, PyBlitzArrayObject* input, PyBlitzArrayObject* output){
  self->cxx->filter(*PyBlitzArrayCxx_AsBlitz<T,D>(input), *PyBlitzArrayCxx_AsBlitz<double,D>(output));
  Py_INCREF(output);
  return PyBlitzArray_AsNumpyArray(output, 0);
}

static PyObject* PyBobIpBaseWeightedGaussian_filter(PyBobIpBaseWeightedGaussianObject* self, PyObject* args, PyObject* kwargs) {
  TRY
  static char* kwlist[] = {c("src"), c("dst"), 0};

  PyBlitzArrayObject* src,* dst = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O&", kwlist, &PyBlitzArray_Converter, &src, &PyBlitzArray_OutputConverter, &dst)) return 0;

  auto src_ = make_safe(src), dst_ = make_xsafe(dst);

  // perform checks on input and output image
  if (src->ndim != 2 && src->ndim != 3){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D or 3D arrays", Py_TYPE(self)->tp_name);
    filter.print_usage();
    return 0;
  }

  if (dst){
    if (dst->ndim != src->ndim){
      PyErr_Format(PyExc_TypeError, "`%s' 'src' and 'dst' shape has to be identical", Py_TYPE(self)->tp_name);
      filter.print_usage();
      return 0;
    }
    if (dst->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "`%s' only processes to arrays of type float", Py_TYPE(self)->tp_name);
      filter.print_usage();
      return 0;
    }
  } else {
    // create output in desired shape
    dst = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, src->ndim, src->shape);
    dst_ = make_safe(dst);
  }

  // finally, extract the features
  switch (src->type_num){
    case NPY_UINT8:   if (src->ndim == 2) return filter_inner<uint8_t,2>(self, src, dst);  else return filter_inner<uint8_t,3>(self, src, dst);
    case NPY_UINT16:  if (src->ndim == 2) return filter_inner<uint16_t,2>(self, src, dst); else return filter_inner<uint16_t,3>(self, src, dst);
    case NPY_FLOAT64: if (src->ndim == 2) return filter_inner<double,2>(self, src, dst);   else return filter_inner<double,3>(self, src, dst);
    default:
      filter.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' processes only images of types uint8, uint16 or float, and not from %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(src->type_num));
      return 0;
  }

  CATCH("cannot perform WeightedGaussian filtering in image", 0)
}


static PyMethodDef PyBobIpBaseWeightedGaussian_methods[] = {
  {
    filter.name(),
    (PyCFunction)PyBobIpBaseWeightedGaussian_filter,
    METH_VARARGS|METH_KEYWORDS,
    filter.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the WeightedGaussian type struct; will be initialized later
PyTypeObject PyBobIpBaseWeightedGaussianType = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseWeightedGaussian(PyObject* module)
{
  // initialize the Gabor wavelet type struct
  PyBobIpBaseWeightedGaussianType.tp_name = WeightedGaussian_doc.name();
  PyBobIpBaseWeightedGaussianType.tp_basicsize = sizeof(PyBobIpBaseWeightedGaussianObject);
  PyBobIpBaseWeightedGaussianType.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseWeightedGaussianType.tp_doc = WeightedGaussian_doc.doc();

  // set the functions
  PyBobIpBaseWeightedGaussianType.tp_new = PyType_GenericNew;
  PyBobIpBaseWeightedGaussianType.tp_init = reinterpret_cast<initproc>(PyBobIpBaseWeightedGaussian_init);
  PyBobIpBaseWeightedGaussianType.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseWeightedGaussian_delete);
  PyBobIpBaseWeightedGaussianType.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseWeightedGaussian_RichCompare);
  PyBobIpBaseWeightedGaussianType.tp_methods = PyBobIpBaseWeightedGaussian_methods;
  PyBobIpBaseWeightedGaussianType.tp_getset = PyBobIpBaseWeightedGaussian_getseters;
  PyBobIpBaseWeightedGaussianType.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseWeightedGaussian_filter);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseWeightedGaussianType) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseWeightedGaussianType);
  return PyModule_AddObject(module, "WeightedGaussian", (PyObject*)&PyBobIpBaseWeightedGaussianType) >= 0;
}

