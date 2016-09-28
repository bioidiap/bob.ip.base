/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Thu Jul  3 17:59:11 CEST 2014
 *
 * @brief Binds the Gaussian class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto Gaussian_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".Gaussian",
  "Objects of this class, after configuration, can perform Gaussian filtering (smoothing) on images",
  "The Gaussian smoothing is done by convolving the image with a vertical and a horizontal smoothing filter."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new Gaussian filter",
    "The Gaussian kernel is generated in both directions independently, using the given standard deviation and the given radius, where the size of the kernels is actually ``2*radius+1``. "
    "When the radius is not given or negative, it will be automatically computed ad ``3*sigma``.\n\n"
    ".. note::\n\n  Since the Gaussian smoothing is done by convolution, a larger radius will lead to longer execution time.",
    true
  )
  .add_prototype("sigma, [radius], [border]","")
  .add_prototype("gaussian", "")
  .add_parameter("sigma", "(double, double)", "The standard deviation of the Gaussian along the y- and x-axes in pixels")
  .add_parameter("radius", "(int, int)", "[default: (-1, -1) -> ``3*sigma`` ] The radius of the Gaussian in both directions -- the size of the kernel is ``2*radius+1``")
  .add_parameter("border", ":py:class:`bob.sp.BorderType`", "[default: ``bob.sp.BorderType.Mirror``] The extrapolation method used by the convolution at the border")
  .add_parameter("gaussian", ":py:class:`bob.ip.base.Gaussian`", "The Gaussian object to use for copy-construction")
);

static int PyBobIpBaseGaussian_init(PyBobIpBaseGaussianObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = Gaussian_doc.kwlist(0);
  char** kwlist2 = Gaussian_doc.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseGaussian_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseGaussianObject* gaussian;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseGaussian_Type, &gaussian)) return -1;

    self->cxx.reset(new bob::ip::base::Gaussian(*gaussian->cxx));
    return 0;
  }

  blitz::TinyVector<double,2> sigma;
  blitz::TinyVector<int,2> radius (-1, -1);
  bob::sp::Extrapolation::BorderType border = bob::sp::Extrapolation::Mirror;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(dd)|(ii)O&", kwlist1, &sigma[0], &sigma[1], &radius[0], &radius[1], &PyBobSpExtrapolationBorder_Converter, &border)){
    Gaussian_doc.print_usage();
    return -1;
  }
  // set the radius
  for (int i = 0; i < 2; ++i) if (radius[i] < 0) radius[i] = std::max(int(sigma[i] * 3 + 0.5), 1);

  self->cxx.reset(new bob::ip::base::Gaussian(radius[0], radius[1], sigma[0], sigma[1], border));
  return 0;

  BOB_CATCH_MEMBER("cannot create Gaussian", -1)
}

static void PyBobIpBaseGaussian_delete(PyBobIpBaseGaussianObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseGaussian_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseGaussian_Type));
}

static PyObject* PyBobIpBaseGaussian_RichCompare(PyBobIpBaseGaussianObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseGaussian_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseGaussianObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare Gaussian objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto sigma = bob::extension::VariableDoc(
  "sigma",
  "(float, float)",
  "The standard deviation of the Gaussian along the y- and x-axes; with read and write access",
  ".. note::\n\n  The :py:attr:`radius` of the kernel is **not** reset by setting the ``sigma`` value."
);
PyObject* PyBobIpBaseGaussian_getSigma(PyBobIpBaseGaussianObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(dd)", self->cxx->getSigmaY(), self->cxx->getSigmaX());
  BOB_CATCH_MEMBER("sigma could not be read", 0)
}
int PyBobIpBaseGaussian_setSigma(PyBobIpBaseGaussianObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<double,2> r;
  if (!PyArg_ParseTuple(value, "dd", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, sigma.name());
    return -1;
  }
  self->cxx->setSigma(r);
  return 0;
  BOB_CATCH_MEMBER("sigma could not be set", -1)
}

static auto radius = bob::extension::VariableDoc(
  "radius",
  "(int, int)",
  "The radius of the Gaussian along the y- and x-axes (size of the kernel=2*radius+1); with read and write access",
  "When setting the radius to a negative value, it will be automatically computed as ``3*``:py:attr:`sigma`."
);
PyObject* PyBobIpBaseGaussian_getRadius(PyBobIpBaseGaussianObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(ii)", self->cxx->getRadiusY(), self->cxx->getRadiusX());
  BOB_CATCH_MEMBER("radius could not be read", 0)
}
int PyBobIpBaseGaussian_setRadius(PyBobIpBaseGaussianObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> r;
  if (!PyArg_ParseTuple(value, "ii", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two integers", Py_TYPE(self)->tp_name, radius.name());
    return -1;
  }
  if (r[0] < 0) r[0] = std::max(int(self->cxx->getSigmaY() * 3 + 0.5), 1);
  if (r[1] < 0) r[1] = std::max(int(self->cxx->getSigmaX() * 3 + 0.5), 1);
  self->cxx->setRadius(r);
  return 0;
  BOB_CATCH_MEMBER("radius could not be set", -1)
}

static auto border = bob::extension::VariableDoc(
  "border",
  ":py:class:`bob.sp.BorderType`",
  "The extrapolation method used by the convolution at the border, with read and write access"
);
PyObject* PyBobIpBaseGaussian_getBorder(PyBobIpBaseGaussianObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getConvBorder());
  BOB_CATCH_MEMBER("border could not be read", 0)
}
int PyBobIpBaseGaussian_setBorder(PyBobIpBaseGaussianObject* self, PyObject* value, void*){
  BOB_TRY
  bob::sp::Extrapolation::BorderType b;
  if (!PyBobSpExtrapolationBorder_Converter(value, &b)) return -1;
  self->cxx->setConvBorder(b);
  return 0;
  BOB_CATCH_MEMBER("border could not be set", -1)
}

static auto kernelY = bob::extension::VariableDoc(
  "kernel_y",
  "array_like (1D, float)",
  "The values of the kernel in vertical direction; read only access"
);
PyObject* PyBobIpBaseGaussian_getKernelY(PyBobIpBaseGaussianObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getKernelY());
  BOB_CATCH_MEMBER("kernel_y could not be read", 0)
}

static auto kernelX = bob::extension::VariableDoc(
  "kernel_x",
  "array_like (1D, float)",
  "The values of the kernel in horizontal direction; read only access"
);
PyObject* PyBobIpBaseGaussian_getKernelX(PyBobIpBaseGaussianObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getKernelX());
  BOB_CATCH_MEMBER("kernel_x could not be read", 0)
}

static PyGetSetDef PyBobIpBaseGaussian_getseters[] = {
    {
      sigma.name(),
      (getter)PyBobIpBaseGaussian_getSigma,
      (setter)PyBobIpBaseGaussian_setSigma,
      sigma.doc(),
      0
    },
    {
      radius.name(),
      (getter)PyBobIpBaseGaussian_getRadius,
      (setter)PyBobIpBaseGaussian_setRadius,
      radius.doc(),
      0
    },
    {
      border.name(),
      (getter)PyBobIpBaseGaussian_getBorder,
      (setter)PyBobIpBaseGaussian_setBorder,
      border.doc(),
      0
    },
    {
      kernelY.name(),
      (getter)PyBobIpBaseGaussian_getKernelY,
      0,
      kernelY.doc(),
      0
    },
    {
      kernelX.name(),
      (getter)PyBobIpBaseGaussian_getKernelX,
      0,
      kernelX.doc(),
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
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like (2D)", "The input image which should be smoothed")
.add_parameter("dst", "array_like (2D, float)", "[default: ``None``] If given, the output will be saved into this image; must be of the same shape as ``src``")
.add_return("dst", "array_like (2D, float)", "The resulting output image, which is the same as ``dst`` (if given)")
;

template <typename T, int D>
static PyObject* filter_inner(PyBobIpBaseGaussianObject* self, PyBlitzArrayObject* input, PyBlitzArrayObject* output){
  self->cxx->filter(*PyBlitzArrayCxx_AsBlitz<T,D>(input), *PyBlitzArrayCxx_AsBlitz<double,D>(output));
  return PyBlitzArray_AsNumpyArray(output, 0);
}

static PyObject* PyBobIpBaseGaussian_filter(PyBobIpBaseGaussianObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = filter.kwlist();

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

  BOB_CATCH_MEMBER("cannot perform Gaussian filtering in image", 0)
}


static PyMethodDef PyBobIpBaseGaussian_methods[] = {
  {
    filter.name(),
    (PyCFunction)PyBobIpBaseGaussian_filter,
    METH_VARARGS|METH_KEYWORDS,
    filter.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobIpBaseGaussian_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseGaussian(PyObject* module)
{
  // initialize the type struct
  PyBobIpBaseGaussian_Type.tp_name = Gaussian_doc.name();
  PyBobIpBaseGaussian_Type.tp_basicsize = sizeof(PyBobIpBaseGaussianObject);
  PyBobIpBaseGaussian_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseGaussian_Type.tp_doc = Gaussian_doc.doc();

  // set the functions
  PyBobIpBaseGaussian_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseGaussian_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseGaussian_init);
  PyBobIpBaseGaussian_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseGaussian_delete);
  PyBobIpBaseGaussian_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseGaussian_RichCompare);
  PyBobIpBaseGaussian_Type.tp_methods = PyBobIpBaseGaussian_methods;
  PyBobIpBaseGaussian_Type.tp_getset = PyBobIpBaseGaussian_getseters;
  PyBobIpBaseGaussian_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseGaussian_filter);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseGaussian_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseGaussian_Type);
  return PyModule_AddObject(module, "Gaussian", (PyObject*)&PyBobIpBaseGaussian_Type) >= 0;
}
