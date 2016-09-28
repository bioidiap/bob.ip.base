/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Thu Nov 27 18:27:33 CET 2014
 *
 * @brief Binds the Wiener class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto Wiener_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".Wiener",
  "A Wiener filter",
  "The Wiener filter is implemented after the description in Part 3.4.3 of [Szeliski2010]_"
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new Wiener filter",
    "Several variants of contructors are possible for contructing a Wiener filter. "
    "They are:\n\n"
    "1. Constructs a new Wiener filter dedicated to images of the given ``size``. The filter is initialized with zero values\n"
    "2. Constructs a new Wiener filter from a set of variance estimates ``Ps`` and a noise level ``Pn``\n"
    "3. Trains the new Wiener filter with the given ``data``\n"
    "4. Copy constructs the given Wiener filter\n"
    "5. Reads the Wiener filter from :py:class:`bob.io.base.HDF5File`",
    true
  )
  .add_prototype("size, Pn, [variance_threshold]", "")
  .add_prototype("Ps, Pn, [variance_threshold]","")
  .add_prototype("data, [variance_threshold]", "")
  .add_prototype("filter", "")
  .add_prototype("hdf5", "")
  .add_parameter("Ps", "array_like<float, 2D>", "Variance Ps estimated at each frequency")
  .add_parameter("Pn", "float", "Noise level Pn")
  .add_parameter("size", "(int, int)", "The shape of the newly created empty filter")
  .add_parameter("data", "array_like<float, 3D>", "The training data, with dimensions ``(#data, height, width)``")
  .add_parameter("variance_threshold", "float", "[default: ``1e-8``] Variance flooring threshold (i.e., the minimum variance value")
  .add_parameter("filter", ":py:class:`bob.ip.base.Wiener`", "The Wiener filter object to use for copy-construction")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "The HDF5 file object to read the Wiener filter from")
);

static int PyBobIpBaseWiener_init(PyBobIpBaseWienerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = Wiener_doc.kwlist(0);
  char** kwlist2 = Wiener_doc.kwlist(1);
  char** kwlist3 = Wiener_doc.kwlist(2);
  char** kwlist4 = Wiener_doc.kwlist(3);
  char** kwlist5 = Wiener_doc.kwlist(4);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k1 = Py_BuildValue("s", kwlist1[0]),* k2 = Py_BuildValue("s", kwlist2[0]),* k4 = Py_BuildValue("s", kwlist4[0]),* k5 = Py_BuildValue("s", kwlist4[0]);
  auto k1_ = make_safe(k1), k2_ = make_safe(k2), k4_ = make_safe(k4), k5_ = make_safe(k5);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseWiener_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k4)))){
    // copy construct
    PyBobIpBaseWienerObject* wiener;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist4, &PyBobIpBaseWiener_Type, &wiener)) return -1;
    self->cxx.reset(new bob::ip::base::Wiener(*wiener->cxx));
    return 0;
  } else if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIoHDF5File_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k5)))){
    // construct from HDF5
    PyBobIoHDF5FileObject* hdf5;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist5, &PyBobIoHDF5File_Converter, &hdf5)) return -1;
    auto hdf5_ = make_safe(hdf5);
    self->cxx.reset(new bob::ip::base::Wiener(*hdf5->f));
    return 0;
  }

  if (nargs >= 2 && ((args && PyTuple_Size(args) >= 1 && (PyTuple_Check(PyTuple_GET_ITEM(args,0)) || (PyList_Check(PyTuple_GET_ITEM(args,0)) && PyList_Size(PyTuple_GET_ITEM(args,0)) == 2))) || (kwargs && PyDict_Contains(kwargs, k1)))){
    // construct from size
    blitz::TinyVector<int,2> size;
    double Pn, thres=1e-8;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)d|d", kwlist1, &size[0], &size[1], &Pn, &thres)) return -1;
    self->cxx.reset(new bob::ip::base::Wiener(size, Pn, thres));
    return 0;
  }

  // construct with data
  PyBlitzArrayObject* data;
  double d1 = 1e-8, d2 = 1e-8;
  if (nargs == 3 || (nargs == 2 && kwargs && PyDict_Contains(kwargs, k2))){
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&d|d", kwlist2, &PyBlitzArray_Converter, &data, &d1, &d2)) return -1;
  } else {
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|d", kwlist3, &PyBlitzArray_Converter, &data, &d1)) return -1;
  }

  auto data_ = make_safe(data);

  if (data->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "%s constructor expects input array of type float", Py_TYPE(self)->tp_name);
    return -1;
  }

  if (data->ndim == 3){
    // construction with data
    self->cxx.reset(new bob::ip::base::Wiener(*PyBlitzArrayCxx_AsBlitz<double,3>(data), d1));
    return 0;
  } else if (data->ndim == 2){
    // construction with Ps and Pn
    self->cxx.reset(new bob::ip::base::Wiener(*PyBlitzArrayCxx_AsBlitz<double,2>(data), d1, d2));
    return 0;
  } else {
    PyErr_Format(PyExc_TypeError, "%s constructor expects input array of 2D or 3D", Py_TYPE(self)->tp_name);
    return -1;
  }

  BOB_CATCH_MEMBER("cannot create Wiener filter", -1)
}

static void PyBobIpBaseWiener_delete(PyBobIpBaseWienerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseWiener_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseWiener_Type));
}

static PyObject* PyBobIpBaseWiener_RichCompare(PyBobIpBaseWienerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseWiener_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseWienerObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare Wiener filter objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto Ps = bob::extension::VariableDoc(
  "Ps",
  "array_like <float, 2D>",
  "Variance Ps estimated at each frequency"
);
PyObject* PyBobIpBaseWiener_getPs(PyBobIpBaseWienerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getPs());
  BOB_CATCH_MEMBER("Ps could not be read", 0)
}
int PyBobIpBaseWiener_setPs(PyBobIpBaseWienerObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, Ps.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(o, "Ps");
  if (!b) return -1;
  self->cxx->setPs(*b);
  return 0;
  BOB_CATCH_MEMBER("Ps could not be set", -1)
}

static auto Pn = bob::extension::VariableDoc(
  "Pn",
  "float",
  "Noise level Pn"
);
PyObject* PyBobIpBaseWiener_getPn(PyBobIpBaseWienerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getPn());
  BOB_CATCH_MEMBER("Pn could not be read", 0)
}
int PyBobIpBaseWiener_setPn(PyBobIpBaseWienerObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyFloat_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a float", Py_TYPE(self)->tp_name, Pn.name());
    return -1;
  }
  self->cxx->setPn(PyFloat_AS_DOUBLE(value));
  return 0;
  BOB_CATCH_MEMBER("Pn could not be set", -1)
}

static auto w = bob::extension::VariableDoc(
  "w",
  "array_like<2D, float>",
  "The Wiener filter W (W=1/(1+Pn/Ps)) (read-only)"
);
PyObject* PyBobIpBaseWiener_getW(PyBobIpBaseWienerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getW());
  BOB_CATCH_MEMBER("border could not be read", 0)
}

static auto size = bob::extension::VariableDoc(
  "size",
  "(int, int)",
  "The size of the filter"
);
PyObject* PyBobIpBaseWiener_getSize(PyBobIpBaseWienerObject* self, void*){
  BOB_TRY
  auto size = self->cxx->getSize();
  return Py_BuildValue("(ii)", size[0], size[1]);
  BOB_CATCH_MEMBER("size could not be read", 0)
}
int PyBobIpBaseWiener_setSize(PyBobIpBaseWienerObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> s;
  if (!PyArg_ParseTuple(value, "ii", &s[0], &s[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, size.name());
    return -1;
  }
  self->cxx->resize(s);
  return 0;
  BOB_CATCH_MEMBER("size could not be set", -1)
}

static auto thres = bob::extension::VariableDoc(
  "variance_threshold",
  "float",
  "Variance flooring threshold"
);
PyObject* PyBobIpBaseWiener_getThres(PyBobIpBaseWienerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getVarianceThreshold());
  BOB_CATCH_MEMBER("variance_threshold could not be read", 0)
}
int PyBobIpBaseWiener_setThres(PyBobIpBaseWienerObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyFloat_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a float", Py_TYPE(self)->tp_name, thres.name());
    return -1;
  }
  self->cxx->setVarianceThreshold(PyFloat_AS_DOUBLE(value));
  return 0;
  BOB_CATCH_MEMBER("variance_threshold could not be set", -1)
}

static PyGetSetDef PyBobIpBaseWiener_getseters[] = {
    {
      Ps.name(),
      (getter)PyBobIpBaseWiener_getPs,
      (setter)PyBobIpBaseWiener_setPs,
      Ps.doc(),
      0
    },
    {
      Pn.name(),
      (getter)PyBobIpBaseWiener_getPn,
      (setter)PyBobIpBaseWiener_setPn,
      Pn.doc(),
      0
    },
    {
      w.name(),
      (getter)PyBobIpBaseWiener_getW,
      0,
      w.doc(),
      0
    },
    {
      size.name(),
      (getter)PyBobIpBaseWiener_getSize,
      (setter)PyBobIpBaseWiener_setSize,
      size.doc(),
      0
    },
    {
      thres.name(),
      (getter)PyBobIpBaseWiener_getThres,
      (setter)PyBobIpBaseWiener_setThres,
      thres.doc(),
      0
    },
    {0}  /* Sentinel */
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto filter = bob::extension::FunctionDoc(
  "filter",
  "Filters the input image",
  "If given, the dst array should have the expected type (numpy.float64) and the same size as the src array.\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like (2D)", "The input image which should be smoothed")
.add_parameter("dst", "array_like (2D, float)", "[default: ``None``] If given, the output will be saved into this image; must be of the same shape as ``src``")
.add_return("dst", "array_like (2D, float)", "The resulting output image, which is the same as ``dst`` (if given)")
;

static PyObject* PyBobIpBaseWiener_filter(PyBobIpBaseWienerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = filter.kwlist();

  PyBlitzArrayObject* src,* dst = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O&", kwlist, &PyBlitzArray_Converter, &src, &PyBlitzArray_OutputConverter, &dst)) return 0;

  auto src_ = make_safe(src), dst_ = make_xsafe(dst);

  // perform checks on input and output image
  if (src->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays", Py_TYPE(self)->tp_name);
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

  // finally, filter the given image
  self->cxx->filter(*PyBlitzArrayCxx_AsBlitz<double,2>(src), *PyBlitzArrayCxx_AsBlitz<double,2>(dst));

  // and return the result
  return Py_BuildValue("O", dst);

  BOB_CATCH_MEMBER("cannot perform Wiener filtering in image", 0)
}

static auto load = bob::extension::FunctionDoc(
  "load",
  "Loads the configuration of the Wiener filter from the given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file opened for reading")
;

static PyObject* PyBobIpBaseWiener_load(PyBobIpBaseWienerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  // get list of arguments
  char** kwlist = load.kwlist();
  PyBobIoHDF5FileObject* hdf5 = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, PyBobIoHDF5File_Converter, &hdf5)){
    load.print_usage();
    return 0;
  }
  auto hdf5_ = make_safe(hdf5);
  self->cxx->load(*hdf5->f);
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("cannot load parametrization", 0)
}

static auto save = bob::extension::FunctionDoc(
  "save",
  "Saves the the configuration of the Wiener filter to the given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing")
;

static PyObject* PyBobIpBaseWiener_save(PyBobIpBaseWienerObject* self, PyObject* args, PyObject* kwargs) {
  // get list of arguments
  char** kwlist = save.kwlist();
  PyBobIoHDF5FileObject* hdf5 = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, PyBobIoHDF5File_Converter, &hdf5)){
    save.print_usage();
    return NULL;
  }

  auto hdf5_ = make_safe(hdf5);
  try{
    self->cxx->save(*hdf5->f);
  } catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }catch (...) {
    PyErr_Format(PyExc_RuntimeError, "%s cannot save parametrization: unknown exception caught", Py_TYPE(self)->tp_name);
    return 0;
  }

  Py_RETURN_NONE;
}


static auto similar = bob::extension::FunctionDoc(
  "is_similar_to",
  "Compares this Wiener filter with the ``other`` one to be approximately the same",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the relative and absolute precision, similarly to :py:func:`numpy.allclose`.",
  true
)
.add_prototype("other, [r_epsilon], [a_epsilon]")
.add_parameter("other", ":py:class:`bob.ip.base.Wiener`", "The other Wiener filter to compare with")
.add_parameter("r_epsilon", "float", "[Default: ``1e-5``] The relative precision")
.add_parameter("a_epsilon", "float", "[Default: ``1e-8``] The absolute precision")
;
static PyObject* PyBobIpBaseWiener_similar(PyBobIpBaseWienerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = similar.kwlist();

  PyBobIpBaseWienerObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|dd", kwlist, &PyBobIpBaseWiener_Type, &other, &r_epsilon, &a_epsilon)) return 0;

  if (self->cxx->is_similar_to(*other->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyMethodDef PyBobIpBaseWiener_methods[] = {
  {
    filter.name(),
    (PyCFunction)PyBobIpBaseWiener_filter,
    METH_VARARGS|METH_KEYWORDS,
    filter.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobIpBaseWiener_load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    save.name(),
    (PyCFunction)PyBobIpBaseWiener_save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    similar.name(),
    (PyCFunction)PyBobIpBaseWiener_similar,
    METH_VARARGS|METH_KEYWORDS,
    similar.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Wiener type struct; will be initialized later
PyTypeObject PyBobIpBaseWiener_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseWiener(PyObject* module)
{
  // initialize the type struct
  PyBobIpBaseWiener_Type.tp_name = Wiener_doc.name();
  PyBobIpBaseWiener_Type.tp_basicsize = sizeof(PyBobIpBaseWienerObject);
  PyBobIpBaseWiener_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseWiener_Type.tp_doc = Wiener_doc.doc();

  // set the functions
  PyBobIpBaseWiener_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseWiener_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseWiener_init);
  PyBobIpBaseWiener_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseWiener_delete);
  PyBobIpBaseWiener_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseWiener_RichCompare);
  PyBobIpBaseWiener_Type.tp_methods = PyBobIpBaseWiener_methods;
  PyBobIpBaseWiener_Type.tp_getset = PyBobIpBaseWiener_getseters;
  PyBobIpBaseWiener_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseWiener_filter);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseWiener_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseWiener_Type);
  return PyModule_AddObject(module, "Wiener", (PyObject*)&PyBobIpBaseWiener_Type) >= 0;
}
