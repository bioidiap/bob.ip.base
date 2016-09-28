/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Jul  7 10:13:37 CEST 2014
 *
 * @brief Binds the SIFT class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

#if HAVE_VLFEAT

static auto VLSIFT_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".VLSIFT",
  "Computes SIFT features using the VLFeat library",
  "For details, please read [Lowe2004]_."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates an object that allows the extraction of VLSIFT descriptors",
    ".. todo:: Explain VLSIFT constructor in more detail.",
    true
  )
  .add_prototype("size, scales, octaves, octave_min, [peak_thres], [edge_thres], [magnif]", "")
  .add_prototype("sift", "")
  .add_parameter("size", "(int, int)", "The height and width of the images to process")
  .add_parameter("scales", "int", "The number of intervals in each octave")
  .add_parameter("octaves", "int", "The number of octaves of the pyramid")
  .add_parameter("octave_min", "int", "The index of the minimum octave")
  .add_parameter("peak_thres", "float", "[default: 0.03] The peak threshold (minimum amount of contrast to accept a keypoint)")
  .add_parameter("edge_thres", "float", "[default: 10.] The edge rejectipon threshold used during keypoint detection")
  .add_parameter("magnif", "float", "[default: 3.] The magnification factor (descriptor size is determined by multiplying the keypoint scale by this factor)")
  .add_parameter("sift", ":py:class:`bob.ip.base.VLSIFT`", "The VLSIFT object to use for copy-construction")
);


static int PyBobIpBaseVLSIFT_init(PyBobIpBaseVLSIFTObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = VLSIFT_doc.kwlist(0);
  char** kwlist2 = VLSIFT_doc.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseVLSIFT_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseVLSIFTObject* sift;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseVLSIFT_Type, &sift)) return -1;

    self->cxx.reset(new bob::ip::base::VLSIFT(*sift->cxx));
    return 0;
  }

  blitz::TinyVector<int,2> size;
  int scales, octaves, octave_min;
  double peak = 0.03, edge = 10., magnif = 3.;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)iii|ddd", kwlist1, &size[0], &size[1], &scales, &octaves, &octave_min, &peak, &edge, &magnif)){
    VLSIFT_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::ip::base::VLSIFT(size[0], size[1], scales, octaves, octave_min, peak, edge, magnif));
  return 0;

  BOB_CATCH_MEMBER("cannot create VLSIFT", -1)
}

static void PyBobIpBaseVLSIFT_delete(PyBobIpBaseVLSIFTObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseVLSIFT_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseVLSIFT_Type));
}

static PyObject* PyBobIpBaseVLSIFT_RichCompare(PyBobIpBaseVLSIFTObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseVLSIFT_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseVLSIFTObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare VLSIFT objects", 0)
}

/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto size = bob::extension::VariableDoc(
  "size",
  "(int, int)",
  "The shape of the images to process, with read and write access"
);
PyObject* PyBobIpBaseVLSIFT_getSize(PyBobIpBaseVLSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(ii)", self->cxx->getHeight(), self->cxx->getWidth());
  BOB_CATCH_MEMBER("size could not be read", 0)
}
int PyBobIpBaseVLSIFT_setSize(PyBobIpBaseVLSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  int h, w;
  if (!PyArg_ParseTuple(value, "ii", &h, &w)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two ints", Py_TYPE(self)->tp_name, size.name());
    return -1;
  }
  self->cxx->setHeight(h);;
  self->cxx->setWidth(w);;
  return 0;
  BOB_CATCH_MEMBER("size could not be set", -1)
}

static auto scales = bob::extension::VariableDoc(
  "scales",
  "int",
  "The number of intervals of the pyramid, with read and write access",
  "Three additional scales will be computed in practice, as this is required for extracting VLSIFT features"
);
PyObject* PyBobIpBaseVLSIFT_getScales(PyBobIpBaseVLSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNIntervals());
  BOB_CATCH_MEMBER("scales could not be read", 0)
}
int PyBobIpBaseVLSIFT_setScales(PyBobIpBaseVLSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, scales.name());
    return -1;
  }
  self->cxx->setNIntervals(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("scales could not be set", -1)
}

static auto octaves = bob::extension::VariableDoc(
  "octaves",
  "int",
  "The number of octaves of the pyramid, with read and write access"
);
PyObject* PyBobIpBaseVLSIFT_getOctaves(PyBobIpBaseVLSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNOctaves());
  BOB_CATCH_MEMBER("octaves could not be read", 0)
}
int PyBobIpBaseVLSIFT_setOctaves(PyBobIpBaseVLSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, octaves.name());
    return -1;
  }
  self->cxx->setNOctaves(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("octaves could not be set", -1)
}

static auto octaveMin = bob::extension::VariableDoc(
  "octave_min",
  "int",
  "The index of the minimum octave, with read and write access"
);
PyObject* PyBobIpBaseVLSIFT_getOctaveMin(PyBobIpBaseVLSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getOctaveMin());
  BOB_CATCH_MEMBER("octave_min could not be read", 0)
}
int PyBobIpBaseVLSIFT_setOctaveMin(PyBobIpBaseVLSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, octaveMin.name());
    return -1;
  }
  self->cxx->setOctaveMin(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("octave_min could not be set", -1)
}

static auto octaveMax = bob::extension::VariableDoc(
  "octave_max",
  "int",
  "The index of the minimum octave, read only access",
  "This is equal to ``octave_min+octaves-1``."
);
PyObject* PyBobIpBaseVLSIFT_getOctaveMax(PyBobIpBaseVLSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getOctaveMax());
  BOB_CATCH_MEMBER("octave_max could not be read", 0)
}

static auto peakThreshold = bob::extension::VariableDoc(
  "peak_threshold",
  "float",
  "The peak threshold (minimum amount of contrast to accept a keypoint), with read and write access"
);
PyObject* PyBobIpBaseVLSIFT_getPeakThreshold(PyBobIpBaseVLSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getPeakThres());
  BOB_CATCH_MEMBER("peak_threshold could not be read", 0)
}
int PyBobIpBaseVLSIFT_setPeakThreshold(PyBobIpBaseVLSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setPeakThres(d);
  return 0;
  BOB_CATCH_MEMBER("peak_threshold could not be set", -1)
}

static auto edgeThreshold = bob::extension::VariableDoc(
  "edge_threshold",
  "float",
  "The edge rejection threshold used during keypoint detection, with read and write access"
);
PyObject* PyBobIpBaseVLSIFT_getEdgeThreshold(PyBobIpBaseVLSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getEdgeThres());
  BOB_CATCH_MEMBER("edge_threshold could not be read", 0)
}
int PyBobIpBaseVLSIFT_setEdgeThreshold(PyBobIpBaseVLSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setEdgeThres(d);
  return 0;
  BOB_CATCH_MEMBER("edge_threshold could not be set", -1)
}

static auto magnif = bob::extension::VariableDoc(
  "magnif",
  "float",
  "The magnification factor for the descriptor"
);
PyObject* PyBobIpBaseVLSIFT_getMagnif(PyBobIpBaseVLSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getMagnif());
  BOB_CATCH_MEMBER("magnif could not be read", 0)
}
int PyBobIpBaseVLSIFT_setMagnif(PyBobIpBaseVLSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setMagnif(d);
  return 0;
  BOB_CATCH_MEMBER("magnif could not be set", -1)
}


static PyGetSetDef PyBobIpBaseVLSIFT_getseters[] = {
    {
      size.name(),
      (getter)PyBobIpBaseVLSIFT_getSize,
      (setter)PyBobIpBaseVLSIFT_setSize,
      size.doc(),
      0
    },
    {
      octaves.name(),
      (getter)PyBobIpBaseVLSIFT_getOctaves,
      (setter)PyBobIpBaseVLSIFT_setOctaves,
      octaves.doc(),
      0
    },
    {
      scales.name(),
      (getter)PyBobIpBaseVLSIFT_getScales,
      (setter)PyBobIpBaseVLSIFT_setScales,
      scales.doc(),
      0
    },
    {
      octaveMin.name(),
      (getter)PyBobIpBaseVLSIFT_getOctaveMin,
      (setter)PyBobIpBaseVLSIFT_setOctaveMin,
      octaveMin.doc(),
      0
    },
    {
      octaveMax.name(),
      (getter)PyBobIpBaseVLSIFT_getOctaveMax,
      0,
      octaveMax.doc(),
      0
    },
    {
      peakThreshold.name(),
      (getter)PyBobIpBaseVLSIFT_getPeakThreshold,
      (setter)PyBobIpBaseVLSIFT_setPeakThreshold,
      peakThreshold.doc(),
      0
    },
    {
      edgeThreshold.name(),
      (getter)PyBobIpBaseVLSIFT_getEdgeThreshold,
      (setter)PyBobIpBaseVLSIFT_setEdgeThreshold,
      edgeThreshold.doc(),
      0
    },
    {
      magnif.name(),
      (getter)PyBobIpBaseVLSIFT_getMagnif,
      (setter)PyBobIpBaseVLSIFT_setMagnif,
      magnif.doc(),
      0
    },
    {0}  /* Sentinel */
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto extract = bob::extension::FunctionDoc(
  "extract",
  "Computes the SIFT features from an input image",
  "A keypoint is specified by a 3- or 4-tuple (y, x, sigma, [orientation]), stored as one row of the given ``keypoints`` parameter. "
  "If the ``keypoints`` are not given, the are detected first. "
  "It returns a list of descriptors, one for each keypoint and orientation. "
  "The first four values are the x, y, sigma and orientation of the values. "
  "The 128 remaining values define the descriptor.\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("src, [keypoints]", "dst")
.add_parameter("src", "array_like (2D, uint8)", "The input image which should be processed")
.add_parameter("keypoints", "array_like (2D, float)", "The keypoints at which the descriptors should be computed")
.add_return("dst", "[array_like (1D, float)]", "The resulting descriptors; the first four values are the x, y, sigma and orientation of the keypoints, the 128 remaining values define the descriptor")
;

static PyObject* PyBobIpBaseVLSIFT_extract(PyBobIpBaseVLSIFTObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = extract.kwlist();

  PyBlitzArrayObject* src,* keypoints = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O&", kwlist, &PyBlitzArray_Converter, &src, &PyBlitzArray_Converter, &keypoints)) return 0;

  auto src_ = make_safe(src);
  auto kp_ = make_xsafe(keypoints);

  // perform checks on input and output image
  if (src->ndim != 2 || src->type_num != NPY_UINT8){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of type uint8", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (keypoints && (keypoints->ndim != 2 || keypoints->type_num != NPY_FLOAT64)){
    PyErr_Format(PyExc_TypeError, "`%s' 'keypoints' must be a 2D arrays of type float", Py_TYPE(self)->tp_name);
    return 0;
  }

  // extract SIFT features
  std::vector<blitz::Array<double,1>> features;
  if (keypoints)
    self->cxx->extract(*PyBlitzArrayCxx_AsBlitz<uint8_t,2>(src), *PyBlitzArrayCxx_AsBlitz<double, 2>(keypoints), features);
  else
    self->cxx->extract(*PyBlitzArrayCxx_AsBlitz<uint8_t,2>(src), features);

  // extract into a list of numpy arrays
  PyObject* dst = PyList_New(features.size());
  auto dst_ = make_safe(dst);
  for (Py_ssize_t i = 0; i < PyList_Size(dst); ++i){
    PyList_SET_ITEM(dst, i, PyBlitzArrayCxx_AsNumpy(features[i]));
  }

  return Py_BuildValue("O", dst);

  BOB_CATCH_MEMBER("cannot extract SIFT features for image", 0)
}


static PyMethodDef PyBobIpBaseVLSIFT_methods[] = {
  {
    extract.name(),
    (PyCFunction)PyBobIpBaseVLSIFT_extract,
    METH_VARARGS|METH_KEYWORDS,
    extract.doc()
  },
  {0} /* Sentinel */
};



// VLDSIFT

static auto VLDSIFT_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".VLDSIFT",
  "Computes dense SIFT features using the VLFeat library",
  "For details, please read [Lowe2004]_."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates an object that allows the extraction of VLDSIFT descriptors",
    ".. todo:: Explain VLDSIFT constructor in more detail.",
    true
  )
  .add_prototype("size, [step], [block_size]", "")
  .add_prototype("sift", "")
  .add_parameter("size", "(int, int)", "The height and width of the images to process")
  .add_parameter("step", "(int, int)", "[default: ``(5, 5)``] The step along the y- and x-axes")
  .add_parameter("block_size", "(int, int)", "[default: ``(5, 5)``] The block size along the y- and x-axes")
  .add_parameter("sift", ":py:class:`bob.ip.base.VLDSIFT`", "The VLDSIFT object to use for copy-construction")
);

static int PyBobIpBaseVLDSIFT_init(PyBobIpBaseVLDSIFTObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = VLDSIFT_doc.kwlist(0);
  char** kwlist2 = VLDSIFT_doc.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseVLDSIFT_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseVLDSIFTObject* sift;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseVLDSIFT_Type, &sift)) return -1;

    self->cxx.reset(new bob::ip::base::VLDSIFT(*sift->cxx));
    return 0;
  }

  blitz::TinyVector<int,2> size, step(5,5), block_size(5,5);

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)|(ii)(ii)", kwlist1, &size[0], &size[1], &step[0], &step[1], &block_size[0], &block_size[1])){
    VLDSIFT_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::ip::base::VLDSIFT(size, step, block_size));
  return 0;

  BOB_CATCH_MEMBER("cannot create VLDSIFT", -1)
}

static void PyBobIpBaseVLDSIFT_delete(PyBobIpBaseVLDSIFTObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseVLDSIFT_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseVLDSIFT_Type));
}

static PyObject* PyBobIpBaseVLDSIFT_RichCompare(PyBobIpBaseVLDSIFTObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseVLDSIFT_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseVLDSIFTObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare VLDSIFT objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto size_ = bob::extension::VariableDoc(
  "size",
  "(int, int)",
  "The shape of the images to process, with read and write access"
);
PyObject* PyBobIpBaseVLDSIFT_getSize(PyBobIpBaseVLDSIFTObject* self, void*){
  BOB_TRY
  auto r = self->cxx->getSize();
  return Py_BuildValue("(ii)", r[0], r[1]);
  BOB_CATCH_MEMBER("size could not be read", 0)
}
int PyBobIpBaseVLDSIFT_setSize(PyBobIpBaseVLDSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> r;
  if (!PyArg_ParseTuple(value, "ii", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two ints", Py_TYPE(self)->tp_name, size.name());
    return -1;
  }
  self->cxx->setSize(r);
  return 0;
  BOB_CATCH_MEMBER("size could not be set", -1)
}

static auto step = bob::extension::VariableDoc(
  "step",
  "(int, int)",
  "The step along both directions, with read and write access"
);
PyObject* PyBobIpBaseVLDSIFT_getStep(PyBobIpBaseVLDSIFTObject* self, void*){
  BOB_TRY
  auto r = self->cxx->getStep();
  return Py_BuildValue("(ii)", r[0], r[1]);
  BOB_CATCH_MEMBER("step could not be read", 0)
}
int PyBobIpBaseVLDSIFT_setStep(PyBobIpBaseVLDSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> r;
  if (!PyArg_ParseTuple(value, "ii", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two ints", Py_TYPE(self)->tp_name, size.name());
    return -1;
  }
  self->cxx->setStep(r);
  return 0;
  BOB_CATCH_MEMBER("step could not be set", -1)
}

static auto blockSize = bob::extension::VariableDoc(
  "block_size",
  "(int, int)",
  "The block size in both directions, with read and write access"
);
PyObject* PyBobIpBaseVLDSIFT_getBlockSize(PyBobIpBaseVLDSIFTObject* self, void*){
  BOB_TRY
  auto r = self->cxx->getBlockSize();
  return Py_BuildValue("(ii)", r[0], r[1]);
  BOB_CATCH_MEMBER("block_size could not be read", 0)
}
int PyBobIpBaseVLDSIFT_setBlockSize(PyBobIpBaseVLDSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> r;
  if (!PyArg_ParseTuple(value, "ii", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two ints", Py_TYPE(self)->tp_name, size.name());
    return -1;
  }
  self->cxx->setBlockSize(r);
  return 0;
  BOB_CATCH_MEMBER("block_size could not be set", -1)
}

static auto useFlatWindow = bob::extension::VariableDoc(
  "use_flat_window",
  "bool",
  "Whether to use a flat window or not (to boost the processing time), with read and write access"
);
PyObject* PyBobIpBaseVLDSIFT_getUseFlatWindow(PyBobIpBaseVLDSIFTObject* self, void*){
  BOB_TRY
  if (self->cxx->getUseFlatWindow()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("use_flat_window could not be read", 0)
}
int PyBobIpBaseVLDSIFT_setUseFlatWindow(PyBobIpBaseVLDSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, useFlatWindow.name());
    return -1;
  }
  self->cxx->setUseFlatWindow(r>0);
  return 0;
  BOB_CATCH_MEMBER("use_flat_window could not be set", -1)
}

static auto windowSize = bob::extension::VariableDoc(
  "window_size",
  "float",
  "The window size, with read and write access"
);
PyObject* PyBobIpBaseVLDSIFT_getWindowSize(PyBobIpBaseVLDSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getWindowSize());
  BOB_CATCH_MEMBER("window_size could not be read", 0)
}
int PyBobIpBaseVLDSIFT_setWindowSize(PyBobIpBaseVLDSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setWindowSize(d);
  return 0;
  BOB_CATCH_MEMBER("window_size could not be set", -1)
}



static PyGetSetDef PyBobIpBaseVLDSIFT_getseters[] = {
    {
      size_.name(),
      (getter)PyBobIpBaseVLDSIFT_getSize,
      (setter)PyBobIpBaseVLDSIFT_setSize,
      size_.doc(),
      0
    },
    {
      step.name(),
      (getter)PyBobIpBaseVLDSIFT_getStep,
      (setter)PyBobIpBaseVLDSIFT_setStep,
      step.doc(),
      0
    },
    {
      blockSize.name(),
      (getter)PyBobIpBaseVLDSIFT_getBlockSize,
      (setter)PyBobIpBaseVLDSIFT_setBlockSize,
      blockSize.doc(),
      0
    },
    {
      useFlatWindow.name(),
      (getter)PyBobIpBaseVLDSIFT_getUseFlatWindow,
      (setter)PyBobIpBaseVLDSIFT_setUseFlatWindow,
      useFlatWindow.doc(),
      0
    },
    {
      windowSize.name(),
      (getter)PyBobIpBaseVLDSIFT_getWindowSize,
      (setter)PyBobIpBaseVLDSIFT_setWindowSize,
      windowSize.doc(),
      0
    },
    {0}  /* Sentinel */
};



/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto outputShape = bob::extension::FunctionDoc(
  "output_shape",
  "Returns the output shape for the current setup",
  "The output shape is a 2-element tuple consisting of the number of keypoints for the current size, and the size of the descriptors",
  true
)
.add_prototype("", "shape")
.add_return("shape", "(int, int)", "The shape of the output array required to call :py:func:`extract`")
;

static PyObject* PyBobIpBaseVLDSIFT_outputShape(PyBobIpBaseVLDSIFTObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char* kwlist[] = {0};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwlist)) return 0;

  return Py_BuildValue("(ii)", self->cxx->getNKeypoints(), self->cxx->getDescriptorSize());

  BOB_CATCH_MEMBER("cannot compute output shape", 0)
}

static auto extract_ = bob::extension::FunctionDoc(
  "extract",
  "Computes the dense SIFT features from an input image, using the VLFeat library",
  "If given, the results are put in the output ``dst``, which should be of type float and allocated in the shape :py:func:`output_shape` method.\n\n"
  ".. todo:: Describe the output of the :py:func:`VLDSIFT.extract` method in more detail.\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like (2D, float32)", "The input image which should be processed")
.add_parameter("dst", "[array_like (2D, float32)]", "The descriptors that should have been allocated in size :py:func:`output_shape`")
.add_return("dst", "array_like (2D, float32)", "The resulting descriptors, if given it will be the same as the ``dst`` parameter")
;

static PyObject* PyBobIpBaseVLDSIFT_extract(PyBobIpBaseVLDSIFTObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = extract_.kwlist();

  PyBlitzArrayObject* src, *dst = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O&", kwlist, &PyBlitzArray_Converter, &src, &PyBlitzArray_OutputConverter, &dst)) return 0;

  auto src_ = make_safe(src), dst_ = make_xsafe(dst);

  // perform checks on input and output image
  if (src->ndim != 2 || src->type_num != NPY_FLOAT32){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of type numpy.float32", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (dst){
    // check that data type is correct and dimensions fit
    if (dst->ndim != 2 || dst->type_num != NPY_FLOAT32){
      PyErr_Format(PyExc_TypeError, "'%s' the 'dst' array must be 2D of type numpy.float32, not %dD of type %s", Py_TYPE(self)->tp_name, (int)dst->ndim, PyBlitzArray_TypenumAsString(dst->type_num));
      return 0;
    }
  } else {
    // create output in the desired dimensions
    Py_ssize_t n[] = {(Py_ssize_t)self->cxx->getNKeypoints(), (Py_ssize_t)self->cxx->getDescriptorSize()};
    dst = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_FLOAT32, 2, n));
    dst_ = make_safe(dst);
  }

  // finally, extract the features
  self->cxx->extract(*PyBlitzArrayCxx_AsBlitz<float,2>(src), *PyBlitzArrayCxx_AsBlitz<float,2>(dst));
  return PyBlitzArray_AsNumpyArray(dst,0);

  BOB_CATCH_MEMBER("cannot extract dense SIFT features for image", 0)
}


static PyMethodDef PyBobIpBaseVLDSIFT_methods[] = {
  {
    outputShape.name(),
    (PyCFunction)PyBobIpBaseVLDSIFT_outputShape,
    METH_VARARGS|METH_KEYWORDS,
    outputShape.doc()
  },
  {
    extract_.name(),
    (PyCFunction)PyBobIpBaseVLDSIFT_extract,
    METH_VARARGS|METH_KEYWORDS,
    extract_.doc()
  },
  {0} /* Sentinel */
};



/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the VLSIFT type struct; will be initialized later
PyTypeObject PyBobIpBaseVLSIFT_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

// Define the VLDSIFT type struct; will be initialized later
PyTypeObject PyBobIpBaseVLDSIFT_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseVLFEAT(PyObject* module)
{
  // VLSIFT
  // initialize the type struct
  PyBobIpBaseVLSIFT_Type.tp_name = VLSIFT_doc.name();
  PyBobIpBaseVLSIFT_Type.tp_basicsize = sizeof(PyBobIpBaseVLSIFTObject);
  PyBobIpBaseVLSIFT_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseVLSIFT_Type.tp_doc = VLSIFT_doc.doc();

  // set the functions
  PyBobIpBaseVLSIFT_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseVLSIFT_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseVLSIFT_init);
  PyBobIpBaseVLSIFT_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseVLSIFT_delete);
  PyBobIpBaseVLSIFT_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseVLSIFT_RichCompare);
  PyBobIpBaseVLSIFT_Type.tp_methods = PyBobIpBaseVLSIFT_methods;
  PyBobIpBaseVLSIFT_Type.tp_getset = PyBobIpBaseVLSIFT_getseters;
  PyBobIpBaseVLSIFT_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseVLSIFT_extract);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseVLSIFT_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseVLSIFT_Type);
  if (PyModule_AddObject(module, "VLSIFT", (PyObject*)&PyBobIpBaseVLSIFT_Type) < 0) return false;


  // VLDSIFT
  // initialize the type struct
  PyBobIpBaseVLDSIFT_Type.tp_name = VLDSIFT_doc.name();
  PyBobIpBaseVLDSIFT_Type.tp_basicsize = sizeof(PyBobIpBaseVLDSIFTObject);
  PyBobIpBaseVLDSIFT_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseVLDSIFT_Type.tp_doc = VLDSIFT_doc.doc();

  // set the functions
  PyBobIpBaseVLDSIFT_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseVLDSIFT_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseVLDSIFT_init);
  PyBobIpBaseVLDSIFT_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseVLDSIFT_delete);
  PyBobIpBaseVLDSIFT_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseVLDSIFT_RichCompare);
  PyBobIpBaseVLDSIFT_Type.tp_methods = PyBobIpBaseVLDSIFT_methods;
  PyBobIpBaseVLDSIFT_Type.tp_getset = PyBobIpBaseVLDSIFT_getseters;
  PyBobIpBaseVLDSIFT_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseVLDSIFT_extract);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseVLDSIFT_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseVLDSIFT_Type);
  return PyModule_AddObject(module, "VLDSIFT", (PyObject*)&PyBobIpBaseVLDSIFT_Type) >= 0;
}

#endif // HAVE_VLFEAT
