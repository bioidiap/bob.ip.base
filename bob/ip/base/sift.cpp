/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Jul  7 10:13:37 CEST 2014
 *
 * @brief Binds the SIFT class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

static auto SIFT_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".SIFT",
  "This class allows after configuration the extraction of SIFT descriptors",
  "For details, please read [Lowe2004]_."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates an object that allows the extraction of SIFT descriptors",
    ".. todo:: Explain SIFT constructor in more detail.\n\n"
    ".. warning:: The order of the parameters ``scales`` and ``octaves`` has changed compared to the old implementation, in order to keep it consistent with :py:class:`bob.ip.base.VLSIFT`!",
    true
  )
  .add_prototype("size, scales, octaves, octave_min, [sigma_n], [sigma0], [contrast_thres], [edge_thres], [norm_thres], [kernel_radius_factor], [border]", "")
  .add_prototype("sift", "")
  .add_parameter("size", "(int, int)", "The height and width of the images to process")
  .add_parameter("scales", "int", "The number of intervals of the pyramid. Three additional scales will be computed in practice, as this is required for extracting SIFT features")
  .add_parameter("octaves", "int", "The number of octaves of the pyramid")
  .add_parameter("octave_min", "int", "The index of the minimum octave")
  .add_parameter("sigma_n", "float", "[default: 0.5] The value sigma_n of the standard deviation for the nominal/initial octave/scale")
  .add_parameter("sigma0", "float", "[default: 1.6] The value sigma0 of the standard deviation for the image of the first octave and first scale")
  .add_parameter("contrast_thres", "float", "[default: 0.03] The contrast threshold used during keypoint detection")
  .add_parameter("edge_thres", "float", "[default: 10.] The edge threshold used during keypoint detection")
  .add_parameter("norm_thres", "float", "[default: 0.2] The norm threshold used during descriptor normalization")
  .add_parameter("kernel_radius_factor", "float", "[default: 4.] Factor used to determine the kernel radii: ``size=2*radius+1``. For each Gaussian kernel, the radius is equal to ``ceil(kernel_radius_factor*sigma_{octave,scale})``")
  .add_parameter("border", ":py:class:`bob.sp.BorderType`", "[default: ``bob.sp.BorderType.Mirror``] The extrapolation method used by the convolution at the border")
  .add_parameter("sift", ":py:class:`bob.ip.base.SIFT`", "The SIFT object to use for copy-construction")
);


static int PyBobIpBaseSIFT_init(PyBobIpBaseSIFTObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = SIFT_doc.kwlist(0);
  char** kwlist2 = SIFT_doc.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseSIFT_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseSIFTObject* sift;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseSIFT_Type, &sift)) return -1;

    self->cxx.reset(new bob::ip::base::SIFT(*sift->cxx));
    return 0;
  }

  blitz::TinyVector<int,2> size;
  int scales, octaves, octave_min;
  double sigma_n = 0.5, sigma0 = 1.6, contrast = 0.03, edge = 10., norm = 0.2, factor = 4.;
  bob::sp::Extrapolation::BorderType border = bob::sp::Extrapolation::Mirror;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)iii|ddddddO&", kwlist1, &size[0], &size[1], &scales, &octaves, &octave_min, &sigma_n, &sigma0, &contrast, &edge, &norm, &factor, &PyBobSpExtrapolationBorder_Converter, &border)){
    SIFT_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::ip::base::SIFT(size[0], size[1], scales, octaves, octave_min, sigma_n, sigma0, contrast, edge, norm, factor, border));
  return 0;

  BOB_CATCH_MEMBER("cannot create SIFT", -1)
}

static void PyBobIpBaseSIFT_delete(PyBobIpBaseSIFTObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseSIFT_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseSIFT_Type));
}

static PyObject* PyBobIpBaseSIFT_RichCompare(PyBobIpBaseSIFTObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseSIFT_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseSIFTObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare SIFT objects", 0)
}

/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto size = bob::extension::VariableDoc(
  "size",
  "(int, int)",
  "The shape of the images to process, with read and write access"
);
PyObject* PyBobIpBaseSIFT_getSize(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(ii)", self->cxx->getHeight(), self->cxx->getWidth());
  BOB_CATCH_MEMBER("size could not be read", 0)
}
int PyBobIpBaseSIFT_setSize(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
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

static auto octaves = bob::extension::VariableDoc(
  "octaves",
  "int",
  "The number of octaves of the pyramid, with read and write access"
);
PyObject* PyBobIpBaseSIFT_getOctaves(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNOctaves());
  BOB_CATCH_MEMBER("octaves could not be read", 0)
}
int PyBobIpBaseSIFT_setOctaves(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, octaves.name());
    return -1;
  }
  self->cxx->setNOctaves(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("octaves could not be set", -1)
}

static auto scales = bob::extension::VariableDoc(
  "scales",
  "int",
  "The number of intervals of the pyramid, with read and write access",
  "Three additional scales will be computed in practice, as this is required for extracting SIFT features"
);
PyObject* PyBobIpBaseSIFT_getScales(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNIntervals());
  BOB_CATCH_MEMBER("scales could not be read", 0)
}
int PyBobIpBaseSIFT_setScales(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, scales.name());
    return -1;
  }
  self->cxx->setNIntervals(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("scales could not be set", -1)
}

static auto octaveMin = bob::extension::VariableDoc(
  "octave_min",
  "int",
  "The index of the minimum octave, with read and write access"
);
PyObject* PyBobIpBaseSIFT_getOctaveMin(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getOctaveMin());
  BOB_CATCH_MEMBER("octave_min could not be read", 0)
}
int PyBobIpBaseSIFT_setOctaveMin(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
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
PyObject* PyBobIpBaseSIFT_getOctaveMax(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getOctaveMax());
  BOB_CATCH_MEMBER("octave_max could not be read", 0)
}

static auto sigmaN = bob::extension::VariableDoc(
  "sigma_n",
  "float",
  "The value sigma_n of the standard deviation for the nominal/initial octave/scale; with read and write access"
);
PyObject* PyBobIpBaseSIFT_getSigmaN(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getSigmaN());
  BOB_CATCH_MEMBER("sigma_n could not be read", 0)
}
int PyBobIpBaseSIFT_setSigmaN(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setSigmaN(d);
  return 0;
  BOB_CATCH_MEMBER("sigma_n could not be set", -1)
}

static auto sigma0 = bob::extension::VariableDoc(
  "sigma0",
  "float",
  "The value sigma0 of the standard deviation for the image of the first octave and first scale"
);
PyObject* PyBobIpBaseSIFT_getSigma0(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getSigma0());
  BOB_CATCH_MEMBER("sigma_0 could not be read", 0)
}
int PyBobIpBaseSIFT_setSigma0(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setSigma0(d);
  return 0;
  BOB_CATCH_MEMBER("sigma_0 could not be set", -1)
}

static auto contrastThreshold = bob::extension::VariableDoc(
  "contrast_threshold",
  "float",
  "The contrast threshold used during keypoint detection"
);
PyObject* PyBobIpBaseSIFT_getContrastThreshold(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getContrastThreshold());
  BOB_CATCH_MEMBER("contrast_threshold could not be read", 0)
}
int PyBobIpBaseSIFT_setContrastThreshold(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setContrastThreshold(d);
  return 0;
  BOB_CATCH_MEMBER("contrast_threshold could not be set", -1)
}

static auto edgeThreshold = bob::extension::VariableDoc(
  "edge_threshold",
  "float",
  "The edge threshold used during keypoint detection"
);
PyObject* PyBobIpBaseSIFT_getEdgeThreshold(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getEdgeThreshold());
  BOB_CATCH_MEMBER("edge_threshold could not be read", 0)
}
int PyBobIpBaseSIFT_setEdgeThreshold(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setEdgeThreshold(d);
  return 0;
  BOB_CATCH_MEMBER("edge_threshold could not be set", -1)
}

static auto normThreshold = bob::extension::VariableDoc(
  "norm_threshold",
  "float",
  "The norm threshold used during keypoint detection"
);
PyObject* PyBobIpBaseSIFT_getNormThreshold(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getNormThreshold());
  BOB_CATCH_MEMBER("norm_threshold could not be read", 0)
}
int PyBobIpBaseSIFT_setNormThreshold(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setNormThreshold(d);
  return 0;
  BOB_CATCH_MEMBER("norm_threshold could not be set", -1)
}

static auto kernelRadiusFactor = bob::extension::VariableDoc(
  "kernel_radius_factor",
  "float",
  "Factor used to determine the kernel radii ``size=2*radius+1``",
  "For each Gaussian kernel, the radius is equal to ``ceil(kernel_radius_factor*sigma_{octave,scale})``"
);
PyObject* PyBobIpBaseSIFT_getKernelRadiusFactor(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getKernelRadiusFactor());
  BOB_CATCH_MEMBER("kernel_radius_factor could not be read", 0)
}
int PyBobIpBaseSIFT_setKernelRadiusFactor(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setKernelRadiusFactor(d);
  return 0;
  BOB_CATCH_MEMBER("kernel_radius_factor could not be set", -1)
}

static auto border = bob::extension::VariableDoc(
  "border",
  ":py:class:`bob.sp.BorderType`",
  "The extrapolation method used by the convolution at the border; with read and write access"
);
PyObject* PyBobIpBaseSIFT_getBorder(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getConvBorder());
  BOB_CATCH_MEMBER("border could not be read", 0)
}
int PyBobIpBaseSIFT_setBorder(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  bob::sp::Extrapolation::BorderType b;
  if (!PyBobSpExtrapolationBorder_Converter(value, &b)) return -1;
  self->cxx->setConvBorder(b);
  return 0;
  BOB_CATCH_MEMBER("border could not be set", -1)
}

static auto blocks = bob::extension::VariableDoc(
  "blocks",
  "int",
  "The number of blocks for the descriptor, with read and write access"
);
PyObject* PyBobIpBaseSIFT_getBlocks(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNBlocks());
  BOB_CATCH_MEMBER("blocks could not be read", 0)
}
int PyBobIpBaseSIFT_setBlocks(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, blocks.name());
    return -1;
  }
  self->cxx->setNBlocks(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("blocks could not be set", -1)
}

static auto bins = bob::extension::VariableDoc(
  "bins",
  "int",
  "The number of bins for the descriptor, with read and write access"
);
PyObject* PyBobIpBaseSIFT_getBins(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNBins());
  BOB_CATCH_MEMBER("bins could not be read", 0)
}
int PyBobIpBaseSIFT_setBins(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, bins.name());
    return -1;
  }
  self->cxx->setNBins(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("bins could not be set", -1)
}

static auto gaussianWindowSize = bob::extension::VariableDoc(
  "gaussian_window_size",
  "float",
  "The Gaussian window size for the descriptor"
);
PyObject* PyBobIpBaseSIFT_getGaussianWindowSize(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getGaussianWindowSize());
  BOB_CATCH_MEMBER("gaussian_window_size could not be read", 0)
}
int PyBobIpBaseSIFT_setGaussianWindowSize(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setGaussianWindowSize(d);
  return 0;
  BOB_CATCH_MEMBER("gaussian_window_size could not be set", -1)
}

static auto magnif = bob::extension::VariableDoc(
  "magnif",
  "float",
  "The magnification factor for the descriptor"
);
PyObject* PyBobIpBaseSIFT_getMagnif(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getMagnif());
  BOB_CATCH_MEMBER("magnif could not be read", 0)
}
int PyBobIpBaseSIFT_setMagnif(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setMagnif(d);
  return 0;
  BOB_CATCH_MEMBER("magnif could not be set", -1)
}

static auto normEpsilon = bob::extension::VariableDoc(
  "norm_epsilon",
  "float",
  "The magnification factor for the descriptor"
);
PyObject* PyBobIpBaseSIFT_getNormEpsilon(PyBobIpBaseSIFTObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getNormEpsilon());
  BOB_CATCH_MEMBER("norm_epsilon could not be read", 0)
}
int PyBobIpBaseSIFT_setNormEpsilon(PyBobIpBaseSIFTObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setNormEpsilon(d);
  return 0;
  BOB_CATCH_MEMBER("norm_epsilon could not be set", -1)
}

static PyGetSetDef PyBobIpBaseSIFT_getseters[] = {
    {
      size.name(),
      (getter)PyBobIpBaseSIFT_getSize,
      (setter)PyBobIpBaseSIFT_setSize,
      size.doc(),
      0
    },
    {
      octaves.name(),
      (getter)PyBobIpBaseSIFT_getOctaves,
      (setter)PyBobIpBaseSIFT_setOctaves,
      octaves.doc(),
      0
    },
    {
      scales.name(),
      (getter)PyBobIpBaseSIFT_getScales,
      (setter)PyBobIpBaseSIFT_setScales,
      scales.doc(),
      0
    },
    {
      octaveMin.name(),
      (getter)PyBobIpBaseSIFT_getOctaveMin,
      (setter)PyBobIpBaseSIFT_setOctaveMin,
      octaveMin.doc(),
      0
    },
    {
      octaveMax.name(),
      (getter)PyBobIpBaseSIFT_getOctaveMax,
      0,
      octaveMax.doc(),
      0
    },
    {
      sigmaN.name(),
      (getter)PyBobIpBaseSIFT_getSigmaN,
      (setter)PyBobIpBaseSIFT_setSigmaN,
      sigmaN.doc(),
      0
    },
    {
      sigma0.name(),
      (getter)PyBobIpBaseSIFT_getSigma0,
      (setter)PyBobIpBaseSIFT_setSigma0,
      sigma0.doc(),
      0
    },
    {
      contrastThreshold.name(),
      (getter)PyBobIpBaseSIFT_getContrastThreshold,
      (setter)PyBobIpBaseSIFT_setContrastThreshold,
      contrastThreshold.doc(),
      0
    },
    {
      edgeThreshold.name(),
      (getter)PyBobIpBaseSIFT_getEdgeThreshold,
      (setter)PyBobIpBaseSIFT_setEdgeThreshold,
      edgeThreshold.doc(),
      0
    },
    {
      normThreshold.name(),
      (getter)PyBobIpBaseSIFT_getNormThreshold,
      (setter)PyBobIpBaseSIFT_setNormThreshold,
      normThreshold.doc(),
      0
    },
    {
      kernelRadiusFactor.name(),
      (getter)PyBobIpBaseSIFT_getKernelRadiusFactor,
      (setter)PyBobIpBaseSIFT_setKernelRadiusFactor,
      kernelRadiusFactor.doc(),
      0
    },
    {
      border.name(),
      (getter)PyBobIpBaseSIFT_getBorder,
      (setter)PyBobIpBaseSIFT_setBorder,
      border.doc(),
      0
    },
    {
      blocks.name(),
      (getter)PyBobIpBaseSIFT_getBlocks,
      (setter)PyBobIpBaseSIFT_setBlocks,
      blocks.doc(),
      0
    },
    {
      bins.name(),
      (getter)PyBobIpBaseSIFT_getBins,
      (setter)PyBobIpBaseSIFT_setBins,
      bins.doc(),
      0
    },
    {
      gaussianWindowSize.name(),
      (getter)PyBobIpBaseSIFT_getGaussianWindowSize,
      (setter)PyBobIpBaseSIFT_setGaussianWindowSize,
      gaussianWindowSize.doc(),
      0
    },
    {
      magnif.name(),
      (getter)PyBobIpBaseSIFT_getMagnif,
      (setter)PyBobIpBaseSIFT_setMagnif,
      magnif.doc(),
      0
    },
    {
      normEpsilon.name(),
      (getter)PyBobIpBaseSIFT_getNormEpsilon,
      (setter)PyBobIpBaseSIFT_setNormEpsilon,
      normEpsilon.doc(),
      0
    },
    {0}  /* Sentinel */
};



/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto setNoSmooth = bob::extension::FunctionDoc(
  "set_sigma0_no_init_smoothing",
  "Sets sigma0 such that there is not smoothing at the first scale of octave_min",
  0,
  true
)
.add_prototype("");

static PyObject* PyBobIpBaseSIFT_setNoSmooth(PyBobIpBaseSIFTObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char* kwlist[] = {0};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwlist)) return 0;

  self->cxx->setSigma0NoInitSmoothing();
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("cannot call set_sigma0_no_init_smoothing", 0)
}

static auto outputShape = bob::extension::FunctionDoc(
  "output_shape",
  "Returns the output shape for the given number of input keypoints",
  0,
  true
)
.add_prototype("keypoints", "shape")
.add_parameter("keypoints", "int", "The number of keypoints that you want to retrieve SIFT features for")
.add_return("shape", "(int, int, int, int)", "The shape of the output array required to call :py:func:`compute_descriptor`")
;

static PyObject* PyBobIpBaseSIFT_outputShape(PyBobIpBaseSIFTObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = outputShape.kwlist();

  int keypoints;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &keypoints)) return 0;

  auto shape = self->cxx->getDescriptorShape();
  return Py_BuildValue("(iiii)", keypoints, shape[0], shape[1], shape[2]);

  BOB_CATCH_MEMBER("cannot compute output shape", 0)
}

static auto computeDescriptor = bob::extension::FunctionDoc(
  "compute_descriptor",
  "Computes SIFT descriptor for a 2D/grayscale image, at the given keypoints",
  "If given, the results are put in the output ``dst``, which output should be of type float and allocated in the shape :py:func:`output_shape` method).\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("src, keypoints, [dst]", "dst")
.add_parameter("src", "array_like (2D)", "The input image which should be processed")
.add_parameter("keypoints", "[:py:class:`bob.ip.base.GSSKeypoint`]", "The keypoints at which the descriptors should be computed")
.add_parameter("dst", "[array_like (4D, float)]", "The descriptors that should have been allocated in size :py:func:`output_shape`")
.add_return("dst", "[array_like (4D, float)]", "The resulting descriptors, if given it will be the same as the ``dst`` parameter")
;

template <typename T>
static PyObject* compute_inner(PyBobIpBaseSIFTObject* self, PyBlitzArrayObject* src, const std::vector<boost::shared_ptr<bob::ip::base::GSSKeypoint> >& keypoints, PyBlitzArrayObject* dst){
  self->cxx->computeDescriptor(*PyBlitzArrayCxx_AsBlitz<T,2>(src), keypoints, *PyBlitzArrayCxx_AsBlitz<double,4>(dst));
  return PyBlitzArray_AsNumpyArray(dst,0);
}

static PyObject* PyBobIpBaseSIFT_computeDescriptor(PyBobIpBaseSIFTObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = computeDescriptor.kwlist();

  PyBlitzArrayObject* src, *dst = 0;
  PyObject* kp;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O!|O&", kwlist, &PyBlitzArray_Converter, &src, &PyList_Type, &kp, &PyBlitzArray_OutputConverter, &dst)) return 0;

  auto src_ = make_safe(src), dst_ = make_xsafe(dst);

  // perform checks on input and output image
  if (src->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays", Py_TYPE(self)->tp_name);
    return 0;
  }

  // get the list of descriptors
  Py_ssize_t size = PyList_GET_SIZE(kp);
  std::vector<boost::shared_ptr<bob::ip::base::GSSKeypoint>> keypoints(size);
  for (Py_ssize_t i = 0; i < size; ++i){
    PyObject* o = PyList_GET_ITEM(kp, i);
    if (!PyBobIpBaseGSSKeypoint_Check(o)){
      PyErr_Format(PyExc_TypeError, "`%s' keypoints must be of type bob.ip.base.GSSKeypoint, but list item %d is not", Py_TYPE(self)->tp_name, (int)i);
    }
    keypoints[i] = reinterpret_cast<PyBobIpBaseGSSKeypointObject*>(o)->cxx;
  }

  if (dst){
    // check that data type is correct and dimensions fit
    if (dst->ndim != 4){
      PyErr_Format(PyExc_TypeError, "'%s' the 'dst' array must be 4D, not %dD", Py_TYPE(self)->tp_name, (int)dst->ndim);
      return 0;
    }
    if (dst->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "'%s': the 'dst' array must be of type float, not %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(dst->type_num));
      return 0;
    }
  } else {
    // create output in the desired dimensions
    auto shape = self->cxx->getDescriptorShape();
    Py_ssize_t n[] = {size, shape[0], shape[1], shape[2]};
    dst = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_FLOAT64, 4, n));
    dst_ = make_safe(dst);
  }

  // finally, extract the features
  switch (src->type_num){
    case NPY_UINT8:   return compute_inner<uint8_t>(self, src, keypoints, dst);
    case NPY_UINT16:  return compute_inner<uint16_t>(self, src, keypoints, dst);
    case NPY_FLOAT64: return compute_inner<double>(self, src, keypoints, dst);
    default:
      PyErr_Format(PyExc_TypeError, "`%s' processes only images of types uint8, uint16 or float, and not %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(src->type_num));
      return 0;
  }

  BOB_CATCH_MEMBER("cannot compute descriptors for image", 0)
}


static PyMethodDef PyBobIpBaseSIFT_methods[] = {
  {
    setNoSmooth.name(),
    (PyCFunction)PyBobIpBaseSIFT_setNoSmooth,
    METH_VARARGS|METH_KEYWORDS,
    setNoSmooth.doc()
  },
  {
    outputShape.name(),
    (PyCFunction)PyBobIpBaseSIFT_outputShape,
    METH_VARARGS|METH_KEYWORDS,
    outputShape.doc()
  },
  {
    computeDescriptor.name(),
    (PyCFunction)PyBobIpBaseSIFT_computeDescriptor,
    METH_VARARGS|METH_KEYWORDS,
    computeDescriptor.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the SIFT type struct; will be initialized later
PyTypeObject PyBobIpBaseSIFT_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseSIFT(PyObject* module)
{
  // initialize the type struct
  PyBobIpBaseSIFT_Type.tp_name = SIFT_doc.name();
  PyBobIpBaseSIFT_Type.tp_basicsize = sizeof(PyBobIpBaseSIFTObject);
  PyBobIpBaseSIFT_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseSIFT_Type.tp_doc = SIFT_doc.doc();

  // set the functions
  PyBobIpBaseSIFT_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseSIFT_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseSIFT_init);
  PyBobIpBaseSIFT_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseSIFT_delete);
  PyBobIpBaseSIFT_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseSIFT_RichCompare);
  PyBobIpBaseSIFT_Type.tp_methods = PyBobIpBaseSIFT_methods;
  PyBobIpBaseSIFT_Type.tp_getset = PyBobIpBaseSIFT_getseters;
  PyBobIpBaseSIFT_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseSIFT_computeDescriptor);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseSIFT_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseSIFT_Type);
  return PyModule_AddObject(module, "SIFT", (PyObject*)&PyBobIpBaseSIFT_Type) >= 0;
}
