/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Fri Jul  4 15:39:36 CEST 2014
 *
 * @brief Binds the GaussianScaleSpace class and related classes to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ GSSKeypoint Section *********************************/
/******************************************************************/

PyTypeObject PyBobIpBaseGSSKeypoint_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

static auto GSSKeypoint_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".GSSKeypoint",
  "Structure to describe a keypoint on the :py:class:`bob.ip.base.GaussianScaleSpace`",
  "It consists of a scale sigma, a location (y,x) and an orientation."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a GSS keypoint",
    0,
    true
  )
  .add_prototype("sigma, location, [orientation]", "")
  .add_parameter("sigma", "float", "The floating point value describing the scale of the keypoint")
  .add_parameter("location", "(float, float)", "The location of the keypoint")
  .add_parameter("orientation", "float", "[default: 0] The orientation of the keypoint (in degrees)")
);

static int PyBobIpBaseGSSKeypoint_init(PyBobIpBaseGSSKeypointObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = GSSKeypoint_doc.kwlist();

  double sigma, orientation = 0.;
  blitz::TinyVector<double,2> location;

  if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "d(dd)|d", kwlist, &sigma, &location[0], &location[1], &orientation))) return 0;

  self->cxx.reset(new bob::ip::base::GSSKeypoint());
  self->cxx->sigma = sigma;
  self->cxx->y = location[0];
  self->cxx->x = location[1];
  // orientation in radians
  self->cxx->orientation = orientation * M_PI / 180.;
  return 0;

  BOB_CATCH_MEMBER("cannot create GSSKeypoint", -1)
}

static void PyBobIpBaseGSSKeypoint_delete(PyBobIpBaseGSSKeypointObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseGSSKeypoint_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseGSSKeypoint_Type));
}

static auto kpSigma = bob::extension::VariableDoc(
  "sigma",
  "float",
  "The floating point value describing the scale of the keypoint, with read and write access"
);
PyObject* PyBobIpBaseGSSKeypoint_getSigma(PyBobIpBaseGSSKeypointObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->sigma);
  BOB_CATCH_MEMBER("sigma could not be read", 0)
}
int PyBobIpBaseGSSKeypoint_setSigma(PyBobIpBaseGSSKeypointObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->sigma = d;
  return 0;
  BOB_CATCH_MEMBER("sigma could not be set", -1)
}

static auto kpLocation = bob::extension::VariableDoc(
  "location",
  "(float, float)",
  "The location (y, x) of the keypoint, with read and write access"
);
PyObject* PyBobIpBaseGSSKeypoint_getLocation(PyBobIpBaseGSSKeypointObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(dd)", self->cxx->y, self->cxx->x);
  BOB_CATCH_MEMBER("location could not be read", 0)
}
int PyBobIpBaseGSSKeypoint_setLocation(PyBobIpBaseGSSKeypointObject* self, PyObject* value, void*){
  BOB_TRY
  double y, x;
  if (!PyArg_ParseTuple(value, "dd", &y, &x)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, kpLocation.name());
    return -1;
  }
  self->cxx->y = y;
  self->cxx->x = x;
  return 0;
  BOB_CATCH_MEMBER("location could not be set", -1)
}

static auto kpOrientation = bob::extension::VariableDoc(
  "orientation",
  "float",
  "The orientation of the keypoint (in degree),  with read and write access"
);
PyObject* PyBobIpBaseGSSKeypoint_getOrientation(PyBobIpBaseGSSKeypointObject* self, void*){
  BOB_TRY
  double o = self->cxx->orientation * 180. / M_PI;
  return Py_BuildValue("d", o);
  BOB_CATCH_MEMBER("orientation could not be read", 0)
}
int PyBobIpBaseGSSKeypoint_setOrientation(PyBobIpBaseGSSKeypointObject* self, PyObject* value, void*){
  BOB_TRY
  double o = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->sigma = o * M_PI / 180.;
  return 0;
  BOB_CATCH_MEMBER("orientation could not be set", -1)
}

static PyGetSetDef PyBobIpBaseGSSKeypoint_getseters[] = {
    {
      kpSigma.name(),
      (getter)PyBobIpBaseGSSKeypoint_getSigma,
      (setter)PyBobIpBaseGSSKeypoint_setSigma,
      kpSigma.doc(),
      0
    },
    {
      kpLocation.name(),
      (getter)PyBobIpBaseGSSKeypoint_getLocation,
      (setter)PyBobIpBaseGSSKeypoint_setLocation,
      kpLocation.doc(),
      0
    },
    {
      kpOrientation.name(),
      (getter)PyBobIpBaseGSSKeypoint_getOrientation,
      (setter)PyBobIpBaseGSSKeypoint_setOrientation,
      kpOrientation.doc(),
      0
    },
    {0}  /* Sentinel */
};


/******************************************************************/
/************ GSSKeypointInfo Section *****************************/
/******************************************************************/

PyTypeObject PyBobIpBaseGSSKeypointInfo_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

static auto GSSKeypointInfo_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".GSSKeypointInfo",
  "This is a companion structure to the :py:class:`bob.ip.base.GSSKeypoint`",
  "It provides additional and practical information such as the octave and scale indices, the integer location ``location = (y,x)``, and eventually the scores associated to the detection step (``peak_score`` and ``edge_score``)"
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a GSS keypoint",
    0,
    true
  )
  .add_prototype("[octave_index], [scale_index], [location], [peak_score], [edge_score]", "")
  .add_parameter("octave_index", "int", "[default: 0] The octave index associated with the keypoint in the :py:class:`bob.ip.base.GaussianScaleSpace` object")
  .add_parameter("scale_index", "int", "[default: 0] The scale index associated with the keypoint in the :py:class:`bob.ip.base.GaussianScaleSpace` object")
  .add_parameter("location", "(int, int)", "[default: ``(0, 0)``] The integer unnormalized location (y,x) of the keypoint")
  .add_parameter("peak_score", "float", "[default: 0] The orientation of the keypoint (in degrees)")
  .add_parameter("edge_score", "float", "[default: 0] The orientation of the keypoint (in degrees)")
);

static int PyBobIpBaseGSSKeypointInfo_init(PyBobIpBaseGSSKeypointInfoObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = GSSKeypointInfo_doc.kwlist();

  int octave = 0, scale = 0, y = 0, x = 0;
  double peak = 0., edge = 0.;

  if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "|ii(ii)dd", kwlist, &octave, &scale, &y, &x, &peak, &edge))) return 0;

  self->cxx.reset(new bob::ip::base::GSSKeypointInfo());
  self->cxx->o = octave;
  self->cxx->s = scale;
  self->cxx->iy = y;
  self->cxx->ix = x;
  self->cxx->peak_score = peak;
  self->cxx->edge_score = edge;
  return 0;

  BOB_CATCH_MEMBER("cannot create GSSKeypointInfo", -1)
}

static void PyBobIpBaseGSSKeypointInfo_delete(PyBobIpBaseGSSKeypointInfoObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseGSSKeypointInfo_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseGSSKeypointInfo_Type));
}


static auto kpiOctaveIndex = bob::extension::VariableDoc(
  "octave_index",
  "int",
  "The octave index associated with the keypoint in the :py:class:`bob.ip.base.GaussianScaleSpace` object, with read and write access"
);
PyObject* PyBobIpBaseGSSKeypointInfo_getOctaveIndex(PyBobIpBaseGSSKeypointInfoObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->o);
  BOB_CATCH_MEMBER("octave_index could not be read", 0)
}
int PyBobIpBaseGSSKeypointInfo_setOctaveIndex(PyBobIpBaseGSSKeypointInfoObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, kpiOctaveIndex.name());
    return -1;
  }
  self->cxx->o = PyInt_AS_LONG(value);
  return 0;
  BOB_CATCH_MEMBER("octave_index could not be set", -1)
}

static auto kpiScaleIndex = bob::extension::VariableDoc(
  "scale_index",
  "int",
  "The scale index associated with the keypoint in the :py:class:`bob.ip.base.GaussianScaleSpace` object, with read and write access"
);
PyObject* PyBobIpBaseGSSKeypointInfo_getScaleIndex(PyBobIpBaseGSSKeypointInfoObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->s);
  BOB_CATCH_MEMBER("scale_index could not be read", 0)
}
int PyBobIpBaseGSSKeypointInfo_setScaleIndex(PyBobIpBaseGSSKeypointInfoObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, kpiScaleIndex.name());
    return -1;
  }
  self->cxx->o = PyInt_AS_LONG(value);
  return 0;
  BOB_CATCH_MEMBER("scale_index could not be set", -1)
}

static auto kpiLocation = bob::extension::VariableDoc(
  "location",
  "(int, int)",
  "The integer unnormalized location (y, x) of the keypoint, with read and write access"
);
PyObject* PyBobIpBaseGSSKeypointInfo_getLocation(PyBobIpBaseGSSKeypointInfoObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(ii)", self->cxx->iy, self->cxx->ix);
  BOB_CATCH_MEMBER("location could not be read", 0)
}
int PyBobIpBaseGSSKeypointInfo_setLocation(PyBobIpBaseGSSKeypointInfoObject* self, PyObject* value, void*){
  BOB_TRY
  int y, x;
  if (!PyArg_ParseTuple(value, "ii", &y, &x)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two ints", Py_TYPE(self)->tp_name, kpiLocation.name());
    return -1;
  }
  self->cxx->iy = y;
  self->cxx->ix = x;
  return 0;
  BOB_CATCH_MEMBER("location could not be set", -1)
}

static auto kpiPeakScore = bob::extension::VariableDoc(
  "peak_score",
  "float",
  "The peak score of the keypoint during the SIFT-like detection step, with read and write access"
);
PyObject* PyBobIpBaseGSSKeypointInfo_getPeakScore(PyBobIpBaseGSSKeypointInfoObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->peak_score);
  BOB_CATCH_MEMBER("peak_score could not be read", 0)
}
int PyBobIpBaseGSSKeypointInfo_setPeakScore(PyBobIpBaseGSSKeypointInfoObject* self, PyObject* value, void*){
  BOB_TRY
  double o = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->peak_score = o;
  return 0;
  BOB_CATCH_MEMBER("peak_score could not be set", -1)
}

static auto kpiEdgeScore = bob::extension::VariableDoc(
  "edge_score",
  "float",
  "The edge score of the keypoint during the SIFT-like detection step, with read and write access"
);
PyObject* PyBobIpBaseGSSKeypointInfo_getEdgeScore(PyBobIpBaseGSSKeypointInfoObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->edge_score);
  BOB_CATCH_MEMBER("edge_score could not be read", 0)
}
int PyBobIpBaseGSSKeypointInfo_setEdgeScore(PyBobIpBaseGSSKeypointInfoObject* self, PyObject* value, void*){
  BOB_TRY
  double o = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->edge_score = o;
  return 0;
  BOB_CATCH_MEMBER("edge_score could not be set", -1)
}

static PyGetSetDef PyBobIpBaseGSSKeypointInfo_getseters[] = {
    {
      kpiOctaveIndex.name(),
      (getter)PyBobIpBaseGSSKeypointInfo_getOctaveIndex,
      (setter)PyBobIpBaseGSSKeypointInfo_setOctaveIndex,
      kpiOctaveIndex.doc(),
      0
    },
    {
      kpiScaleIndex.name(),
      (getter)PyBobIpBaseGSSKeypointInfo_getScaleIndex,
      (setter)PyBobIpBaseGSSKeypointInfo_setScaleIndex,
      kpiScaleIndex.doc(),
      0
    },
    {
      kpiLocation.name(),
      (getter)PyBobIpBaseGSSKeypointInfo_getLocation,
      (setter)PyBobIpBaseGSSKeypointInfo_setLocation,
      kpiLocation.doc(),
      0
    },
    {
      kpiPeakScore.name(),
      (getter)PyBobIpBaseGSSKeypointInfo_getPeakScore,
      (setter)PyBobIpBaseGSSKeypointInfo_setPeakScore,
      kpiPeakScore.doc(),
      0
    },
    {
      kpiEdgeScore.name(),
      (getter)PyBobIpBaseGSSKeypointInfo_getEdgeScore,
      (setter)PyBobIpBaseGSSKeypointInfo_setEdgeScore,
      kpiEdgeScore.doc(),
      0
    },
    {0}  /* Sentinel */
};


/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto GaussianScaleSpace_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".GaussianScaleSpace",
  "This class allows after configuration the generation of Gaussian Pyramids that can be used to extract SIFT features",
  "For details, please read [Lowe2004]_."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new DCT features extractor",
    ".. todo:: Explain GaussianScaleSpace constructor in more detail.\n\n"
    ".. warning:: The order of the parameters ``scales`` and ``octaves`` has changed compared to the old implementation, in order to keep it consistent with :py:class:`bob.ip.base.VLSIFT`!",
    true
  )
  .add_prototype("size, scales, octaves, octave_min, [sigma_n], [sigma0], [kernel_radius_factor], [border]", "")
  .add_prototype("gss", "")
  .add_parameter("size", "(int, int)", "The height and width of the images to process")
  .add_parameter("scales", "int", "The number of intervals of the pyramid. Three additional scales will be computed in practice, as this is required for extracting SIFT features")
  .add_parameter("octaves", "int", "The number of octaves of the pyramid")
  .add_parameter("octave_min", "int", "The index of the minimum octave")
  .add_parameter("sigma_n", "float", "[default: 0.5] The value sigma_n of the standard deviation for the nominal/initial octave/scale")
  .add_parameter("sigma0", "float", "[default: 1.6] The value sigma0 of the standard deviation for the image of the first octave and first scale")
  .add_parameter("kernel_radius_factor", "float", "[default: 4.] Factor used to determine the kernel radii: ``size=2*radius+1``. For each Gaussian kernel, the radius is equal to ``ceil(kernel_radius_factor*sigma_{octave,scale})``")
  .add_parameter("border", ":py:class:`bob.sp.BorderType`", "[default: ``bob.sp.BorderType.Mirror``] The extrapolation method used by the convolution at the border")
  .add_parameter("gss", ":py:class:`bob.ip.base.GaussianScaleSpace`", "The GaussianScaleSpace object to use for copy-construction")
);


static int PyBobIpBaseGaussianScaleSpace_init(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = GaussianScaleSpace_doc.kwlist(0);
  char** kwlist2 = GaussianScaleSpace_doc.kwlist(1);;

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseGaussianScaleSpace_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseGaussianScaleSpaceObject* gss;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseGaussianScaleSpace_Type, &gss)) return -1;

    self->cxx.reset(new bob::ip::base::GaussianScaleSpace(*gss->cxx));
    return 0;
  }

  blitz::TinyVector<int,2> size;
  int scales,octaves, octave_min;
  double sigma_n = 0.5, sigma0 = 1.6, factor = 4.;
  bob::sp::Extrapolation::BorderType border = bob::sp::Extrapolation::Mirror;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)iii|dddO&", kwlist1, &size[0], &size[1], &scales, &octaves, &octave_min, &sigma_n, &sigma0, &factor, &PyBobSpExtrapolationBorder_Converter, &border)){
    GaussianScaleSpace_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::ip::base::GaussianScaleSpace(size[0], size[1], scales, octaves, octave_min, sigma_n, sigma0, factor, border));
  return 0;

  BOB_CATCH_MEMBER("cannot create GaussianScaleSpace", -1)
}

static void PyBobIpBaseGaussianScaleSpace_delete(PyBobIpBaseGaussianScaleSpaceObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseGaussianScaleSpace_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseGaussianScaleSpace_Type));
}

static PyObject* PyBobIpBaseGaussianScaleSpace_RichCompare(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseGaussianScaleSpace_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseGaussianScaleSpaceObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare GaussianScaleSpace objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto size = bob::extension::VariableDoc(
  "size",
  "(int, int)",
  "The shape of the images to process, with read and write access"
);
PyObject* PyBobIpBaseGaussianScaleSpace_getSize(PyBobIpBaseGaussianScaleSpaceObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(ii)", self->cxx->getHeight(), self->cxx->getWidth());
  BOB_CATCH_MEMBER("size could not be read", 0)
}
int PyBobIpBaseGaussianScaleSpace_setSize(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* value, void*){
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
PyObject* PyBobIpBaseGaussianScaleSpace_getOctaves(PyBobIpBaseGaussianScaleSpaceObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNOctaves());
  BOB_CATCH_MEMBER("octaves could not be read", 0)
}
int PyBobIpBaseGaussianScaleSpace_setOctaves(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* value, void*){
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
PyObject* PyBobIpBaseGaussianScaleSpace_getScales(PyBobIpBaseGaussianScaleSpaceObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNIntervals());
  BOB_CATCH_MEMBER("scales could not be read", 0)
}
int PyBobIpBaseGaussianScaleSpace_setScales(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* value, void*){
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
PyObject* PyBobIpBaseGaussianScaleSpace_getOctaveMin(PyBobIpBaseGaussianScaleSpaceObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getOctaveMin());
  BOB_CATCH_MEMBER("octave_min could not be read", 0)
}
int PyBobIpBaseGaussianScaleSpace_setOctaveMin(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* value, void*){
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
  "This is equal to octave_min+n_octaves-1."
);
PyObject* PyBobIpBaseGaussianScaleSpace_getOctaveMax(PyBobIpBaseGaussianScaleSpaceObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getOctaveMax());
  BOB_CATCH_MEMBER("octave_max could not be read", 0)
}

static auto sigmaN = bob::extension::VariableDoc(
  "sigma_n",
  "float",
  "The value sigma_n of the standard deviation for the nominal/initial octave/scale; with read and write access"
);
PyObject* PyBobIpBaseGaussianScaleSpace_getSigmaN(PyBobIpBaseGaussianScaleSpaceObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getSigmaN());
  BOB_CATCH_MEMBER("sigma_n could not be read", 0)
}
int PyBobIpBaseGaussianScaleSpace_setSigmaN(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* value, void*){
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
PyObject* PyBobIpBaseGaussianScaleSpace_getSigma0(PyBobIpBaseGaussianScaleSpaceObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getSigma0());
  BOB_CATCH_MEMBER("sigma_0 could not be read", 0)
}
int PyBobIpBaseGaussianScaleSpace_setSigma0(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setSigma0(d);
  return 0;
  BOB_CATCH_MEMBER("sigma_0 could not be set", -1)
}

static auto kernelRadiusFactor = bob::extension::VariableDoc(
  "kernel_radius_factor",
  "float",
  "Factor used to determine the kernel radii ``size=2*radius+1``",
  "For each Gaussian kernel, the radius is equal to ``ceil(kernel_radius_factor*sigma_{octave,scale})``"
);
PyObject* PyBobIpBaseGaussianScaleSpace_getKernelRadiusFactor(PyBobIpBaseGaussianScaleSpaceObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getKernelRadiusFactor());
  BOB_CATCH_MEMBER("kernel_radius_factor could not be read", 0)
}
int PyBobIpBaseGaussianScaleSpace_setKernelRadiusFactor(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* value, void*){
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
PyObject* PyBobIpBaseGaussianScaleSpace_getBorder(PyBobIpBaseGaussianScaleSpaceObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getConvBorder());
  BOB_CATCH_MEMBER("border could not be read", 0)
}
int PyBobIpBaseGaussianScaleSpace_setBorder(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* value, void*){
  BOB_TRY
  bob::sp::Extrapolation::BorderType b;
  if (!PyBobSpExtrapolationBorder_Converter(value, &b)) return -1;
  self->cxx->setConvBorder(b);
  return 0;
  BOB_CATCH_MEMBER("border could not be set", -1)
}


static PyGetSetDef PyBobIpBaseGaussianScaleSpace_getseters[] = {
    {
      size.name(),
      (getter)PyBobIpBaseGaussianScaleSpace_getSize,
      (setter)PyBobIpBaseGaussianScaleSpace_setSize,
      size.doc(),
      0
    },
    {
      octaves.name(),
      (getter)PyBobIpBaseGaussianScaleSpace_getOctaves,
      (setter)PyBobIpBaseGaussianScaleSpace_setOctaves,
      octaves.doc(),
      0
    },
    {
      scales.name(),
      (getter)PyBobIpBaseGaussianScaleSpace_getScales,
      (setter)PyBobIpBaseGaussianScaleSpace_setScales,
      scales.doc(),
      0
    },
    {
      octaveMin.name(),
      (getter)PyBobIpBaseGaussianScaleSpace_getOctaveMin,
      (setter)PyBobIpBaseGaussianScaleSpace_setOctaveMin,
      octaveMin.doc(),
      0
    },
    {
      octaveMax.name(),
      (getter)PyBobIpBaseGaussianScaleSpace_getOctaveMax,
      0,
      octaveMax.doc(),
      0
    },
    {
      sigmaN.name(),
      (getter)PyBobIpBaseGaussianScaleSpace_getSigmaN,
      (setter)PyBobIpBaseGaussianScaleSpace_setSigmaN,
      sigmaN.doc(),
      0
    },
    {
      sigma0.name(),
      (getter)PyBobIpBaseGaussianScaleSpace_getSigma0,
      (setter)PyBobIpBaseGaussianScaleSpace_setSigma0,
      sigma0.doc(),
      0
    },
    {
      kernelRadiusFactor.name(),
      (getter)PyBobIpBaseGaussianScaleSpace_getKernelRadiusFactor,
      (setter)PyBobIpBaseGaussianScaleSpace_setKernelRadiusFactor,
      kernelRadiusFactor.doc(),
      0
    },
    {
      border.name(),
      (getter)PyBobIpBaseGaussianScaleSpace_getBorder,
      (setter)PyBobIpBaseGaussianScaleSpace_setBorder,
      border.doc(),
      0
    },
    {0}  /* Sentinel */
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto getGaussian = bob::extension::FunctionDoc(
  "get_gaussian",
  "Returns the Gaussian at index/interval/scale i",
  0,
  true
)
.add_prototype("index", "gaussian")
.add_parameter("index", "int", "The index of the scale for which the Gaussian should be retrieved")
.add_return("gaussian", ":py:class:`bob.ip.base.Gaussian`", "The Gaussian at the given index")
;

static PyObject* PyBobIpBaseGaussianScaleSpace_getGaussian(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = getGaussian.kwlist();

  int index;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &index)) return 0;

  PyBobIpBaseGaussianObject* gaussian = (PyBobIpBaseGaussianObject*)PyBobIpBaseGaussian_Type.tp_alloc(&PyBobIpBaseGaussian_Type, 0);
  gaussian->cxx = self->cxx->getGaussian(index);
  return Py_BuildValue("N", gaussian);

  BOB_CATCH_MEMBER("cannot get Gaussian", 0)
}

static auto setNoSmooth = bob::extension::FunctionDoc(
  "set_sigma0_no_init_smoothing",
  "Sets sigma0 such that there is not smoothing at the first scale of octave_min",
  0,
  true
)
.add_prototype("");

static PyObject* PyBobIpBaseGaussianScaleSpace_setNoSmooth(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char* kwlist[] = {0};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwlist)) return 0;

  self->cxx->setSigma0NoInitSmoothing();
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("cannot call set_sigma0_no_init_smoothing", 0)
}

static auto allocateOutput = bob::extension::FunctionDoc(
  "allocate_output",
  "Allocates a python list of arrays for the Gaussian pyramid",
  0,
  true
)
.add_prototype("", "pyramid")
.add_return("pyramid", "[array_like(3D, float)]", "A list of output arrays in the size required to call :py:func`process`")
;

static PyObject* _allocate(PyBobIpBaseGaussianScaleSpaceObject* self){

  // get the number of octaves to process
  Py_ssize_t size = self->cxx->getOctaveMax()+1;
  PyObject* list = PyList_New(size);
  auto list_ = make_safe(list);

  for (Py_ssize_t i = 0; i < size; ++i){
    // allocate memory for the current octave in the desired size
    const blitz::TinyVector<int,3> shape = self->cxx->getOutputShape(i);
    Py_ssize_t o[] = {shape[0], shape[1], shape[2]};
    PyObject* array = PyBlitzArray_SimpleNew(NPY_FLOAT64, 3, o);
    PyList_SET_ITEM(list, i, PyBlitzArray_NUMPY_WRAP(array));
  }

  return Py_BuildValue("O", list);
}

static PyObject* PyBobIpBaseGaussianScaleSpace_allocateOutput(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char* kwlist[] = {0};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwlist)) return 0;

  return _allocate(self);

  BOB_CATCH_MEMBER("cannot allocate output", 0)
}

static auto process = bob::extension::FunctionDoc(
  "process",
  "Computes a Gaussian Pyramid for an input 2D image",
  "If given, the results are put in the output ``dst``, which output should already be allocated and of the correct size (using the :py:func:`allocate_output` method).\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("src, [dst]", "dst")
.add_parameter("src", "array_like (2D)", "The input image which should be processed")
.add_parameter("dst", "[array_like (3D, float)]", "The Gaussian pyramid that should have been allocated with :py:func:`allocate_output`")
.add_return("dst", "[array_like (3D, float)]", "The resulting Gaussian pyramid, if given it will be the same as the ``dst`` parameter")
;

template <typename T>
static void process_inner(PyBobIpBaseGaussianScaleSpaceObject* self, PyBlitzArrayObject* input, std::vector<blitz::Array<double,3>>& dst){
  self->cxx->process(*PyBlitzArrayCxx_AsBlitz<T,2>(input), dst);
}

static PyObject* PyBobIpBaseGaussianScaleSpace_process(PyBobIpBaseGaussianScaleSpaceObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = process.kwlist();

  PyBlitzArrayObject* src;
  PyObject* dst = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O!", kwlist, &PyBlitzArray_Converter, &src, &PyList_Type, &dst)) return 0;

  auto src_ = make_safe(src);
  auto dst_ = make_xsafe(dst);

  // perform checks on input and output image
  if (src->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays", Py_TYPE(self)->tp_name);
    return 0;
  }

  // check output
  Py_ssize_t size = self->cxx->getOctaveMax()+1;
  if (dst){
    if (PyList_Size(dst) != size){
      PyErr_Format(PyExc_TypeError, "`%s' The given output list needs to have %d elements, but has %d", Py_TYPE(self)->tp_name, (int)PyList_Size(dst),(int) size);
      return 0;
    }
  } else {
    // create output in desired shape
    dst = _allocate(self);
    dst_ = make_safe(dst);
  }

  // convert output to list of arrays
  std::vector<blitz::Array<double,3>> output(size);
  for (Py_ssize_t i = 0; i < size; ++i){
    // get array
    PyBlitzArrayObject* array = 0;
    if (!PyBlitzArray_OutputConverter(PyList_GET_ITEM(dst, i), &array)){
      PyErr_Format(PyExc_TypeError, "'%s' process cannot convert the given dst array at index %d in the list",  Py_TYPE(self)->tp_name, (int)i);
      return 0;
    }
    // check array
    auto array_ = make_safe(array);
    if (array->type_num != NPY_FLOAT64 || array->ndim != 3){
      PyErr_Format(PyExc_TypeError, "'%s' the dst arrays for the process function must be 3D and of type float, but in index %d it is not",  Py_TYPE(self)->tp_name, (int)i);
      return 0;
    }
    // reference array
    output[i].reference(*PyBlitzArrayCxx_AsBlitz<double,3>(array));
  }

  // finally, extract the features
  switch (src->type_num){
    case NPY_UINT8:   process_inner<uint8_t>(self, src, output); break;
    case NPY_UINT16:  process_inner<uint16_t>(self, src, output); break;
    case NPY_FLOAT64: process_inner<double>(self, src, output); break;
    default:
      process.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' processes only images of types uint8, uint16 or float, and not %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(src->type_num));
      return 0;
  }

  return Py_BuildValue("O", dst);

  BOB_CATCH_MEMBER("cannot process image", 0)
}


static PyMethodDef PyBobIpBaseGaussianScaleSpace_methods[] = {
  {
    getGaussian.name(),
    (PyCFunction)PyBobIpBaseGaussianScaleSpace_getGaussian,
    METH_VARARGS|METH_KEYWORDS,
    getGaussian.doc()
  },
  {
    setNoSmooth.name(),
    (PyCFunction)PyBobIpBaseGaussianScaleSpace_setNoSmooth,
    METH_VARARGS|METH_KEYWORDS,
    setNoSmooth.doc()
  },
  {
    allocateOutput.name(),
    (PyCFunction)PyBobIpBaseGaussianScaleSpace_allocateOutput,
    METH_VARARGS|METH_KEYWORDS,
    allocateOutput.doc()
  },
  {
    process.name(),
    (PyCFunction)PyBobIpBaseGaussianScaleSpace_process,
    METH_VARARGS|METH_KEYWORDS,
    process.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the GaussianScaleSpace type struct; will be initialized later
PyTypeObject PyBobIpBaseGaussianScaleSpace_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseGaussianScaleSpace(PyObject* module)
{
  // initialize GSSKeypoint
  PyBobIpBaseGSSKeypoint_Type.tp_name = GSSKeypoint_doc.name();
  PyBobIpBaseGSSKeypoint_Type.tp_basicsize = sizeof(PyBobIpBaseGSSKeypointObject);
  PyBobIpBaseGSSKeypoint_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseGSSKeypoint_Type.tp_doc = GSSKeypoint_doc.doc();
  PyBobIpBaseGSSKeypoint_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseGSSKeypoint_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseGSSKeypoint_init);
  PyBobIpBaseGSSKeypoint_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseGSSKeypoint_delete);
  PyBobIpBaseGSSKeypoint_Type.tp_getset = PyBobIpBaseGSSKeypoint_getseters;

  // initialize GSSKeypointInfo
  PyBobIpBaseGSSKeypointInfo_Type.tp_name = GSSKeypointInfo_doc.name();
  PyBobIpBaseGSSKeypointInfo_Type.tp_basicsize = sizeof(PyBobIpBaseGSSKeypointInfoObject);
  PyBobIpBaseGSSKeypointInfo_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseGSSKeypointInfo_Type.tp_doc = GSSKeypointInfo_doc.doc();
  PyBobIpBaseGSSKeypointInfo_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseGSSKeypointInfo_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseGSSKeypointInfo_init);
  PyBobIpBaseGSSKeypointInfo_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseGSSKeypointInfo_delete);
  PyBobIpBaseGSSKeypointInfo_Type.tp_getset = PyBobIpBaseGSSKeypointInfo_getseters;

  // initialize the type struct
  PyBobIpBaseGaussianScaleSpace_Type.tp_name = GaussianScaleSpace_doc.name();
  PyBobIpBaseGaussianScaleSpace_Type.tp_basicsize = sizeof(PyBobIpBaseGaussianScaleSpaceObject);
  PyBobIpBaseGaussianScaleSpace_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseGaussianScaleSpace_Type.tp_doc = GaussianScaleSpace_doc.doc();

  // set the functions
  PyBobIpBaseGaussianScaleSpace_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseGaussianScaleSpace_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseGaussianScaleSpace_init);
  PyBobIpBaseGaussianScaleSpace_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseGaussianScaleSpace_delete);
  PyBobIpBaseGaussianScaleSpace_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseGaussianScaleSpace_RichCompare);
  PyBobIpBaseGaussianScaleSpace_Type.tp_methods = PyBobIpBaseGaussianScaleSpace_methods;
  PyBobIpBaseGaussianScaleSpace_Type.tp_getset = PyBobIpBaseGaussianScaleSpace_getseters;
  PyBobIpBaseGaussianScaleSpace_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseGaussianScaleSpace_process);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseGSSKeypoint_Type) < 0) return false;
  if (PyType_Ready(&PyBobIpBaseGSSKeypointInfo_Type) < 0) return false;
  if (PyType_Ready(&PyBobIpBaseGaussianScaleSpace_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseGSSKeypoint_Type);
  if (PyModule_AddObject(module, "GSSKeypoint", (PyObject*)&PyBobIpBaseGSSKeypoint_Type) < 0) return false;
  Py_INCREF(&PyBobIpBaseGSSKeypointInfo_Type);
  if (PyModule_AddObject(module, "GSSKeypointInfo", (PyObject*)&PyBobIpBaseGSSKeypointInfo_Type) < 0) return false;
  Py_INCREF(&PyBobIpBaseGaussianScaleSpace_Type);
  return PyModule_AddObject(module, "GaussianScaleSpace", (PyObject*)&PyBobIpBaseGaussianScaleSpace_Type) >= 0;
}
