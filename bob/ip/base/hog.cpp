/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Jun 30 19:58:28 CEST 2014
 *
 * @brief Binds the HOG class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns false if object is NULL */

/******************************************************************/
/************ Enumerations Section ********************************/
/******************************************************************/

auto GradientMagnitude_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".GradientMagnitude",
  "Gradient 'magnitude' used",
  "Possible values are:\n\n"
  "* ``Magnitude``: L2 magnitude over X and Y\n"
  "* ``MagnitudeSquare``: Square of the L2 magnitude\n"
  "* ``SqrtMagnitude``: Square root of the L2 magnitude"
);

static PyObject* createGradientMagnitude() {
  auto retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  auto entries = PyDict_New();
  if (!entries) return 0;
  auto entries_ = make_safe(entries);

  if (insert_item_string(retval, entries, "Magnitude", bob::ip::base::GradientMagnitudeType::Magnitude) < 0) return 0;
  if (insert_item_string(retval, entries, "MagnitudeSquare", bob::ip::base::GradientMagnitudeType::MagnitudeSquare) < 0) return 0;
  if (insert_item_string(retval, entries, "SqrtMagnitude", bob::ip::base::GradientMagnitudeType::SqrtMagnitude) < 0) return 0;
  if (PyDict_SetItemString(retval, "entries", entries) < 0) return 0;

  return Py_BuildValue("O", retval);
}

int PyBobIpBaseGradientMagnitude_Converter(PyObject* o, bob::ip::base::GradientMagnitudeType* b) {
  if (PyString_Check(o)){
    PyObject* dict = PyBobIpBaseGradientMagnitude_Type.tp_dict;
    if (!PyDict_Contains(dict, o)){
      PyErr_Format(PyExc_ValueError, "gradient magnitude type parameter must be set to one of the integer values defined in `%s'", PyBobIpBaseGradientMagnitude_Type.tp_name);
      return 0;
    }
    o = PyDict_GetItem(dict, o);
  }

  Py_ssize_t v = PyNumber_AsSsize_t(o, PyExc_OverflowError);
  if (v == -1 && PyErr_Occurred()) return 0;

  if (v >= 0 && v < bob::ip::base::GradientMagnitudeType::MagnitudeType_Count){
    *b = static_cast<bob::ip::base::GradientMagnitudeType>(v);
    return 1;
  }

  PyErr_Format(PyExc_ValueError, "gradient magnitude type parameter must be set to one of the str or int values defined in `%s'", PyBobIpBaseGradientMagnitude_Type.tp_name);
  return 0;
}

static int PyBobIpBaseGradientMagnitude_init(PyObject* self, PyObject*, PyObject*) {
  PyErr_Format(PyExc_NotImplementedError, "cannot initialize C++ enumeration bindings `%s' - use one of the class' attached attributes instead", Py_TYPE(self)->tp_name);
  return -1;
}

auto BlockNorm_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".BlockNorm",
  "Enumeration that defines the norm that is used for normalizing the descriptor blocks",
  "Possible values are:\n\n"
  "* ``L2``: Euclidean norm\n"
  "* ``L2Hys``: L2 norm with clipping of high values\n"
  "* ``L1``: L1 norm (Manhattan distance)\n"
  "* ``L1sqrt``: Square root of the L1 norm\n"
  "* ``Nonorm``: no norm used"
);

static PyObject* createBlockNorm() {
  auto retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  auto entries = PyDict_New();
  if (!entries) return 0;
  auto entries_ = make_safe(entries);

  if (insert_item_string(retval, entries, "L2", bob::ip::base::BlockNorm::L2) < 0) return 0;
  if (insert_item_string(retval, entries, "L2Hys", bob::ip::base::BlockNorm::L2Hys) < 0) return 0;
  if (insert_item_string(retval, entries, "L1", bob::ip::base::BlockNorm::L1) < 0) return 0;
  if (insert_item_string(retval, entries, "L1sqrt", bob::ip::base::BlockNorm::L1sqrt) < 0) return 0;
  if (insert_item_string(retval, entries, "Nonorm", bob::ip::base::BlockNorm::Nonorm) < 0) return 0;
  if (PyDict_SetItemString(retval, "entries", entries) < 0) return 0;

  return Py_BuildValue("O", retval);
}

int PyBobIpBaseBlockNorm_Converter(PyObject* o, bob::ip::base::BlockNorm* b) {
  if (PyString_Check(o)){
    PyObject* dict = PyBobIpBaseBlockNorm_Type.tp_dict;
    if (!PyDict_Contains(dict, o)){
      PyErr_Format(PyExc_ValueError, "block norm type parameter parameter must be set to one of the str or int values defined in `%s'", PyBobIpBaseBlockNorm_Type.tp_name);
      return 0;
    }
    o = PyDict_GetItem(dict, o);
  }

  Py_ssize_t v = PyNumber_AsSsize_t(o, PyExc_OverflowError);
  if (v == -1 && PyErr_Occurred()) return 0;

  if (v >= 0 && v < bob::ip::base::BlockNorm::BlockNorm_Count){
    *b = static_cast<bob::ip::base::BlockNorm>(v);
    return 1;
  }

  PyErr_Format(PyExc_ValueError, "block norm type parameter must be set to one of the str or int values defined in `%s'", PyBobIpBaseBlockNorm_Type.tp_name);
  return 0;
}

static int PyBobIpBaseBlockNorm_init(PyObject* self, PyObject*, PyObject*) {
  PyErr_Format(PyExc_NotImplementedError, "cannot initialize C++ enumeration bindings `%s' - use one of the class' attached attributes instead", Py_TYPE(self)->tp_name);
  return -1;
}

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto HOG_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".HOG",
  "Objects of this class, after configuration, can extract Histogram of Oriented Gradients (HOG) descriptors.",
  "This implementation relies on the article of [Dalal2005]_. "
  "A few remarks:\n\n"
  "* Only single channel inputs (a.k.a. grayscale) are considered. "
  "Therefore, it does not take the maximum gradient over several channels as proposed in the above article.\n"
  "* Gamma/Color normalization is not part of the descriptor computation. "
  "However, this can easily be done (using this library) before extracting the descriptors.\n"
  "* Gradients are computed using standard 1D centered gradient (except at the borders where the gradient is uncentered [-1 1]). "
  "This is the method which achieved best performance reported in the article. "
  "To avoid too many uncentered gradients to be used, the gradients are computed on the full image prior to the cell decomposition. "
  "This implies that extra-pixels at each boundary of the cell are contributing to the gradients, although these pixels are not located inside the cell.\n"
  "* R-HOG blocks (rectangular) normalization is supported, but not C-HOG blocks (circular).\n"
  "* Due to the similarity with the SIFT descriptors, this can also be used to extract dense-SIFT features.\n"
  "* The first bin of each histogram is always centered around 0. "
  "  This implies that the orientations are in ``[0-e,180-e]`` rather than [0,180], with ``e`` being half the angle size of a bin (same with [0,360])."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new HOG extractor",
    0,
    true
  )
  .add_prototype("image_size, [bins], [full_orientation], [cell_size], [cell_overlap], [block_size], [block_overlap]", "")
  .add_prototype("hog", "")
  .add_parameter("image_size", "(int, int)", "The size of the input image to process.")
  .add_parameter("bins", "int", "[default: 8] Dimensionality of a cell descriptor (i.e. the number of bins)")
  .add_parameter("full_orientation", "bool", "[default: ``False``] Whether the range ``[0,360]`` is used or only ``[0,180]``")
  .add_parameter("cell_size", "(int, int)", "[default: ``(4,4)``] The size of a cell.")
  .add_parameter("cell_overlap", "(int, int)", "[default: ``(0,0)``] The overlap between cells.")
  .add_parameter("block_size", "(int, int)", "[default: ``(4,4)``] The size of a block (in terms of cells).")
  .add_parameter("block_overlap", "(int, int)", "[default: ``(0,0)``] The overlap between blocks (in terms of cells).")
  .add_parameter("hog", ":py:class:`bob.ip.base.HOG`", "Another HOG object to copy")
);

static int PyBobIpBaseHOG_init(PyBobIpBaseHOGObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = HOG_doc.kwlist(0);
  char** kwlist2 = HOG_doc.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseHOG_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseHOGObject* hog;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseHOG_Type, &hog)) return -1;

    self->cxx.reset(new bob::ip::base::HOG(*hog->cxx));
    return 0;
  }

  blitz::TinyVector<int,2> image_size, cell_size(4,4), cell_overlap(0,0), block_size(4,4), block_overlap(0,0);
  int bins = 8;
  PyObject* full_orientation = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)|iO!(ii)(ii)(ii)(ii)", kwlist1, &image_size[0], &image_size[1], &bins, &PyBool_Type, &full_orientation, &cell_size[0], &cell_size[1], &cell_overlap[0], &cell_overlap[1], &block_size[0], &block_size[1], &block_overlap[0], &block_overlap[1])){
    HOG_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::ip::base::HOG(image_size[0], image_size[1], bins, f(full_orientation), cell_size[0], cell_size[1], cell_overlap[0], cell_overlap[1], block_size[0], block_size[1], block_overlap[0], block_overlap[1]));
  return 0;

  BOB_CATCH_MEMBER("cannot create HOG object", -1)
}

static void PyBobIpBaseHOG_delete(PyBobIpBaseHOGObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseHOG_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseHOG_Type));
}

static PyObject* PyBobIpBaseHOG_RichCompare(PyBobIpBaseHOGObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseHOG_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseHOGObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare HOG objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto imageSize = bob::extension::VariableDoc(
  "image_size",
  "(int, int)",
  "The size of the input image to process., with read and write access"
);
PyObject* PyBobIpBaseHOG_getImageSize(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(ii)", self->cxx->getHeight(), self->cxx->getWidth());
  BOB_CATCH_MEMBER("image_size could not be read", 0)
}
int PyBobIpBaseHOG_setImageSize(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> r;
  if (!PyArg_ParseTuple(value, "ii", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two ints", Py_TYPE(self)->tp_name, imageSize.name());
    return -1;
  }
  self->cxx->setSize(r[0], r[1]);
  return 0;
  BOB_CATCH_MEMBER("image_size could not be set", -1)
}

static auto magnitudeType = bob::extension::VariableDoc(
  "magnitude_type",
  ":py:class:`bob.ip.base.GradientMagnitude`",
  "Type of the magnitude to consider for the descriptors, with read and write access"
);
PyObject* PyBobIpBaseHOG_getMagnitudeType(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getGradientMagnitudeType());
  BOB_CATCH_MEMBER("magnitude_type could not be read", 0)
}
int PyBobIpBaseHOG_setMagnitudeType(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  bob::ip::base::GradientMagnitudeType b;
  if (!PyBobIpBaseGradientMagnitude_Converter(value, &b)) return -1;
  self->cxx->setGradientMagnitudeType(b);
  return 0;
  BOB_CATCH_MEMBER("magnitude_type could not be set", -1)
}

static auto bins = bob::extension::VariableDoc(
  "bins",
  "int",
  "Dimensionality of a cell descriptor (i.e. the number of bins), with read and write access"
);
PyObject* PyBobIpBaseHOG_getBins(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getCellDim());
  BOB_CATCH_MEMBER("bins could not be read", 0)
}
int PyBobIpBaseHOG_setBins(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, bins.name());
    return -1;
  }
  self->cxx->setCellDim(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("bins could not be set", -1)
}

static auto fullOrientation = bob::extension::VariableDoc(
  "full_orientation",
  "bool",
  "Whether the range [0,360] is used or not ([0,180] otherwise), with read and write access"
);
PyObject* PyBobIpBaseHOG_getFullOrientation(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  if (self->cxx->getFullOrientation()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("full_orientation could not be read", 0)
}
int PyBobIpBaseHOG_setFullOrientation(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, fullOrientation.name());
    return -1;
  }
  self->cxx->setFullOrientation(r>0);
  return 0;
  BOB_CATCH_MEMBER("full_orientation could not be set", -1)
}

static auto cellSize = bob::extension::VariableDoc(
  "cell_size",
  "(int, int)",
  "Size of a cell, with read and write access"
);
PyObject* PyBobIpBaseHOG_getCellSize(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(ii)", self->cxx->getCellHeight(), self->cxx->getCellWidth());
  BOB_CATCH_MEMBER("cell_size could not be read", 0)
}
int PyBobIpBaseHOG_setCellSize(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> r;
  if (!PyArg_ParseTuple(value, "ii", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two ints", Py_TYPE(self)->tp_name, cellSize.name());
    return -1;
  }
  self->cxx->setCellSize(r[0], r[1]);
  return 0;
  BOB_CATCH_MEMBER("cell_size could not be set", -1)
}

static auto cellOverlap = bob::extension::VariableDoc(
  "cell_overlap",
  "(int, int)",
  "Overlap between cells, with read and write access"
);
PyObject* PyBobIpBaseHOG_getCellOverlap(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(ii)", self->cxx->getCellOverlapHeight(), self->cxx->getCellOverlapWidth());
  BOB_CATCH_MEMBER("cell_overlap could not be read", 0)
}
int PyBobIpBaseHOG_setCellOverlap(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> r;
  if (!PyArg_ParseTuple(value, "ii", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two ints", Py_TYPE(self)->tp_name, cellOverlap.name());
    return -1;
  }
  self->cxx->setCellOverlap(r[0], r[1]);
  return 0;
  BOB_CATCH_MEMBER("cell_overlap could not be set", -1)
}

static auto blockSize = bob::extension::VariableDoc(
  "block_size",
  "(int, int)",
  "Size of a block (in terms of cells), with read and write access"
);
PyObject* PyBobIpBaseHOG_getBlockSize(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(ii)", self->cxx->getBlockHeight(), self->cxx->getBlockWidth());
  BOB_CATCH_MEMBER("block_size could not be read", 0)
}
int PyBobIpBaseHOG_setBlockSize(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> r;
  if (!PyArg_ParseTuple(value, "ii", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two ints", Py_TYPE(self)->tp_name, blockSize.name());
    return -1;
  }
  self->cxx->setBlockSize(r[0], r[1]);
  return 0;
  BOB_CATCH_MEMBER("block_size could not be set", -1)
}

static auto blockOverlap = bob::extension::VariableDoc(
  "block_overlap",
  "(int, int)",
  "Overlap between blocks (in terms of cells), with read and write access"
);
PyObject* PyBobIpBaseHOG_getBlockOverlap(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  return Py_BuildValue("(ii)", self->cxx->getBlockOverlapHeight(), self->cxx->getBlockOverlapWidth());
  BOB_CATCH_MEMBER("block_overlap could not be read", 0)
}
int PyBobIpBaseHOG_setBlockOverlap(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> r;
  if (!PyArg_ParseTuple(value, "ii", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two ints", Py_TYPE(self)->tp_name, blockOverlap.name());
    return -1;
  }
  self->cxx->setBlockOverlap(r[0], r[1]);
  return 0;
  BOB_CATCH_MEMBER("block_overlap could not be set", -1)
}

static auto blockNorm = bob::extension::VariableDoc(
  "block_norm",
  ":py:class:`bob.ip.base.BlockNorm`",
  "The type of norm used for normalizing blocks, with read and write access"
);
PyObject* PyBobIpBaseHOG_getBlockNorm(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getBlockNorm());
  BOB_CATCH_MEMBER("block_norm could not be read", 0)
}
int PyBobIpBaseHOG_setBlockNorm(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  bob::ip::base::BlockNorm b;
  if (!PyBobIpBaseBlockNorm_Converter(value, &b)) return -1;
  self->cxx->setBlockNorm(b);
  return 0;
  BOB_CATCH_MEMBER("block_norm could not be set", -1)
}

static auto blockNormEps = bob::extension::VariableDoc(
  "block_norm_eps",
  "float",
  "Epsilon value used to avoid division by zeros when normalizing the blocks, read and write access"
);
PyObject* PyBobIpBaseHOG_getBlockNormEps(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getBlockNormEps());
  BOB_CATCH_MEMBER("block_norm_eps could not be read", 0)
}
int PyBobIpBaseHOG_setBlockNormEps(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyFloat_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a float", Py_TYPE(self)->tp_name, blockNormEps.name());
    return -1;
  }
  self->cxx->setBlockNormEps(PyFloat_AS_DOUBLE(value));
  return 0;
  BOB_CATCH_MEMBER("block_norm_eps could not be set", -1)
}

static auto blockNormThreshold = bob::extension::VariableDoc(
  "block_norm_threshold",
  "float",
  "Threshold used to perform the clipping during the block normalization, with read and write access"
);
PyObject* PyBobIpBaseHOG_getBlockNormThreshold(PyBobIpBaseHOGObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getBlockNormThreshold());
  BOB_CATCH_MEMBER("block_norm_threshold could not be read", 0)
}
int PyBobIpBaseHOG_setBlockNormThreshold(PyBobIpBaseHOGObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setBlockNormThreshold(d);
  return 0;
  BOB_CATCH_MEMBER("block_norm_threshold could not be set", -1)
}

static PyGetSetDef PyBobIpBaseHOG_getseters[] = {
    {
      imageSize.name(),
      (getter)PyBobIpBaseHOG_getImageSize,
      (setter)PyBobIpBaseHOG_setImageSize,
      imageSize.doc(),
      0
    },
    {
      magnitudeType.name(),
      (getter)PyBobIpBaseHOG_getMagnitudeType,
      (setter)PyBobIpBaseHOG_setMagnitudeType,
      magnitudeType.doc(),
      0
    },
    {
      bins.name(),
      (getter)PyBobIpBaseHOG_getBins,
      (setter)PyBobIpBaseHOG_setBins,
      bins.doc(),
      0
    },
    {
      fullOrientation.name(),
      (getter)PyBobIpBaseHOG_getFullOrientation,
      (setter)PyBobIpBaseHOG_setFullOrientation,
      fullOrientation.doc(),
      0
    },
    {
      cellSize.name(),
      (getter)PyBobIpBaseHOG_getCellSize,
      (setter)PyBobIpBaseHOG_setCellSize,
      cellSize.doc(),
      0
    },
    {
      cellOverlap.name(),
      (getter)PyBobIpBaseHOG_getCellOverlap,
      (setter)PyBobIpBaseHOG_setCellOverlap,
      cellOverlap.doc(),
      0
    },
    {
      blockSize.name(),
      (getter)PyBobIpBaseHOG_getBlockSize,
      (setter)PyBobIpBaseHOG_setBlockSize,
      blockSize.doc(),
      0
    },
    {
      blockOverlap.name(),
      (getter)PyBobIpBaseHOG_getBlockOverlap,
      (setter)PyBobIpBaseHOG_setBlockOverlap,
      blockOverlap.doc(),
      0
    },
    {
      blockNorm.name(),
      (getter)PyBobIpBaseHOG_getBlockNorm,
      (setter)PyBobIpBaseHOG_setBlockNorm,
      blockNorm.doc(),
      0
    },
    {
      blockNormEps.name(),
      (getter)PyBobIpBaseHOG_getBlockNormEps,
      (setter)PyBobIpBaseHOG_setBlockNormEps,
      blockNormEps.doc(),
      0
    },
    {
      blockNormThreshold.name(),
      (getter)PyBobIpBaseHOG_getBlockNormThreshold,
      (setter)PyBobIpBaseHOG_setBlockNormThreshold,
      blockNormThreshold.doc(),
      0
    },
    {0}  /* Sentinel */
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto outputShape = bob::extension::FunctionDoc(
  "output_shape",
  "Gets the descriptor output size given the current parameters and size",
  "In detail, it returns (number of blocks along Y, number of blocks along X, number of bins)",
  true
)
.add_prototype("", "shape")
.add_return("shape", "(int, int, int)", "The shape of the output array required to call :py:func:`extract`")
;

static PyObject* PyBobIpBaseHOG_outputShape(PyBobIpBaseHOGObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char* kwlist[] = {0};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwlist)) return 0;

  auto shape = self->cxx->getOutputShape();
  return Py_BuildValue("(iii)", shape[0], shape[1], shape[2]);

  BOB_CATCH_MEMBER("cannot compute output shape", 0)
}

static auto disableBlockNorm = bob::extension::FunctionDoc(
  "disable_block_normalization",
  "Disable block normalization",
  "This is performed by setting parameters such that the cells are not further processed, i.e.:\n\n"
  "* :py:attr:`block_size` ``= (1, 1)``\n"
  "* :py:attr:`block_overlap` ``= (0, 0)``\n"
  "* :py:attr:`block_norm` ``=`` :py:attr:`bob.ip.base.BlockNorm.Nonorm`",
  true
)
.add_prototype("")
;

static PyObject* PyBobIpBaseHOG_disableBlockNorm(PyBobIpBaseHOGObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char* kwlist[] = {0};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwlist)) return 0;

  self->cxx->disableBlockNormalization();
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("cannot disable block normalization", 0)
}

static auto computeHistogram = bob::extension::FunctionDoc(
  "compute_histogram",
  "Computes an Histogram of Gradients for a given 'cell'",
  "The inputs are the gradient magnitudes and the orientations for each pixel of the cell",
  true
)
.add_prototype("magnitude, orientation, [histogram]", "histogram")
.add_parameter("magnitude", "array_like (2D, float)", "The input array with the gradient magnitudes")
.add_parameter("orientation", "array_like (2D, float)", "The input array with the orientations")
.add_parameter("histogram", "array_like (1D, float)", "[default = None] If given, the result will be written to this histogram; must be of size :py:attr:`bins`")
.add_return("histogram", "array_like (1D, float)", "The resulting histogram; same as input ``histogram``, if given")
;

static PyObject* PyBobIpBaseHOG_computeHistogram(PyBobIpBaseHOGObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = computeHistogram.kwlist();

  PyBlitzArrayObject* mag,* ori,* hist = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|O&", kwlist, &PyBlitzArray_Converter, &mag, &PyBlitzArray_Converter, &ori, &PyBlitzArray_OutputConverter, &hist)) return 0;

  auto mag_ = make_safe(mag), ori_ = make_safe(ori), hist_ = make_xsafe(hist);

  // perform checks on input
  if (mag->ndim != 2 || ori->ndim != 2 || mag->type_num != NPY_FLOAT64 || ori->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of type float", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (hist){
    // check that data type is correct and dimensions fit
    if (hist->ndim != 1 || hist->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "'%s' the 'hist' array must be 1D and of type float, not %dD and type %s", Py_TYPE(self)->tp_name, (int)hist->ndim, PyBlitzArray_TypenumAsString(hist->type_num));
      return 0;
    }
  } else {
    // create output in the desired dimensions
    Py_ssize_t n[] = {static_cast<Py_ssize_t>(self->cxx->getCellDim())};
    hist = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, n));
    hist_ = make_safe(hist);
  }

  // call the function
  self->cxx->computeHistogram(*PyBlitzArrayCxx_AsBlitz<double,2>(mag), *PyBlitzArrayCxx_AsBlitz<double,2>(ori), *PyBlitzArrayCxx_AsBlitz<double,1>(hist));

  // return the histogram
  return PyBlitzArray_AsNumpyArray(hist, 0);

  BOB_CATCH_MEMBER("cannot compute histogram", 0)
}

static auto extract = bob::extension::FunctionDoc(
  "extract",
  "Extract the HOG descriptors",
  "This extracts HOG descriptors from the input image. "
  "The output is 3D, the first two dimensions being the y- and x- indices of the block, and the last one the index of the bin (among the concatenated cell histograms for this block).\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("input, [output]", "output")
.add_parameter("input", "array_like (2D)", "The input image to extract HOG features from")
.add_parameter("output", "array_like (3D, float)", "[default: ``None``] If given, the container to extract the HOG features to; must be of size :py:func:`output_shape`")
.add_return("output", "array_like(2D, float)", "The resulting HOG features, same as parameter ``output``, if given")
;

template <typename T>
static PyObject* extract_inner(PyBobIpBaseHOGObject* self, PyBlitzArrayObject* input, PyBlitzArrayObject* output){
  blitz::Array<double,2> input_;
  if (typeid(T) == typeid(double))
    input_.reference(*PyBlitzArrayCxx_AsBlitz<double,2>(input));
  else
    input_.reference(bob::core::array::cast<double>(*PyBlitzArrayCxx_AsBlitz<T,2>(input)));
  self->cxx->extract(input_, *PyBlitzArrayCxx_AsBlitz<double,3>(output));
  return PyBlitzArray_AsNumpyArray(output, 0);
}

static PyObject* PyBobIpBaseHOG_extract(PyBobIpBaseHOGObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = extract.kwlist();

  PyBlitzArrayObject* input,* output = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O&", kwlist, &PyBlitzArray_Converter, &input, &PyBlitzArray_OutputConverter, &output)) return 0;

  auto input_ = make_safe(input), output_ = make_xsafe(output);

  // perform checks on input
  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (output){
    // check that data type is correct and dimensions fit
    if (output->ndim != 3 || output->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "'%s' the 'output' array must be 3D and of type float, not %dD and type %s", Py_TYPE(self)->tp_name, (int)output->ndim, PyBlitzArray_TypenumAsString(output->type_num));
      return 0;
    }
  } else {
    // create output in the desired dimensions
    auto shape = self->cxx->getOutputShape();
    Py_ssize_t n[] = {shape[0], shape[1], shape[2]};
    output = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_FLOAT64, 3, n));
    output_ = make_safe(output);
  }

  // finally, process the data
  switch (input->type_num){
    case NPY_UINT8:   return extract_inner<uint8_t>(self, input, output);
    case NPY_UINT16:  return extract_inner<uint16_t>(self, input, output);
    case NPY_FLOAT64: return extract_inner<double>(self, input, output);
    default:
      PyErr_Format(PyExc_TypeError, "`%s' input array of type %s are currently not supported", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(input->type_num));
      extract.print_usage();
      return 0;
  }

  BOB_CATCH_MEMBER("cannot extract HOG features", 0)
}

static PyMethodDef PyBobIpBaseHOG_methods[] = {
  {
    outputShape.name(),
    (PyCFunction)PyBobIpBaseHOG_outputShape,
    METH_VARARGS|METH_KEYWORDS,
    outputShape.doc()
  },
  {
    computeHistogram.name(),
    (PyCFunction)PyBobIpBaseHOG_computeHistogram,
    METH_VARARGS|METH_KEYWORDS,
    computeHistogram.doc()
  },
  {
    disableBlockNorm.name(),
    (PyCFunction)PyBobIpBaseHOG_disableBlockNorm,
    METH_VARARGS|METH_KEYWORDS,
    disableBlockNorm.doc()
  },
  {
    extract.name(),
    (PyCFunction)PyBobIpBaseHOG_extract,
    METH_VARARGS|METH_KEYWORDS,
    extract.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the type structs; will be initialized later
PyTypeObject PyBobIpBaseGradientMagnitude_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

PyTypeObject PyBobIpBaseBlockNorm_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

PyTypeObject PyBobIpBaseHOG_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseHOG(PyObject* module)
{

  // GradientMagnitude
  PyBobIpBaseGradientMagnitude_Type.tp_name = GradientMagnitude_doc.name();
  PyBobIpBaseGradientMagnitude_Type.tp_basicsize = sizeof(PyBobIpBaseGradientMagnitude_Type);
  PyBobIpBaseGradientMagnitude_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseGradientMagnitude_Type.tp_doc = GradientMagnitude_doc.doc();
  PyBobIpBaseGradientMagnitude_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseGradientMagnitude_init);
  PyBobIpBaseGradientMagnitude_Type.tp_dict = createGradientMagnitude();

  if (PyType_Ready(&PyBobIpBaseGradientMagnitude_Type) < 0) return false;
  Py_INCREF(&PyBobIpBaseGradientMagnitude_Type);
  if (PyModule_AddObject(module, "GradientMagnitude", (PyObject*)&PyBobIpBaseGradientMagnitude_Type) < 0) return false;

  // BlockNorm
  PyBobIpBaseBlockNorm_Type.tp_name = BlockNorm_doc.name();
  PyBobIpBaseBlockNorm_Type.tp_basicsize = sizeof(PyBobIpBaseBlockNorm_Type);
  PyBobIpBaseBlockNorm_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseBlockNorm_Type.tp_doc = BlockNorm_doc.doc();
  PyBobIpBaseBlockNorm_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseBlockNorm_init);
  PyBobIpBaseBlockNorm_Type.tp_dict = createBlockNorm();

  if (PyType_Ready(&PyBobIpBaseBlockNorm_Type) < 0) return false;
  Py_INCREF(&PyBobIpBaseBlockNorm_Type);
  if (PyModule_AddObject(module, "BlockNorm", (PyObject*)&PyBobIpBaseBlockNorm_Type) < 0) return false;

  // initialize the type struct
  PyBobIpBaseHOG_Type.tp_name = HOG_doc.name();
  PyBobIpBaseHOG_Type.tp_basicsize = sizeof(PyBobIpBaseHOGObject);
  PyBobIpBaseHOG_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseHOG_Type.tp_doc = HOG_doc.doc();

  // set the functions
  PyBobIpBaseHOG_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseHOG_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseHOG_init);
  PyBobIpBaseHOG_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseHOG_delete);
  PyBobIpBaseHOG_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseHOG_RichCompare);
  PyBobIpBaseHOG_Type.tp_methods = PyBobIpBaseHOG_methods;
  PyBobIpBaseHOG_Type.tp_getset = PyBobIpBaseHOG_getseters;
  PyBobIpBaseHOG_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseHOG_extract);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseHOG_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseHOG_Type);
  return PyModule_AddObject(module, "HOG", (PyObject*)&PyBobIpBaseHOG_Type) >= 0;
}
