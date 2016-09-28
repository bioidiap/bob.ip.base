/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Wed Jul  2 14:38:18 CEST 2014
 *
 * @brief Binds the DCTFeatures class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

static inline bool t(PyObject* o){return o == 0 || PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns true if object is NULL */
static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns false if object is NULL */

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto DCTFeatures_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".DCTFeatures",
  "Objects of this class, after configuration, can extract DCT features.",
  "The DCT feature extraction is described in more detail in [Sanderson2002]_. "
  "This class also supports block normalization and DCT coefficient normalization."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new DCT features extractor",
    ".. todo:: Explain DCTFeatures constructor in more detail.",
    true
  )
  .add_prototype("coefficients, block_size, [block_overlap], [normalize_block], [normalize_dct], [square_pattern]", "")
  .add_prototype("dct_features", "")
  .add_parameter("coefficients", "int", "The number of DCT coefficients;\n\n.. note::\n\n  the real number of DCT coefficient returned by the extractor is ``coefficients-1`` when the block normalization is enabled by setting ``normalize_block=True`` (as the first coefficient is always 0 in this case)")
  .add_parameter("block_size", "(int, int)", "The size of the blocks, in which the image is decomposed")
  .add_parameter("block_overlap", "(int, int)", "[default: ``(0, 0)``] The overlap of the blocks")
  .add_parameter("normalize_block", "bool", "[default: ``False``] Normalize each block to zero mean and unit variance before extracting DCT coefficients? In this case, the first coefficient will always be zero and hence will not be returned")
  .add_parameter("normalize_dct", "bool", "[default: ``False``] Normalize DCT coefficients to zero mean and unit variance after the DCT extraction?")
  .add_parameter("square_pattern", "bool", "[default: False] Select, whether a zigzag pattern or a square pattern is used for the DCT extraction; for a square pattern, the number of DCT coefficients must be a square integer")
  .add_parameter("dct_features", ":py:class:`bob.ip.base.DCTFeatures`", "The DCTFeatures object to use for copy-construction")
);


static int PyBobIpBaseDCTFeatures_init(PyBobIpBaseDCTFeaturesObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = DCTFeatures_doc.kwlist(0);
  char** kwlist2 = DCTFeatures_doc.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseDCTFeatures_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseDCTFeaturesObject* dct;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseDCTFeatures_Type, &dct)) return -1;

    self->cxx.reset(new bob::ip::base::DCTFeatures(*dct->cxx));
    return 0;
  }

  int coefs;
  blitz::TinyVector<int,2> block_size, block_overlap(0,0);
  PyObject* norm_block = 0,* norm_dct = 0,* square = 0;

  if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "i(ii)|(ii)O!O!O!", kwlist1,
        &coefs, &block_size[0], &block_size[1], &block_overlap[0], &block_overlap[1],
        &PyBool_Type, &norm_block, &PyBool_Type, &norm_dct, &PyBool_Type, &square))
  ){
    DCTFeatures_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::ip::base::DCTFeatures(coefs, block_size[0], block_size[1], block_overlap[0], block_overlap[1], f(norm_block), f(norm_dct), f(square)));
  return 0;

  BOB_CATCH_MEMBER("cannot create DCTFeatures", -1)
}

static void PyBobIpBaseDCTFeatures_delete(PyBobIpBaseDCTFeaturesObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseDCTFeatures_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseDCTFeatures_Type));
}

static PyObject* PyBobIpBaseDCTFeatures_RichCompare(PyBobIpBaseDCTFeaturesObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseDCTFeatures_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseDCTFeaturesObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare DCTFeatures objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto coefficients = bob::extension::VariableDoc(
  "coefficients",
  "int",
  "The number of DCT coefficients, with read and write access",
  ".. note::\n\n  The real number of DCT coefficient returned by the extractor is ``coefficients-1`` when the block normalization is enabled (as the first coefficient is always 0 in this case)"
);
PyObject* PyBobIpBaseDCTFeatures_getCoefficients(PyBobIpBaseDCTFeaturesObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNDctCoefs());
  BOB_CATCH_MEMBER("coefficients could not be read", 0)
}
int PyBobIpBaseDCTFeatures_setCoefficients(PyBobIpBaseDCTFeaturesObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, coefficients.name());
    return -1;
  }
  self->cxx->setNDctCoefs(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("coefficients could not be set", -1)
}

static auto blockSize = bob::extension::VariableDoc(
  "block_size",
  "(int, int)",
  "The size of each block for the block decomposition, with read and write access"
);
PyObject* PyBobIpBaseDCTFeatures_getBlockSize(PyBobIpBaseDCTFeaturesObject* self, void*){
  BOB_TRY
  auto s = self->cxx->getBlockSize();
  return Py_BuildValue("(ii)", s[0], s[1]);
  BOB_CATCH_MEMBER("block_size could not be read", 0)
}
int PyBobIpBaseDCTFeatures_setBlockSize(PyBobIpBaseDCTFeaturesObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> s;
  if (!PyArg_ParseTuple(value, "ii", &s[0], &s[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, blockSize.name());
    return -1;
  }
  self->cxx->setBlockSize(s);
  return 0;
  BOB_CATCH_MEMBER("block_size could not be set", -1)
}

static auto blockOverlap = bob::extension::VariableDoc(
  "block_overlap",
  "(int, int)",
  "The block overlap in both vertical and horizontal direction of the Multi-Block-DCTFeatures extractor, with read and write access",
  ".. note::\n\n  The ``block_overlap`` must be smaller than the :py:attr:`block_size`."
);
PyObject* PyBobIpBaseDCTFeatures_getBlockOverlap(PyBobIpBaseDCTFeaturesObject* self, void*){
  BOB_TRY
  auto s = self->cxx->getBlockOverlap();
  return Py_BuildValue("(ii)", s[0], s[1]);
  BOB_CATCH_MEMBER("block_overlap could not be read", 0)
}
int PyBobIpBaseDCTFeatures_setBlockOverlap(PyBobIpBaseDCTFeaturesObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> s;
  if (!PyArg_ParseTuple(value, "ii", &s[0], &s[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, blockOverlap.name());
    return -1;
  }
  self->cxx->setBlockOverlap(s);
  return 0;
  BOB_CATCH_MEMBER("block_overlap could not be set", -1)
}

static auto normalizeBlock = bob::extension::VariableDoc(
  "normalize_block",
  "bool",
  "Normalize each block to zero mean and unit variance before extracting DCT coefficients (read and write access)",
  ".. note::\n\n  In case ``normalize_block`` is set to ``True`` the first coefficient will always be zero and, hence, will not be returned."
);
PyObject* PyBobIpBaseDCTFeatures_getNormalizeBlock(PyBobIpBaseDCTFeaturesObject* self, void*){
  BOB_TRY
  if (self->cxx->getNormalizeBlock()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("normalize_block could not be read", 0)
}
int PyBobIpBaseDCTFeatures_setNormalizeBlock(PyBobIpBaseDCTFeaturesObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, normalizeBlock.name());
    return -1;
  }
  self->cxx->setNormalizeBlock(r>0);
  return 0;
  BOB_CATCH_MEMBER("normalize_block could not be set", -1)
}

static auto normalizeDCT = bob::extension::VariableDoc(
  "normalize_dct",
  "bool",
  "Normalize DCT coefficients to zero mean and unit variance after the DCT extraction (read and write access)"
);
PyObject* PyBobIpBaseDCTFeatures_getNormalizeDCT(PyBobIpBaseDCTFeaturesObject* self, void*){
  BOB_TRY
  if (self->cxx->getNormalizeDct()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("normalize_dct could not be read", 0)
}
int PyBobIpBaseDCTFeatures_setNormalizeDCT(PyBobIpBaseDCTFeaturesObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, normalizeDCT.name());
    return -1;
  }
  self->cxx->setNormalizeDct(r>0);
  return 0;
  BOB_CATCH_MEMBER("normalize_dct could not be set", -1)
}

static auto squarePattern = bob::extension::VariableDoc(
  "square_pattern",
  "bool",
  "Tells whether a zigzag pattern or a square pattern is used for the DCT extraction (read and write access)?",
  ".. note::\n\n  For a square pattern, the number of DCT coefficients must be a square integer."
);
PyObject* PyBobIpBaseDCTFeatures_getSquarePattern(PyBobIpBaseDCTFeaturesObject* self, void*){
  BOB_TRY
  if (self->cxx->getSquarePattern()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("square_pattern could not be read", 0)
}
int PyBobIpBaseDCTFeatures_setSquarePattern(PyBobIpBaseDCTFeaturesObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, squarePattern.name());
    return -1;
  }
  self->cxx->setSquarePattern(r>0);
  return 0;
  BOB_CATCH_MEMBER("square_pattern could not be set", -1)
}

static auto normEpsilon = bob::extension::VariableDoc(
  "normalization_epsilon",
  "float",
  "The epsilon value to avoid division-by-zero when performing block or DCT coefficient normalization (read and write access)",
  "The default value for this epsilon is ``10 * sys.float_info.min``, and usually there is little necessity to change that."
);
PyObject* PyBobIpBaseDCTFeatures_getNormEpsilon(PyBobIpBaseDCTFeaturesObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getNormEpsilon());
  BOB_CATCH_MEMBER("normalization_epsilon could not be read", 0)
}
int PyBobIpBaseDCTFeatures_setNormEpsilon(PyBobIpBaseDCTFeaturesObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyFloat_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a float", Py_TYPE(self)->tp_name, normEpsilon.name());
    return -1;
  }
  self->cxx->setNormEpsilon(PyFloat_AS_DOUBLE(value));
  return 0;
  BOB_CATCH_MEMBER("normalization_epsilon could not be set", -1)
}



static PyGetSetDef PyBobIpBaseDCTFeatures_getseters[] = {
    {
      coefficients.name(),
      (getter)PyBobIpBaseDCTFeatures_getCoefficients,
      (setter)PyBobIpBaseDCTFeatures_setCoefficients,
      coefficients.doc(),
      0
    },
    {
      blockSize.name(),
      (getter)PyBobIpBaseDCTFeatures_getBlockSize,
      (setter)PyBobIpBaseDCTFeatures_setBlockSize,
      blockSize.doc(),
      0
    },
    {
      blockOverlap.name(),
      (getter)PyBobIpBaseDCTFeatures_getBlockOverlap,
      (setter)PyBobIpBaseDCTFeatures_setBlockOverlap,
      blockOverlap.doc(),
      0
    },
    {
      normalizeBlock.name(),
      (getter)PyBobIpBaseDCTFeatures_getNormalizeBlock,
      (setter)PyBobIpBaseDCTFeatures_setNormalizeBlock,
      normalizeBlock.doc(),
      0
    },
    {
      normalizeDCT.name(),
      (getter)PyBobIpBaseDCTFeatures_getNormalizeDCT,
      (setter)PyBobIpBaseDCTFeatures_setNormalizeDCT,
      normalizeDCT.doc(),
      0
    },
    {
      squarePattern.name(),
      (getter)PyBobIpBaseDCTFeatures_getSquarePattern,
      (setter)PyBobIpBaseDCTFeatures_setSquarePattern,
      squarePattern.doc(),
      0
    },
    {
      normEpsilon.name(),
      (getter)PyBobIpBaseDCTFeatures_getNormEpsilon,
      (setter)PyBobIpBaseDCTFeatures_setNormEpsilon,
      normEpsilon.doc(),
      0
    },
    {0}  /* Sentinel */
};

/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto outputShape = bob::extension::FunctionDoc(
  "output_shape",
  "This function returns the shape of the DCT output for the given input",
  "The blocks can be split into either a 2D array of shape ``(block_index, coefficients)`` by setting ``flat=True``, or into a 3D array of shape ``(block_index_y, block_index_x, coefficients)`` with ``flat=False``.",
  true
)
.add_prototype("input, [flat]", "dct_shape")
.add_prototype("shape, [flat]", "dct_shape")
.add_parameter("input", "array_like (2D)", "The input image for which DCT features should be extracted")
.add_parameter("shape", "(int, int)", "The shape of the input image for which DCT features should be extracted")
.add_parameter("flat", "bool", "[default: ``True``] The ``flat`` parameter is used to decide whether 2D (``flat = True``) or 3D (``flat = False``) output shape is generated")
.add_return("dct_shape", "(int, int) or (int, int, int)", "The shape of the DCT features image that is required in a call to :py:func:`extract`")
;

static PyObject* PyBobIpBaseDCTFeatures_outputShape(PyBobIpBaseDCTFeaturesObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = outputShape.kwlist(0);
  char** kwlist2 = outputShape.kwlist(1);

  blitz::TinyVector<int,2> shape;
  PyObject* flat = 0; // is_integral_image
  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (
    (kwargs && PyDict_Contains(kwargs, k)) ||
    (args && (PyTuple_Check(PyTuple_GetItem(args, 0)) || PyList_Check(PyTuple_GetItem(args, 0))))
  ){
    // by shape
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)|O!", kwlist2, &shape[0], &shape[1], &PyBool_Type, &flat)){
      outputShape.print_usage();
      return 0;
    }
  } else {
    // by image
    PyBlitzArrayObject* image = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O!", kwlist1, &PyBlitzArray_Converter, &image, &PyBool_Type, &flat)){
      outputShape.print_usage();
      return 0;
    }
    auto _ = make_safe(image);
    if (image->ndim != 2) {
      outputShape.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' only accepts 2-dimensional arrays (not %" PY_FORMAT_SIZE_T "dD arrays)", Py_TYPE(self)->tp_name, image->ndim);
      return 0;
    }
    shape[0] = image->shape[0];
    shape[1] = image->shape[1];
  }
  if (t(flat)){
    auto dct_shape = self->cxx->get2DOutputShape(shape);
    return Py_BuildValue("(ii)", dct_shape[0], dct_shape[1]);
  } else {
    auto dct_shape = self->cxx->get3DOutputShape(shape);
    return Py_BuildValue("(iii)", dct_shape[0], dct_shape[1], dct_shape[2]);
  }

  BOB_CATCH_MEMBER("cannot get DCT features output shape", 0)
}

static auto extract = bob::extension::FunctionDoc(
  "extract",
  "Extracts DCT features from either uint8, uint16 or double arrays",
  "The input array is a 2D array/grayscale image. "
  "The destination array, if given, should be a 2D or 3D array of type float64 and allocated with the correct dimensions (see :py:func:`output_shape`). "
  "If the destination array is not given (first version), it is generated in the required size. "
  "The blocks can be split into either a 2D array of shape ``(block_index, coefficients)`` by setting ``flat=True``, or into a 3D array of shape ``(block_index_y, block_index_x, coefficients)`` with ``flat=False``.\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("input, [flat]", "output")
.add_prototype("input, output")
.add_parameter("input", "array_like (2D)", "The input image for which DCT features should be extracted")
.add_parameter("flat", "bool", "[default: ``True``] The ``flat`` parameter is used to decide whether 2D (``flat = True``) or 3D (``flat = False``) output shape is generated")
.add_parameter("output", "array_like (2D, float)", "The output image that need to be of shape :py:func:`output_shape`")
.add_return("output", "array_like (2D, float)", "The resulting DCT features")
;

template <typename T>
static void extract_inner(PyBobIpBaseDCTFeaturesObject* self, PyBlitzArrayObject* input, PyBlitzArrayObject* output){
  if (output->ndim == 2)
    self->cxx->extract(*PyBlitzArrayCxx_AsBlitz<T,2>(input), *PyBlitzArrayCxx_AsBlitz<double,2>(output));
  else
    self->cxx->extract(*PyBlitzArrayCxx_AsBlitz<T,2>(input), *PyBlitzArrayCxx_AsBlitz<double,3>(output));
}

static PyObject* PyBobIpBaseDCTFeatures_extract(PyBobIpBaseDCTFeaturesObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist1 = extract.kwlist(0);
  char** kwlist2 = extract.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  PyBlitzArrayObject* input,* output = 0;
  PyObject* flat = 0;

  PyObject* k = Py_BuildValue("s", kwlist1[1]);
  auto k_ = make_safe(k);
  if (nargs == 1 || (nargs == 2 && (
     (args && PyTuple_Size(args) == 2 && PyBool_Check(PyTuple_GET_ITEM(args,1))) ||
     (kwargs && PyDict_Contains(kwargs, k))
  ))){
    // with flat
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O!", kwlist1, &PyBlitzArray_Converter, &input, &PyBool_Type, &flat)) return 0;
  } else if (nargs == 2){
    // with output
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&", kwlist2, &PyBlitzArray_Converter, &input, &PyBlitzArray_OutputConverter, &output)) return 0;
  } else {
    // at least one argument is required
    extract.print_usage();
    PyErr_Format(PyExc_TypeError, "`%s' extract called with an unsupported number of arguments", Py_TYPE(self)->tp_name);
    return 0;
  }

  auto input_ = make_safe(input), output_ = make_xsafe(output);

  // perform checks on input and output image
  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only extracts from 2D arrays", Py_TYPE(self)->tp_name);
    extract.print_usage();
    return 0;
  }

  bool return_out = false;
  if (output){
    if (output->ndim != 2 && output->ndim != 3){
      PyErr_Format(PyExc_TypeError, "`%s' only extracts to 2D or 3D arrays", Py_TYPE(self)->tp_name);
      extract.print_usage();
      return 0;
    }
    if (output->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "`%s' extract requires the output image to of type float64, not of type %s),", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(output->type_num));
      extract.print_usage();
      return 0;
    }
  } else {
    // create output in desired shape
    blitz::TinyVector<int,2> input_shape(input->shape[0], input->shape[1]);
    if (t(flat)){
      auto shape = self->cxx->get2DOutputShape(input_shape);
      Py_ssize_t osize[] = {shape[0], shape[1]};
      output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, 2, osize);
    } else {
      auto shape = self->cxx->get3DOutputShape(input_shape);
      Py_ssize_t osize[] = {shape[0], shape[1], shape[2]};
      output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, 3, osize);
    }
    output_ = make_safe(output);
    return_out = true;
  }

  // finally, extract the features
  switch (input->type_num){
    case NPY_UINT8:   extract_inner<uint8_t>(self, input, output); break;
    case NPY_UINT16:  extract_inner<uint16_t>(self, input, output); break;
    case NPY_FLOAT64: extract_inner<double>(self, input, output); break;
    default:
      extract.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' extracts only from images of types uint8, uint16 or float, and not from %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(input->type_num));
      return 0;
  }

  if (return_out){
    return PyBlitzArray_AsNumpyArray(output, 0);
  } else {
    Py_RETURN_NONE;
  }

  BOB_CATCH_MEMBER("cannot extract DCTFeatures from image", 0)
}

#if 0
// TODO load and write DCTFeatures objects
static auto load = bob::extension::FunctionDoc(
  "load",
  "Loads the parametrization of the DCTFeatures extractor from the given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file opened for reading")
;

static PyObject* PyBobIpBaseDCTFeatures_load(PyBobIpBaseDCTFeaturesObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  // get list of arguments
  char* kwlist[] = {c("hdf5"), NULL};
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
  "Saves the the parametrization of the DCTFeatures extractor to the given HDF5 file",
  ".. warning:: For the time being, the look-up-table is **not saved**. "
  "If you have set the :py:attr:`look_up_table` by hand, it is lost.",
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing")
;

static PyObject* PyBobIpBaseDCTFeatures_save(PyBobIpBaseDCTFeaturesObject* self, PyObject* args, PyObject* kwargs) {
  // get list of arguments
  char* kwlist[] = {c("hdf5"), NULL};
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
#endif

static PyMethodDef PyBobIpBaseDCTFeatures_methods[] = {
  {
    outputShape.name(),
    (PyCFunction)PyBobIpBaseDCTFeatures_outputShape,
    METH_VARARGS|METH_KEYWORDS,
    outputShape.doc()
  },
  {
    extract.name(),
    (PyCFunction)PyBobIpBaseDCTFeatures_extract,
    METH_VARARGS|METH_KEYWORDS,
    extract.doc()
  },
#if 0
  {
    load.name(),
    (PyCFunction)PyBobIpBaseDCTFeatures_load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    save.name(),
    (PyCFunction)PyBobIpBaseDCTFeatures_save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
#endif
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the DCTFeatures type struct; will be initialized later
PyTypeObject PyBobIpBaseDCTFeatures_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseDCTFeatures(PyObject* module)
{
  // initialize the type struct
  PyBobIpBaseDCTFeatures_Type.tp_name = DCTFeatures_doc.name();
  PyBobIpBaseDCTFeatures_Type.tp_basicsize = sizeof(PyBobIpBaseDCTFeaturesObject);
  PyBobIpBaseDCTFeatures_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseDCTFeatures_Type.tp_doc = DCTFeatures_doc.doc();

  // set the functions
  PyBobIpBaseDCTFeatures_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseDCTFeatures_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseDCTFeatures_init);
  PyBobIpBaseDCTFeatures_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseDCTFeatures_delete);
  PyBobIpBaseDCTFeatures_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseDCTFeatures_RichCompare);
  PyBobIpBaseDCTFeatures_Type.tp_methods = PyBobIpBaseDCTFeatures_methods;
  PyBobIpBaseDCTFeatures_Type.tp_getset = PyBobIpBaseDCTFeatures_getseters;
  PyBobIpBaseDCTFeatures_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseDCTFeatures_extract);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseDCTFeatures_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseDCTFeatures_Type);
  return PyModule_AddObject(module, "DCTFeatures", (PyObject*)&PyBobIpBaseDCTFeatures_Type) >= 0;
}
