/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Thu Jul 10 13:40:53 CEST 2014
 *
 * @brief Binds the GLCM class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"
#include <bob.ip.base/GLCM.h>

/******************************************************************/
/************ Enumerations Section ********************************/
/******************************************************************/

typedef enum {
  angular_second_moment,
  energy,
  variance,
  contrast,
  auto_correlation,
  correlation,
  correlation_m,
  inv_diff_mom,
  sum_avg,
  sum_var,
  sum_entropy,
  entropy,
  diff_var,
  diff_entropy,
  dissimilarity,
  homogeneity,
  cluster_prom,
  cluster_shade,
  max_prob,
  inf_meas_corr1,
  inf_meas_corr2,
  inv_diff,
  inv_diff_norm,
  inv_diff_mom_norm,
  GLCMProperty_Count
} GLCMProperty;


PyTypeObject PyBobIpBaseGLCMProperty_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

auto GLCMProperty_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".GLCMProperty",
  "Enumeration that defines the properties of GLCM, to be used in :py:func:`bob.ip.base.GLCM.properties_by_name`",
  "Possible values are:\n\n"
  "* ``'angular_second_moment'`` [1] / energy [6]\n"
  "* ``'energy'`` [4]\n"
  "* ``'variance'`` (sum of squares) [1]\n"
  "* ``'contrast'`` [1], [6]\n"
  "* ``'auto_correlation'`` [2]\n"
  "* ``'correlation'`` [1]\n"
  "* ``'correlation_matlab'`` as in MATLAB Image Processing Toolbox method graycoprops() [6]\n"
  "* ``'inverse_difference_moment'`` [1] = homogeneity [2], homop[5]\n"
  "* ``'sum_average'`` [1]\n"
  "* ``'sum_variance'`` [1]\n"
  "* ``'sum_entropy'`` [1]\n"
  "* ``'entropy'`` [1]\n"
  "* ``'difference_variance'`` [4]\n"
  "* ``'difference_entropy'`` [1]\n"
  "* ``'dissimilarity'`` [4]\n"
  "* ``'homogeneity'`` [6]\n"
  "* ``'cluster_prominence'`` [2]\n"
  "* ``'cluster_shade'`` [2]\n"
  "* ``'maximum_probability'`` [2]\n"
  "* ``'information_measure_of_correlation_1'`` [1]\n"
  "* ``'information_measure_of_correlation_2'`` [1]\n"
  "* ``'inverse_difference'`` (INV) is homom [3]\n"
  "* ``'inverse_difference_normalized'`` (INN) [3]\n"
  "* ``'inverse_difference_moment_normalized'`` [3]\n\n"
  "The references from above are as follows:\n\n"
  "* [1] R. M. Haralick, K. Shanmugam, I. Dinstein; Textural Features for Image classification, in IEEE Transactions on Systems, Man and Cybernetics, vol.SMC-3, No. 6, p. 610-621.\n"
  "* [2] L. Soh and C. Tsatsoulis; Texture Analysis of SAR Sea Ice Imagery Using Gray Level Co-Occurrence Matrices, IEEE Transactions on Geoscience and Remote Sensing, vol. 37, no. 2, March 1999.\n"
  "* [3] D A. Clausi, An analysis of co-occurrence texture statistics as a function of grey level quantization, Can. J. Remote Sensing, vol. 28, no.1, pp. 45-62, 2002.\n"
  "* [4] http://murphylab.web.cmu.edu/publications/boland/boland_node26.html\n"
  "* [5] http://www.mathworks.com/matlabcentral/fileexchange/22354-glcmfeatures4-m-vectorized-version-of-glcmfeatures1-m-with-code-changes\n"
  "* [6] http://www.mathworks.ch/ch/help/images/ref/graycoprops.html"
);

static PyObject* createGLCMProperty() {
  BOB_TRY
  auto retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  auto entries = PyDict_New();
  if (!entries) return 0;
  auto entries_ = make_safe(entries);

  if (insert_item_string(retval, entries, "angular_second_moment", GLCMProperty::angular_second_moment) < 0) return 0;
  if (insert_item_string(retval, entries, "energy", GLCMProperty::energy) < 0) return 0;
  if (insert_item_string(retval, entries, "variance", GLCMProperty::variance) < 0) return 0;
  if (insert_item_string(retval, entries, "contrast", GLCMProperty::contrast) < 0) return 0;
  if (insert_item_string(retval, entries, "auto_correlation", GLCMProperty::auto_correlation) < 0) return 0;
  if (insert_item_string(retval, entries, "correlation", GLCMProperty::correlation) < 0) return 0;
  if (insert_item_string(retval, entries, "correlation_matlab", GLCMProperty::correlation_m) < 0) return 0;
  if (insert_item_string(retval, entries, "inverse_difference_moment", GLCMProperty::inv_diff_mom) < 0) return 0;
  if (insert_item_string(retval, entries, "sum_average", GLCMProperty::sum_avg) < 0) return 0;
  if (insert_item_string(retval, entries, "sum_variance", GLCMProperty::sum_var) < 0) return 0;
  if (insert_item_string(retval, entries, "sum_entropy", GLCMProperty::sum_entropy) < 0) return 0;
  if (insert_item_string(retval, entries, "entropy", GLCMProperty::entropy) < 0) return 0;
  if (insert_item_string(retval, entries, "difference_variance", GLCMProperty::diff_var) < 0) return 0;
  if (insert_item_string(retval, entries, "difference_entropy", GLCMProperty::diff_entropy) < 0) return 0;
  if (insert_item_string(retval, entries, "dissimilarity", GLCMProperty::dissimilarity) < 0) return 0;
  if (insert_item_string(retval, entries, "homogeneity", GLCMProperty::homogeneity) < 0) return 0;
  if (insert_item_string(retval, entries, "cluster_prominence", GLCMProperty::cluster_prom) < 0) return 0;
  if (insert_item_string(retval, entries, "cluster_shade", GLCMProperty::cluster_shade) < 0) return 0;
  if (insert_item_string(retval, entries, "maximum_probability", GLCMProperty::max_prob) < 0) return 0;
  if (insert_item_string(retval, entries, "information_measure_of_correlation_1", GLCMProperty::inf_meas_corr1) < 0) return 0;
  if (insert_item_string(retval, entries, "information_measure_of_correlation_2", GLCMProperty::inf_meas_corr2) < 0) return 0;
  if (insert_item_string(retval, entries, "inverse_difference", GLCMProperty::inv_diff) < 0) return 0;
  if (insert_item_string(retval, entries, "inverse_difference_normalized", GLCMProperty::inv_diff_norm) < 0) return 0;
  if (insert_item_string(retval, entries, "inverse_difference_moment_normalized", GLCMProperty::inv_diff_mom_norm) < 0) return 0;
  if (PyDict_SetItemString(retval, "entries", entries) < 0) return 0;

  return Py_BuildValue("O", retval);
  BOB_CATCH_FUNCTION("create glmc", 0)
}

static int PyBobIpBaseGLCMProperty_Converter(PyObject* o, GLCMProperty* b) {
  BOB_TRY
  if (PyString_Check(o)){
    PyObject* dict = PyBobIpBaseGLCMProperty_Type.tp_dict;
    if (!PyDict_Contains(dict, o)){
      PyErr_Format(PyExc_ValueError, "GLCMProperty parameter must be set to one of the str or int values defined in `%s'", PyBobIpBaseGLCMProperty_Type.tp_name);
      return 0;
    }
    o = PyDict_GetItem(dict, o);
  }

  Py_ssize_t v = PyNumber_AsSsize_t(o, PyExc_OverflowError);
  if (v == -1 && PyErr_Occurred()) return 0;

  if (v >= 0 && v < GLCMProperty::GLCMProperty_Count){
    *b = static_cast<GLCMProperty>(v);
    return 1;
  }

  PyErr_Format(PyExc_ValueError, "block norm type parameter must be set to one of the str or int values defined in `%s'", PyBobIpBaseGLCMProperty_Type.tp_name);
  return 0;
  BOB_CATCH_FUNCTION("property converter", 0)
}

static int PyBobIpBaseGLCMProperty_init(PyObject* self, PyObject*, PyObject*) {
  PyErr_Format(PyExc_NotImplementedError, "cannot initialize C++ enumeration bindings `%s' - use one of the class' attached attributes instead", Py_TYPE(self)->tp_name);
  return -1;
}



/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto GLCM_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".GLCM",
  "Objects of this class, after configuration, can compute Grey-Level Co-occurence Matrix of an image",
  "This class allows to extract a Grey-Level Co-occurence Matrix (GLCM) [Haralick1973]_. "
  "A thorough tutorial about GLCM and the textural (so-called Haralick) properties that can be derived from it, can be found at: http://www.fp.ucalgary.ca/mhallbey/tutorial.htm. "
  "A MatLab implementation can be found at: http://www.mathworks.ch/ch/help/images/ref/graycomatrix.html"
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructor",
    0,
    true
  )
  .add_prototype("[levels], [min_level], [max_level], [dtype]","")
  .add_prototype("quantization_table","")
  .add_prototype("glcm", "")
  .add_parameter("dtype", ":py:class:`numpy.dtype`", "[default: ``numpy.uint8``] The data-type for the GLCM class")
  .add_parameter("glcm", ":py:class:`bob.ip.base.GLCM`", "The GLCM object to use for copy-construction")
);

static int PyBobIpBaseGLCM_init(PyBobIpBaseGLCMObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = GLCM_doc.kwlist(0);
  char** kwlist2 = GLCM_doc.kwlist(1);
  char** kwlist3 = GLCM_doc.kwlist(2);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  self->prop.reset(new bob::ip::base::GLCMProp());

  PyObject* k = Py_BuildValue("s", kwlist3[0]),* k1 = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k), k1_ = make_safe(k1);
  if (nargs == 1){
    if (((args && PyTuple_Size(args) == 1 && PyBobIpBaseGLCM_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
      // copy construct
      PyBobIpBaseGLCMObject* glcm;
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist3, &PyBobIpBaseGLCM_Type, &glcm)) return -1;

      self->type_num = glcm->type_num;
      switch (self->type_num){
        case NPY_UINT8: self->cxx.reset(new bob::ip::base::GLCM<uint8_t>(*reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(glcm->cxx.get()))); break;
        case NPY_UINT16: self->cxx.reset(new bob::ip::base::GLCM<uint16_t>(*reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(glcm->cxx.get()))); break;
        case NPY_FLOAT64: self->cxx.reset(new bob::ip::base::GLCM<double>(*reinterpret_cast<bob::ip::base::GLCM<double>*>(glcm->cxx.get()))); break;
        default:
          PyErr_Format(PyExc_TypeError, "`%s' can only be created from uint8, uint16 or float", Py_TYPE(self)->tp_name);
          return -1;
      }
      return 0;
    }
    if (((args && PyTuple_Size(args) == 1 && !PyInt_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k1)))){
      // from quantization table
      PyBlitzArrayObject* table;
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist2, &PyBlitzArray_Converter, &table)) return -1;
      auto table_ = make_safe(table);
      self->type_num = table->type_num;
      switch (self->type_num){
        case NPY_UINT8: self->cxx.reset(new bob::ip::base::GLCM<uint8_t>(*PyBlitzArrayCxx_AsBlitz<uint8_t, 1>(table))); break;
        case NPY_UINT16: self->cxx.reset(new bob::ip::base::GLCM<uint16_t>(*PyBlitzArrayCxx_AsBlitz<uint16_t, 1>(table))); break;
        case NPY_FLOAT64: self->cxx.reset(new bob::ip::base::GLCM<double>(*PyBlitzArrayCxx_AsBlitz<double, 1>(table))); break;
        default:
          PyErr_Format(PyExc_TypeError, "`%s' can only be created from quantization tables of type uint8, uint16 or float", Py_TYPE(self)->tp_name);
          return -1;
      }
      return 0;
    }
  }

  // extract arguments
  PyObject* levels = 0,* min = 0,* max = 0;
  self->type_num = NPY_UINT8;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOO&", kwlist1, &levels, &min, &max, &PyBlitzArray_TypenumConverter, &self->type_num)){
    GLCM_doc.print_usage();
    return -1;
  }

  if (self->type_num != NPY_UINT8 && self->type_num != NPY_UINT16 && self->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' dtype parameter can only be one of type uint8, uint16 or float", Py_TYPE(self)->tp_name);
    return -1;
  }

  if (self->type_num == NPY_FLOAT64 && (levels == 0 || min == 0 || max == 0)){
    PyErr_Format(PyExc_TypeError, "`%s' for dtype 'float' levels, min and max must be specified!", Py_TYPE(self)->tp_name);
    return -1;
  }

  if ((min == 0) != (max == 0)){
    PyErr_Format(PyExc_TypeError, "`%s' min_level and max_level can only be specified at the same time", Py_TYPE(self)->tp_name);
    return -1;
  }

  if (levels == 0){
    // default constructor
    switch (self->type_num){
      case NPY_UINT8: self->cxx.reset(new bob::ip::base::GLCM<uint8_t>()); break;
      case NPY_UINT16: self->cxx.reset(new bob::ip::base::GLCM<uint16_t>()); break;
      default: break;// already handled
    }
  } else if (min == 0){
    // constructor with levels
    switch (self->type_num){
      case NPY_UINT8: self->cxx.reset(new bob::ip::base::GLCM<uint8_t>(static_cast<uint8_t>(PyInt_AS_LONG(levels)))); break;
      case NPY_UINT16: self->cxx.reset(new bob::ip::base::GLCM<uint16_t>(static_cast<uint16_t>(PyInt_AS_LONG(levels)))); break;
      default: break; // already handled
    }
  } else {
    // constructor with all three parameters
    switch (self->type_num){
      case NPY_UINT8: self->cxx.reset(new bob::ip::base::GLCM<uint8_t>(static_cast<uint8_t>(PyInt_AS_LONG(levels)), static_cast<uint8_t>(PyInt_AS_LONG(min)), static_cast<uint8_t>(PyInt_AS_LONG(max)))); break;
      case NPY_UINT16: self->cxx.reset(new bob::ip::base::GLCM<uint16_t>(static_cast<uint16_t>(PyInt_AS_LONG(levels)), static_cast<uint16_t>(PyInt_AS_LONG(min)), static_cast<uint16_t>(PyInt_AS_LONG(max)))); break;
      case NPY_FLOAT64: self->cxx.reset(new bob::ip::base::GLCM<double>(static_cast<uint8_t>(PyFloat_AsDouble(levels)), static_cast<uint8_t>(PyFloat_AsDouble(min)), static_cast<uint8_t>(PyFloat_AsDouble(max)))); break;
      default: break; // already handled
    }
  }

  return PyErr_Occurred() ? -1 : 0;

  BOB_CATCH_MEMBER("cannot create GLCM", -1)
}

static void PyBobIpBaseGLCM_delete(PyBobIpBaseGLCMObject* self) {
  self->prop.reset();
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseGLCM_Check(PyObject* o) {
  return PyObject_TypeCheck(o, &PyBobIpBaseGLCM_Type);
}

static PyObject* PyBobIpBaseGLCM_RichCompare(PyBobIpBaseGLCMObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseGLCM_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto o = reinterpret_cast<PyBobIpBaseGLCMObject*>(other);
  switch (op) {
    case Py_EQ:
      if (self->type_num != o->type_num) Py_RETURN_FALSE;
      switch (self->type_num){
        case NPY_UINT8: if (*reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get()) == *reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(o->cxx.get())) Py_RETURN_TRUE; else Py_RETURN_FALSE;
        case NPY_UINT16: if (*reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get()) == *reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(o->cxx.get())) Py_RETURN_TRUE; else Py_RETURN_FALSE;
        case NPY_FLOAT64: if (*reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get()) == *reinterpret_cast<bob::ip::base::GLCM<double>*>(o->cxx.get())) Py_RETURN_TRUE; else Py_RETURN_FALSE;
        default: break; // not possible
      }
      break;
    case Py_NE:
      if (self->type_num != o->type_num) Py_RETURN_TRUE;
      switch (self->type_num){
        case NPY_UINT8: if (*reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get()) == *reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(o->cxx.get())) Py_RETURN_FALSE; else Py_RETURN_TRUE;
        case NPY_UINT16: if (*reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get()) == *reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(o->cxx.get())) Py_RETURN_FALSE; else Py_RETURN_TRUE;
        case NPY_FLOAT64: if (*reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get()) == *reinterpret_cast<bob::ip::base::GLCM<double>*>(o->cxx.get())) Py_RETURN_FALSE; else Py_RETURN_TRUE;
        default: break; // not possible
      }
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }

  PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
  return 0;

  BOB_CATCH_MEMBER("cannot compare GLCM objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto dtype = bob::extension::VariableDoc(
  "dtype",
  ":py:class:`numpy.dtype`",
  "The data type, which was used in the constructor",
  "Only images of this data type can be processed in the :py:func:`extract` function."
);
PyObject* PyBobIpBaseGLCM_getDtype(PyBobIpBaseGLCMObject* self, void*){
  BOB_TRY
  PyArray_Descr* dtype = PyArray_DescrNewFromType(self->type_num);
  auto dtype_ = make_safe(dtype);
  return Py_BuildValue("O", dtype);
  BOB_CATCH_MEMBER("dtype could not be read", 0)
}

static auto offset = bob::extension::VariableDoc(
  "offset",
  "array_like (2D, int)",
  "The offset specifying the column and row distance between pixel pairs",
  "The shape of this array is (num_offsets, 2), where num_offsets is the total number of offsets to be taken into account when computing GLCM."
);
PyObject* PyBobIpBaseGLCM_getOffset(PyBobIpBaseGLCMObject* self, void*){
  BOB_TRY
  switch (self->type_num){
    case NPY_UINT8: return PyBlitzArrayCxx_AsConstNumpy(reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->getOffset());
    case NPY_UINT16: return PyBlitzArrayCxx_AsConstNumpy(reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->getOffset());
    case NPY_FLOAT64: return PyBlitzArrayCxx_AsConstNumpy(reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->getOffset());
    default: return 0;
  }
  BOB_CATCH_MEMBER("offset could not be read", 0)
}
int PyBobIpBaseGLCM_setOffset(PyBobIpBaseGLCMObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* v;
  if (!PyBlitzArray_Converter(value, &v)) return 0;
  auto v_ = make_safe(v);
  blitz::Array<int32_t, 2>* array = PyBlitzArrayCxx_AsBlitz<int32_t, 2>(v, "offset");
  if (!array) return -1;
  switch (self->type_num){
    case NPY_UINT8: reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->setOffset(*array); break;
    case NPY_UINT16: reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->setOffset(*array); break;
    case NPY_FLOAT64: reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->setOffset(*array); break;
    default: return -1;
  }
  return 0;
  BOB_CATCH_MEMBER("offset could not be set", -1)
}

static auto quantizationTable = bob::extension::VariableDoc(
  "quantization_table",
  "array_like (1D)",
  "The thresholds of the quantization"
  "Each element corresponds to the lower boundary of the particular quantization level. "
  "E.g.. array([ 0,  5, 10]) means quantization in 3 levels. "
  "Input values in the range [0,4] will be quantized to level 0, input values in the range[5,9] will be quantized to level 1 and input values in the range [10-max_level] will be quantized to level 2."
);
PyObject* PyBobIpBaseGLCM_getQuantizationTable(PyBobIpBaseGLCMObject* self, void*){
  BOB_TRY
  switch (self->type_num){
    case NPY_UINT8: return PyBlitzArrayCxx_AsConstNumpy(reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->getQuantizationTable());
    case NPY_UINT16: return PyBlitzArrayCxx_AsConstNumpy(reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->getQuantizationTable());
    case NPY_FLOAT64: return PyBlitzArrayCxx_AsConstNumpy(reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->getQuantizationTable());
    default: return 0;
  }
  BOB_CATCH_MEMBER("quantization_table could not be read", 0)
}

static auto levels = bob::extension::VariableDoc(
  "levels",
  "int",
  "Specifies the number of gray-levels to use when scaling the gray values in the input image",
  "This is the number of the values in the first and second dimension in the GLCM matrix. "
  "The default is the total number of gray values permitted by the type of the input image."
);
PyObject* PyBobIpBaseGLCM_getLevels(PyBobIpBaseGLCMObject* self, void*){
  BOB_TRY
  switch (self->type_num){
    case NPY_UINT8: return Py_BuildValue("i", reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->getNumLevels());
    case NPY_UINT16: return Py_BuildValue("i", reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->getNumLevels());
    case NPY_FLOAT64: return Py_BuildValue("i", reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->getNumLevels());
    default: return 0;
  }
  BOB_CATCH_MEMBER("levels could not be read", 0)
}

static auto maxLevel = bob::extension::VariableDoc(
  "max_level",
  "int",
  "Gray values greater than or equal to this value are scaled to :py:attr:`levels`"
  " The default is the maximum gray-level permitted by the type of input image."
);
PyObject* PyBobIpBaseGLCM_getMaxLevel(PyBobIpBaseGLCMObject* self, void*){
  BOB_TRY
  switch (self->type_num){
    case NPY_UINT8: return Py_BuildValue("i", reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->getMaxLevel());
    case NPY_UINT16: return Py_BuildValue("i", reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->getMaxLevel());
    case NPY_FLOAT64: return Py_BuildValue("i", reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->getMaxLevel());
    default: return 0;
  }
  BOB_CATCH_MEMBER("max_level could not be read", 0)
}

static auto minLevel = bob::extension::VariableDoc(
  "min_level",
  "int",
  "Gray values smaller than or equal to this value are scaled to 0"
  "The default is the minimum gray-level permitted by the type of input image."
);
PyObject* PyBobIpBaseGLCM_getMinLevel(PyBobIpBaseGLCMObject* self, void*){
  BOB_TRY
  switch (self->type_num){
    case NPY_UINT8: return Py_BuildValue("i", reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->getMinLevel());
    case NPY_UINT16: return Py_BuildValue("i", reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->getMinLevel());
    case NPY_FLOAT64: return Py_BuildValue("i", reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->getMinLevel());
    default: return 0;
  }
  BOB_CATCH_MEMBER("min_level could not be read", 0)
}

static auto symmetric = bob::extension::VariableDoc(
  "symmetric",
  "bool",
  "Tells whether a zigzag pattern or a square pattern is used for the DCT extraction (read and write access)?",
  ".. note::\n\n  For a square pattern, the number of DCT coefficients must be a square integer."
);
PyObject* PyBobIpBaseGLCM_getSymmetric(PyBobIpBaseGLCMObject* self, void*){
  BOB_TRY
  switch (self->type_num){
    case NPY_UINT8: if (reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->getSymmetric()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case NPY_UINT16: if (reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->getSymmetric()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case NPY_FLOAT64: if (reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->getSymmetric()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    default: return 0;
  }
  BOB_CATCH_MEMBER("symmetric could not be read", 0)
}
int PyBobIpBaseGLCM_setSymmetric(PyBobIpBaseGLCMObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, symmetric.name());
    return -1;
  }
  switch (self->type_num){
    case NPY_UINT8: reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->setSymmetric(r>0); return 0;
    case NPY_UINT16: reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->setSymmetric(r>0); return 0;
    case NPY_FLOAT64: reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->setSymmetric(r>0); return 0;
    default: return -1;
  }

  BOB_CATCH_MEMBER("symmetric could not be set", -1)
}

static auto normalized = bob::extension::VariableDoc(
  "normalized",
  "bool",
  "Tells whether a zigzag pattern or a square pattern is used for the DCT extraction (read and write access)?",
  ".. note::\n\n  For a square pattern, the number of DCT coefficients must be a square integer."
);
PyObject* PyBobIpBaseGLCM_getNormalized(PyBobIpBaseGLCMObject* self, void*){
  BOB_TRY
  switch (self->type_num){
    case NPY_UINT8: if (reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->getNormalized()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case NPY_UINT16: if (reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->getNormalized()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case NPY_FLOAT64: if (reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->getNormalized()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    default: return 0;
  }
  BOB_CATCH_MEMBER("normalized could not be read", 0)
}
int PyBobIpBaseGLCM_setNormalized(PyBobIpBaseGLCMObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, normalized.name());
    return -1;
  }
  switch (self->type_num){
    case NPY_UINT8: reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->setNormalized(r>0); return 0;
    case NPY_UINT16: reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->setNormalized(r>0); return 0;
    case NPY_FLOAT64: reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->setNormalized(r>0); return 0;
    default: return -1;
  }

  BOB_CATCH_MEMBER("normalized could not be set", -1)
}

static PyGetSetDef PyBobIpBaseGLCM_getseters[] = {
    {
      dtype.name(),
      (getter)PyBobIpBaseGLCM_getDtype,
      0,
      dtype.doc(),
      0
    },
    {
      offset.name(),
      (getter)PyBobIpBaseGLCM_getOffset,
      (setter)PyBobIpBaseGLCM_setOffset,
      offset.doc(),
      0
    },
    {
      quantizationTable.name(),
      (getter)PyBobIpBaseGLCM_getQuantizationTable,
      0,
      quantizationTable.doc(),
      0
    },
    {
      levels.name(),
      (getter)PyBobIpBaseGLCM_getLevels,
      0,
      levels.doc(),
      0
    },
    {
      maxLevel.name(),
      (getter)PyBobIpBaseGLCM_getMaxLevel,
      0,
      maxLevel.doc(),
      0
    },
    {
      minLevel.name(),
      (getter)PyBobIpBaseGLCM_getMinLevel,
      0,
      minLevel.doc(),
      0
    },
    {
      symmetric.name(),
      (getter)PyBobIpBaseGLCM_getSymmetric,
      (setter)PyBobIpBaseGLCM_setSymmetric,
      symmetric.doc(),
      0
    },
    {
      normalized.name(),
      (getter)PyBobIpBaseGLCM_getNormalized,
      (setter)PyBobIpBaseGLCM_setNormalized,
      normalized.doc(),
      0
    },
    {0}  /* Sentinel */
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto outputShape = bob::extension::FunctionDoc(
  "output_shape",
  "Get the shape of the GLCM matrix goven the input image",
  "The shape has 3 dimensions: two for the number of gray levels, and one for the number of offsets",
  true
)
.add_prototype("", "shape")
.add_return("shape", "(int, int, int)", "The shape of the output array required to call :py:func:`extract`")
;

blitz::TinyVector<int,3> _getShape(PyBobIpBaseGLCMObject* self){
  blitz::TinyVector<int,3> shape;
  switch (self->type_num){
    case NPY_UINT8: shape = reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->getGLCMShape(); break;
    case NPY_UINT16: shape = reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->getGLCMShape(); break;
    case NPY_FLOAT64: shape = reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->getGLCMShape(); break;
    default: return 0;
  }
  return shape;
}

static PyObject* PyBobIpBaseGLCM_outputShape(PyBobIpBaseGLCMObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char* kwlist[] = {0};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwlist)) return 0;

  auto shape = _getShape(self);
  return Py_BuildValue("(iii)", shape[0], shape[1], shape[2]);

  BOB_CATCH_MEMBER("cannot compute output shape", 0)
}

static auto extract = bob::extension::FunctionDoc(
  "extract",
  "Extracts the GLCM matrix from the given input image",
  "If given, the output array should have the expected type (numpy.float64) and the size as defined by :py:func:`output_shape` .\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("input, [output]", "output")
.add_parameter("input", "array_like (2D)", "The input image to extract GLCM features from")
.add_parameter("output", "array_like (3D, float)", "[default: ``None``] If given, the output will be saved into this array; must be of the shape as :py:func:`output_shape`")
.add_return("output", "array_like (3D, float)", "The resulting output data, which is the same as the parameter ``output`` (if given)")
;

static PyObject* PyBobIpBaseGLCM_extract(PyBobIpBaseGLCMObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = extract.kwlist();

  PyBlitzArrayObject* input,* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O&", kwlist, &PyBlitzArray_Converter, &input, &PyBlitzArray_OutputConverter, &output)) return 0;

  auto input_ = make_safe(input), output_ = make_xsafe(output);

  // perform checks on input and output image
  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D or 3D arrays", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (input->type_num != self->type_num){
    PyErr_Format(PyExc_TypeError, "`%s' can only process images of type %s (see 'dtype' in constructor) and not %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num), PyBlitzArray_TypenumAsString(input->type_num));
    return 0;
  }

  if (output){
    if (output->ndim != 3 || output->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "`%s' 'output' must be 3D and of type float, not %dD and type %s", Py_TYPE(self)->tp_name, (int)output->ndim, PyBlitzArray_TypenumAsString(output->type_num));
      return 0;
    }
  } else {
    // create output in desired shape
    auto shape = _getShape(self);
    Py_ssize_t n[] = {shape[0], shape[1], shape[2]};
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, 3, n);
    output_ = make_safe(output);
  }

  // finally, extract the features
  switch (self->type_num){
    case NPY_UINT8: reinterpret_cast<bob::ip::base::GLCM<uint8_t>*>(self->cxx.get())->extract(*PyBlitzArrayCxx_AsBlitz<uint8_t,2>(input), *PyBlitzArrayCxx_AsBlitz<double,3>(output)); break;
    case NPY_UINT16: reinterpret_cast<bob::ip::base::GLCM<uint16_t>*>(self->cxx.get())->extract(*PyBlitzArrayCxx_AsBlitz<uint16_t,2>(input), *PyBlitzArrayCxx_AsBlitz<double,3>(output)); break;
    case NPY_FLOAT64: reinterpret_cast<bob::ip::base::GLCM<double>*>(self->cxx.get())->extract(*PyBlitzArrayCxx_AsBlitz<double,2>(input), *PyBlitzArrayCxx_AsBlitz<double,3>(output)); break;
    default: return 0;
  }

  return PyBlitzArray_AsNumpyArray(output, 0);

  BOB_CATCH_MEMBER("cannot extract GLCM matrix from image", 0)
}


static auto propertiesByName = bob::extension::FunctionDoc(
  "properties_by_name",
  "Query the properties of GLCM by specifying a name",
  "Returns a list of numpy.array of the queried properties. "
  "Please see the documentation of :py:class:`bob.ip.base.GLCMProperty` for details on the possible properties.",
  true
)
.add_prototype("glcm_matrix, prop_names", "prop_values")
.add_parameter("glcm_matrix", "array_like (3D, float)", "The result of the GLCM extraction")
.add_parameter("prop_names", "[:py:class:`bob.ip.base.GLCMProperty`]", "[default: ``None``] A list of GLCM properties; either by value (int) or by name (str)")
.add_return("prop_values", "[array_like (1D, float)]", "The GLCM properties for the given ``prop_names``")
;

static PyObject* PyBobIpBaseGLCM_propertiesByName(PyBobIpBaseGLCMObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = propertiesByName.kwlist();

  PyBlitzArrayObject* matrix;
  PyObject* list;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O!", kwlist, &PyBlitzArray_Converter, &matrix, &PyList_Type, &list)) return 0;

  auto matrix_ = make_safe(matrix);

  // perform checks on input and output image
  if (matrix->ndim != 3 || matrix->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only accepts 3D arrays of type float, and not %dD arrays of type %s", Py_TYPE(self)->tp_name, (int)matrix->ndim, PyBlitzArray_TypenumAsString(matrix->type_num));
    return 0;
  }

  auto m = PyBlitzArrayCxx_AsBlitz<double,3>(matrix);
  Py_ssize_t n[] = {self->prop->get_prop_shape(*m)[0]};

  Py_ssize_t len = PyList_Size(list);
  PyObject* result = PyList_New(len);
  auto result_ = make_safe(result);

  for (Py_ssize_t i = 0; i < len; ++i){
    GLCMProperty p;
    // try to convert element to
    if (!PyBobIpBaseGLCMProperty_Converter(PyList_GET_ITEM(list, i), &p)) return 0;
    // compute output
    auto values = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, n);
    auto values_ = make_safe(values);
    auto v = PyBlitzArrayCxx_AsBlitz<double,1>(values);
    switch (p){
      case GLCMProperty::angular_second_moment: self->prop->angular_second_moment(*m, *v); break;
      case GLCMProperty::energy: self->prop->energy(*m, *v); break;
      case GLCMProperty::variance: self->prop->variance(*m, *v); break;
      case GLCMProperty::contrast: self->prop->contrast(*m, *v); break;
      case GLCMProperty::auto_correlation: self->prop->auto_correlation(*m, *v); break;
      case GLCMProperty::correlation: self->prop->correlation(*m, *v); break;
      case GLCMProperty::correlation_m: self->prop->correlation_m(*m, *v); break;
      case GLCMProperty::inv_diff_mom: self->prop->inv_diff_mom(*m, *v); break;
      case GLCMProperty::sum_avg: self->prop->sum_avg(*m, *v); break;
      case GLCMProperty::sum_var: self->prop->sum_var(*m, *v); break;
      case GLCMProperty::sum_entropy: self->prop->sum_entropy(*m, *v); break;
      case GLCMProperty::entropy: self->prop->entropy(*m, *v); break;
      case GLCMProperty::diff_var: self->prop->diff_var(*m, *v); break;
      case GLCMProperty::diff_entropy: self->prop->diff_entropy(*m, *v); break;
      case GLCMProperty::dissimilarity: self->prop->dissimilarity(*m, *v); break;
      case GLCMProperty::homogeneity: self->prop->homogeneity(*m, *v); break;
      case GLCMProperty::cluster_prom: self->prop->cluster_prom(*m, *v); break;
      case GLCMProperty::cluster_shade: self->prop->cluster_shade(*m, *v); break;
      case GLCMProperty::max_prob: self->prop->max_prob(*m, *v); break;
      case GLCMProperty::inf_meas_corr1: self->prop->inf_meas_corr1(*m, *v); break;
      case GLCMProperty::inf_meas_corr2: self->prop->inf_meas_corr2(*m, *v); break;
      case GLCMProperty::inv_diff: self->prop->inv_diff(*m, *v); break;
      case GLCMProperty::inv_diff_norm: self->prop->inv_diff_norm(*m, *v); break;
      case GLCMProperty::inv_diff_mom_norm: self->prop->inv_diff_mom_norm(*m, *v); break;
      case GLCMProperty::GLCMProperty_Count: return 0; // cannot happen
    }
    // set list item
    PyList_SET_ITEM(result, i, Py_BuildValue("N", PyBlitzArray_AsNumpyArray(values, 0)));
  }

  return Py_BuildValue("O", result);

  BOB_CATCH_MEMBER("cannot extract GLCM matrix from image", 0)
}


static PyMethodDef PyBobIpBaseGLCM_methods[] = {
  {
    outputShape.name(),
    (PyCFunction)PyBobIpBaseGLCM_outputShape,
    METH_VARARGS|METH_KEYWORDS,
    outputShape.doc()
  },
  {
    extract.name(),
    (PyCFunction)PyBobIpBaseGLCM_extract,
    METH_VARARGS|METH_KEYWORDS,
    extract.doc()
  },
  {
    propertiesByName.name(),
    (PyCFunction)PyBobIpBaseGLCM_propertiesByName,
    METH_VARARGS|METH_KEYWORDS,
    propertiesByName.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the GLCM type struct; will be initialized later
PyTypeObject PyBobIpBaseGLCM_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseGLCM(PyObject* module)
{
  // GLCMProperty
  PyBobIpBaseGLCMProperty_Type.tp_name = GLCMProperty_doc.name();
  PyBobIpBaseGLCMProperty_Type.tp_basicsize = sizeof(PyBobIpBaseGLCMProperty_Type);
  PyBobIpBaseGLCMProperty_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseGLCMProperty_Type.tp_doc = GLCMProperty_doc.doc();
  PyBobIpBaseGLCMProperty_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseGLCMProperty_init);
  PyBobIpBaseGLCMProperty_Type.tp_dict = createGLCMProperty();

  if (PyType_Ready(&PyBobIpBaseGLCMProperty_Type) < 0) return false;
  Py_INCREF(&PyBobIpBaseGLCMProperty_Type);
  if (PyModule_AddObject(module, "GLCMProperty", (PyObject*)&PyBobIpBaseGLCMProperty_Type) < 0) return false;

  // initialize the type struct
  PyBobIpBaseGLCM_Type.tp_name = GLCM_doc.name();
  PyBobIpBaseGLCM_Type.tp_basicsize = sizeof(PyBobIpBaseGLCMObject);
  PyBobIpBaseGLCM_Type.tp_flags = Py_TPFLAGS_DEFAULT |  Py_TPFLAGS_BASETYPE;
  PyBobIpBaseGLCM_Type.tp_doc = GLCM_doc.doc();

  // set the functions
  PyBobIpBaseGLCM_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseGLCM_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseGLCM_init);
  PyBobIpBaseGLCM_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseGLCM_delete);
  PyBobIpBaseGLCM_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseGLCM_RichCompare);
  PyBobIpBaseGLCM_Type.tp_methods = PyBobIpBaseGLCM_methods;
  PyBobIpBaseGLCM_Type.tp_getset = PyBobIpBaseGLCM_getseters;
  PyBobIpBaseGLCM_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseGLCM_extract);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseGLCM_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseGLCM_Type);
  // here we actually bind the _GLCM class, which will be sub-typed in python later on (I cannot set attributes in C++ classes directly)
  return PyModule_AddObject(module, "_GLCM", (PyObject*)&PyBobIpBaseGLCM_Type) >= 0;
}
