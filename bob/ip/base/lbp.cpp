/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Tue Jun 24 14:03:17 CEST 2014
 *
 * @brief Binds the LBP class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns false if object is NULL */

// ELBP type conversion
static const std::map<std::string, bob::ip::base::ELBPType> E = {{"regular",  bob::ip::base::ELBP_REGULAR}, {"transitional", bob::ip::base::ELBP_TRANSITIONAL}, {"direction-coded", bob::ip::base::ELBP_DIRECTION_CODED}};
static inline bob::ip::base::ELBPType e(const std::string& o){            /* converts string to ELBP type */
  auto it = E.find(o);
  if (it == E.end()) throw std::runtime_error("The given LBP type '" + o + "' is not known; choose one of ('regular', 'transitional', 'direction-coded')");
  else return it->second;
}
static inline const std::string& e(bob::ip::base::ELBPType o){            /* converts ELBP type to string */
  for (auto it = E.begin(); it != E.end(); ++it) if (it->second == o) return it->first;
  throw std::runtime_error("The given LBP type is not known");
}

// Border handling
static const std::map<std::string, bob::ip::base::LBPBorderHandling> B = {{"shrink",  bob::ip::base::LBP_BORDER_SHRINK}, {"wrap", bob::ip::base::LBP_BORDER_WRAP}};
static inline bob::ip::base::LBPBorderHandling b(const std::string& o){  /* converts string to border handling */
  auto it = B.find(o);
  if (it == B.end()) throw std::runtime_error("The given border handling '" + o + "' is not known; choose one of ('shrink', 'wrap')");
  else return it->second;
}
static inline const std::string& b(bob::ip::base::LBPBorderHandling o){            /* converts border handling to string */
  for (auto it = B.begin(); it != B.end(); ++it) if (it->second == o) return it->first;
  throw std::runtime_error("The given border handling is not known");
}

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto LBP_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".LBP",
  "A class that extracts local binary patterns in various types",
  "The implementation is based on [Atanasoaei2012]_, where all the different types of LBP features are defined in more detail."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates an LBP extractor with the given parametrization",
    "Basically, the LBP configuration can be split into three parts.\n\n"
    "1. Which pixels are compared how:\n\n"
    "   * The number of neighbors (might be 4, 8 or 16)\n"
    "   * Circular or rectangular offset positions around the center, or even Multi-Block LBP (MB-LBP)\n"
    "   * Compare the pixels to the center pixel or to the average\n\n"
    "2. How to generate the bit strings from the pixels (this is handled by the ``elbp_type`` parameter):\n\n"
    "   * ``'regular'``: Choose one bit for each comparison of the neighboring pixel with the central pixel\n"
    "   * ``'transitional'``: Compare only the neighboring pixels and skip the central one\n"
    "   * ``'direction-coded'``: Compute a 2-bit code for four directions\n\n"
    "3. How to cluster the generated bit strings to compute the final LBP code:\n\n"
    "   * ``uniform``: Only uniform LBP codes (with less than two bit-changes between 0 and 1) are considered; all other strings are combined into one LBP code\n"
    "   * ``rotation_invariant``: Rotation invariant LBP codes are generated, e.g., bit strings ``00110000`` and ``00000110`` will lead to the same LBP code\n\n"
    "This clustering is done using a look-up-table, which you can also set yourself using the :py:attr:`look_up_table` attribute. "
    "The maximum code that will be generated can be read from the :py:attr:`max_label` attribute.\n\n"
    "Finally, the border handling of the image can be selected. "
    "With the ``'shrink'`` option, no LBP code is computed for the border pixels and the resulting image is :math:`2\\times` ``radius`` or :math:`3\\times` ``block_size`` :math:`-1` pixels smaller in both directions, see :py:func:`lbp_shape`. "
    "The ``'wrap'`` option will wrap around the border and no truncation is performed.\n\n"
    ".. note::\n\n  To compute MB-LBP features, it is possible to compute an integral image before to speed up the calculation.",
    true
  )
  .add_prototype("neighbors, [radius], [circular], [to_average], [add_average_bit], [uniform], [rotation_invariant], [elbp_type], [border_handling]", "")
  .add_prototype("neighbors, radius_y, radius_x, [circular], [to_average], [add_average_bit], [uniform], [rotation_invariant], [elbp_type], [border_handling]", "")
  .add_prototype("neighbors, block_size, [block_overlap], [to_average], [add_average_bit], [uniform], [rotation_invariant], [elbp_type], [border_handling]", "")
  .add_prototype("lbp", "")
  .add_prototype("hdf5", "")
  .add_parameter("neighbors", "int", "The number of neighboring pixels that should be taken into account; possible values: 4, 8, 16")
  .add_parameter("radius", "float", "[default: 1.] The radius of the LBP in both vertical and horizontal direction together")
  .add_parameter("radius_y, radius_x", "float", "The radius of the LBP in both vertical and horizontal direction separately")
  .add_parameter("block_size", "(int, int)", "If set, multi-block LBP's with the given block size will be extracted")
  .add_parameter("block_overlap", "(int, int)", "[default: ``(0, 0)``] Multi-block LBP's with the given block overlap will be extracted")
  .add_parameter("circular", "bool", "[default: ``False``] Extract neighbors on a circle or on a square?")
  .add_parameter("to_average", "bool", "[default: ``False``] Compare the neighbors to the average of the pixels instead of the central pixel?")
  .add_parameter("add_average_bit", "bool", "[default: False] (only useful if to_average is True) Add another bit to compare the central pixel to the average of the pixels?")
  .add_parameter("uniform", "bool", "[default: ``False``] Extract uniform LBP features?")
  .add_parameter("rotation_invariant", "bool", "[default: ``False``] Extract rotation invariant LBP features?")
  .add_parameter("elbp_type", "str", "[default: ``'regular'``] Which type of LBP codes should be computed; possible values: ('regular', 'transitional', 'direction-coded'), see :py:attr:`elbp_type`")
  .add_parameter("border_handling", "str", "[default: ``'shrink'``] How should the borders of the image be treated; possible values: ('shrink', 'wrap'), see :py:attr:`border_handling`")
  .add_parameter("lbp", ":py:class:`bob.ip.base.LBP`", "Another LBP object to copy")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file to read the LBP configuration from")
);


static int PyBobIpBaseLBP_init(PyBobIpBaseLBPObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = LBP_doc.kwlist(0);
  char** kwlist2 = LBP_doc.kwlist(1);
  char** kwlist3 = LBP_doc.kwlist(2);
  char** kwlist4 = LBP_doc.kwlist(3);
  char** kwlist5 = LBP_doc.kwlist(4);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  int neighbors;
  double radius = 1., r_y, r_x;
  blitz::TinyVector<int,2> block_size, block_overlap(0,0);
  PyObject* circular = 0,* to_average = 0,* add_average_bit = 0,* uniform = 0,* rotation_invariant = 0;
  const char* elbp_type = "regular",* border_handling = "shrink";

  if (!nargs){
    // at least one argument is required
    LBP_doc.print_usage();
    PyErr_Format(PyExc_TypeError, "`%s' constructor requires at least one parameter", Py_TYPE(self)->tp_name);
    return -1;
  } // nargs == 0

  if (nargs == 1){
    // three different possible ways to call
    PyObject* k4 = Py_BuildValue("s", kwlist4[0]),* k5 = Py_BuildValue("s", kwlist5[0]);
    auto k4_ = make_safe(k4), k5_ = make_safe(k5);
    if (
      (kwargs && PyDict_Contains(kwargs, k5)) ||
      (args && PyTuple_Size(args) && PyBobIoHDF5File_Check(PyTuple_GetItem(args, 0)))
    ){
      // create from HDF5 file
      PyBobIoHDF5FileObject* hdf5;
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist5, &PyBobIoHDF5File_Converter, &hdf5)){
        LBP_doc.print_usage();
        return -1;
      }
      auto hdf5_ = make_safe(hdf5);
      self->cxx.reset(new bob::ip::base::LBP(*hdf5->f));
      return 0;
    } else if (
      (kwargs && PyDict_Contains(kwargs, k4)) ||
      (args && PyTuple_Size(args) && PyBobIpBaseLBP_Check(PyTuple_GetItem(args, 0)))
    ){
      // copy constructor
      PyBobIpBaseLBPObject* lbp;
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist4, &PyBobIpBaseLBP_Type, &lbp)){
        LBP_doc.print_usage();
        return -1;
      }
      self->cxx.reset(new bob::ip::base::LBP(*lbp->cxx));
      return 0;
    } else {
      // first variant with default radius
      char* kwlist_t[] = {kwlist1[0], NULL};
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist_t, &neighbors)){
        LBP_doc.print_usage();
        return -1;
      }
      self->cxx.reset(new bob::ip::base::LBP(neighbors, radius, f(circular), f(to_average), f(add_average_bit), f(uniform), f(rotation_invariant), e(elbp_type), b(border_handling)));
      return 0;
    }
  } // nargs == 1

  // more than one parameter; check the second one
  PyObject* k2 =  Py_BuildValue("s", kwlist2[2]),* k3 =  Py_BuildValue("s", kwlist3[1]);
  auto k2_ = make_safe(k2), k3_ = make_safe(k3);
  if (
    (kwargs && PyDict_Contains(kwargs, k3)) ||
    (args && PyTuple_Size(args) > 1 && PySequence_Check(PyTuple_GetItem(args, 1)))
  ){
    // MB-LBP
    if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "i(ii)|(ii)O!O!O!O!ss", kwlist3,
          &neighbors, &block_size[0], &block_size[1], &block_overlap[0], &block_overlap[1],
          &PyBool_Type, &to_average, &PyBool_Type, &add_average_bit, &PyBool_Type, &uniform, &PyBool_Type, &rotation_invariant,
          &elbp_type, &border_handling))
    ){
      LBP_doc.print_usage();
      return -1;
    }
    self->cxx.reset(new bob::ip::base::LBP(neighbors, block_size, block_overlap, f(to_average), f(add_average_bit), f(uniform), f(rotation_invariant), e(elbp_type), b(border_handling)));
    return 0;
  } else {
    // check if one or two radii are given
    if (
      (kwargs && PyDict_Contains(kwargs, k2)) ||
      (args && PyTuple_Size(args) > 2 && PyFloat_CheckExact(PyTuple_GetItem(args, 2)))
    ){
      // two radii
      if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "idd|O!O!O!O!O!ss", kwlist2,
            &neighbors, &r_y, &r_x,
            &PyBool_Type, &circular, &PyBool_Type, &to_average, &PyBool_Type, &add_average_bit, &PyBool_Type, &uniform, &PyBool_Type, &rotation_invariant,
            &elbp_type, &border_handling))
      ){
        LBP_doc.print_usage();
        return -1;
      }
      self->cxx.reset(new bob::ip::base::LBP(neighbors, r_y, r_x, f(circular), f(to_average), f(add_average_bit), f(uniform), f(rotation_invariant), e(elbp_type), b(border_handling)));
      return 0;
    } else {
      // only one radius
      if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "i|dO!O!O!O!O!ss", kwlist1,
            &neighbors, &radius,
            &PyBool_Type, &circular, &PyBool_Type, &to_average, &PyBool_Type, &add_average_bit, &PyBool_Type, &uniform, &PyBool_Type, &rotation_invariant,
            &elbp_type, &border_handling))
      ){
        LBP_doc.print_usage();
        return -1;
      }
      self->cxx.reset(new bob::ip::base::LBP(neighbors, radius, f(circular), f(to_average), f(add_average_bit), f(uniform), f(rotation_invariant), e(elbp_type), b(border_handling)));
      return 0;
    }
  }

  BOB_CATCH_MEMBER("cannot create LBP extractor", -1)
}

static void PyBobIpBaseLBP_delete(PyBobIpBaseLBPObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseLBP_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseLBP_Type));
}

int PyBobIpBaseLBP_Converter(PyObject* o, PyBobIpBaseLBPObject** a) {
  if (!PyBobIpBaseLBP_Check(o)) return 0;
  Py_INCREF(o);
  *a = reinterpret_cast<PyBobIpBaseLBPObject*>(o);
  return 1;
}



static PyObject* PyBobIpBaseLBP_RichCompare(PyBobIpBaseLBPObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseLBP_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseLBPObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare LBP objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto radius = bob::extension::VariableDoc(
  "radius",
  "float",
  "The radius of the round or square LBP extractor, with read and write access"
);
PyObject* PyBobIpBaseLBP_getRadius(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getRadius());
  BOB_CATCH_MEMBER("radius could not be read", 0)
}
int PyBobIpBaseLBP_setRadius(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  double d = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) return -1;
  self->cxx->setRadius(d);
  return 0;
  BOB_CATCH_MEMBER("radius could not be set", -1)
}

static auto radii = bob::extension::VariableDoc(
  "radii",
  "(float, float)",
  "The radii in both vertical and horizontal direction of the elliptical or rectangular LBP extractor, with read and write access"
);
PyObject* PyBobIpBaseLBP_getRadii(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  auto r = self->cxx->getRadii();
  return Py_BuildValue("(dd)", r[0], r[1]);
  BOB_CATCH_MEMBER("radii could not be read", 0)
}
int PyBobIpBaseLBP_setRadii(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<double,2> r;
  if (!PyArg_ParseTuple(value, "dd", &r[0], &r[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, radii.name());
    return -1;
  }
  self->cxx->setRadii(r);
  return 0;
  BOB_CATCH_MEMBER("radii could not be set", -1)
}

static auto blockSize = bob::extension::VariableDoc(
  "block_size",
  "(int, int)",
  "The block size in both vertical and horizontal direction of the Multi-Block-LBP extractor, with read and write access"
);
PyObject* PyBobIpBaseLBP_getBlockSize(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  auto s = self->cxx->getBlockSize();
  return Py_BuildValue("(ii)", s[0], s[1]);
  BOB_CATCH_MEMBER("block size could not be read", 0)
}
int PyBobIpBaseLBP_setBlockSize(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> s;
  if (!PyArg_ParseTuple(value, "ii", &s[0], &s[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, blockSize.name());
    return -1;
  }
  self->cxx->setBlockSize(s);
  return 0;
  BOB_CATCH_MEMBER("block size could not be set", -1)
}

static auto blockOverlap = bob::extension::VariableDoc(
  "block_overlap",
  "(int, int)",
  "The block overlap in both vertical and horizontal direction of the Multi-Block-LBP extractor, with read and write access",
  ".. note::\n\n  The ``block_overlap`` must be smaller than the :py:attr:`block_size`. "
  "To set both the block size and the block overlap at the same time, use the :py:func:`set_block_size_and_overlap` function."
);
PyObject* PyBobIpBaseLBP_getBlockOverlap(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  auto s = self->cxx->getBlockOverlap();
  return Py_BuildValue("(ii)", s[0], s[1]);
  BOB_CATCH_MEMBER("block overlap could not be read", 0)
}
int PyBobIpBaseLBP_setBlockOverlap(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  blitz::TinyVector<int,2> s;
  if (!PyArg_ParseTuple(value, "ii", &s[0], &s[1])){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a tuple of two floats", Py_TYPE(self)->tp_name, blockOverlap.name());
    return -1;
  }
  self->cxx->setBlockOverlap(s);
  return 0;
  BOB_CATCH_MEMBER("block overlap could not be set", -1)
}

static auto points = bob::extension::VariableDoc(
  "points",
  "int",
  "The number of neighbors (usually 4, 8 or 16), with read and write access",
  ".. note::\n\n  The ``block_overlap`` must be smaller than the :py:attr:`block_size`. "
  "To set both the block size and the block overlap at the same time, use the :py:func:`set_block_size_and_overlap` function."
);
PyObject* PyBobIpBaseLBP_getPoints(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNNeighbours());
  BOB_CATCH_MEMBER("points could not be read", 0)
}
int PyBobIpBaseLBP_setPoints(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, points.name());
    return -1;
  }
  self->cxx->setNNeighbours(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("points could not be set", -1)
}

static auto circular = bob::extension::VariableDoc(
  "circular",
  "bool",
  "Should circular or rectangular LBP's be extracted (read and write access)?"
);
PyObject* PyBobIpBaseLBP_getCircular(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  if (self->cxx->getCircular()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("circular could not be read", 0)
}
int PyBobIpBaseLBP_setCircular(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, circular.name());
    return -1;
  }
  self->cxx->setCircular(r>0);
  return 0;
  BOB_CATCH_MEMBER("circular could not be set", -1)
}

static auto toAverage = bob::extension::VariableDoc(
  "to_average",
  "bool",
  "Should the neighboring pixels be compared with the average of all pixels, or to the central one (read and write access)?"
);
PyObject* PyBobIpBaseLBP_getToAverage(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  if (self->cxx->getToAverage()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("to_average could not be read", 0)
}
int PyBobIpBaseLBP_setToAverage(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, toAverage.name());
    return -1;
  }
  self->cxx->setToAverage(r>0);
  return 0;
  BOB_CATCH_MEMBER("to_average could not be set", -1)
}

static auto addAverage = bob::extension::VariableDoc(
  "add_average_bit",
  "bool",
  "Should the bit for the comparison of the central pixel with the average be added as well (read and write access)?"
);
PyObject* PyBobIpBaseLBP_getAddAverage(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  if (self->cxx->getAddAverageBit()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("add_average_bit could not be read", 0)
}
int PyBobIpBaseLBP_setAddAverage(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, addAverage.name());
    return -1;
  }
  self->cxx->setAddAverageBit(r>0);
  return 0;
  BOB_CATCH_MEMBER("add_average_bit could not be set", -1)
}

static auto uniform = bob::extension::VariableDoc(
  "uniform",
  "bool",
  "Should uniform LBP patterns be extracted (read and write access)?",
  "Uniform LBP patterns are those bit strings, where only up to two changes from 0 to 1 and vice versa are allowed. "
  "Hence, ``00111000`` is a uniform pattern, while ``00110011`` is not. "
  "All non-uniform bit strings will be collected in a single LBP code."
);
PyObject* PyBobIpBaseLBP_getUniform(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  if (self->cxx->getUniform()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("uniform could not be read", 0)
}
int PyBobIpBaseLBP_setUniform(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, uniform.name());
    return -1;
  }
  self->cxx->setUniform(r>0);
  return 0;
  BOB_CATCH_MEMBER("uniform could not be set", -1)
}

static auto rotationInvariant = bob::extension::VariableDoc(
  "rotation_invariant",
  "bool",
  "Should rotation invariant LBP patterns be extracted (read and write access)?",
  "Rotation invariant LBP codes collects all patterns that have the same bit string with shifts. "
  "Hence, ``00111000`` and ``10000011`` will result in the same LBP code."
);
PyObject* PyBobIpBaseLBP_getRotationInvariant(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  if (self->cxx->getRotationInvariant()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("rotation_invariant could not be read", 0)
}
int PyBobIpBaseLBP_setRotationInvariant(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  int r = PyObject_IsTrue(value);
  if (r < 0){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a bool", Py_TYPE(self)->tp_name, rotationInvariant.name());
    return -1;
  }
  self->cxx->setRotationInvariant(r>0);
  return 0;
  BOB_CATCH_MEMBER("rotation_invariant could not be set", -1)
}

static auto elbpType = bob::extension::VariableDoc(
  "elbp_type",
  "str",
  "The type of LBP bit string that should be extracted (read and write access)",
  "Possible values are: ('regular', 'transitional', 'direction-coded')"
);
PyObject* PyBobIpBaseLBP_getELBPType(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  return Py_BuildValue("s", e(self->cxx->get_eLBP()).c_str());
  BOB_CATCH_MEMBER("elbp_type could not be read", 0)
}
int PyBobIpBaseLBP_setELBPType(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyString_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an str", Py_TYPE(self)->tp_name, elbpType.name());
    return -1;
  }
  self->cxx->set_eLBP(e(PyString_AS_STRING(value)));
  return 0;
  BOB_CATCH_MEMBER("elbp_type could not be set", -1)
}

static auto borderHandling = bob::extension::VariableDoc(
  "border_handling",
  "str",
  "The type of border handling that should be applied (read and write access)",
  "Possible values are: ('shrink', 'wrap')"
);
PyObject* PyBobIpBaseLBP_getBorderHandling(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  return Py_BuildValue("s", b(self->cxx->getBorderHandling()).c_str());
  BOB_CATCH_MEMBER("border_handling could not be read", 0)
}
int PyBobIpBaseLBP_setBorderHandling(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyString_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an str", Py_TYPE(self)->tp_name, borderHandling.name());
    return -1;
  }
  self->cxx->setBorderHandling(b(PyString_AS_STRING(value)));
  return 0;
  BOB_CATCH_MEMBER("border_handling could not be set", -1)
}

static auto lookUpTable = bob::extension::VariableDoc(
  "look_up_table",
  "array_like (1D, uint16)",
  "The look up table that defines, which bit string is converted into which LBP code (read and write access)",
  "Depending on the values of :py:attr:`uniform` and :py:attr:`rotation_invariant`, bit strings might be converted into different LBP codes. "
  "Since this attribute is writable, you can define a look-up-table for LBP codes yourself.\n\n"
  ".. warning:: For the time being, the look up tables are **not** saved by the :py:func:`save` function!"
);
PyObject* PyBobIpBaseLBP_getLUT(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsNumpy(self->cxx->getLookUpTable());
  BOB_CATCH_MEMBER("look_up_table could not be read", 0)
}
int PyBobIpBaseLBP_setLUT(PyBobIpBaseLBPObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* lut = 0;
  if (!PyBlitzArray_Converter(value, &lut)) return -1;
  auto lut_ = make_safe(lut);

  if (lut->type_num != NPY_UINT16 || lut->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports uint16 1D arrays for property %s", Py_TYPE(self)->tp_name, lookUpTable.name());
    return -1;
  }
  self->cxx->setLookUpTable(*PyBlitzArrayCxx_AsBlitz<uint16_t,1>(lut));
  return 0;
  BOB_CATCH_MEMBER("look_up_table could not be set", -1)
}

static auto maxLabel = bob::extension::VariableDoc(
  "max_label",
  "int",
  "The number of different LBP code that are extracted (read access only)",
  "The codes themselves are uint16 numbers in the range ``[0, max_label - 1]``. "
  "Depending on the values of :py:attr:`uniform` and :py:attr:`rotation_invariant`, bit strings might be converted into different LBP codes."
);
PyObject* PyBobIpBaseLBP_getMaxLabel(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getMaxLabel());
  BOB_CATCH_MEMBER("max_label could not be read", 0)
}

static auto relativePositions = bob::extension::VariableDoc(
  "relative_positions",
  "array_like (2D, float)",
  "The list of neighbor positions, with which the central pixel is compared (read access only)",
  "The list is defined as relative positions, where the central pixel is considered to be at ``(0, 0)``."
);
PyObject* PyBobIpBaseLBP_getRelativePositions(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getRelativePositions());
  BOB_CATCH_MEMBER("relative_positions could not be read", 0)
}

static auto offset = bob::extension::VariableDoc(
  "offset",
  "(int, int)",
  "The offset in the image, where the first LBP code can be extracted (read access only)",
  ".. note::\n\n  When extracting LBP features from an image with a specific ``shape``, positions might be in range ``[offset, shape - offset[`` only. "
  "Otherwise, an exception will be raised."
);
PyObject* PyBobIpBaseLBP_getOffset(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  auto o = self->cxx->getOffset();
  return Py_BuildValue("(ii)", o[0], o[1]);
  BOB_CATCH_MEMBER("offset could not be read", 0)
}

static auto isMulti = bob::extension::VariableDoc(
  "is_multi_block_lbp",
  "bool",
  "Is the current configuration of the LBP extractor set up to extract Multi-Block LBP's (read access only)?"
);
PyObject* PyBobIpBaseLBP_getIsMulti(PyBobIpBaseLBPObject* self, void*){
  BOB_TRY
  if (self->cxx->isMultiBlockLBP()) Py_RETURN_TRUE; else Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("is_multi_block_lbp could not be read", 0)
}

static PyGetSetDef PyBobIpBaseLBP_getseters[] = {
    {
      radius.name(),
      (getter)PyBobIpBaseLBP_getRadius,
      (setter)PyBobIpBaseLBP_setRadius,
      radius.doc(),
      0
    },
    {
      radii.name(),
      (getter)PyBobIpBaseLBP_getRadii,
      (setter)PyBobIpBaseLBP_setRadii,
      radii.doc(),
      0
    },
    {
      blockSize.name(),
      (getter)PyBobIpBaseLBP_getBlockSize,
      (setter)PyBobIpBaseLBP_setBlockSize,
      blockSize.doc(),
      0
    },
    {
      blockOverlap.name(),
      (getter)PyBobIpBaseLBP_getBlockOverlap,
      (setter)PyBobIpBaseLBP_setBlockOverlap,
      blockOverlap.doc(),
      0
    },
    {
      points.name(),
      (getter)PyBobIpBaseLBP_getPoints,
      (setter)PyBobIpBaseLBP_setPoints,
      points.doc(),
      0
    },
    {
      circular.name(),
      (getter)PyBobIpBaseLBP_getCircular,
      (setter)PyBobIpBaseLBP_setCircular,
      circular.doc(),
      0
    },
    {
      toAverage.name(),
      (getter)PyBobIpBaseLBP_getToAverage,
      (setter)PyBobIpBaseLBP_setToAverage,
      toAverage.doc(),
      0
    },
    {
      addAverage.name(),
      (getter)PyBobIpBaseLBP_getAddAverage,
      (setter)PyBobIpBaseLBP_setAddAverage,
      addAverage.doc(),
      0
    },
    {
      uniform.name(),
      (getter)PyBobIpBaseLBP_getUniform,
      (setter)PyBobIpBaseLBP_setUniform,
      uniform.doc(),
      0
    },
    {
      rotationInvariant.name(),
      (getter)PyBobIpBaseLBP_getRotationInvariant,
      (setter)PyBobIpBaseLBP_setRotationInvariant,
      rotationInvariant.doc(),
      0
    },
    {
      elbpType.name(),
      (getter)PyBobIpBaseLBP_getELBPType,
      (setter)PyBobIpBaseLBP_setELBPType,
      elbpType.doc(),
      0
    },
    {
      borderHandling.name(),
      (getter)PyBobIpBaseLBP_getBorderHandling,
      (setter)PyBobIpBaseLBP_setBorderHandling,
      borderHandling.doc(),
      0
    },
    {
      lookUpTable.name(),
      (getter)PyBobIpBaseLBP_getLUT,
      (setter)PyBobIpBaseLBP_setLUT,
      lookUpTable.doc(),
      0
    },
    {
      maxLabel.name(),
      (getter)PyBobIpBaseLBP_getMaxLabel,
      0,
      maxLabel.doc(),
      0
    },
    {
      relativePositions.name(),
      (getter)PyBobIpBaseLBP_getRelativePositions,
      0,
      relativePositions.doc(),
      0
    },
    {
      offset.name(),
      (getter)PyBobIpBaseLBP_getOffset,
      0,
      offset.doc(),
      0
    },
    {
      isMulti.name(),
      (getter)PyBobIpBaseLBP_getIsMulti,
      0,
      isMulti.doc(),
      0
    },
    {0}  /* Sentinel */
};



/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto setBlockSizeAndOverlap = bob::extension::FunctionDoc(
  "set_block_size_and_overlap",
  "This function sets the block size and the block overlap for MB-LBP features at the same time",
  0,
  true
)
.add_prototype("block_size, block_overlap")
.add_parameter("block_size", "(int, int)", "Multi-block LBP's with the given block size will be extracted")
.add_parameter("block_overlap", "(int, int)", "Multi-block LBP's with the given block overlap will be extracted")
;

static PyObject* PyBobIpBaseLBP_setBlockSizeAndOverlap(PyBobIpBaseLBPObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = setBlockSizeAndOverlap.kwlist();

  blitz::TinyVector<int,2> size, overlap;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)(ii)", kwlist, &size[0], &size[1], &overlap[0], &overlap[1])){
    setBlockSizeAndOverlap.print_usage();
    return 0;
  }
  self->cxx->setBlockSizeAndOverlap(size, overlap);
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("cannot set block size and overlap", 0)
}

static auto getShape = bob::extension::FunctionDoc(
  "lbp_shape",
  "This function returns the shape of the LBP image for the given image",
  "In case the :py:attr:`border_handling` is ``'shrink'`` the image resolution will be reduced, depending on the LBP configuration. "
  "This function will return the desired output shape for the given input image or input shape.",
  true
)
.add_prototype("input, is_integral_image", "lbp_shape")
.add_prototype("shape, is_integral_image", "lbp_shape")
.add_parameter("input", "array_like (2D)", "The input image for which LBP features should be extracted")
.add_parameter("shape", "(int, int)", "The shape of the input image for which LBP features should be extracted")
.add_parameter("is_integral_image", "bool", "[default: ``False``] Is the given image (shape) an integral image?")
.add_return("lbp_shape", "(int, int)", "The shape of the LBP image that is required in a call to :py:func:`extract`")
;

static PyObject* PyBobIpBaseLBP_getShape(PyBobIpBaseLBPObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = getShape.kwlist(0);
  char** kwlist2 = getShape.kwlist(1);

  blitz::TinyVector<int,2> shape;
  PyObject* iii = 0; // is_integral_image
  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (
    (kwargs && PyDict_Contains(kwargs, k)) ||
    (args && PyTuple_Size(args) && (PyTuple_Check(PyTuple_GetItem(args, 0)) || PyList_Check(PyTuple_GetItem(args, 0))))
  ){
    // by shape
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)|O!", kwlist2, &shape[0], &shape[1], &PyBool_Type, &iii)){
      getShape.print_usage();
      return 0;
    }
  } else {
    // by image
    PyBlitzArrayObject* image = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O!", kwlist1, &PyBlitzArray_Converter, &image, &PyBool_Type, &iii)){
      getShape.print_usage();
      return 0;
    }
    auto _ = make_safe(image);
    if (image->ndim != 2) {
      getShape.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' only accepts 2-dimensional arrays (not %" PY_FORMAT_SIZE_T "dD arrays)", Py_TYPE(self)->tp_name, image->ndim);
      return 0;
    }
    shape[0] = image->shape[0];
    shape[1] = image->shape[1];
  }
  auto lbp_shape = self->cxx->getLBPShape(shape, f(iii));
  return Py_BuildValue("(ii)", lbp_shape[0], lbp_shape[1]);

  BOB_CATCH_MEMBER("cannot get LBP output shape", 0)
}

static auto extract = bob::extension::FunctionDoc(
  "extract",
  "This function extracts the LBP features from an image",
  "LBP features can be extracted either for the whole image, or at a single location in the image. "
  "When MB-LBP features will be extracted, an integral image will be computed to speed up the calculation. "
  "The integral image calculation can be done **before** this function is called, and the integral image can be passed to this function directly. "
  "In this case, please set the ``is_integral_image`` parameter to ``True``.\n\n"
  ".. note::\n\n  The `__call__` function is an alias for this method.",
  true
)
.add_prototype("input, [is_integral_image]", "output")
.add_prototype("input, position, [is_integral_image]", "code")
.add_prototype("input, output, [is_integral_image]")
.add_parameter("input", "array_like (2D)", "The input image for which LBP features should be extracted")
.add_parameter("position", "(int, int)", "The position in the ``input`` image, where the LBP code should be extracted; assure that you don't try to provide positions outside of the :py:attr:`offset`")
.add_parameter("output", "array_like (2D, uint16)", "The output image that need to be of shape :py:func:`lbp_shape`")
.add_parameter("is_integral_image", "bool", "[default: ``False``] Is the given ``input`` image an integral image?")
.add_return("output", "array_like (2D, uint16)", "The resulting image of LBP codes")
.add_return("code", "uint16", "The resulting LBP code at the given position in the image")
;

template <typename T>
static PyObject* extract_inner(PyBobIpBaseLBPObject* self, PyBlitzArrayObject* input, const blitz::TinyVector<int,2>& position, bool iii){
  uint16_t v = self->cxx->extract(*PyBlitzArrayCxx_AsBlitz<T,2>(input), position[0], position[1], iii);
  return Py_BuildValue("H", v);
}
template <typename T>
static PyObject* extract_inner(PyBobIpBaseLBPObject* self, PyBlitzArrayObject* input, PyBlitzArrayObject* output, bool iii, bool ret_img){
  self->cxx->extract(*PyBlitzArrayCxx_AsBlitz<T,2>(input), *PyBlitzArrayCxx_AsBlitz<uint16_t,2>(output), iii);
  if (ret_img){
    return PyBlitzArray_AsNumpyArray(output, 0);
  } else {
    Py_RETURN_NONE;
  }
}

static PyObject* PyBobIpBaseLBP_extract(PyBobIpBaseLBPObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist1 = extract.kwlist(0);
  char** kwlist2 = extract.kwlist(1);
  char** kwlist3 = extract.kwlist(2);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if (!nargs){
    // at least one argument is required
    extract.print_usage();
    PyErr_Format(PyExc_TypeError, "`%s' extract requires at least the ``input`` parameter", Py_TYPE(self)->tp_name);
    return 0;
  } // nargs == 0

  int how = 0;

  PyObject* k1 = Py_BuildValue("s", kwlist1[1]),* k2 = Py_BuildValue("s", kwlist2[1]);
  auto k1_ = make_safe(k1), k2_ = make_safe(k2);
  if (nargs == 1) how = 1;
  else if (nargs == 2){
    if ((args && PyTuple_Size(args) == 2 && PyBool_Check(PyTuple_GET_ITEM(args,1))) || (kwargs && PyDict_Contains(kwargs, k1))) how = 1;
    else if ((args && PyTuple_Size(args) == 2 && (PyTuple_Check(PyTuple_GET_ITEM(args,1)) || PyList_Check(PyTuple_GET_ITEM(args,1)))) || (kwargs && PyDict_Contains(kwargs, k2))) how = 2;
    else how = 3;
  } else if (nargs == 3){
    if ((args && PyTuple_Size(args) >= 2 && (PyTuple_Check(PyTuple_GET_ITEM(args,1)) || PyList_Check(PyTuple_GET_ITEM(args,1)))) || (kwargs && PyDict_Contains(kwargs, k2))) how = 2;
    else how = 3;
  }
  else {
    extract.print_usage();
    PyErr_Format(PyExc_TypeError, "`%s' extract has maximum 3 parameters", Py_TYPE(self)->tp_name);
    return 0;
  }

  PyBlitzArrayObject* input = 0,* output = 0;
  PyObject* iii = 0; // is_integral_image
  auto input_ = make_xsafe(input);
  auto output_ = make_xsafe(output);
  blitz::TinyVector<int,2> position;

  // get the command line parameters
  switch (how){
    case 1:
      // input image only
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O!", kwlist1, &PyBlitzArray_Converter, &input, &PyBool_Type, &iii)){
        extract.print_usage();
        return 0;
      }
      break;
    case 2:
      // with position
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&(ii)|O!", kwlist2, &PyBlitzArray_Converter, &input, &position[0], &position[1], &PyBool_Type, &iii)){
        extract.print_usage();
        return 0;
      }
      break;
    case 3:
      // with input and output image
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|O!", kwlist3, &PyBlitzArray_Converter, &input, &PyBlitzArray_OutputConverter, &output, &PyBool_Type, &iii)){
        extract.print_usage();
        return 0;
      }
      output_ = make_safe(output);
      break;
  }
  input_ = make_safe(input);
  // perform checks on input and output image
  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only extracts from 2D arrays", Py_TYPE(self)->tp_name);
    extract.print_usage();
    return 0;
  }
  auto shape = self->cxx->getLBPShape(blitz::TinyVector<int,2>(input->shape[0], input->shape[1]), f(iii));
  if (output){
    if (output->ndim != 2){
      PyErr_Format(PyExc_TypeError, "`%s' only extracts to 2D arrays", Py_TYPE(self)->tp_name);
      extract.print_usage();
      return 0;
    }
    if (output->shape[0] != shape[0] || output->shape[1] != shape[1]){
      PyErr_Format(PyExc_TypeError, "`%s' requires the shape of the output image to be (%d, %d), but it is (%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d),", Py_TYPE(self)->tp_name, shape[0], shape[1], output->shape[0], output->shape[1]);
      extract.print_usage();
      return 0;
    }
  } else if (how == 1) {
    Py_ssize_t osize[] = {shape[0], shape[1]};
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_UINT16, 2, osize);
    output_ = make_safe(output);
  }

  // finally, extract the features
  switch (input->type_num){
    case NPY_UINT8:   return how == 2 ? extract_inner<uint8_t>(self, input, position, f(iii))  : extract_inner<uint8_t>(self, input, output, f(iii), how == 1);
    case NPY_UINT16:  return how == 2 ? extract_inner<uint16_t>(self, input, position, f(iii)) : extract_inner<uint16_t>(self, input, output, f(iii), how == 1);
    case NPY_FLOAT64: return how == 2 ? extract_inner<double>(self, input, position, f(iii))   : extract_inner<double>(self, input, output, f(iii), how == 1);
    default:
      extract.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' extracts only from images of types uint8, uint16 or float, and not from %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(input->type_num));
      return 0;
  }
  BOB_CATCH_MEMBER("cannot extract LBP from image", 0)
}

static auto load = bob::extension::FunctionDoc(
  "load",
  "Loads the parametrization of the LBP extractor from the given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file opened for reading")
;

static PyObject* PyBobIpBaseLBP_load(PyBobIpBaseLBPObject* self, PyObject* args, PyObject* kwargs) {
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
  "Saves the the parametrization of the LBP extractor to the given HDF5 file",
  ".. warning:: For the time being, the look-up-table is **not saved**. "
  "If you have set the :py:attr:`look_up_table` by hand, it is lost.",
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing")
;

static PyObject* PyBobIpBaseLBP_save(PyBobIpBaseLBPObject* self, PyObject* args, PyObject* kwargs) {
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


static PyMethodDef PyBobIpBaseLBP_methods[] = {
  {
    setBlockSizeAndOverlap.name(),
    (PyCFunction)PyBobIpBaseLBP_setBlockSizeAndOverlap,
    METH_VARARGS|METH_KEYWORDS,
    setBlockSizeAndOverlap.doc()
  },
  {
    getShape.name(),
    (PyCFunction)PyBobIpBaseLBP_getShape,
    METH_VARARGS|METH_KEYWORDS,
    getShape.doc()
  },
  {
    extract.name(),
    (PyCFunction)PyBobIpBaseLBP_extract,
    METH_VARARGS|METH_KEYWORDS,
    extract.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobIpBaseLBP_load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    save.name(),
    (PyCFunction)PyBobIpBaseLBP_save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the LBP type struct; will be initialized later
PyTypeObject PyBobIpBaseLBP_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseLBP(PyObject* module)
{
  // initialize the type struct
  PyBobIpBaseLBP_Type.tp_name = LBP_doc.name();
  PyBobIpBaseLBP_Type.tp_basicsize = sizeof(PyBobIpBaseLBPObject);
  PyBobIpBaseLBP_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseLBP_Type.tp_doc = LBP_doc.doc();

  // set the functions
  PyBobIpBaseLBP_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseLBP_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseLBP_init);
  PyBobIpBaseLBP_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseLBP_delete);
  PyBobIpBaseLBP_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseLBP_RichCompare);
  PyBobIpBaseLBP_Type.tp_methods = PyBobIpBaseLBP_methods;
  PyBobIpBaseLBP_Type.tp_getset = PyBobIpBaseLBP_getseters;
  PyBobIpBaseLBP_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseLBP_extract);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseLBP_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseLBP_Type);
  return PyModule_AddObject(module, "LBP", (PyObject*)&PyBobIpBaseLBP_Type) >= 0;
}
