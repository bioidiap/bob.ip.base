/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Wed Jun 25 18:28:03 CEST 2014
 *
 * @brief Binds the LBPTop class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto LBPTop_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".LBPTop",
  "A class that extracts local binary patterns (LBP) in three orthogonal planes (TOP)",
  "The LBPTop class is designed to calculate the LBP-Top coefficients given a set of images. "
  "The workflow is as follows:\n\n"
  ".. todo:: UPDATE as this is not true\n\n"
  "1. You initialize the class, defining the radius and number of points in each of the three directions: XY, XT, YT for the LBP calculations\n"
  "2. For each image you have in the frame sequence, you push into the class\n"
  "3. An internal FIFO queue (length = radius in T direction) keeps track of the current image and their order. "
  "As a new image is pushed in, the oldest on the queue is pushed out.\n"
  "4. After pushing an image, you read the current LBP-Top coefficients and may save it somewhere."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new LBPTop object",
    "For all three directions, the LBP objects need to be specified. "
    "The radii for the three LBP classes must be consistent, i.e., ``xy.radii[1] == xt.radii[1]``, ``xy.radii[0] == yt.radii[1]`` and ``xt.radii[0] == yt.radii[0]``.\n\n"
    ".. warning::\n\n"
    "   The order of the ``radius_x`` and ``radius_y`` parameters are not ``(radius_x, radius_y)`` in the :py:class:`LBP` constructor, but ``(radius_y, radius_x)``. "
    "   Hence, to get an ``x`` radius 2 and ``y`` radius 3, you need to use ``xy = bob.ip.base.LBP(8, 3, 2)`` or more specifically ``xy = bob.ip.base.LBP(8, radius_x=2, radius_y=3)``. "
    "   The same applies for ``xt`` and ``yt``.",
    true
  )
  .add_prototype("xy, xt, yt", "")
  .add_parameter("xy", ":py:class:`bob.ip.base.LBP`", "The 2D LBP-XY plane configuration")
  .add_parameter("xt", ":py:class:`bob.ip.base.LBP`", "The 2D LBP-XT plane configuration")
  .add_parameter("yt", ":py:class:`bob.ip.base.LBP`", "The 2D LBP-YT plane configuration")
);


static int PyBobIpBaseLBPTop_init(PyBobIpBaseLBPTopObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = LBPTop_doc.kwlist();

  PyBobIpBaseLBPObject* xy,* xt,* yt;
  if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!", kwlist, &PyBobIpBaseLBP_Type, &xy, &PyBobIpBaseLBP_Type, &xt, &PyBobIpBaseLBP_Type, &yt))){
    LBPTop_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::ip::base::LBPTop(xy->cxx, xt->cxx, yt->cxx));
  return 0;

  BOB_CATCH_MEMBER("cannot create LBPTop operator", -1)
}

static void PyBobIpBaseLBPTop_delete(PyBobIpBaseLBPTopObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpBaseLBPTop_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpBaseLBPTop_Type));
}

/** TODO
static PyObject* PyBobIpBaseLBPTop_RichCompare(PyBobIpBaseLBPTopObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobIpBaseLBPTop_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpBaseLBPTopObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx == *other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx == *other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare LBPTop objects", 0)
}
*/

/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto xy = bob::extension::VariableDoc(
  "xy",
  ":py:class:`bob.ip.base.LBP`",
  "The 2D LBP-XY plane configuration"
);
PyObject* PyBobIpBaseLBPTop_getXY(PyBobIpBaseLBPTopObject* self, void*){
  BOB_TRY
  PyBobIpBaseLBPObject* lbp = (PyBobIpBaseLBPObject*)PyBobIpBaseLBP_Type.tp_alloc(&PyBobIpBaseLBP_Type, 0);
  lbp->cxx = self->cxx->getXY();
  return Py_BuildValue("N", lbp);
  BOB_CATCH_MEMBER("xy could not be read", 0)
}

static auto xt = bob::extension::VariableDoc(
  "xt",
  ":py:class:`bob.ip.base.LBP`",
  "The 2D LBP-XT plane configuration"
);
PyObject* PyBobIpBaseLBPTop_getXT(PyBobIpBaseLBPTopObject* self, void*){
  BOB_TRY
  PyBobIpBaseLBPObject* lbp = (PyBobIpBaseLBPObject*)PyBobIpBaseLBP_Type.tp_alloc(&PyBobIpBaseLBP_Type, 0);
  lbp->cxx = self->cxx->getXT();
  return Py_BuildValue("N", lbp);
  BOB_CATCH_MEMBER("xt could not be read", 0)
}

static auto yt = bob::extension::VariableDoc(
  "yt",
  ":py:class:`bob.ip.base.LBP`",
  "The 2D LBP-XT plane configuration"
);
PyObject* PyBobIpBaseLBPTop_getYT(PyBobIpBaseLBPTopObject* self, void*){
  BOB_TRY
  PyBobIpBaseLBPObject* lbp = (PyBobIpBaseLBPObject*)PyBobIpBaseLBP_Type.tp_alloc(&PyBobIpBaseLBP_Type, 0);
  lbp->cxx = self->cxx->getYT();
  return Py_BuildValue("N", lbp);
  BOB_CATCH_MEMBER("yt could not be read", 0)
}


static PyGetSetDef PyBobIpBaseLBPTop_getseters[] = {
    {
      xy.name(),
      (getter)PyBobIpBaseLBPTop_getXY,
      0,
      xy.doc(),
      0
    },
    {
      xt.name(),
      (getter)PyBobIpBaseLBPTop_getXT,
      0,
      xt.doc(),
      0
    },
    {
      yt.name(),
      (getter)PyBobIpBaseLBPTop_getYT,
      0,
      yt.doc(),
      0
    },
    {0}  /* Sentinel */
};



/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto process = bob::extension::FunctionDoc(
  "process",
  "This function processes the given set of images and extracts the three orthogonal planes",
  "The given 3D input array represents a set of **gray-scale** images and returns (by argument) the three LBP planes calculated. "
  "The 3D array has to be arranged in this way:\n\n"
  "1. First dimension: time\n"
  "2. Second dimension: frame height\n"
  "3. Third dimension: frame width\n\n"
  "The central pixel is the point where the LBP planes intersect/have to be calculated from.",
  true
)
.add_prototype("input, xy, xt, yt")
.add_parameter("input", "array_like (3D)", "The input set of gray-scale images for which LBPTop features should be extracted")
.add_parameter("xy, xt, yt", "array_like (3D, uint16)", "The result of the LBP operator in the XY, XT and YT plane (frame), for the central frame of the input array")
;

template <typename T>
static PyObject* process_inner(PyBobIpBaseLBPTopObject* self, PyBlitzArrayObject* input, PyBlitzArrayObject* xy, PyBlitzArrayObject* xt, PyBlitzArrayObject* yt){
  self->cxx->process(*PyBlitzArrayCxx_AsBlitz<T,3>(input), *PyBlitzArrayCxx_AsBlitz<uint16_t,3>(xy), *PyBlitzArrayCxx_AsBlitz<uint16_t,3>(xt), *PyBlitzArrayCxx_AsBlitz<uint16_t,3>(yt));
  Py_RETURN_NONE;
}

static PyObject* PyBobIpBaseLBPTop_process(PyBobIpBaseLBPTopObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = process.kwlist();

  PyBlitzArrayObject* input,* xy,* xt,* yt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&O&O&", kwlist, &PyBlitzArray_Converter, &input, &PyBlitzArray_OutputConverter, &xy, &PyBlitzArray_OutputConverter, &xt, &PyBlitzArray_OutputConverter, &yt)){
    process.print_usage();
    return 0;
  }
  auto input_ = make_safe(input), xy_ = make_safe(xy), xt_ = make_safe(xt), yt_ = make_safe(yt);

  // checks
  if (xy->ndim != 3 || xt->ndim != 3 || yt->ndim != 3 ||  xy->type_num != NPY_UINT16 || xt->type_num != NPY_UINT16 || yt->type_num != NPY_UINT16){
    PyErr_Format(PyExc_TypeError, "`%s' only extracts to 3D arrays of type uint16", Py_TYPE(self)->tp_name);
    return 0;
  }
  if (input->ndim != 3){
    PyErr_Format(PyExc_TypeError, "`%s' only extracts from 3D arrays", Py_TYPE(self)->tp_name);
    return 0;
  }

  switch (input->type_num){
    case NPY_UINT8: return process_inner<uint8_t>(self, input, xy, xt, yt);
    case NPY_UINT16: return process_inner<uint16_t>(self, input, xy, xt, yt);
    case NPY_FLOAT64: return process_inner<double>(self, input, xy, xt, yt);
    default:
      process.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' processes only images of types uint8, uint16 or float, and not from %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(input->type_num));
      return 0;
  }

  BOB_CATCH_MEMBER("cannot process LBPTop", 0)
}

static PyMethodDef PyBobIpBaseLBPTop_methods[] = {
  {
    process.name(),
    (PyCFunction)PyBobIpBaseLBPTop_process,
    METH_VARARGS|METH_KEYWORDS,
    process.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the TBPTop type struct; will be initialized later
PyTypeObject PyBobIpBaseLBPTop_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpBaseLBPTop(PyObject* module)
{
  // initialize the type struct
  PyBobIpBaseLBPTop_Type.tp_name = LBPTop_doc.name();
  PyBobIpBaseLBPTop_Type.tp_basicsize = sizeof(PyBobIpBaseLBPTopObject);
  PyBobIpBaseLBPTop_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpBaseLBPTop_Type.tp_doc = LBPTop_doc.doc();

  // set the functions
  PyBobIpBaseLBPTop_Type.tp_new = PyType_GenericNew;
  PyBobIpBaseLBPTop_Type.tp_init = reinterpret_cast<initproc>(PyBobIpBaseLBPTop_init);
  PyBobIpBaseLBPTop_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseLBPTop_delete);
//  PyBobIpBaseLBPTop_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseLBPTop_RichCompare);
  PyBobIpBaseLBPTop_Type.tp_methods = PyBobIpBaseLBPTop_methods;
  PyBobIpBaseLBPTop_Type.tp_getset = PyBobIpBaseLBPTop_getseters;
  PyBobIpBaseLBPTop_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseLBPTop_process);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpBaseLBPTop_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpBaseLBPTop_Type);
  return PyModule_AddObject(module, "LBPTop", (PyObject*)&PyBobIpBaseLBPTop_Type) >= 0;
}
