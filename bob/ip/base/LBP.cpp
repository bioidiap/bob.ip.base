/**
 * @author Manuel Guenther <manuel.guenthr@idiap.ch>
 * @date Tue Jun 24 14:03:17 CEST 2014
 *
 * @brief Binds the LBP class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_IP_GABOR_MODULE

#include "main.h"
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>


static inline char* c(const char* o){return const_cast<char*>(o);}
static inline bool b(PyObject* o){return !o || PyObject_IsTrue(o);}
static inline bob::ip::ELBPType e(const std::string& o){
  if (o == "regular") return bob::ip::ELBP_REGULAR;
  else if (o == "transitional") return bob::ip::ELBP_TRANSITIONAL;
  else if (o == "direction-coded") return bob::ip::ELBP_DIRECTION_CODED;
  else throw std::runtime_error("The given LBP type '" + o + "' is not known; choose one of ('regular', 'transitional', 'direction-coded')");
}
static inline bob::ip::LBPBorderHandling h(const std::string& o){
  if (o == "shrink") return bob::ip::LBP_BORDER_SHRINK;
  else if (o == "wrap") return bob::ip::LBP_BORDER_WRAP;
  else throw std::runtime_error("The given border handling '" + o + "' is not known; choose one of ('shrink', 'wrap')");
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
    "Finally, the border handling of the image can be selected. "
    "With the ``'shrink'`` option, no LBP code is computed for the border pixels and the resulting image is :math:`2\\times` radius or :math:`3\\times` blocksize :math:`-1` pixels smaller in both directions, see :py:func:`get_lbp_shape`. "
    "The ``'wrap'`` option will wrap around the border and no truncation is performed.\n\n"
    ".. note:: To compute MB-LBP features, it is possible to compute an integral image before to speed up the calculation.",
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


static int PyBobIpLBP_init(PyBobIpLBPObject* self, PyObject* args, PyObject* kwargs) {

  char* kwlist1[] = {c("neighbors"), c("radius"), c("circular"), c("to_average"), c("add_average_bit"), c("uniform"), c("rotation_invariant"), c("elbp_type"), c("border_handling"), NULL};
  char* kwlist2[] = {c("neighbors"), c("radius_y"), c("radius_x"), c("circular"), c("to_average"), c("add_average_bit"), c("uniform"), c("rotation_invariant"), c("elbp_type"), c("border_handling"), NULL};
  char* kwlist3[] = {c("neighbors"), c("block_size"), c("block_overlap"), c("to_average"), c("add_average_bit"), c("uniform"), c("rotation_invariant"), c("elbp_type"), c("border_handling"), NULL};
  char* kwlist4[] = {c("lbp"), NULL};
  char* kwlist5[] = {c("hdf5"), NULL};

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  int neighbors;
  double radius = 1., r_y, r_x;
  blitz::TinyVector<int,2> block_size, block_overlap(0,0);
  PyObject* circular = 0,* to_average = 0,* add_average_bit = 0,* uniform = 0,* rotation_invariant = 0;
  const char* elbp_type = "regular",* border_handling = "shrink";

  try{

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
        (args && PyBobIoHDF5File_Check(PyTuple_GetItem(args, 0)))
      ){
        // create from HDF5 file
        PyBobIoHDF5FileObject* hdf5;
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist5, &PyBobIoHDF5File_Converter, &hdf5)){
          LBP_doc.print_usage();
          return -1;
        }
        auto hdf5_ = make_safe(hdf5);
        self->cxx.reset(new bob::ip::LBP(*hdf5->f));
        return 0;
      } else if (
        (kwargs && PyDict_Contains(kwargs, k4)) ||
        (args && PyBobIpLBP_Check(PyTuple_GetItem(args, 0)))
      ){
        // copy constructor
        PyBobIpLBPObject* lbp;
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist4, &PyBobIpLBPType, &lbp)){
          LBP_doc.print_usage();
          return -1;
        }
        self->cxx.reset(new bob::ip::LBP(*lbp->cxx));
        return 0;
      } else {
        // first variant with default radius
        char* kwlist_t[] = {kwlist1[0], NULL};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist_t, &neighbors)){
          LBP_doc.print_usage();
          return -1;
        }
        self->cxx.reset(new bob::ip::LBP(neighbors, radius, b(circular), b(to_average), b(add_average_bit), b(uniform), b(rotation_invariant), e(elbp_type), h(border_handling)));
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
      self->cxx.reset(new bob::ip::LBP(neighbors, block_size, block_overlap, b(to_average), b(add_average_bit), b(uniform), b(rotation_invariant), e(elbp_type), h(border_handling)));
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
        self->cxx.reset(new bob::ip::LBP(neighbors, r_y, r_x, b(circular), b(to_average), b(add_average_bit), b(uniform), b(rotation_invariant), e(elbp_type), h(border_handling)));
        return 0;
      } else {
        // only one radius
        if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "id|O!O!O!O!O!ss", kwlist1,
              &neighbors, &radius,
              &PyBool_Type, &circular, &PyBool_Type, &to_average, &PyBool_Type, &add_average_bit, &PyBool_Type, &uniform, &PyBool_Type, &rotation_invariant,
              &elbp_type, &border_handling))
        ){
          LBP_doc.print_usage();
          return -1;
        }
        self->cxx.reset(new bob::ip::LBP(neighbors, radius, b(circular), b(to_average), b(add_average_bit), b(uniform), b(rotation_invariant), e(elbp_type), h(border_handling)));
        return 0;
      }
    }
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "%s cannot transform input: unknown exception caught", Py_TYPE(self)->tp_name);
    return -1;
  }
}

static void PyBobIpLBP_delete(PyBobIpLBPObject* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpLBP_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpLBPType));
}

static PyObject* PyBobIpLBP_RichCompare(PyBobIpLBPObject* self, PyObject* other, int op) {

  if (!PyBobIpLBP_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpLBPObject*>(other);
  try{
    switch (op) {
      case Py_EQ:
        if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
      case Py_NE:
        if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
      default:
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "%s cannot compare LBP objects: unknown exception caught", Py_TYPE(self)->tp_name);
  }
  return 0;
}


#if 0
/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto getWavelet_doc = bob::extension::VariableDoc(
  "wavelet",
  "2D-array complex",
  "The image representation of the Gabor wavelet in frequency domain",
  "The image representation is generated on the fly (since the original data format is different), the data format is float. "
  "To obtain the image representation in spatial domain, please perform a :py:func:`bob.sp.ifft()` on the returned image."
);

PyObject* PyBobIpLBP_getWavelet(PyBobIpLBPObject* self, void*){
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->waveletImage());
}


static PyGetSetDef PyBobIpLBP_getseters[] = {
    {
      getWavelet_doc.name(),
      (getter)PyBobIpLBP_getWavelet,
      0,
      getWavelet_doc.doc(),
      0
    },
    {0}  /* Sentinel */
};
#endif



/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto getShape = bob::extension::FunctionDoc(
  "get_lbp_shape",
  "This function returns the shape of the LBP image for the given image",
  "In case the :py:attr:`border_handling` is ``'shrink'`` the image resolution will be reduced, depending on the LBP configuration. "
  "This function will return the desired output shape for the given input image or input shape. "
  "If the given image is an integral image",
  true
)
.add_prototype("input, is_integral_image", "lbp_shape")
.add_prototype("shape, is_integral_image", "lbp_shape")
.add_parameter("input", "array_like (2D)", "The input image for which LBP features should be extracted")
.add_parameter("shape", "(int, int)", "The shape of the input image for which LBP features should be extracted")
.add_parameter("shape", "bool", "Is the given image (shape) an integral image?")
.add_return("lbp_shape", "(int, int)", "The shape of the LBP image that is required in a call to :py:func:`extract`")
;

static PyObject* PyBobIpLBP_getShape(PyBobIpLBPObject* self, PyObject* args, PyObject* kwargs) {

  static char* kwlist1[] = {c("input"), 0};
  static char* kwlist2[] = {c("shape"), 0};

  try{
    blitz::TinyVector<int,2> shape;
    PyObject* k = Py_BuildValue("s", kwlist2[0]);
    if (
      (kwargs && PyDict_Contains(kwargs, k)) ||
      (args && PySequence_Check(PyTuple_GetItem(args, 0)))
    ){
      // by shape
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)", kwlist2, &shape[0], &shape[1])){
        getShape.print_usage();
        return 0;
      }
    } else {
      // by image
      PyBlitzArrayObject* image = 0;
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist1, &PyBlitzArray_Converter, &image)){
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
    auto lbp_shape = self->cxx->getLBPShape(shape);
    return Py_BuildValue("(ii)", lbp_shape[0], lbp_shape[1]);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "%s cannot get LBP output shape: unknown exception caught", Py_TYPE(self)->tp_name);
  }
  return 0;
}

static auto extract = bob::extension::FunctionDoc(
  "extract",
  "This function extracts the LBP features from an image",
  "LBP features can be extracted either for the whole image, or at a single location in the image. "
  "When MB-LBP features will be extracted, an integral image will be computed to speed up the calculation. "
  "The integral image calculation can be done **before** this function is called, and the integral image can be passed to this function directly. "
  "In this case, please set the ``is_integral_image`` parameter to ``True``.",
  true
)
.add_prototype("input, [is_integral_image]", "output")
.add_prototype("input, y, x, [is_integral_image]", "code")
.add_prototype("input, output, [is_integral_image]")
.add_parameter("input", "array_like (2D)", "The input image for which LBP features should be extracted")
.add_parameter("y, x", "int", "The position in the ``input`` image, where the LBP code should be extracted")
.add_parameter("output", "array_like (2D, uint16)", "The output image that need to be of shape :py:func:`get_lbp_shape`")
.add_parameter("is_integral_image", "bool", "[default: ``False``] Is the given ``input`` image an integral image?")
.add_return("output", "array_like (2D, uint16)", "The resulting image of LBP codes")
.add_return("code", "uint16", "The resulting LBP code at the given position in the image")
;

template <typename T>
static PyObject* extract_inner(PyBobIpLBPObject* self, PyBlitzArrayObject* input, int y, int x, bool iii){
  uint16_t v = self->cxx->extract_(*PyBlitzArrayCxx_AsBlitz<T,2>(input), y, x, iii);
  return Py_BuildValue("H", v);
}
template <typename T>
static PyObject* extract_inner(PyBobIpLBPObject* self, PyBlitzArrayObject* input, PyBlitzArrayObject* output, bool iii){
  self->cxx->extract_(*PyBlitzArrayCxx_AsBlitz<T,2>(input), *PyBlitzArrayCxx_AsBlitz<uint16_t,2>(output), iii);
  Py_INCREF(output);
  return Py_BuildValue("O", output);
}

static PyObject* PyBobIpLBP_extract(PyBobIpLBPObject* self, PyObject* args, PyObject* kwargs) {

  static char* kwlist1[] = {c("input"), c("is_integral_image"), 0};
  static char* kwlist2[] = {c("input"), c("y"), c("x"), c("is_integral_image"), 0};
  static char* kwlist3[] = {c("input"), c("output"), c("is_integral_image"), 0};

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  try{
    if (!nargs){
      // at least one argument is required
      LBP_doc.print_usage();
      PyErr_Format(PyExc_TypeError, "`%s' extract requires at least the ``input`` parameter", Py_TYPE(self)->tp_name);
      return 0;
    } // nargs == 0

    int how = 0;

    PyObject* k1 = Py_BuildValue("s", kwlist1[1]),* k2 = Py_BuildValue("s", kwlist2[2]);
    auto k1_ = make_safe(k1), k2_ = make_safe(k2);
    if (nargs == 4) how = 2;
    else if (nargs == 3){
      if ((args && PyTuple_Size(args) == 3 && PyInt_Check(PyTuple_GET_ITEM(args,2))) || (kwargs && PyDict_Contains(kwargs, k2))) how = 2;
      else how = 3;
    } else if (nargs == 2){
      if ((args && PyTuple_Size(args) == 2 && PyBool_Check(PyTuple_GET_ITEM(args,1))) || (kwargs && PyDict_Contains(kwargs, k1))) how = 1;
      else how = 3;
    }

    PyBlitzArrayObject* input = 0,* output = 0;
    PyObject* iii = 0; // is_integral_image
    auto input_ = make_xsafe(input);
    auto output_ = make_xsafe(output);
    int y, x;

    // get the command line paramters
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
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&ii|O!", kwlist2, &PyBlitzArray_Converter, &input, &y, &x, &PyBool_Type, &iii)){
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
    // perform checks on input and output image
    if (input->ndim != 2){
      PyErr_Format(PyExc_TypeError, "`%s' only extracts from 2D arrays", Py_TYPE(self)->tp_name);
      extract.print_usage();
      return 0;
    }
    auto shape = self->cxx->getLBPShape(blitz::TinyVector<int,2>(input->shape[0], input->shape[1]), b(iii));
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
    } else if (how == 3) {
      Py_ssize_t osize[] = {shape[0], shape[1]};
      output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_UINT16, 2, osize);
      output_ = make_safe(output);
    }

    // finally, extract the features
    switch (input->type_num){
      case NPY_UINT8:   return how == 2 ? extract_inner<uint8_t>(self, input, y, x, b(iii))  : extract_inner<uint8_t>(self, input, output, b(iii));
      case NPY_UINT16:  return how == 2 ? extract_inner<uint16_t>(self, input, y, x, b(iii)) : extract_inner<uint16_t>(self, input, output, b(iii));
      case NPY_FLOAT64: return how == 2 ? extract_inner<double>(self, input, y, x, b(iii))   : extract_inner<double>(self, input, output, b(iii));
      default:
        extract.print_usage();
        PyErr_Format(PyExc_TypeError, "`%s' extracts only from images of types uint8, uint16 or float, and not from %s", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(input->type_num));
        return 0;
    }
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "%s cannot get LBP output shape: unknown exception caught", Py_TYPE(self)->tp_name);
  }
  return 0;
}

static PyMethodDef PyBobIpLBP_methods[] = {
  {
    getShape.name(),
    (PyCFunction)PyBobIpLBP_getShape,
    METH_VARARGS|METH_KEYWORDS,
    getShape.doc()
  },
  {
    extract.name(),
    (PyCFunction)PyBobIpLBP_extract,
    METH_VARARGS|METH_KEYWORDS,
    extract.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gabor wavelet type struct; will be initialized later
PyTypeObject PyBobIpLBPType = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpLBP(PyObject* module)
{

  // initialize the Gabor wavelet type struct
  PyBobIpLBPType.tp_name = LBP_doc.name();
  PyBobIpLBPType.tp_basicsize = sizeof(PyBobIpLBPObject);
  PyBobIpLBPType.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpLBPType.tp_doc = LBP_doc.doc();

  // set the functions
  PyBobIpLBPType.tp_new = PyType_GenericNew;
  PyBobIpLBPType.tp_init = reinterpret_cast<initproc>(PyBobIpLBP_init);
  PyBobIpLBPType.tp_dealloc = reinterpret_cast<destructor>(PyBobIpLBP_delete);
  PyBobIpLBPType.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpLBP_RichCompare);
  PyBobIpLBPType.tp_methods = PyBobIpLBP_methods;
//  PyBobIpLBPType.tp_getset = PyBobIpLBP_getseters;
//  PyBobIpLBPType.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpLBP_transform);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpLBPType) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&PyBobIpLBPType);
  return PyModule_AddObject(module, "LBP", (PyObject*)&PyBobIpLBPType) >= 0;
}

