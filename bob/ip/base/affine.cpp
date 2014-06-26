/**
 * @author Manuel Guenther <manuel.guenthr@idiap.ch>
 * @date Wed Jun 25 18:28:03 CEST 2014
 *
 * @brief Binds auxiliary functions of bob::ip::base class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


#include "main.h"
#include "cpp/Affine.h"


bob::extension::FunctionDoc s_maxRectInMask = bob::extension::FunctionDoc(
  "max_rect_in_mask",
  "Given a 2D mask (a 2D blitz array of booleans), compute the maximum rectangle which only contains true values.",
  "The resulting rectangle contains the coordinates in the following order:\n\n"
  "0/ The y-coordinate of the top left corner\n\n"
  "1/ The x-coordinate of the top left corner\n\n"
  "2/ The height of the rectangle\n\n"
  "3/ The width of the rectangle"
)
.add_prototype("mask", "rect")
.add_parameter("mask", "array_like (2D, bool)", "The mask of boolean values, e.g., as a result of :py:func:`bob.ip.base.GeomNorm.process`")
.add_return("rect", "(int, int, int, int)", "The resulting rectangle: (top, left, height, width)")
;

PyObject* PyBobIpBase_maxRectInMask(PyObject*, PyObject* args, PyObject* kwds) {
  TRY
  /* Parses input arguments in a single shot */
  static char* kwlist[] = {c("mask"), NULL};

  PyBlitzArrayObject* mask = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBlitzArray_Converter, &mask)) return 0;

  auto input_ = make_safe(mask);

  if (mask->ndim != 2 || mask->type_num != NPY_BOOL) {
    PyErr_Format(PyExc_TypeError, "max_rect_in_mask: the mask must be 2D and of boolean type");
    return 0;
  }

  auto rect = bob::ip::base::maxRectInMask(*PyBlitzArrayCxx_AsBlitz<bool, 2>(mask));

  return Py_BuildValue("(iiii)", rect[0], rect[1], rect[2], rect[3]);

  CATCH_("in max_rect_in_mask", 0)
}

