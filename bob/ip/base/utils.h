/**
 * @author André Anjos <andre.anjos@idiap.ch>
 * @date Fri  4 Apr 15:20:24 2014 CEST
 *
 * @brief Helpers for color conversion
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>

int check_and_allocate(Py_ssize_t input_dims, Py_ssize_t output_dims,
    boost::shared_ptr<PyBlitzArrayObject>& input,
    boost::shared_ptr<PyBlitzArrayObject>& output);

int check_scalar(const char* s, PyObject* v);

int check_scalars(const char* s1, PyObject* v1, const char* s2, PyObject* v2,
    const char* s3, PyObject* v3);
