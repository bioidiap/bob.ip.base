/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Jun 23 19:18:25 CEST 2014
 *
 * @brief Header file for bindings to bob::ip
 */

#ifndef BOB_IP_BASE_MAIN_H
#define BOB_IP_BASE_MAIN_H

#include <Python.h>

#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

#include <bob/ip/LBP.h>

// LBP
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::LBP> cxx;
} PyBobIpLBPObject;

extern PyTypeObject PyBobIpLBPType;

bool init_BobIpLBP(PyObject* module);

int PyBobIpLBP_Check(PyObject* o);



// zigzag
PyObject* PyBobIpBase_zigzag(PyObject*, PyObject*, PyObject*);

extern bob::extension::FunctionDoc s_zigzag;

#endif // BOB_IP_BASE_MAIN_H
