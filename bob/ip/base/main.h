/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Jun 23 19:18:25 CEST 2014
 *
 * @brief Header file for bindings to bob::ip
 */

#ifndef BOB_IP_BASE_MAIN_H
#define BOB_IP_BASE_MAIN_H

#include <Python.h>

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

#include "cpp/LBP.h"
#include "cpp/LBPTop.h"
#include "cpp/LBPHS.h"
#include "cpp/GeomNorm.h"


#if PY_VERSION_HEX >= 0x03000000
#define PyInt_Check PyLong_Check
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyString_AS_STRING PyUnicode_AsUTF8
#define PyString_Check PyUnicode_Check
#endif

#define TRY try{

#define CATCH(message,ret) }\
  catch (std::exception& e) {\
    PyErr_SetString(PyExc_RuntimeError, e.what());\
    return ret;\
  } \
  catch (...) {\
    PyErr_Format(PyExc_RuntimeError, "%s " message ": unknown exception caught", Py_TYPE(self)->tp_name);\
    return ret;\
  }

#define CATCH_(message, ret) }\
  catch (std::exception& e) {\
    PyErr_SetString(PyExc_RuntimeError, e.what());\
    return ret;\
  } \
  catch (...) {\
    PyErr_Format(PyExc_RuntimeError, message ": unknown exception caught");\
    return ret;\
  }

static inline char* c(const char* o){return const_cast<char*>(o);}  /* converts const char* to char* */


// GeomNorm
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::GeomNorm> cxx;
} PyBobIpBaseGeomNormObject;

extern PyTypeObject PyBobIpBaseGeomNormType;
bool init_BobIpBaseGeomNorm(PyObject* module);
int PyBobIpBaseGeomNorm_Check(PyObject* o);

// affine functions
PyObject* PyBobIpBase_maxRectInMask(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_maxRectInMask;



// LBP
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::LBP> cxx;
} PyBobIpBaseLBPObject;

extern PyTypeObject PyBobIpBaseLBPType;
bool init_BobIpBaseLBP(PyObject* module);
int PyBobIpBaseLBP_Check(PyObject* o);


// LBP-Top
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::LBPTop> cxx;
} PyBobIpBaseLBPTopObject;

extern PyTypeObject PyBobIpBaseLBPTopType;
bool init_BobIpBaseLBPTop(PyObject* module);
int PyBobIpBaseLBPTop_Check(PyObject* o);


// LBPHS
PyObject* PyBobIpBase_lbphs(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_lbphs;
PyObject* PyBobIpBase_lbphsOutputShape(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_lbphsOutputShape;


// integral
PyObject* PyBobIpBase_integral(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_integral;


// zigzag
PyObject* PyBobIpBase_zigzag(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_zigzag;


#endif // BOB_IP_BASE_MAIN_H
