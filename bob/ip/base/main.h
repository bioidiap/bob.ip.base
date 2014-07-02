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
#include "cpp/DCTFeatures.h"
#include "cpp/GeomNorm.h"
#include "cpp/FaceEyesNorm.h"


#if PY_VERSION_HEX >= 0x03000000
#define PyInt_Check PyLong_Check
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyString_Check PyUnicode_Check
#define PyString_AS_STRING(x) PyBytes_AS_STRING(make_safe(PyUnicode_AsUTF8String(x)).get())
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

// FaceEyesNorm
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::FaceEyesNorm> cxx;
} PyBobIpBaseFaceEyesNormObject;

extern PyTypeObject PyBobIpBaseFaceEyesNormType;
bool init_BobIpBaseFaceEyesNorm(PyObject* module);
int PyBobIpBaseFaceEyesNorm_Check(PyObject* o);

// .. scaling
PyObject* PyBobIpBase_scale(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_scale;
PyObject* PyBobIpBase_getScaledOutputShape(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_getScaledOutputShape;
// .. rotating
PyObject* PyBobIpBase_rotate(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_rotate;
PyObject* PyBobIpBase_getRotatedOutputShape(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_getRotatedOutputShape;

// mask functions (in Affine.h)
PyObject* PyBobIpBase_maxRectInMask(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_maxRectInMask;
PyObject* PyBobIpBase_extrapolateMask(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_extrapolateMask;


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


// DCTFeatures
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::DCTFeatures> cxx;
} PyBobIpBaseDCTFeaturesObject;

extern PyTypeObject PyBobIpBaseDCTFeaturesType;
bool init_BobIpBaseDCTFeatures(PyObject* module);
int PyBobIpBaseDCTFeatures_Check(PyObject* o);


// block
PyObject* PyBobIpBase_block(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_block;
PyObject* PyBobIpBase_blockOutputShape(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_blockOutputShape;

// LBPHS
PyObject* PyBobIpBase_lbphs(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_lbphs;
PyObject* PyBobIpBase_lbphsOutputShape(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_lbphsOutputShape;


// integral
PyObject* PyBobIpBase_integral(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_integral;


// histogram
PyObject* PyBobIpBase_histogram(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_histogram;
PyObject* PyBobIpBase_histogramEqualization(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_histogramEqualization;

// zigzag
PyObject* PyBobIpBase_zigzag(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_zigzag;


#endif // BOB_IP_BASE_MAIN_H
