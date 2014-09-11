/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Tue  5 Nov 12:22:48 2013
 *
 * @brief Python API for bob::io::base
 */

#ifndef BOB_IP_BASE_API_H
#define BOB_IP_BASE_API_H

/* Define Module Name and Prefix for other Modules
   Note: We cannot use BOB_EXT_* macros here, unfortunately */
#define BOB_IP_BASE_PREFIX    "bob.ip.base"
#define BOB_IP_BASE_FULL_NAME "bob.ip.base._library"

#include <Python.h>

#include <bob.ip.base/config.h>
#include <bob.ip.base/LBP.h>

#include <boost/shared_ptr.hpp>

/*******************
 * C API functions *
 *******************/

/* Enum defining entries in the function table */
enum _PyBobIpBase_ENUM{
  PyBobIpBase_APIVersion_NUM = 0,
  // LBP bindings
  PyBobIpBaseLBP_Type_NUM,
  PyBobIpBaseLBP_Check_NUM,
  PyBobIpBaseLBP_Converter_NUM,
  // Total number of C API pointers
  PyBobIpBase_API_pointers
};

/**************
 * Versioning *
 **************/

/**********************************
 * Bindings for bob.ip.base.LBP *
 **********************************/

// LBP
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::LBP> cxx;
} PyBobIpBaseLBPObject;



#ifdef BOB_IP_BASE_MODULE

  /* This section is used when compiling `bob.io.base' itself */

  /**************
   * Versioning *
   **************/

  extern int PyBobIpBase_APIVersion;

  /**********************************
   * Bindings for bob.ip.base.LBP *
   **********************************/

  extern PyTypeObject PyBobIpBaseLBP_Type;
  int PyBobIpBaseLBP_Check(PyObject*);
  int PyBobIpBaseLBP_Converter(PyObject*, PyBobIpBaseLBPObject**);

#else // BOB_IP_BASE_MODULE

  /* This section is used in modules that use `bob.io.base's' C-API */

#if defined(NO_IMPORT_ARRAY)
  extern void **PyBobIpBase_API;
#elif defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyBobIpBase_API;
#else
  static void **PyBobIpBase_API=NULL;
#endif

  /**************
   * Versioning *
   **************/

#define PyBobIpBase_APIVersion (*(int *)PyBobIpBase_API[PyBobIpBase_APIVersion_NUM])

  /********************************
   * Bindings for bob.ip.base.LBP *
   ********************************/

#define PyBobIpBaseLBP_Type (*(PyTypeObject *)PyBobIpBase_API[PyBobIpBaseLBP_Type_NUM])
#define PyBobIpBaseLBP_Check (*(int (*)(PyObject*)) PyBobIpBase_API[PyBobIpBaseLBP_Check_NUM])
#define PyBobIpBaseLBP_Converter (*(int (*)(PyObject*, PyBobIpBaseLBPObject**)) PyBobIpBase_API[PyBobIpBaseLBP_Converter_NUM])

#if !defined(NO_IMPORT_ARRAY)

  /**
   * Returns -1 on error, 0 on success.
   */
  static int import_bob_ip_base(void) {

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOB_IP_BASE_FULL_NAME);

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

#if PY_VERSION_HEX >= 0x02070000
    if (PyCapsule_CheckExact(c_api_object)) {
      PyBobIpBase_API = (void **)PyCapsule_GetPointer(c_api_object, PyCapsule_GetName(c_api_object));
    }
#else
    if (PyCObject_Check(c_api_object)) {
      PyBobIpBase_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }
#endif

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    if (!PyBobIpBase_API) {
      PyErr_SetString(PyExc_ImportError, "cannot find C/C++ API "
#if PY_VERSION_HEX >= 0x02070000
          "capsule"
#else
          "cobject"
#endif
          " at `" BOB_IP_BASE_FULL_NAME "._C_API'");
      return -1;
    }

    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyBobIpBase_API[PyBobIpBase_APIVersion_NUM];

    if (BOB_IP_BASE_API_VERSION != imported_version) {
      PyErr_Format(PyExc_ImportError, BOB_IP_BASE_FULL_NAME " import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOB_IP_BASE_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

#endif //!defined(NO_IMPORT_ARRAY)

#endif /* BOB_IP_BASE_MODULE */

#endif /* BOB_IP_BASE_API_H */
