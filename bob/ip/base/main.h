/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Jun 23 19:18:25 CEST 2014
 *
 * @brief Header file for bindings to bob::ip
 */

#ifndef BOB_IP_BASE_MAIN_H
#define BOB_IP_BASE_MAIN_H

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.core/random_api.h>
#include <bob.sp/api.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

#define BOB_IP_BASE_MODULE
#include <bob.ip.base/api.h>

#include <bob.ip.base/LBPTop.h>
#include <bob.ip.base/DCTFeatures.h>
#include <bob.ip.base/TanTriggs.h>
#include <bob.ip.base/Gaussian.h>
#include <bob.ip.base/MultiscaleRetinex.h>
#include <bob.ip.base/WeightedGaussian.h>
#include <bob.ip.base/SelfQuotientImage.h>
#include <bob.ip.base/GaussianScaleSpace.h>
#include <bob.ip.base/SIFT.h>
#include <bob.ip.base/HOG.h>
#include <bob.ip.base/GeomNorm.h>
#include <bob.ip.base/FaceEyesNorm.h>
#include <bob.ip.base/GLCM.h>
#include <bob.ip.base/Wiener.h>


/// inserts the given key, value pair into the given dictionaries
static inline int insert_item_string(PyObject* dict, PyObject* entries, const char* key, Py_ssize_t value){
  auto v = make_safe(Py_BuildValue("n", value));
  if (PyDict_SetItemString(dict, key, v.get()) < 0) return -1;
  return PyDict_SetItemString(entries, key, v.get());
}


// GeomNorm
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::GeomNorm> cxx;
} PyBobIpBaseGeomNormObject;

extern PyTypeObject PyBobIpBaseGeomNorm_Type;
bool init_BobIpBaseGeomNorm(PyObject* module);
int PyBobIpBaseGeomNorm_Check(PyObject* o);

// FaceEyesNorm
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::FaceEyesNorm> cxx;
} PyBobIpBaseFaceEyesNormObject;

extern PyTypeObject PyBobIpBaseFaceEyesNorm_Type;
bool init_BobIpBaseFaceEyesNorm(PyObject* module);
int PyBobIpBaseFaceEyesNorm_Check(PyObject* o);

// .. scaling
PyObject* PyBobIpBase_scale(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_scale;
PyObject* PyBobIpBase_scaledOutputShape(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_scaledOutputShape;
// .. rotating
PyObject* PyBobIpBase_rotate(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_rotate;
PyObject* PyBobIpBase_rotatedOutputShape(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_rotatedOutputShape;

// mask functions (in Affine.h)
PyObject* PyBobIpBase_maxRectInMask(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_maxRectInMask;
PyObject* PyBobIpBase_extrapolateMask(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_extrapolateMask;


// LBP
bool init_BobIpBaseLBP(PyObject* module);


// LBP-Top
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::LBPTop> cxx;
} PyBobIpBaseLBPTopObject;

extern PyTypeObject PyBobIpBaseLBPTop_Type;
bool init_BobIpBaseLBPTop(PyObject* module);
int PyBobIpBaseLBPTop_Check(PyObject* o);


// DCTFeatures
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::DCTFeatures> cxx;
} PyBobIpBaseDCTFeaturesObject;

extern PyTypeObject PyBobIpBaseDCTFeatures_Type;
bool init_BobIpBaseDCTFeatures(PyObject* module);
int PyBobIpBaseDCTFeatures_Check(PyObject* o);



// gamma correction
PyObject* PyBobIpBase_gammaCorrection(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_gammaCorrection;
// Tan-Triggs
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::TanTriggs> cxx;
} PyBobIpBaseTanTriggsObject;

extern PyTypeObject PyBobIpBaseTanTriggs_Type;
bool init_BobIpBaseTanTriggs(PyObject* module);
int PyBobIpBaseTanTriggs_Check(PyObject* o);


// Gaussian
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::Gaussian> cxx;
} PyBobIpBaseGaussianObject;

extern PyTypeObject PyBobIpBaseGaussian_Type;
bool init_BobIpBaseGaussian(PyObject* module);
int PyBobIpBaseGaussian_Check(PyObject* o);


// MultiscaleRetinex
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::MultiscaleRetinex> cxx;
} PyBobIpBaseMultiscaleRetinexObject;

extern PyTypeObject PyBobIpBaseMultiscaleRetinex_Type;
bool init_BobIpBaseMultiscaleRetinex(PyObject* module);
int PyBobIpBaseMultiscaleRetinex_Check(PyObject* o);


// WeightedGaussian
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::WeightedGaussian> cxx;
} PyBobIpBaseWeightedGaussianObject;

extern PyTypeObject PyBobIpBaseWeightedGaussian_Type;
bool init_BobIpBaseWeightedGaussian(PyObject* module);
int PyBobIpBaseWeightedGaussian_Check(PyObject* o);


// SelfQuotientImage
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::SelfQuotientImage> cxx;
} PyBobIpBaseSelfQuotientImageObject;

extern PyTypeObject PyBobIpBaseSelfQuotientImage_Type;
bool init_BobIpBaseSelfQuotientImage(PyObject* module);
int PyBobIpBaseSelfQuotientImage_Check(PyObject* o);


// SIFT...
// .. Gaussian Scale Space Keypoint
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::GSSKeypoint> cxx;
} PyBobIpBaseGSSKeypointObject;

extern PyTypeObject PyBobIpBaseGSSKeypoint_Type;
int PyBobIpBaseGSSKeypoint_Check(PyObject* o);

// .. Gaussian Scale Space KeypointInfo
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::GSSKeypointInfo> cxx;
} PyBobIpBaseGSSKeypointInfoObject;

extern PyTypeObject PyBobIpBaseGSSKeypointInfo_Type;
int PyBobIpBaseGSSKeypointInfo_Check(PyObject* o);

// .. Gaussian Scale Space
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::GaussianScaleSpace> cxx;
} PyBobIpBaseGaussianScaleSpaceObject;

extern PyTypeObject PyBobIpBaseGaussianScaleSpace_Type;
bool init_BobIpBaseGaussianScaleSpace(PyObject* module);
int PyBobIpBaseGaussianScaleSpace_Check(PyObject* o);

// .. SIFT
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::SIFT> cxx;
} PyBobIpBaseSIFTObject;

extern PyTypeObject PyBobIpBaseSIFT_Type;
bool init_BobIpBaseSIFT(PyObject* module);
int PyBobIpBaseSIFT_Check(PyObject* o);

#if HAVE_VLFEAT
// .. VLSIFT
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::VLSIFT> cxx;
} PyBobIpBaseVLSIFTObject;

extern PyTypeObject PyBobIpBaseVLSIFT_Type;
int PyBobIpBaseVLSIFT_Check(PyObject* o);

// .. VLDSIFT
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::VLDSIFT> cxx;
} PyBobIpBaseVLDSIFTObject;

extern PyTypeObject PyBobIpBaseVLDSIFT_Type;
int PyBobIpBaseVLDSIFT_Check(PyObject* o);

bool init_BobIpBaseVLFEAT(PyObject* module);
#endif // HAVE_VLFEAT


// HOG...
// .. GradientMagnitude
extern PyTypeObject PyBobIpBaseGradientMagnitude_Type;
int PyBobIpBaseGradientMagnitude_Check(PyObject* o);
int PyBobIpBaseGradientMagnitude_Converter(PyObject* o, bob::ip::base::GradientMagnitudeType* b);

// .. BlockNorm
extern PyTypeObject PyBobIpBaseBlockNorm_Type;
int PyBobIpBaseBlockNorm_Check(PyObject* o);
int PyBobIpBaseBlockNorm_Converter(PyObject* o, bob::ip::base::BlockNorm* b);

// .. HOG
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::HOG> cxx;
} PyBobIpBaseHOGObject;

extern PyTypeObject PyBobIpBaseHOG_Type;
int PyBobIpBaseHOG_Check(PyObject* o);

bool init_BobIpBaseHOG(PyObject* module);



// GLCM
typedef struct {
  PyObject_HEAD
  int type_num;
  boost::shared_ptr<void> cxx; // will be casted in each call
  boost::shared_ptr<bob::ip::base::GLCMProp> prop; // the GLMC property handler
} PyBobIpBaseGLCMObject;

extern PyTypeObject PyBobIpBaseGLCM_Type;
int PyBobIpBaseGLCM_Check(PyObject* o);

bool init_BobIpBaseGLCM(PyObject* module);


// .. Wiener filter
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::base::Wiener> cxx;
} PyBobIpBaseWienerObject;

extern PyTypeObject PyBobIpBaseWiener_Type;
int PyBobIpBaseWiener_Check(PyObject* o);

bool init_BobIpBaseWiener(PyObject* module);


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

// filtering
PyObject* PyBobIpBase_median(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_median;

PyObject* PyBobIpBase_sobel(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc s_sobel;

#endif // BOB_IP_BASE_MAIN_H
