/**
 * @date Fri Sep 30 16:56:06 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IP_BASE_WIENER_H
#define BOB_IP_BASE_WIENER_H

#include <blitz/array.h>
#include <complex>
#include <bob.io.base/HDF5File.h>
#include <bob.sp/FFT2D.h>

namespace bob { namespace ip { namespace base {

/**
 * @brief A Wiener filter, which can be used to denoise a signal,
 * by comparing with a statistical model of the noiseless signal.\n
 *
 * Reference:\n
 * Computer Vision: Algorithms and Applications, Richard Szeliski
 * (Part 3.4.3)
 */
class Wiener
{
  public: //api
    /**
     * @brief Constructor, builds a new Wiener filter. Wiener filter is
     * initialized with the given size, Ps being sets to the variance
     * threshold.
     */
    Wiener(const blitz::TinyVector<int,2>& size, const double Pn, const double variance_threshold=1e-8);

    /**
     * @brief Builds a new filter with the given variance estimate Ps and
     * noise level Pn.
     */
    Wiener(const blitz::Array<double,2>& Ps, const double Pn, const double variance_threshold=1e-8);

    /**
     * @brief Trains a new Wiener filter with the given data
     */
    Wiener(const blitz::Array<double,3>& data, const double variance_threshold=1e-8);

    /**
     * @brief Copy constructor
     */
    Wiener(const Wiener& other);

    /**
     * @brief Loads a Wiener filter from file
     */
    Wiener(bob::io::base::HDF5File& config);

    /**
     * @brief Assignment operator
     */
    Wiener& operator=(const Wiener& other);

    /**
     * @brief Equal to
     */
    bool operator==(const Wiener& other) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const Wiener& other) const;

    /**
     * @brief Is similar to
     */
    bool is_similar_to(const Wiener& b, const double r_epsilon=1e-5, const double a_epsilon=1e-8) const;

    /**
     * @brief Loads filter from file
     */
    void load(bob::io::base::HDF5File& config);

    /**
     * @brief Saves filter to file
     */
    void save(bob::io::base::HDF5File& config) const;

    /**
     * @brief Trains the Wiener filter to perform the filtering.
     */
    void train(const blitz::Array<double,3>& data);

    /**
     * @brief filters the given input image
     *
     * The input and output are NOT checked for compatibility each time. It
     * is your responsibility to do it.
     */
    void filter_(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const;

    /**
     * @brief filters the given input image
     *
     * The input and output are checked for compatibility each time the
     * filtering method is applied.
     */
    void filter(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const;

    /**
     * @brief Resizes the filter and preserves the data
     */
    void resize(const blitz::TinyVector<int,2>& size);

    /**
     * @brief Returns the current variance Ps estimated at each frequency
     */
    const blitz::Array<double, 2>& getPs() const {return m_Ps;}

     /**
      * @brief Returns the current variance threshold applied to Ps
      */
    double getVarianceThreshold() const {return m_variance_threshold;}

     /**
      * @brief Returns the current noise level Pn
      */
    double getPn() const {return m_Pn;}

    /**
     * @brief Returns the size of the filter/input
     */
    blitz::TinyVector<int,2> getSize() const {return m_W.shape();}

    /**
     * @brief Returns the current Wiener filter (in the frequency domain).
     */
    const blitz::Array<double, 2>& getW() const {return m_W;}

    /**
     * @brief Sets the current variance Ps estimated at each frequency.
     * This will also update the Wiener filter, using thresholded values.
     */
    void setPs(const blitz::Array<double,2>& Ps);

    /**
     * @brief Sets the current variance threshold to be used.
     * This will also update the Wiener filter
     */
    void setVarianceThreshold(const double variance_threshold);

    /**
     * @brief Sets the current noise level Pn to be considered.
     * This will update the Wiener filter
     */
    void setPn(const double Pn) {m_Pn = Pn; computeW();}


  private: //representation
    void computeW(); /// Compute the Wiener filter using Pn, Ps, etc.
    void applyVarianceThreshold(); /// Apply variance flooring threshold

    blitz::Array<double, 2> m_Ps; ///< variance at each frequency estimated empirically
    double m_variance_threshold; ///< Threshold on Ps values when computing the Wiener filter
                                 ///  (to avoid division by zero)
    double m_Pn; ///< variance of the noise
    blitz::Array<double, 2> m_W; ///< Wiener filter in the frequency domain (W=1/(1+Pn/Ps))
    bob::sp::FFT2D m_fft;
    bob::sp::IFFT2D m_ifft;

    mutable blitz::Array<std::complex<double>, 2> m_buffer1; ///< a buffer for speed
    mutable blitz::Array<std::complex<double>, 2> m_buffer2; ///< a buffer for speed
};

} } } // namespaces

#endif /* BOB_IP_BASE_WIENER_H */
