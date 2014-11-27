/**
 * @date Fri Sep 30 16:56:06 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements a Wiener filter
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.core/array_copy.h>
#include <bob.core/cast.h>
#include <bob.ip.base/Wiener.h>
#include <complex>

bob::ip::base::Wiener::Wiener(const blitz::Array<double,2>& Ps, const double Pn, const double variance_threshold)
: m_Ps(bob::core::array::ccopy(Ps)),
  m_variance_threshold(variance_threshold),
  m_Pn(Pn),
  m_W(m_Ps.extent(0),m_Ps.extent(1)),
  m_fft(m_Ps.extent(0),m_Ps.extent(1)),
  m_ifft(m_Ps.extent(0),m_Ps.extent(1)),
  m_buffer1(m_Ps.extent(0),m_Ps.extent(1)),
  m_buffer2(m_Ps.extent(0),m_Ps.extent(1))
{
  computeW();
}

bob::ip::base::Wiener::Wiener(const blitz::TinyVector<int,2>& size, const double Pn, const double variance_threshold)
: m_Ps(size),
  m_variance_threshold(variance_threshold),
  m_Pn(Pn),
  m_W(size),
  m_fft(size[0], size[1]),
  m_ifft(size[0], size[1]),
  m_buffer1(0,0), m_buffer2(0,0)
{
  m_Ps = 1.;
  computeW();
}

bob::ip::base::Wiener::Wiener(const blitz::Array<double,3>& data, const double variance_threshold)
: m_variance_threshold(variance_threshold)
{
  train(data);
}

bob::ip::base::Wiener::Wiener(const bob::ip::base::Wiener& other)
: m_Ps(bob::core::array::ccopy(other.m_Ps)),
  m_variance_threshold(other.m_variance_threshold),
  m_Pn(other.m_Pn),
  m_W(bob::core::array::ccopy(other.m_W)),
  m_fft(other.m_fft),
  m_ifft(other.m_ifft),
  m_buffer1(m_Ps.extent(0),m_Ps.extent(1)),
  m_buffer2(m_Ps.extent(0),m_Ps.extent(1))
{
}

bob::ip::base::Wiener::Wiener(bob::io::base::HDF5File& config){
  load(config);
}

bob::ip::base::Wiener& bob::ip::base::Wiener::operator=(const bob::ip::base::Wiener& other){
  if (this != &other)
  {
    m_Ps.reference(bob::core::array::ccopy(other.m_Ps));
    m_Pn = other.m_Pn;
    m_variance_threshold = other.m_variance_threshold;
    m_W.reference(bob::core::array::ccopy(other.m_W));
    m_fft.setShape(m_Ps.extent(0),m_Ps.extent(1));
    m_ifft.setShape(m_Ps.extent(0),m_Ps.extent(1));
    m_buffer1.resize(m_Ps.shape());
    m_buffer2.resize(m_Ps.shape());
  }
  return *this;
}

bool bob::ip::base::Wiener::operator==(const bob::ip::base::Wiener& b) const{
  return bob::core::array::isEqual(m_Ps, b.m_Ps) &&
         m_variance_threshold == b.m_variance_threshold &&
         m_Pn == b.m_Pn &&
         bob::core::array::isEqual(m_W, b.m_W);
}

bool bob::ip::base::Wiener::operator!=(const bob::ip::base::Wiener& b) const{
  return !(this->operator==(b));
}

bool bob::ip::base::Wiener::is_similar_to(const bob::ip::base::Wiener& b, const double r_epsilon, const double a_epsilon) const{
  return bob::core::array::isClose(m_Ps, b.m_Ps, r_epsilon, a_epsilon) &&
         bob::core::isClose(m_variance_threshold, b.m_variance_threshold, r_epsilon, a_epsilon) &&
         bob::core::isClose(m_Pn, b.m_Pn, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_W, b.m_W, r_epsilon, a_epsilon);
}

void bob::ip::base::Wiener::load(bob::io::base::HDF5File& config){
  //reads all data directly into the member variables
  m_Ps.reference(config.readArray<double,2>("Ps"));
  m_Pn = config.read<double>("Pn");
  m_variance_threshold = config.read<double>("variance_threshold");
  m_W.reference(config.readArray<double,2>("W"));
  m_fft.setShape(m_Ps.extent(0),m_Ps.extent(1));
  m_ifft.setShape(m_Ps.extent(0),m_Ps.extent(1));
  m_buffer1.resize(m_Ps.shape());
  m_buffer2.resize(m_Ps.shape());
}

void bob::ip::base::Wiener::resize(const blitz::TinyVector<int,2>& size){
  m_Ps.resizeAndPreserve(size);
  m_W.resizeAndPreserve(size);
  m_fft.setShape(size[0], size[1]);
  m_ifft.setShape(size[0], size[1]);
  m_buffer1.resizeAndPreserve(size);
  m_buffer2.resizeAndPreserve(size);
}

void bob::ip::base::Wiener::save(bob::io::base::HDF5File& config) const{
  config.setArray("Ps", m_Ps);
  config.set("Pn", m_Pn);
  config.set("variance_threshold", m_variance_threshold);
  config.setArray("W", m_W);
}

void bob::ip::base::Wiener::train(const blitz::Array<double,3>& ar){
  // Data is checked now and conforms, just proceed w/o any further checks.
  const size_t n_samples = ar.extent(0);
  const size_t height = ar.extent(1);
  const size_t width = ar.extent(2);

  // resize with the new dimensions
  resize(blitz::TinyVector<int,2>(height, width));

  // Loads the data
  blitz::Array<double,3> data(height, width, n_samples);
  blitz::Array<std::complex<double>,2> sample_fft(height, width);
  blitz::Range all = blitz::Range::all();
  for (size_t i=0; i<n_samples; ++i) {
    blitz::Array<double,2> sample = ar(i,all,all);
    blitz::Array<std::complex<double>,2> sample_c = bob::core::array::cast<std::complex<double> >(sample);
    m_fft(sample_c, sample_fft);
    data(all,all,i) = blitz::abs(sample_fft);
  }
  // Computes the mean of the training data
  blitz::Array<double,2> tmp(height,width);
  blitz::thirdIndex k;
  tmp = blitz::mean(data,k);
  // Removes the mean from the data
  for (size_t i=0; i<n_samples; ++i) {
    data(all,all,i) -= tmp;
  }
  // Computes power of 2 values
  data *= data;
  // Sums to get the variance
  tmp = blitz::sum(data,k) / n_samples;

  // sets the Wiener filter with the results:
  setPs(tmp);
}

void bob::ip::base::Wiener::computeW(){
  // W = 1 / (1 + Pn / Ps_thresholded)
  m_W = 1. / (1. + m_Pn / m_Ps);
}


void bob::ip::base::Wiener::filter_(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const{
  m_fft(bob::core::array::cast<std::complex<double> >(input), m_buffer1);
  m_buffer1 *= m_W;
  m_ifft(m_buffer1, m_buffer2);
  output = blitz::abs(m_buffer2);
}

void bob::ip::base::Wiener::filter(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const{
  if (m_W.extent(0) != input.extent(0)) { //checks input
    boost::format m("number of input rows (%d) is not compatible with internal weight matrix (%d)");
    m % input.extent(0) % m_W.extent(0);
    throw std::runtime_error(m.str());
  }
  if (m_W.extent(1) != input.extent(1)) { //checks input
    boost::format m("number of input columns (%d) is not compatible with internal weight matrix (%d)");
    m % input.extent(1) % m_W.extent(1);
    throw std::runtime_error(m.str());
  }
  if (m_W.extent(0) != output.extent(0)) { //checks output
    boost::format m("number of output rows (%d) is not compatible with internal weight matrix (%d)");
    m % output.extent(0) % m_W.extent(0);
    throw std::runtime_error(m.str());
  }
  if (m_W.extent(1) != output.extent(1)) { //checks output
    boost::format m("number of output columns (%d) is not compatible with internal weight matrix (%d)");
    m % output.extent(1) % m_W.extent(1);
    throw std::runtime_error(m.str());
  }
  filter_(input, output);
}

void bob::ip::base::Wiener::setVarianceThreshold(const double variance_threshold){
  m_variance_threshold = variance_threshold;
  applyVarianceThreshold();
  computeW();
}

void bob::ip::base::Wiener::setPs(const blitz::Array<double,2>& Ps){
  if (m_Ps.extent(0) != Ps.extent(0)) {
    boost::format m("number of rows (%d) for input `Ps' does not match the expected (internal) size (%d)");
    m % Ps.extent(0) % m_Ps.extent(0);
    throw std::runtime_error(m.str());
  }
  if (m_Ps.extent(1) != Ps.extent(1)) {
    boost::format m("number of columns (%d) for input `Ps' does not match the expected (internal) size (%d)");
    m % Ps.extent(1) % m_Ps.extent(1);
    throw std::runtime_error(m.str());
  }
  m_Ps = Ps;
  computeW();
}

void bob::ip::base::Wiener::applyVarianceThreshold(){
  m_Ps = blitz::where(m_Ps < m_variance_threshold, m_variance_threshold, m_Ps);
}
