/**
 * @date Tue Apr 26 19:20:57 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago Freitas Pereira <Tiago.Pereira@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * This class can be used to calculate the LBP-Top  of a set of image frames
 * representing a video sequence (c.f. Dynamic Texture Recognition Using Local
 * Binary Patterns with an Application to Facial Expression from Zhao &
 * Pietikäinen, IEEE Trans. on PAMI, 2007). This is the implementation file.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <stdexcept>
#include <boost/format.hpp>
#include <bob.ip.base/LBPTop.h>

bob::ip::base::LBPTop::LBPTop(
    boost::shared_ptr<LBP> lbp_xy,
    boost::shared_ptr<LBP> lbp_xt,
    boost::shared_ptr<LBP> lbp_yt
):
  m_lbp_xy(lbp_xy),
  m_lbp_xt(lbp_xt),
  m_lbp_yt(lbp_yt)
{
  /*
   * Checking the inputs. The radius in XY,XT and YT must be the same
   * Note: radii are in opposity direction lbp_xy has radii stored in (y,x), aso.
   */
  if(lbp_xy->getRadii()[1]!=lbp_xt->getRadii()[1]) {
    throw std::runtime_error((boost::format("The X radii of R_xy (%f) and R_xt (%f) do not match")% lbp_xy->getRadii()[1] % lbp_xt->getRadii()[1]).str());
  }

  if(lbp_xy->getRadii()[0]!=lbp_yt->getRadii()[1]) {
    throw std::runtime_error((boost::format("The Y radii R_xy (%f) and R_yt (%f) do not match")% lbp_xy->getRadii()[0] % lbp_yt->getRadii()[1]).str());
  }

  if(lbp_xt->getRadii()[0]!=lbp_yt->getRadii()[0]) {
    throw std::runtime_error((boost::format("The T radii R_xt (%f) and R_yt (%f) do not match")% lbp_xt->getRadii()[0] % lbp_yt->getRadii()[0]).str());
  }

}

bob::ip::base::LBPTop::LBPTop(const LBPTop& other)
: m_lbp_xy(other.m_lbp_xy),
  m_lbp_xt(other.m_lbp_xt),
  m_lbp_yt(other.m_lbp_yt)
{
}

bob::ip::base::LBPTop::~LBPTop() { }

bob::ip::base::LBPTop& bob::ip::base::LBPTop::operator= (const LBPTop& other) {
  m_lbp_xy = other.m_lbp_xy;
  m_lbp_xt = other.m_lbp_xt;
  m_lbp_yt = other.m_lbp_yt;
  return *this;
}
