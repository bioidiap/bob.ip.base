/**
 * @author Manuel Günther <manuel.guenther@idiap.ch>
 * @date Thu Jun 26 09:33:10 CEST 2014
 *
 * This file defines functions and classes for affine image transformations
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */


#include <bob.ip.base/Affine.h>

static bool isTrue(const blitz::Array<bool,2>& mask, int y0, int x0, int y1, int x1)
{
  for(int j=y0; j<=y1; ++j)
    for(int i=x0; i<=x1; ++i)
      if(!mask(j,i))
        return false;
  return true;
}


const blitz::TinyVector<int,4> bob::ip::base::maxRectInMask(const blitz::Array<bool,2>& mask){
  const int height = mask.extent(0);
  const int width = mask.extent(1);
  blitz::TinyVector<int,4> cur_sol = 0;
  int cur_max_area = 0;

  for( int y0=0; y0<height; ++y0)
  {
    // Find first true value
    int i_min = 0;
    bool found_true = false;
    for(int x0=0; x0<width; ++x0)
      if(mask(y0,x0))
      {
        i_min=x0;
        found_true = true;
        break;
      }
    if(!found_true)
      continue;

    // Find last true value
    int i_max = width;
    for(int x0=i_min+1; x0<width; ++x0)
      if(!mask(y0,x0))
      {
        i_max = x0;
        break;
      }

    // Find largest rectangle visiting all the possible combinations
    for(int x0=i_min; x0<i_max; ++x0)
    {
      int cur_i_max = i_max;
      for(int y1=y0; y1<height; ++y1)
        for(int x1=x0; x1<cur_i_max; ++x1)
        {
          if(isTrue(mask,y1,x0,y1,x1))
          {
            int area = (y1-y0+1)*(x1-x0+1);
            if( area > cur_max_area )
            {
              cur_max_area = area;
              cur_sol(0) = y0;
              cur_sol(1) = x0;
              cur_sol(2) = y1-y0+1;
              cur_sol(3) = x1-x0+1;
            }
          }
          else
          {
            if(x1 < cur_i_max)
              cur_i_max = x1;
            break;
          }
        }
    }
  }

  return cur_sol;
}



