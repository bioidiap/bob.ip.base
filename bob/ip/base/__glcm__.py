from ._library import _GLCM, GLCMProperty

# In this file, we use some python magic to add the properties as functions

# define the class
class GLCM (_GLCM):
  def __init__(self, *args, **kwargs):
    _GLCM.__init__(self, *args, **kwargs)

  # copy the documentation from the base class
  __doc__ = _GLCM.__doc__


import sys
func = ".__func__" if sys.version_info[0] < 3 else ""

# for each property, add a function that returns the property
# internally, this function will simply call the properties_by_name function of the C++ code
for prop, enum in GLCMProperty.entries.items():
  exec("def %s(self, input): return self.properties_by_name(input, [%d])[0]" % (prop, enum))
  exec("GLCM.%s = %s" % (prop, prop))
  exec("GLCM.%s%s.__doc__ = '''%s(input) -> property\n\nComputes the %s property\n\n**Parameters**\n\n``input`` : *array_like (3D, float)*\n\n  The result of the :py:func:`extract` function\n\n**Returns**\n\n``property`` :*array_like (1D, float)*\n\n  The resulting '%s' property'''" % (prop, func, prop, prop, prop))
  exec("del %s" % prop)

