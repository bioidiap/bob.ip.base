import math


def get_angle_to_horizontal(right, left):
  """get_angle_to_horizontal(right, left) -> angle

  Get the angle needed to level out (horizontally) two points.

  **Parameters**

  ``right``, ``left`` : *(float, float)*
    The two points to level out horizontically.

  **Returns**

  angle : *float*
    The angle **in degrees** between the left and the right point
  """

  return math.atan2(right[0] - left[0], right[1] - left[1]) * 108. / math.pi



