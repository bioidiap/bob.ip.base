import math
import numpy

def angle_to_horizontal(right, left):
  """angle_to_horizontal(right, left) -> angle

  Get the angle needed to level out (horizontally) two points.

  **Parameters**

  ``right``, ``left`` : *(float, float)*
    The two points to level out horizontically.

  **Returns**

  angle : *float*
    The angle **in degrees** between the left and the right point
  """

  return math.atan2(right[0] - left[0], right[1] - left[1]) * 180. / math.pi


def flip(src, dst = None):
  """flip(src, [dst]) -> dst

  Flip a 2D or 3D array/image upside-down.
  If given, the destination array ``dst`` should have the same size and type as the source array.

  **Parameters**

  ``src`` : *array_like (2D or 3D)*
    The source image to flip.

  ``dst`` : *array_like (2D or 3D)*
    If given, the destination to flip ``src`` to.

  **Returns**

  ``dst`` : *array_like (2D or 3D)*
    The flipped image
  """
  if dst is None:
    dst = numpy.ndarray(src.shape, src.dtype)
  dst[...,:,:] = src[...,::-1,:]
  return dst


def flop(src, dst = None):
  """flop(src, [dst]) -> dst

  Flip a 2D or 3D array/image left-right.
  If given, the destination array ``dst`` should have the same size and type as the source array.

  **Parameters**

  ``src`` : *array_like (2D or 3D)*
    The source image to flip.

  ``dst`` : *array_like (2D or 3D)*
    If given, the destination to flip ``src`` to.

  **Returns**

  ``dst`` : *array_like (2D or 3D)*
    The flipped image
  """
  if dst is None:
    dst = numpy.ndarray(src.shape, src.dtype)
  dst[...,:] = src[...,::-1]
  return dst


def crop(src, crop_offset, crop_size = None, dst = None, src_mask = None, dst_mask = None, fill_pattern = 0):
  """crop(src, crop_offset, crop_size, [dst], [src_mask], [dst_mask], [fill_pattern]) -> dst

  Crops the given image ``src`` image to the given offset (might be negative) and to the given size (might be greater than ``src`` image).

  Either crop_size or dst need to be specified.
  When masks are given, the need to be of the same size as the ``src`` and ``dst`` parameters.
  When crop regions are outside the image, the cropped image will contain ``fill_pattern`` and the mask will be set to ``False``

  **Parameters**

  ``src`` : *array_like (2D or 3D)*
    The source image to flip.

  ``crop_offset`` : *(int, int)*
    The position in ``src`` coordinates to start cropping; might be negative

  ``crop_size`` : *(int, int)*
    The size of the cropped image; might be omitted when the ``dst`` is given

  ``dst`` : *array_like (2D or 3D)*
    If given, the destination to crop ``src`` to.

  ``src_mask``, ``dst_mask``: *array_like(bool, 2D or 3D)*
    Masks that define, where ``src`` and ``dst`` are valid

  ``fill_pattern``: *number*
    [default: 0] The value to set outside the croppable area

  **Returns**

  ``dst`` : *array_like (2D or 3D)*
    The cropped image
  """
  # check parameters
  if dst is None:
    dst = numpy.ndarray(src.shape[:-2] + crop_size, src.dtype)
  elif crop_size is None:
    crop_size = dst.shape[-2:]
  else:
    assert crop_size == dst.shape[-2:]

  # get the real borders of both the source and the destination image
  src_shape = src.shape[-2:]
  src_offset = [max(crop_offset[i], 0) for i in range(2)]
  src_size = [min(crop_size[i] + crop_offset[i], src_shape[i]) for i in range(2)]
  dst_offset = [max(-crop_offset[i], 0) for i in range(2)]
  dst_size = [min(dst_offset[i] + crop_size[i], src_size[i] + dst_offset[i] - src_offset[i]) for i in range(2)]

  # copy data
  dst.fill(fill_pattern)
  dst[...,dst_offset[0]:dst_size[0], dst_offset[1]:dst_size[1]] = src[...,src_offset[0]:src_size[0], src_offset[1]:src_size[1]]

  # set mask data
  if src_mask is not None and dst_mask is not None:
    assert src_mask.shape == src.shape
    assert dst_mask.shape == dst.shape
    dst_mask.fill(False)
    dst_mask[...,dst_offset[0]:dst_size[0], dst_offset[1]:dst_size[1]] = src_mask[...,src_offset[0]:src_size[0], src_offset[1]:src_size[1]]

  return dst


def shift(src, offset, dst = None, src_mask = None, dst_mask = None, fill_pattern = 0):
  """shift(src, offset, [dst], [src_mask], [dst_mask], [fill_pattern]) -> dst

  Shifts the given image ``src`` image with the given offset (might be negative).

  If ``dst`` is specified, the image is shifted into the ``dst`` image.
  Ideally, ``dst`` should have the same size as ``src``, but other sizes work as well.
  When ``dst`` is ``None`` (the default), it is created in the same size as ``src``.
  When masks are given, the need to be of the same size as the ``src`` and ``dst`` parameters.
  When shift to regions are outside the image, the shifted image will contain ``fill_pattern`` and the mask will be set to ``False``

  **Parameters**

  ``src`` : *array_like (2D or 3D)*
    The source image to flip.

  ``crop_offset`` : *(int, int)*
    The position in ``src`` coordinates to start cropping; might be negative

  ``crop_size`` : *(int, int)*
    The size of the cropped image; might be omitted when the ``dst`` is given

  ``dst`` : *array_like (2D or 3D)*
    If given, the destination to crop ``src`` to.

  ``src_mask``, ``dst_mask``: *array_like(bool, 2D or 3D)*
    Masks that define, where ``src`` and ``dst`` are valid

  ``fill_pattern``: *number*
    [default: 0] The value to set outside the croppable area

  **Returns**

  ``dst`` : *array_like (2D or 3D)*
    The cropped image
  """
  # check parameters
  if dst is None:
    dst = numpy.ndarray(src.shape, src.dtype)

  # shift image by cropping
  return crop(src, offset, dst=dst, src_mask=src_mask, dst_mask=dst_mask, fill_pattern=fill_pattern)


def block_generator(input, block_size, block_overlap=(0, 0)):
  """Performs a block decomposition of a 2D or 3D array/image

  It works exactly as :any:`bob.ip.base.block` except that it yields the blocks
  one by one instead of concatenating them. It also works with color images.

  Parameters
  ----------
  input : :any:`numpy.ndarray`
      A 2D array (Height, Width) or a color image (Bob format: Channels,
      Height, Width).
  block_size : (:obj:`int`, :obj:`int`)
      The size of the blocks in which the image is decomposed.
  block_overlap : (:obj:`int`, :obj:`int`), optional
      The overlap of the blocks.

  Yields
  ------
  array_like
      A block view of the image. Modifying the blocks will change the original
      image as well. This is different from :any:`bob.ip.base.block`.

  Raises
  ------
  ValueError
      If the block_overlap is not smaller than block_size.
      If the block_size is bigger than the image size.
  """
  block_h, block_w = block_size
  overlap_h, overlap_w = block_overlap
  img_h, img_w = input.shape[-2:]

  if overlap_h >= block_h or overlap_w >= block_w:
    raise ValueError(
        "block_overlap: {} must be smaller than block_size: {}.".format(
            block_overlap, block_size))
  if img_h < block_h or img_w < block_w:
    raise ValueError(
        "block_size: {} must be smaller than the image size: {}.".format(
            block_size, input.shape[-2:]))

  # Determine the number of block per row and column
  size_ov_h = block_h - overlap_h
  size_ov_w = block_w - overlap_w

  # Perform the block decomposition
  for h in range(0, img_h - block_h + 1, size_ov_h):
    for w in range(0, img_w - block_w + 1, size_ov_w):
      yield input[..., h: h + block_h, w: w + block_w]
