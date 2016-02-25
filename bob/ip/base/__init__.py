# import Libraries of other lib packages
import bob.io.base
import bob.sp
import bob.math

# import our own Library
import bob.extension
bob.extension.load_bob_library('bob.ip.base', __file__)

from ._library import *
from .__glcm__ import *
from .__functions__ import *

from . import version
from .version import module as __version__
from .version import api as __api_version__


def get_config():
  """Returns a string containing the configuration information.
  """
  return bob.extension.get_config(__name__, version.externals, version.api)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
