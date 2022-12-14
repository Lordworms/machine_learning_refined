#!/usr/bin/env python
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)
  
for importer, modname, ispkg in pkgutil.walk_packages(
        path=__path__,
        prefix=__name__+'.',
        onerror=(lambda x: None)):
    __import__(modname)
  

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')