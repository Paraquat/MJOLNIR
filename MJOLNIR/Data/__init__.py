"""This is the data module inteded to be used as a stand-alone data reduction and visualization.
"""
try: # Imports needed for Windows generation of installers of MJOLNIRGui
    import numpy.random.common
    import numpy.random.bounded_integers
    import numpy.random.entropy  # noqa: F401  needed for MJOLNIRGui Windows installer
except ImportError:
    pass
