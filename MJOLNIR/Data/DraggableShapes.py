"""Cut parameters from shapes drawn on plots: rectangles and circles on a Q plane, rectangles on
a Q-E plot. Each function reads a matplotlib patch and returns keyword arguments, named as for
the DataSet cuts (cut_1d, cut_constant_q); the caller adds what the shape does not give (e.g.
the energy range of a Q plane). MJOLNIR-GUI uses them for its interactive cuts.

The draggable shapes themselves (moving, selecting and drawing them) are in
_interactiveSettings; this module had an older copy of them, which nothing used.

On a Q plane the shape is in Qx, Qy, and with a sample the positions are converted to H, K, L
(rlu). On a Q-E plot x is the position along the cut (0 at its start, 1 at its end, see
Plotting) and y the energy: positions are returned in those plot coordinates, for the caller
to convert with ax.calculatePosition."""
import numpy as np

def corner2center(x, y, width, angle):
    """Centre of the bottom edge of a rectangle with corner x, y, width and angle (degrees)"""
    angleRad = np.deg2rad(angle+90)
    # Unit vector along the height of the rectangle, and half the width orthogonal to it
    dirvec = np.array([np.cos(angleRad), np.sin(angleRad)])
    ortho = np.array([dirvec[1], -dirvec[0]])*width*0.5
    return np.array([x, y])+ortho

def center2corner(x, y, width, angle):
    """Corner of a rectangle from the centre of its bottom edge (inverse of corner2center)"""
    angleRad = np.deg2rad(angle+90)
    dirvec = np.array([np.cos(angleRad), np.sin(angleRad)])
    ortho = np.array([dirvec[1], -dirvec[0]])*width*0.5
    return np.array([x, y])-ortho

def _energy_range(rect, rounding):
    # The rectangle's height may be negative when drawn downwards
    y0 = rect.get_xy()[1]
    energies = [y0, y0+rect.get_height()]
    return {'energy_min': np.round(np.min(energies), rounding),
            'energy_max': np.round(np.max(energies), rounding)}

def extractCut1DPropertiesRectangle(rect, sample=None, rounding=4):
    """cut_1d parameters of a rectangle on a Q plane: from the centre of its bottom edge (q1)
    to the centre of its top edge (q2), its width as the cut width. The energy range is the
    plane's, for the caller to add"""
    xy = np.asarray(rect.get_xy(), dtype=float)
    width = rect.get_width()
    angleRad = np.deg2rad(rect.angle+90)
    # Unit vector along the height of the rectangle
    dirvec = np.array([np.cos(angleRad), np.sin(angleRad)])
    ortho = np.array([dirvec[1], -dirvec[0]])
    q1 = xy+ortho*0.5*width
    q2 = q1+dirvec*rect.get_height()
    if sample is not None:
        q1 = np.asarray(sample.calculateQxQyToHKL(*q1))
        q2 = np.asarray(sample.calculateQxQyToHKL(*q2))
    return {'q1': np.round(q1, rounding),
            'q2': np.round(q2, rounding),
            'width': np.round(np.abs(width), rounding),
            'min_pixel': 0.05,
            'rlu': sample is not None, # H, K, L only with a sample
            'constant_bins': False}

def extractCut1DPropertiesRectanglePerpendicular(rect, sample=None, rounding=4):
    """Parameters of a rectangle on a Q-E plot for a cut across the Q-E cut: its centre (plot
    coordinates), energy range and width (along the plot's x)"""
    xy = np.asarray(rect.get_xy(), dtype=float)
    center = xy+0.5*np.array([rect.get_width(), rect.get_height()])
    return {'center': np.round(center, rounding),
            **_energy_range(rect, rounding),
            'width': np.round(np.abs(rect.get_width()), rounding),
            'rlu': sample is not None,
            'constant_bins': False}

def extractCut1DPropertiesRectangleHorizontal(rect, sample=None, rounding=4):
    """Parameters of a rectangle on a Q-E plot for a cut along the Q-E cut: start (q1) and end
    (q2) at the middle of its height (plot coordinates), and its energy range"""
    xy = np.asarray(rect.get_xy(), dtype=float)
    width = rect.get_width()
    middle = 0.5*rect.get_height()
    return {'q1': np.round(xy+np.array([0.0, middle]), rounding),
            'q2': np.round(xy+np.array([width, middle]), rounding),
            **_energy_range(rect, rounding),
            'rlu': sample is not None,
            'constant_bins': False}

def extractCut1DPropertiesRectangleVertical(rect, sample=None, rounding=4):
    """cut_constant_q parameters of a rectangle on a Q-E plot: its centre along the plot's x
    (q, plot coordinate) and its energy range"""
    xy = rect.get_xy()
    return {'q': np.round(xy[0]+0.5*rect.get_width(), rounding),
            **_energy_range(rect, rounding),
            'rlu': sample is not None,
            'constant_bins': False}

def extractCut1DPropertiesCircle(circ, sample=None, rounding=4):
    """cut_constant_q parameters of a circle on a Q plane: its centre as q and its diameter as
    width (cut_constant_q's width is the diameter)"""
    center = np.asarray(circ.center, dtype=float)
    if sample is not None:
        center = np.asarray(sample.calculateQxQyToHKL(*center))
    return {'q': np.round(center, rounding),
            'width': np.round(2*np.abs(circ.radius), rounding),
            'rlu': sample is not None,
            'constant_bins': False}
