"""Instrumental resolution of CAMEA (the old DataSet.calculateResolutionMatrix and
Geometry.Instrument.calculateResoultionMatrix): the analytical 4D resolution of a three-axis
spectrometer with focusing monochromator and analysers, C. Eckold and O. Sobolev, NIM A 752
(2014) 54, doi:10.1016/j.nima.2014.03.019 (Geometry.eck_Flipped.calc_eck).

Only CAMEA's geometry is described here; other instruments are not supported yet.
"""
import numpy as np

from MJOLNIR import _tools
from MJOLNIR import TasUBlibDEG as TasUBlib
from MJOLNIR.Geometry.eck_Flipped import (get_E, get_scattering_angle, get_mono_angle,
                                          get_angle_ki_Q, get_angle_kf_Q, calc_eck)

# Sample to analyser distance of each CAMEA analyser [m], lowest Ef first
CAMEA_ANALYSER_DISTANCES = np.array([0.9300, 0.9939, 1.0569, 1.1195, 1.1827, 1.2456, 1.3098,
                                     1.3747])

def _a3(qx: float, qy: float, Ei: float, Ef: float, a3_offset: float) -> float:
    """Sample rotation A3 [degrees] measuring Qx, Qy (Geometry.Instrument.converterToA3A4)"""
    direction = np.array([qx, qy, 0.0])/np.linalg.norm([qx, qy])
    rotation = np.linalg.inv(TasUBlib.buildTVMatrix(direction, np.array([0.0, 0.0, 1.0])))
    cos_sgl = np.sqrt(rotation[0, 0]**2 + rotation[1, 0]**2)
    omega = TasUBlib.arctan2d(rotation[1, 0]/cos_sgl, rotation[0, 0]/cos_sgl)
    ki, kf = np.sqrt(Ei)*_tools.factorsqrtEK, np.sqrt(Ef)*_tools.factorsqrtEK
    cos_2theta = (ki**2 + kf**2 - qx**2 - qy**2)/(2.0*ki*kf)
    A4 = TasUBlib.arccosd(cos_2theta)
    return -omega - TasUBlib.calcTheta(ki, kf, A4) + a3_offset

def camea_resolution_matrix(qx: float, qy: float, Ei: float, Ef: float, analyser: int,
                            a3_offset: float = 0.0) -> np.ndarray:
    """Resolution matrix M (4 x 4, in Qx, Qy, Qz [1/AA] and energy [meV]) at Qx, Qy for Ei and Ef
    measured by analyser (0 to 7): a point at distance x from the centre has probability
    exp(-x M x / 2). a3_offset rotates Qx, Qy as the sample's (sample.theta)"""
    if not 0 <= analyser < len(CAMEA_ANALYSER_DISTANCES):
        raise ValueError(f'CAMEA has analysers 0 to {len(CAMEA_ANALYSER_DISTANCES) - 1}; got {analyser}')
    cm2A = 1e8
    min2rad = 1.0/60.0/180.0*np.pi
    d_mono = d_ana = 3.355
    ki, kf = np.sqrt(Ei)*TasUBlib.factorsqrtEK, np.sqrt(Ef)*TasUBlib.factorsqrtEK
    Q = np.linalg.norm([qx, qy])
    params = {
        # scattering triangle
        'ki': ki, 'kf': kf, 'E': get_E(ki, kf), 'Q': Q,
        # angles
        'twotheta': get_scattering_angle(ki, kf, Q), 'thetam': get_mono_angle(ki, d_mono),
        'thetaa': get_mono_angle(kf, d_ana), 'angle_ki_Q': get_angle_ki_Q(ki, kf, Q),
        'angle_kf_Q': get_angle_kf_Q(ki, kf, Q),
        # scattering senses
        'mono_sense': 1.0, 'sample_sense': -1.0, 'ana_sense': 1.0,
        # distances
        'dist_src_mono': 120.0*cm2A, 'dist_mono_sample': 160.0*cm2A,
        'dist_sample_ana': CAMEA_ANALYSER_DISTANCES[analyser]*100.0*cm2A,
        'dist_ana_det': 70.0*cm2A,
        # component sizes
        'src_w': 2.0*cm2A, 'src_h': 3.0*cm2A, 'mono_w': 26.0*cm2A, 'mono_h': 17.0*cm2A,
        'det_w': 1.27*cm2A, 'det_h': 1.0*cm2A, 'ana_w': 5.0*cm2A, 'ana_h': 5.0*cm2A,
        # focusing
        'mono_curvh': 1.0, 'mono_curvv': 1.0, 'ana_curvh': 1.0, 'ana_curvv': 0.0,
        'mono_is_optimally_curved_h': True, 'mono_is_optimally_curved_v': True,
        'ana_is_optimally_curved_h': True, 'ana_is_optimally_curved_v': False,
        'mono_is_curved_h': True, 'mono_is_curved_v': True, 'ana_is_curved_h': True,
        'ana_is_curved_v': False,
        # collimation
        'coll_h_pre_mono': 9999.0*min2rad, 'coll_v_pre_mono': 9999.0*min2rad,
        'coll_h_pre_sample': 9999.0*min2rad, 'coll_v_pre_sample': 9999.0*min2rad,
        'coll_h_post_sample': 9999.0*min2rad, 'coll_v_post_sample': 60.0*min2rad,
        'coll_h_post_ana': 9999.0*min2rad, 'coll_v_post_ana': 9999.0*min2rad,
        # guide
        'use_guide': True, 'guide_div_h': 120.0*min2rad, 'guide_div_v': 120.0*min2rad,
        # mosaics
        'mono_mosaic': 42.0*min2rad, 'mono_mosaic_v': 42.0*min2rad,
        'ana_mosaic': 42.0*min2rad, 'ana_mosaic_v': 42.0*min2rad,
        # crystal reflectivities (always 1 so far)
        'dmono_refl': 1.0, 'dana_effic': 1.0,
        # off-centre scattering (calculated, not used in the ellipses)
        'pos_x': 0.0, 'pos_y': 0.0, 'pos_z': 0.0,
    }
    M = calc_eck(params)['reso']
    # From the scattering triangle's frame to Qx, Qy: a rotation by -A3 in the plane
    rotation = np.eye(4)
    rotation[:2, :2] = _tools.Rot(-_a3(qx, qy, Ei, Ef, a3_offset)).reshape(2, 2)
    return rotation @ M @ rotation.T

def project(M: np.ndarray, direction1, direction2) -> tuple[np.ndarray, np.ndarray]:
    """The resolution ellipse of M in the plane of two directions (4D, in Qx, Qy, Qz and energy,
    of unit length), integrated over the others: its axes (eigenvectors as columns, in the
    plane's coordinates) and widths (sigma along each axis)"""
    P = np.zeros((3, 5))
    P[0, :-1] = direction1
    P[1, :-1] = direction2
    P[-1, -1] = -1
    Q = np.diag([0.0, 0.0, 0.0, 0.0, -1.0])
    Q[:4, :4] = M
    C = np.linalg.inv(P @ np.linalg.inv(Q) @ P.T)
    eigenvalues, eigenvectors = np.linalg.eig(C[:2, :2])
    return eigenvectors, np.power(eigenvalues, -0.5)
