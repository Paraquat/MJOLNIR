"""Conversion between reciprocal lattice units (H, K, L) and the instrument frame (Qx, Qy)
with the UB matrix of a sample"""
import copy
import numpy as np

def convert_to_qxqy(sample, hkl) -> np.ndarray:
    """Convert HKL point(s), shape (3,) or (..., 3), to Qx, Qy with the UB matrix of sample"""
    hkl = np.asarray(hkl)
    if hkl.shape[-1] != 3:
        raise AttributeError(f'Provided HKL point is not 3D. Received: {hkl}')
    qx, qy, qz = np.einsum('ij,...j->i...', sample.orientationMatrix, hkl)
    return np.array([qx, qy]).T

def convert_to_hkl(sample, qxqy) -> np.ndarray:
    """Convert Qx, Qy point(s), shape (2,) or (..., 2), to HKL with the UB matrix of sample"""
    qxqy = np.asarray(qxqy)
    if qxqy.shape[-1] != 2:
        raise AttributeError(f'Provided QxQy point is not 2D. Received: {qxqy}')
    qxqyqz = np.concatenate([qxqy, np.zeros((*qxqy.shape[:-1], 1))], axis=-1)
    h, k, l = np.einsum('ij,...j->i...', sample.orientationMatrixINV, qxqyqz)
    return np.array([h, k, l]).T

def align_to_lattice(sample):
    """Copy of sample in the frame rotated by its RotMat, in which Qx, Qy follow the reciprocal
    lattice (the frame of RLUAxes.createQAxis, and of DataSet.gather_q(align_to_lattice=True))"""
    aligned = copy.deepcopy(sample)
    aligned.convert = np.einsum('ij,j...->i...', aligned.RotMat, aligned.convert)
    aligned.convertinv = np.linalg.inv(aligned.convert)
    aligned.orientationMatrix = np.dot(aligned.RotMat3D, aligned.orientationMatrix)
    aligned.orientationMatrixINV = np.linalg.inv(aligned.orientationMatrix)
    aligned.theta = 0.0
    return aligned
