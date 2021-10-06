import numpy as np
from numba import njit

L = np.array(
    [0, 0.055, 0.315, 0.045, 0.108, 0.005, 0.034, 0.015, 0.088, 0.204])

class ForwKinematicsModel():
    def __init__(self) -> None:
        self.L = L
        self.frame3 = np.array([0.0, 0.0, 0.0, 1.0])

    def calc_pos(self, pred_angles):
        return calc_pos_numba(pred_angles, self.frame3)


@njit
def calc_pos_numba(pred_angles, frame3):
    pred_pos = np.zeros(pred_angles.shape)

    for i, angle_tripple in enumerate(pred_angles):
        trans2to0 = trans1to0(angle_tripple[0]) @ trans2to1(angle_tripple[1])
        trans3to0 = trans2to0 @ trans3to2(angle_tripple[2])
        pred_pos[i, :] = (trans3to0 @ frame3)[:-1]
    return pred_pos

@njit
def trans3to2(theta3):
    return np.array([
        [np.cos(theta3), -np.sin(theta3), 0.0, L[9]*np.sin(theta3)],
        [np.sin(theta3), np.cos(theta3), 0.0, -L[9]*np.cos(theta3)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
@njit
def trans2to1(theta2):
    return np.array([
        [np.cos(theta2), -np.sin(theta2), 0.0, L[7]*np.cos(theta2) + L[8]*np.sin(theta2)],
        [np.sin(theta2), np.cos(theta2), 0.0, L[7] * np.sin(theta2) - L[8]*np.cos(theta2)],
        [0.0, 0.0, 1.0, -L[5]],
        [0.0, 0.0, 0.0, 1.0]])

@njit
def trans1to0(theta1):
    return np.array([
        [np.cos(theta1), 0.0, np.sin(theta1), L[6]*np.cos(theta1) + L[4]*np.sin(theta1)],
        [np.sin(theta1), 0.0, -np.cos(theta1), L[6] * np.sin(theta1) - L[4]*np.cos(theta1)],
        [0.0, 1.0, 0.0, L[2]+L[3]],
        [0.0, 0.0, 0.0, 1.0]])
