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

def test_function():
    print("Testing class: forw_kinematics_model()")

    fkm = ForwKinematicsModel()
    nr_datapoints = 5  # To handle multiple datapoints

    # Test 1
    one_array = [0, 0, 0]
    angles = np.array([one_array, ]*nr_datapoints)

    fkm_answer = np.round(fkm.calc_pos(angles), 5) 
    manual_answer = np.round(np.array(
        [fkm.L[6]+fkm.L[7],
        -fkm.L[4]+fkm.L[5],
        fkm.L[2]+fkm.L[3]-fkm.L[8]-fkm.L[9]]), 5)
    print(fkm_answer)

    assert (fkm_answer[:, 0] == manual_answer[0]).all(
    ), f"Test1 x: {manual_answer[0]} is not equal to values in {fkm_answer[:, 0]}"
    assert (fkm_answer[:, 1] == manual_answer[1]).all(
    ), f"Test1 y: {manual_answer[1]} is not equal to values in {fkm_answer[:, 1]}"
    assert (fkm_answer[:, 2] == manual_answer[2]).all(
    ), f"Test1 z: {manual_answer[2]} is not equal to values in {fkm_answer[:, 2]}"

    # Test 2
    one_array = [np.pi, 0, 0]
    angles = np.array([one_array, ]*nr_datapoints)

    fkm_answer = np.round(fkm.calc_pos(angles), 5)
    manual_answer = np.round(np.array(
        [-(fkm.L[6]+fkm.L[7]),
        fkm.L[4]-fkm.L[5],
        fkm.L[2]+fkm.L[3]-fkm.L[8]-fkm.L[9]]), 5)

    assert (fkm_answer[:, 0] == manual_answer[0]).all(
    ), f"Test2 x: {manual_answer[0]} is not equal to values in {fkm_answer[:, 0]}"
    assert (fkm_answer[:, 1] == manual_answer[1]).all(
    ), f"Test2 y: {manual_answer[1]} is not equal to values in {fkm_answer[:, 1]}"
    assert (fkm_answer[:, 2] == manual_answer[2]).all(
    ), f"Test2 z: {manual_answer[2]} is not equal to values in {fkm_answer[:, 2]}"

    # Test 3
    one_array = [0, np.pi, 0]
    angles = np.array([one_array, ]*nr_datapoints)

    fkm_answer = np.round(fkm.calc_pos(angles), 5)
    manual_answer = np.round(np.array(
        [fkm.L[6]-fkm.L[7],
        -fkm.L[4]+fkm.L[5],
        fkm.L[2]+fkm.L[3]+fkm.L[8]+fkm.L[9]]), 5)
                                    
    assert (fkm_answer[:, 0] == manual_answer[0]).all(
    ), f"Test3 x: {manual_answer[0]} is not equal to values in {fkm_answer[:, 0]}"
    assert (fkm_answer[:, 1] == manual_answer[1]).all(
    ), f"Test3 y: {manual_answer[1]} is not equal to values in {fkm_answer[:, 1]}"
    assert (fkm_answer[:, 2] == manual_answer[2]).all(
    ), f"Test3 z: {manual_answer[2]} is not equal to values in {fkm_answer[:, 2]}"

    # Test 4
    one_array = [0, 0, np.pi/2]
    angles = np.array([one_array, ]*nr_datapoints)

    fkm_answer = np.round(fkm.calc_pos(angles), 5)
    manual_answer = np.round(np.array(
        [fkm.L[6]+fkm.L[7]+fkm.L[9],
        -fkm.L[4]+fkm.L[5],
        fkm.L[2]+fkm.L[3]-fkm.L[8]]), 5)

    assert (fkm_answer[:, 0] == manual_answer[0]).all(
    ), f"Test4 x: {manual_answer[0]} is not equal to values in {fkm_answer[:, 0]}"
    assert (fkm_answer[:, 1] == manual_answer[1]).all(
    ), f"Test4 y: {manual_answer[1]} is not equal to values in {fkm_answer[:, 1]}"
    assert (fkm_answer[:, 2] == manual_answer[2]).all(
    ), f"Test4 z: {manual_answer[2]} is not equal to values in {fkm_answer[:, 2]}"

    # Test 5
    one_array = [np.pi/2, 0, 0]
    angles = np.array([one_array, ]*nr_datapoints)

    fkm_answer = np.round(fkm.calc_pos(angles), 5)
    manual_answer = np.round(np.array(
        [fkm.L[4]-fkm.L[5], 
         fkm.L[6] + fkm.L[7],
         fkm.L[2]+fkm.L[3]-fkm.L[8]-fkm.L[9]]), 5)

    assert (fkm_answer[:, 0] == manual_answer[0]).all(
    ), f"Test5 x: {manual_answer[0]} is not equal to values in {fkm_answer[:, 0]}"
    assert (fkm_answer[:, 1] == manual_answer[1]).all(
    ), f"Test5 y: {manual_answer[1]} is not equal to values in {fkm_answer[:, 1]}"
    assert (fkm_answer[:, 2] == manual_answer[2]).all(
    ), f"Test5 z: {manual_answer[2]} is not equal to values in {fkm_answer[:, 2]}"

    print("All test passed!")


def calculate_from_angle_set(angles):
    assert len(angles) == 3, f"Please give one set of angles in array, got: {angles}"

    fkm = ForwKinematicsModel()
    return fkm.calc_pos(np.array([angles]))[0]

if __name__=="__main__":
    test_function()

    angles = [np.pi/4, np.pi/3, np.pi/6]
    print(f"Calculating position in frame 0 with angle set: {angles}")
    print(f"Position [x,y,z]: {calculate_from_angle_set(angles)}")

