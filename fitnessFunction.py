import numpy as np
from forwKinematicsModel import ForwKinematicsModel
from numba import njit
import math


class FittnessFunction(ForwKinematicsModel):
    def __init__(self, constant_registers, nr_variable_registers, nr_datapoints, datapoint_registers = []) -> None:
        super().__init__()

        # Boundaries
        # Table is the restriction in z-axis
        self.max_pos = np.array([
            np.sqrt((self.L[6] + self.L[8] + self.L[9]) ** 2 + self.L[4]**2),  # x
            np.sqrt((self.L[6] + self.L[8] + self.L[9]) ** 2 + self.L[4]**2),  # y
            self.L[1] + self.L[2] + self.L[3] + \
            self.L[8] + self.L[9]  # z
            ])  # meter
        
        self.min_pos = np.array([
            -(-self.L[1] + self.L[9]),  # x
            -np.sqrt((self.L[6] + self.L[8] + self.L[9]) ** 2 + self.L[4]**2),  # y
            self.L[2] + self.L[3] - self.L[8] - self.L[9]  # z
            ])  # meter
        
        self.max_angle = np.array([np.pi, np.pi, np.pi/2])
        self.min_angle = np.array([0, 0, 0])

        # Registers & operators
        assert(nr_variable_registers >= 3)

        self.nr_datapoints = nr_datapoints
        one_register = np.concatenate((np.zeros(nr_variable_registers), np.array(constant_registers)))
        self.registers = np.array([one_register, ]*nr_datapoints)
        
        self.nr_registers = self.registers.shape[1]
        self.nr_variable_registers = nr_variable_registers
        self.nr_constant_registers = len(constant_registers)
        
        if len(datapoint_registers) > 1:
            self.registers = datapoint_registers
        else:
            self.randomize_input_pos()

    def randomize_input_pos(self):
        variable_registers = np.random.rand(self.nr_datapoints, self.nr_variable_registers)

        # restrict input
        variable_registers[:, :3] = self.min_pos + variable_registers[:, :3]*(self.max_pos - self.min_pos)
        
        # restrict other values
        min_max = 5
        variable_registers[:, 3:] = -min_max + variable_registers[:, 3:]*(min_max - (-min_max))

        self.registers[:,:self.nr_variable_registers] = variable_registers
    
    def decode_cromosome(self, chromosome):
        registers_all = np.copy(self.registers)
        angles = decode_cromosome_numba(chromosome, registers_all, self.max_angle)
        if np.isnan(angles).any():
            np.save(f"result/error_register.npy", self.registers)
            np.save(f"result/error_cromosome.npy", chromosome)
        assert(not np.isnan(angles).any())
        return angles

    def get_error(self, pred_angles):
        real_pos = np.copy(self.registers[:, 0:3])
        pred_pos = self.calc_pos(pred_angles)

        error = np.linalg.norm(real_pos - pred_pos, axis=1)
        error[error == 0] = 1/1000000

        return error

@njit
def decode_cromosome_numba(chromosome, registers_all, max_angle):
    len_cromosome = len(chromosome)
    for registers in registers_all:
        for gene_i in range(0, len_cromosome-1, 4):
            operator_f = int(chromosome[gene_i])
            destination_i = int(chromosome[gene_i+1])
            operand1 = registers[int(chromosome[gene_i+2])]
            if operand1 > 1000000.0:
                operand1 = 1000000.0
            elif operand1 < -1000000.0:
                operand1 = -1000000.0
            operand2 = registers[int(chromosome[gene_i+3])]
            if operand2 > 1000000.0:
                operand2 = 1000000.0
            elif operand2 < -1000000.0:
                operand2 = -1000000.0
            if operator_f == 0:
                registers[destination_i] = operand1 + operand2
            elif operator_f == 1:
                registers[destination_i] = operand1 - operand2
            elif operator_f == 2:
                registers[destination_i] = operand1 * operand2
            elif operator_f == 3:
                if operand2 == 0:
                    registers[destination_i] = 1000000
                else:
                    registers[destination_i] = operand1 / operand2
            elif operator_f == 4:
                registers[destination_i] = np.cos(operand1)
            elif operator_f == 5:
                registers[destination_i] = np.sin(operand1)
            elif operator_f == 6:
                registers[destination_i] = np.arcsin(operand1%1)
            elif operator_f == 7:
                registers[destination_i] = np.arccos(operand1%1)

    return np.absolute(np.remainder(registers_all[:, :3], max_angle))

def test_class():
    print("Testing class: fitness_function()")

    # Test settings
    nr_datapoints = 4
    r_one = np.array([0.45526203,  0.694249,   0.90758016,
                 0.55059288,  1, - 1, 3.14159265])
    r = np.array([r_one, ]*nr_datapoints)
    chromosome = np.array([3., 0., 4., 2., 1., 3., 4., 1.])

    #### TEST: CALCULATION OF ANGLES ####

    # Using fittness_function
    constant_registers = r[0, -3:]
    nr_variable_registers = r.shape[1]-len(constant_registers)
    ff = FittnessFunction(constant_registers=constant_registers,
                        nr_variable_registers=nr_variable_registers,
                        nr_datapoints=nr_datapoints)
    ff.registers = np.copy(r)
    ff_angles = ff.decode_cromosome(chromosome)

    # + - * /
    # manual calculations
    actual = np.array(r_one[0:3])

    r_one[0] = r_one[4] / r_one[2]
    r_one[3] = r_one[4] - r_one[1]

    manual_angles = np.array([r_one[0], r_one[1], r_one[2]])
    max_angle = np.array([np.pi, np.pi, np.pi/2])
    min_angle = np.array([0, 0, 0])
    manual_angles = np.minimum(np.maximum(manual_angles, min_angle), max_angle)

    # Assert
    assert (ff_angles[:, 0] == manual_angles[0]).all(), f"Test angle 1: {manual_angles[0]} is not {ff_angles[:, 0]}"
    assert (ff_angles[:, 1] == manual_angles[1]).all(), f"Test angle 2: {manual_angles[1]} is not {ff_angles[:, 1]}"
    assert (ff_angles[:, 2] == manual_angles[2]).all(), f"Test angle 3: {manual_angles[2]} is not {ff_angles[:, 2]}"
    #### TEST: CALCULATION OF ERROR ####

    # Calc Error using fittness function
    ff_error = ff.get_error(ff_angles)

    # Calc Error manual
    fkm = ForwKinematicsModel()
    manual_positions = fkm.calc_pos(np.array([manual_angles]))
    manual_error = float(np.linalg.norm(actual - manual_positions))
    # Assert error
    assert (ff_error == manual_error).all(), f"Test error: {manual_error} is not {ff_error}"

    print("All test passed!")

if __name__=="__main__":
    test_class()
