import numpy as np
from forwKinematicsModel import ForwKinematicsModel
from numba import njit


class FittnessFunction(ForwKinematicsModel):
    def __init__(self, constant_registers, variable_registers, nr_datapoints) -> None:
        super().__init__()

        # Boundaries
        self.sphere_center_upper = np.array([0, 0, self.L[2] + self.L[3]])
        self.sphere_radius_outer = np.sqrt((self.L[6] + self.L[8] + self.L[9]) ** 2 + (self.L[4]-self.L[5])**2)
        self.sphere_radius_inner_upper = np.sqrt(np.sqrt(self.L[8] ** 2 + (self.L[7] + self.L[9])**2) ** 2 + (self.L[4]-self.L[5])**2)
        self.sphere_center_lower = np.array([0, 0, self.L[2] + self.L[3] - self.L[8]])
        self.sphere_radius_inner_lower = np.sqrt(self.L[9] **2 + (self.L[4] - self.L[5]) ** 2)
        self.upper_lower_limit = self.L[2] + self.L[3] - self.L[8]

        self.max_pos = np.array([
            self.sphere_radius_outer,  # x
            self.sphere_radius_outer,  # y
            self.L[2] + self.L[3] + self.L[8] + self.L[9]  # z
            ])  # meter
        
        self.min_pos = np.array([
            -(self.L[6] + self.L[8] + self.L[9]),  # x
            self.L[6] - self.L[7] - self.L[9],  # y
            self.L[2] + self.L[3] - self.L[8] - self.L[9]  # z
            ])  # meter
        
        self.max_angle = np.array([np.pi, np.pi, np.pi/2])
        self.min_angle = np.array([0, 0, 0])

        # Registers & operators
        self.nr_datapoints = nr_datapoints
        self.constant_registers = constant_registers
        one_register = np.concatenate((np.zeros(3), variable_registers, np.array(constant_registers)))
        self.registers = np.array([one_register, ]*nr_datapoints)

        self.nr_registers = self.registers.shape[1]
        self.nr_variable_registers = self.nr_registers - \
            len(self.constant_registers)
        self.randomize_input_pos()
        
        assert(self.nr_variable_registers >= 3)

    def randomize_input_pos(self):
        # restrict input values (positions)
        points = self.min_pos + np.random.rand(self.nr_datapoints, 3)*(self.max_pos - self.min_pos)

        value_outer = (self.sphere_radius_outer ** 2)
        value_inner_upper = (self.sphere_radius_inner_upper ** 2)
        value_inner_lower = (self.sphere_radius_inner_lower ** 2)
        for i,point in enumerate(points):
            value_1 = np.sum(np.square(point - self.sphere_center_upper))
            value_2 = np.sum(np.square(point - self.sphere_center_lower))
                    
            while (value_1 > value_outer) or \
                    ((point[2] > self.upper_lower_limit) and (value_1 < value_inner_upper)) or \
                    ((point[2] <= self.upper_lower_limit) and (value_2 < value_inner_lower)):
                       
                point = self.min_pos + \
                    np.random.rand(3)*(self.max_pos - self.min_pos)
                value_1 = np.sum(
                    np.square(point - self.sphere_center_upper))
                value_2 = np.sum(
                    np.square(point - self.sphere_center_lower))
            points[i] = point

        self.registers[:, :3] = points
    
    def decode_cromosome(self, chromosome):
        angles = decode_cromosome_numba(chromosome, np.copy(self.registers), self.max_angle)
        
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
            # Get instruction data
            operator_f = int(chromosome[gene_i])
            destination_i = int(chromosome[gene_i+1])
            operand1 = registers[int(chromosome[gene_i+2])]
            operand2 = registers[int(chromosome[gene_i+3])]

            # Restrict
            if operand1 > 1000000.0:
                operand1 = 1000000.0
            elif operand1 < -1000000.0:
                operand1 = -1000000.0
            if operand2 > 1000000.0:
                operand2 = 1000000.0
            elif operand2 < -1000000.0:
                operand2 = -1000000.0

            # Perform instruction
            if operator_f == 0:
                result = operand1 + operand2
            elif operator_f == 1:
                result = operand1 - operand2
            elif operator_f == 2:
                result = operand1 * operand2
            elif operator_f == 3:
                if operand2 == 0:
                    result = 1000000.0
                else:
                    result = operand1 / operand2
            elif operator_f == 4:
                result = np.cos(operand1)
            elif operator_f == 5:
                result = np.sin(operand1)
            elif operator_f == 6:
                result = np.arccos((operand1 % 2)-1)
            elif operator_f == 7:
                result = np.arcsin((operand1 % 2)-1)

            # Update register
            registers[destination_i] = result

    # Return angles restricted to max_angles 
    return np.mod(registers_all[:, :3], max_angle)

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
