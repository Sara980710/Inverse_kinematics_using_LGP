import numpy as np
from fitnessFunction import FittnessFunction


chromosome = np.load("result/error_cromosome.npy")
original_values = np.load("result/error_register.npy")

print(chromosome)
print(original_values)

ff = FittnessFunction(
    constant_registers=original_values[:, 1],
    nr_variable_registers=original_values.shape[1],
    nr_datapoints=original_values.shape[0])
ff.registers = original_values


predicted_angles = ff.decode_cromosome(chromosome)
