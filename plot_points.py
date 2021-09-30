import numpy as np
from matplotlib import pyplot as plt
from fitnessFunction import FittnessFunction

#cromosome = np.load("cromosome.npy")
#original_values = np.load("original_registers.npy")
cromosome = np.load("result2/15_chromosome.npy")
original_values = np.load("result2/15_original_registers.npy")

ff = FittnessFunction(
    constant_registers=np.copy(original_values[:,1]),
    nr_variable_registers=original_values.shape[1],
    nr_datapoints=original_values.shape[0], 
    datapoint_registers = original_values)


predicted_angles = ff.decode_cromosome(np.copy(cromosome))
pred_pos = ff.calc_pos(predicted_angles)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
m = 'o'
ax.scatter(original_values[:, 0], original_values[:, 1], original_values[:,2], marker=m, label="original")
ax.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], marker=m, label="predicted")
ax.legend()
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
ax.set_title(f"Positions")

diff = ff.get_error(np.copy(predicted_angles))
diff.sort()


fig, ax = plt.subplots()
ax.scatter(range(len(diff)), diff, s=5)
ax.set_xlabel('datapoints')
ax.set_ylabel('error')
ax.set_title(f"Max diff = {np.average(diff)}")

plt.show()
