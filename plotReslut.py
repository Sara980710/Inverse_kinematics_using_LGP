import numpy as np
from matplotlib import pyplot as plt
from fitnessFunction import FittnessFunction
from scipy.io import loadmat


def plot_result(chromosome, constant_registers, variable_registers, nr_datapoints):

    # Calculate fitness
    ff = FittnessFunction(constant_registers,
                          variable_registers, nr_datapoints)

    # Plot positions
    predicted_angles = ff.decode_cromosome(np.copy(chromosome))
    pred_pos = ff.calc_pos(predicted_angles)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ff.registers[:, 0], ff.registers[:, 1],
               ff.registers[:, 2], marker='o', label="original")
    ax.scatter(pred_pos[:, 0], pred_pos[:, 1],
               pred_pos[:, 2], marker='o', label="predicted")
    ax.legend()
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.set_title(f"Positions")

    # Plot error
    diff = ff.get_error(predicted_angles)
    diff.sort()


    fig, ax = plt.subplots()
    ax.scatter(range(len(diff)), diff, s=5)
    ax.set_xlabel('datapoints')
    ax.set_ylabel('error')
    ax.set_title(f"Max diff = {np.max(diff)}, {int(sum(i < 0.1 for i in diff)/len(diff)*100)}% under 0.1")

    plt.show()


if __name__=="__main__":
    folder = "results"
    run_nr = "7"
    chromosome = loadmat(f"{folder}/{run_nr}_chromosome.mat")["data"][0]
    constant_registers = loadmat(
        f"{folder}/{run_nr}_original_registers.mat")["constant_registers"][0]
    variable_registers = loadmat(
        f"{folder}/{run_nr}_original_registers.mat")["variable_registers"][0]

    plot_result(chromosome, constant_registers, variable_registers, 1000)
