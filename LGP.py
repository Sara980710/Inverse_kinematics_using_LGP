import numpy as np
import random
from fitnessFunction import FittnessFunction
from matplotlib import pyplot as plt


class LGP(FittnessFunction):
    def __init__(self, population_size, nr_variable_registers, constant_registers,
                 nr_datapoints, max_len_cromosome, nr_operators) -> None:

        # Settings
        self.len_instruction = 4 
        self.min_len_cromosome = 8
        self.max_len_cromosome = max_len_cromosome
        self.nr_operators = nr_operators

        # Check
        # 4 genes per operation
        assert(max_len_cromosome % self.len_instruction == 0)
        assert(max_len_cromosome >= self.min_len_cromosome)
        assert(nr_variable_registers >=3) # for 3 angles/ 3 inputs
        assert(nr_operators <= 8)  # [+ - * / sin cos arcsin arccos]

        # Keep track
        self.generation = 0
        self.best_fitness = 0
        self.best_individual_i = 0
        self.best_error = 100000
        self.overall_best = 0

        # Function handler
        super().__init__(
            constant_registers = constant_registers,
            nr_variable_registers=nr_variable_registers,
            nr_datapoints=nr_datapoints,
            )

        # Polulation
        self.population_size = population_size
        self.population = self.init_population()

        

    def init_population(self):
        population = {}

        for individual in range(self.population_size):
            len_cromosome = random.choice(range(self.min_len_cromosome, self.max_len_cromosome + 1, 4))
            chromosome = np.zeros(len_cromosome)

            for gene_i in range(0,len_cromosome-1,4):
                chromosome[gene_i] = np.random.randint(0, self.nr_operators) # Operator
                chromosome[gene_i+1] = np.random.randint(0, self.nr_variable_registers) # Destination register
                chromosome[gene_i+2] = np.random.randint(0, self.nr_registers) # Operand 1
                chromosome[gene_i+3] = np.random.randint(0, self.nr_registers) # Operand 2

            population[individual] = {'chromosome':chromosome}

        return population

    def calculate_fitness(self, penalty_term, mode):

        for i,individual in self.population.items():
            # Calculate error
            predicted_angles = self.decode_cromosome(individual["chromosome"])
            error_all = self.get_error(predicted_angles)

            # Mode
            if mode == "max":
                error = np.max(error_all)
            elif mode == "average":
                error = np.average(error_all)
            
            # Calculate fitness
            fitness = 1/error

            if len(individual["chromosome"]) > self.max_len_cromosome:
                fitness = fitness*penalty_term

            individual["fitness"] = fitness

            # Save the best chromosome
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual_i = i
                self.best_error = error
                #if fitness > self.overall_best:
                    #print("----------------------NEW BEST!--------------------------")

    def tournament_selection(self, tournament_size, tournament_parameter):
        # Select random idnividuals 
        selected = np.zeros((2, tournament_size))
        selected[0, :] = np.random.randint(self.population_size, size=tournament_size)
        selected[1,:] = [self.population[i]["fitness"] for i in selected[0, :]]

        # Sort them ascending 
        selected = selected[:, selected[1, :].argsort()] 

        # Iterate backwards
        # and return the selected individual's index
        for i in range(selected.shape[1]-1, 0, -1):
            if np.random.uniform() <= tournament_parameter:
                return selected[0,i]
        return selected[0, 0] 
        
    def two_point_crossover(self, ind1, ind2):
        old_ind_1 = np.copy(self.population[ind1]["chromosome"])
        old_ind_2 = np.copy(self.population[ind2]["chromosome"])

        nr_instructions_ind1 = len(old_ind_1)/4
        nr_instructions_ind2 = len(old_ind_2)/4

        # Chose the crossover points
        points_ind1 = np.sort(np.random.randint(nr_instructions_ind1, size=2))*4
        points_ind2 = np.sort(np.random.randint(nr_instructions_ind2, size=2))*4

        # Create new individuals by swapping the middle parts
        new_individual1 = np.concatenate((old_ind_1[:points_ind1[0]], 
                                         old_ind_2[points_ind2[0]:points_ind2[1]],
                                         old_ind_1[points_ind1[1]:]), axis = None)

        new_individual2 = np.concatenate((old_ind_2[:points_ind2[0]], 
                                         old_ind_1[points_ind1[0]:points_ind1[1]],
                                         old_ind_2[points_ind2[1]:]), axis=None)

        return new_individual1, new_individual2

    def mutation(self, chromosome):
        # Get the mutation probability
        mutation_prob = 1/len(chromosome)
        
        # Mutate
        for gene_i in range(0,len(chromosome)-1,4):
            if np.random.uniform() <= mutation_prob:
                chromosome[gene_i] = np.random.randint(0, self.nr_operators) # Operator
            if np.random.uniform() <= mutation_prob:
                chromosome[gene_i+1] = np.random.randint(0, self.nr_variable_registers) # Destination register
            if np.random.uniform() <= mutation_prob:
                chromosome[gene_i+2] = np.random.randint(0, self.nr_registers) # Operand 1
            if np.random.uniform() <= mutation_prob:
                chromosome[gene_i+3] = np.random.randint(0, self.nr_registers) # Operand 2

        return chromosome

def main():
    
    # Settings
    file_nr = "result2/14_"
    nr_generations = 1000
    tournament_parameter = 0.75
    tournament_size = 10
    crossover_probability = 0.5
    penalty_term = 0.1
    mode = "max"
    #mode = "average"
    best_individual_copies = 1

    constant_registers = [-1, np.pi, 10]
    
    # Create LGP
    lgp = LGP(population_size=300,
            nr_variable_registers=10,
              constant_registers=constant_registers,
            max_len_cromosome=4*100,
            nr_datapoints=100,
            nr_operators=8
            )
    
    # Save information about the training
    original = lgp.registers
    np.save(f"{file_nr}original_registers.npy", lgp.registers)
    file1 = open(f"{file_nr}info.txt", "w")
    file1.write(
                 f"{tournament_parameter = }" + "\n" +\
                 f"{tournament_size = }" + "\n" +
                 f"{crossover_probability = }" + "\n" +
                 f"{penalty_term = }" + "\n" +
                 f"{best_individual_copies = }" + "\n" +
                 f"population_size = {lgp.population_size}" + "\n" +\
                 f"nr_variable_registers = {lgp.nr_variable_registers}" + "\n" +\
                 f"registers = {lgp.registers[0, -len(constant_registers):]}" + "\n" +
                 f"max_len_cromosome = {lgp.max_len_cromosome}" + "\n" +\
                 f"nr_datapoints = {lgp.nr_datapoints}" + "\n" +\
                 f"nr_operators = {lgp.nr_operators}" + "\n" +
                 f"{mode = }")
    file1.close()


    print("")
    print("---------------------------------------------")
    print("Starting LGP!")
    print("---------------------------------------------")
    print("")

    while (lgp.generation <= nr_generations):

        # Fitness calculation
        lgp.calculate_fitness(penalty_term, mode)

        if lgp.generation % 10 == 0:
            # Print
            print(f"Generation: {lgp.generation}")
            print(f"----Best individual nr: {lgp.best_individual_i}")
            print(f"----Error: {lgp.best_error}")
            print(f"----Best fitness: {lgp.best_fitness}")

            # Save information about training
            np.save(f"{file_nr}chromosome.npy", lgp.population[lgp.best_individual_i]['chromosome'])
            file1 = open(f"{file_nr}result.txt", "w")
            file1.write(f"Generation: {lgp.generation}" + "\n" +\
                        f"----Best individual nr: {lgp.best_individual_i}" + "\n" +\
                        f"----Error: {lgp.best_error}" + "\n" +\
                        f"----Best fitness: {lgp.best_fitness}" + "\n")
            file1.close()

        # Create new population
        new_population = {}

        for i in range(0, lgp.population_size, 2):
            # Tournament
            individual_i_1 = lgp.tournament_selection(
                tournament_size, tournament_parameter)
            individual_i_2 = lgp.tournament_selection(
                tournament_size, tournament_parameter)

            # Crossover
            if np.random.uniform() <= crossover_probability:
                individual_1_cromo, individual_2_cromo = lgp.two_point_crossover(
                    individual_i_1, individual_i_2)
            else:
                individual_1_cromo = np.copy(
                    lgp.population[individual_i_1]["chromosome"])
                individual_2_cromo = np.copy(
                    lgp.population[individual_i_2]["chromosome"])

            # Mutation
            individual_1_cromo = lgp.mutation(individual_1_cromo)
            individual_2_cromo = lgp.mutation(individual_2_cromo)

            # Put in next population
            new_population[i] = {'chromosome': individual_1_cromo}
            new_population[i+1] = {'chromosome': individual_2_cromo}

        # Elitism
        for i in range(best_individual_copies):
            new_population[i] = {'chromosome': np.copy(
                lgp.population[lgp.best_individual_i]['chromosome'])}

        # Update 
        lgp.population = new_population
        lgp.generation += 1

    print("")
    print("---------------------------------------------")
    print("LGP Done!")
    print("---------------------------------------------")
    print("")

    lgp.calculate_fitness(penalty_term, mode)

    # Print final result
    print(f"Result generation: {lgp.generation}")
    print(f"----Best fitness: {lgp.best_fitness}")
    print(f"----Error: {lgp.best_error}")
    print(f"----Individual nr: {lgp.best_individual_i}")
    print(f"----chromosome: {lgp.population[lgp.best_individual_i]['chromosome']}")

    # Save training information
    np.save(f"{file_nr}chromosome.npy",
            lgp.population[lgp.best_individual_i]['chromosome'])
    file1 = open(f"{file_nr}result.txt", "w")
    file1.write(f"Generation: {lgp.generation}" + "\n" +\
                f"----Best individual nr: {lgp.best_individual_i}" + "\n" +\
                f"----Error: {lgp.best_error}" + "\n" +\
                f"----Best fitness: {lgp.best_fitness}" + "\n")
    file1.close()

    

    ff = FittnessFunction(
        constant_registers=np.copy(lgp.registers[:, 1]),
        nr_variable_registers=lgp.registers.shape[1],
        nr_datapoints=lgp.registers.shape[0],
        datapoint_registers=lgp.registers)

    predicted_angles = ff.decode_cromosome(
        np.copy(lgp.population[lgp.best_individual_i]['chromosome']))
    pred_pos = ff.calc_pos(predicted_angles)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    m = 'o'
    ax.scatter(lgp.registers[:, 0], lgp.registers[:, 1],
               lgp.registers[:, 2], marker=m, label="original")
    ax.scatter(pred_pos[:, 0], pred_pos[:, 1],
            pred_pos[:, 2], marker=m, label="predicted")
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
    ax.set_title(f"Max diff = {np.max(diff)}")

    plt.show()

    assert((original == lgp.registers).all())

def main_multiple():

    # Settings
    
    m_nr_generations = [1000, 2000]
    m_tournament_parameter = [0.75, 0.5]
    m_tournament_size = [10, 2]
    m_crossover_probability = [0.5, 0.7]
    penalty_term = 0.1
    m_mode = ["max", "average"]
    #mode = "average"
    m_best_individual_copies = [1, 2]

    m_constant_registers = [[-1, np.pi, 10], [-1, np.pi, 1]]
    m_nr_variable_registers = [100, 10, 3]

    run = 0

    for nr_generations in m_nr_generations:
        for tournament_parameter in m_tournament_parameter:
            for tournament_size in m_tournament_size:
                for crossover_probability in m_crossover_probability:
                    for mode in m_mode:
                        for best_individual_copies in m_best_individual_copies:
                            for constant_registers  in m_constant_registers:
                                for nr_variable_registers in m_nr_variable_registers:
                                    run = run +1
                                    file_nr = f"result2/{run}_"

                                    # Create LGP
                                    lgp = LGP(population_size=300,
                                            nr_variable_registers=nr_variable_registers,
                                            constant_registers=constant_registers,
                                            max_len_cromosome=4*100,
                                            nr_datapoints=100,
                                            nr_operators=8
                                            )

                                    # Save information about the training
                                    np.save(f"{file_nr}original_registers.npy", lgp.registers)
                                    file1 = open(f"{file_nr}info.txt", "w")
                                    file1.write(
                                        f"{tournament_parameter = }" + "\n" +
                                        f"{tournament_size = }" + "\n" +
                                        f"{crossover_probability = }" + "\n" +
                                        f"{penalty_term = }" + "\n" +
                                        f"{best_individual_copies = }" + "\n" +
                                        f"population_size = {lgp.population_size}" + "\n" +
                                        f"nr_variable_registers = {lgp.nr_variable_registers}" + "\n" +
                                        f"registers = {lgp.registers[0, -len(constant_registers):]}" + "\n" +
                                        f"max_len_cromosome = {lgp.max_len_cromosome}" + "\n" +
                                        f"nr_datapoints = {lgp.nr_datapoints}" + "\n" +
                                        f"nr_operators = {lgp.nr_operators}" + "\n" +
                                        f"{mode = }")
                                    file1.close()

                                    print("")
                                    print("---------------------------------------------")
                                    print("Starting LGP!")
                                    print("---------------------------------------------")
                                    print("")

                                    while (lgp.generation <= nr_generations):

                                        # Fitness calculation
                                        lgp.calculate_fitness(penalty_term, mode)

                                        if lgp.generation % 100 == 0:
                                            # Print
                                            print(f"Generation: {lgp.generation}")
                                            print(f"----Best individual nr: {lgp.best_individual_i}")
                                            print(f"----Error: {lgp.best_error}")
                                            print(f"----Best fitness: {lgp.best_fitness}")

                                            # Save information about training
                                            np.save(f"{file_nr}chromosome.npy",
                                                    lgp.population[lgp.best_individual_i]['chromosome'])
                                            file1 = open(f"{file_nr}result.txt", "w")
                                            file1.write(f"Generation: {lgp.generation}" + "\n" +
                                                        f"----Best individual nr: {lgp.best_individual_i}" + "\n" +
                                                        f"----Error: {lgp.best_error}" + "\n" +
                                                        f"----Best fitness: {lgp.best_fitness}" + "\n")
                                            file1.close()

                                        # Create new population
                                        new_population = {}

                                        for i in range(0, lgp.population_size, 2):
                                            # Tournament
                                            individual_i_1 = lgp.tournament_selection(
                                                tournament_size, tournament_parameter)
                                            individual_i_2 = lgp.tournament_selection(
                                                tournament_size, tournament_parameter)

                                            # Crossover
                                            if np.random.uniform() <= crossover_probability:
                                                individual_1_cromo, individual_2_cromo = lgp.two_point_crossover(
                                                    individual_i_1, individual_i_2)
                                            else:
                                                individual_1_cromo = np.copy(
                                                    lgp.population[individual_i_1]["chromosome"])
                                                individual_2_cromo = np.copy(
                                                    lgp.population[individual_i_2]["chromosome"])

                                            # Mutation
                                            individual_1_cromo = lgp.mutation(individual_1_cromo)
                                            individual_2_cromo = lgp.mutation(individual_2_cromo)

                                            # Put in next population
                                            new_population[i] = {'chromosome': individual_1_cromo}
                                            new_population[i+1] = {'chromosome': individual_2_cromo}

                                        # Elitism
                                        for i in range(best_individual_copies):
                                            new_population[i] = {'chromosome': np.copy(
                                                lgp.population[lgp.best_individual_i]['chromosome'])}

                                        # Update
                                        lgp.population = new_population
                                        lgp.generation += 1

                                    print("")
                                    print("---------------------------------------------")
                                    print("LGP Done!")
                                    print("---------------------------------------------")
                                    print("")

                                    lgp.calculate_fitness(penalty_term, mode)

                                    # Print final result
                                    print(f"Result generation: {lgp.generation}")
                                    print(f"----Best fitness: {lgp.best_fitness}")
                                    print(f"----Error: {lgp.best_error}")
                                    print(f"----Individual nr: {lgp.best_individual_i}")
                                    print(
                                        f"----chromosome: {lgp.population[lgp.best_individual_i]['chromosome']}")

                                    # Save training information
                                    np.save(f"{file_nr}chromosome.npy",
                                            lgp.population[lgp.best_individual_i]['chromosome'])
                                    file1 = open(f"{file_nr}result.txt", "w")
                                    file1.write(f"Generation: {lgp.generation}" + "\n" +
                                                f"----Best individual nr: {lgp.best_individual_i}" + "\n" +
                                                f"----Error: {lgp.best_error}" + "\n" +
                                                f"----Best fitness: {lgp.best_fitness}" + "\n")
                                    file1.close()

                                    predicted_angles = lgp.decode_cromosome(
                                        np.copy(lgp.population[lgp.best_individual_i]['chromosome']))

                                    diff = lgp.get_error(np.copy(predicted_angles))
                                    diff.sort()

                                    print(f"Max diff = {np.max(diff)}")

if __name__=="__main__":
    main()
    #main_multiple()
    
