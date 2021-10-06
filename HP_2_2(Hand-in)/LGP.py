import numpy as np
import random
from scipy.io import savemat

from fitnessFunction import FittnessFunction
from plotResult import plot_result


class LGP(FittnessFunction):
    def __init__(self, population_size, variable_registers, constant_registers,
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
        assert(nr_operators <= 8)  # [+ - * / sin cos arcsin arccos]

        # Keep track
        self.generation = 0

        # Function handler
        super().__init__(
            constant_registers,
            variable_registers,
            nr_datapoints,
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

    def calculate_fitness(self, penalty_term):
        self.best_fitness = 0

        for individual in self.population.values():
            chromosome = individual["chromosome"]

            # Calculate error
            predicted_angles = self.decode_cromosome(chromosome)
            error_all = self.get_error(predicted_angles)

            # RMS-error
            error = np.sqrt(np.sum(np.square(error_all))/self.nr_datapoints)
            
            # Calculate fitness
            fitness = 1/error

            if len(chromosome) > self.max_len_cromosome:
                fitness = fitness*penalty_term

            individual["fitness"] = fitness

            # Save the best chromosome
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_error = error
                self.best_chromosome = chromosome

    def tournament_selection(self, tournament_size, tournament_parameter):
        # Select random idnividuals 
        selected = np.zeros((2, tournament_size))
        selected[0, :] = np.random.randint(self.population_size, size=tournament_size)
        selected[1, :] = [self.population[i]["fitness"] for i in selected[0, :]]

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


def train_LGP(population_size, nr_generations,
              variable_registers, constant_registers,
              max_len_cromosome, nr_datapoints, nr_operators, 
              tournament_parameter, tournament_size, 
              crossover_probability, penalty_term,
              best_individual_copies, file_nr, plot_frequency):
    # Create LGP
    lgp = LGP(population_size=population_size,
              variable_registers=variable_registers,
              constant_registers=constant_registers,
              max_len_cromosome=max_len_cromosome,
              nr_datapoints=nr_datapoints,
              nr_operators=nr_operators
            )
    
    # Save information about the training
    savemat(f"{file_nr}original_registers.mat", {"constant_registers":constant_registers, "variable_registers":variable_registers})
    file1 = open(f"{file_nr}info.txt", "w")
    file1.write(
                 f"{tournament_parameter = }" + "\n" +\
                 f"{tournament_size = }" + "\n" +
                 f"{crossover_probability = }" + "\n" +
                 f"{penalty_term = }" + "\n" +
                 f"{best_individual_copies = }" + "\n" +
                 f"population_size = {lgp.population_size}" + "\n" +\
                 f"{variable_registers = }" + "\n" +\
                 f"registers = {lgp.registers[0, -len(constant_registers):]}" + "\n" +
                 f"max_len_cromosome = {lgp.max_len_cromosome}" + "\n" +\
                 f"nr_datapoints = {lgp.nr_datapoints}" + "\n" +\
                 f"nr_operators = {lgp.nr_operators}")
    file1.close()


    print("")
    print("---------------------------------------------")
    print("Starting LGP!")
    print("---------------------------------------------")
    print("")

    best_fitness_run = 0

    while (lgp.generation <= nr_generations):

        # Fitness calculation
        lgp.calculate_fitness(penalty_term)

        if lgp.generation % plot_frequency == 0:
            # Print
            print(f"Generation: {lgp.generation}")
            print(f"----Error: {lgp.best_error}")
            print(f"----Best fitness: {lgp.best_fitness}")

            if best_fitness_run < lgp.best_fitness:
                best_fitness_run = lgp.best_fitness
                savemat(f"{file_nr}chromosome.mat", {"data": lgp.best_chromosome})

                file1 = open(f"{file_nr}result.txt", "w")
                file1.write(f"Generation: {lgp.generation}" + "\n" +
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
            new_population[i] = {'chromosome': np.copy(lgp.best_chromosome)}

        # Update 
        lgp.population = new_population
        lgp.generation += 1

    print("")
    print("---------------------------------------------")
    print("LGP Done!")
    print("---------------------------------------------")
    print("")

    lgp.calculate_fitness(penalty_term)

    # Print final result
    print(f"Result generation: {lgp.generation}")
    print(f"----Best fitness: {lgp.best_fitness}")
    print(f"----Error: {lgp.best_error}")
    print(f"----chromosome: {lgp.best_chromosome}")

    # Save training information
    savemat(f"{file_nr}chromosome.mat", {"data": lgp.best_chromosome})
    
    file1 = open(f"{file_nr}result.txt", "w")
    file1.write(f"Generation: {lgp.generation}" + "\n" +\
                f"----Error: {lgp.best_error}" + "\n" +\
                f"----Best fitness: {lgp.best_fitness}" + "\n")
    file1.close()
    return lgp

def main():
    variable_registers = np.random.rand(4)
    constant_registers = [np.pi, -1, 2]

    lgp = train_LGP(tournament_parameter=0.75,
            nr_generations=100000,
            tournament_size = 5,
            crossover_probability = 0.3,
            penalty_term = 0.1,
            best_individual_copies = 3,
            population_size = 300,
            variable_registers = variable_registers,
            constant_registers=constant_registers,
            max_len_cromosome = 400,
            nr_datapoints = 200,
            nr_operators = 8,
            file_nr="result/1_",
            plot_frequency=10)

    # Evaluate 
    plot_result(lgp.best_chromosome, constant_registers,
                variable_registers, lgp.nr_datapoints)

if __name__=="__main__":
    main()
    
