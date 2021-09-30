import numpy as np
import random
from fitnessFunction import FittnessFunction
import timeit
from numba import njit

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
        self.penalty_done = False

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
            cromosome = np.zeros(len_cromosome)

            for gene_i in range(0,len_cromosome-1,4):
                cromosome[gene_i] = np.random.randint(0, self.nr_operators) # Operator
                cromosome[gene_i+1] = np.random.randint(0, self.nr_variable_registers) # Destination register
                cromosome[gene_i+2] = np.random.randint(0, self.nr_registers) # Operand 1
                cromosome[gene_i+3] = np.random.randint(0, self.nr_registers) # Operand 2

            population[individual] = {'cromosome':cromosome}

        return population

    def calculate_fitness(self, penalty_len, penalty_angle, best_individual_copies, mode):

        # Decode cromosome and calculate error and fitness
        if not best_individual_copies:
            self.best_fitness = 0
        for i,individual in self.population.items():
            predicted_angles = self.decode_cromosome(individual["cromosome"])
            error = self.get_error(predicted_angles)

            if mode == "max":
                error = np.max(error)
            elif mode == "average":
                error = np.average(error)
            fitness = 1/error

            if len(individual["cromosome"]) > self.max_len_cromosome:
                fitness = fitness*penalty_len
            
            if (predicted_angles > self.max_angle).any() or (predicted_angles < self.min_angle).any():
                fitness = fitness*penalty_angle

            individual["fitness"] = fitness

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual_i = i
                self.best_error = error
                if fitness != 1/error:
                    self.penalty_done = True
                else:
                    self.penalty_done = False
                print("----------------------NEW BEST!--------------------------")


    def tournament_selection(self, tournament_size, tournament_parameter):
        selected = np.zeros((2, tournament_size))

        selected[0, :] = np.random.randint(self.population_size, size=tournament_size)
        selected[1,:] = [self.population[i]["fitness"] for i in selected[0, :]]

        selected = selected[:, selected[1, :].argsort()] # sort

        for i in range(selected.shape[1]-1, 0, -1):
            if np.random.uniform() <= tournament_parameter:
                return selected[0,i]
        return selected[0, 0]

        
    def two_point_crossover(self, ind1, ind2):
        old_ind_1 = np.copy(self.population[ind1]["cromosome"])
        old_ind_2 = np.copy(self.population[ind2]["cromosome"])

        nr_instructions_ind1 = len(old_ind_1)/4
        nr_instructions_ind2 = len(old_ind_2)/4

        points_ind1 = np.sort(np.random.randint(nr_instructions_ind1, size=2))*4
        points_ind2 = np.sort(np.random.randint(nr_instructions_ind2, size=2))*4

        new_individual1 = np.concatenate((old_ind_1[:points_ind1[0]], 
                                         old_ind_2[points_ind2[0]:points_ind2[1]],
                                         old_ind_1[points_ind1[1]:]), axis = None)

        new_individual2 = np.concatenate((old_ind_2[:points_ind2[0]], 
                                         old_ind_1[points_ind1[0]:points_ind1[1]],
                                         old_ind_2[points_ind2[1]:]), axis=None)

        return new_individual1, new_individual2

    def mutation(self, cromosome):
        return mutation_numba(cromosome, self.nr_operators, self.nr_variable_registers, self.nr_registers)

@njit
def mutation_numba(cromosome, nr_operators, nr_variable_registers, nr_registers):
    mutation_prob = 1/len(cromosome)

    for gene_i in range(0, len(cromosome)-1, 4):
        if np.random.uniform() <= mutation_prob:
            cromosome[gene_i] = np.random.randint(0, nr_operators)  # Operator
        if np.random.uniform() <= mutation_prob:
            # Destination register
            cromosome[gene_i +
                        1] = np.random.randint(0, nr_variable_registers)
        if np.random.uniform() <= mutation_prob:
            # Operand 1
            cromosome[gene_i+2] = np.random.randint(0, nr_registers)
        if np.random.uniform() <= mutation_prob:
            # Operand 2
            cromosome[gene_i+3] = np.random.randint(0, nr_registers)

    return cromosome

def main():
    
    file_nr = "result/7_"
    nr_generations = 300

    tournament_parameter = 0.75
    tournament_size = 4
    crossover_probability = 0.25
    penalty_len = 0.1
    penalty_angle = 1
    mode = "max"
    #mode = "average"

    best_individual_copies = 1

    constant_registers = [1, -1, np.pi, 10]
    lgp = LGP(population_size=100,
            nr_variable_registers=4,
              constant_registers=constant_registers,
            max_len_cromosome=1*100,
            nr_datapoints=50,
            nr_operators=7
            )
    
    np.save(f"{file_nr}original_registers.npy", lgp.registers)

    file1 = open(f"{file_nr}info.txt", "w")
    file1.write(
                 f"tournament_parameter = {tournament_parameter}" + "\n" +\
                 f"tournament_size = {tournament_size}" + "\n" +
                 f"crossover_probability = {crossover_probability}" + "\n" +\
                 f"penalty_len = {penalty_len}" + "\n" +\
                 f"penalty_angle = {penalty_angle}" + "\n" +\
                 f"best_individual_copies = {best_individual_copies}" + "\n" +\
                 f"population_size = {lgp.population_size}" + "\n" +\
                 f"nr_variable_registers = {lgp.nr_variable_registers}" + "\n" +\
                 f"registers = {lgp.registers[0, -len(constant_registers):]}" + "\n" +
                 f"max_len_cromosome = {lgp.max_len_cromosome}" + "\n" +\
                 f"nr_datapoints = {lgp.nr_datapoints}" + "\n" +\
                 f"nr_operators = {lgp.nr_operators}" + "\n" +
                 f"mode = {mode}")
    file1.close()

    print("")
    print("---------------------------------------------")
    print("Starting LGP!")
    print("---------------------------------------------")
    print("")


    lasttime = timeit.default_timer()

    while (lgp.generation <= nr_generations):

        lgp.calculate_fitness(penalty_len, penalty_angle,
                              best_individual_copies, mode)

        if lgp.generation % 10 == 0:
            print(f"Generation: {lgp.generation}")
            print(f"----Best individual nr: {lgp.best_individual_i}")
            print(f"----Error: {lgp.best_error}")
            print(f"----Best fitness: {lgp.best_fitness}")
            print(f"----Penalty: {lgp.penalty_done}")

            np.save(f"{file_nr}cromosome.npy", lgp.population[lgp.best_individual_i]['cromosome'])
            file1 = open(f"{file_nr}result.txt", "w")
            file1.write(f"Generation: {lgp.generation}" + "\n" +\
                        f"----Best individual nr: {lgp.best_individual_i}" + "\n" +\
                        f"----Error: {lgp.best_error}" + "\n" +\
                        f"----Best fitness: {lgp.best_fitness}" + "\n" +\
                        f"----Penalty: {lgp.penalty_done}" + "\n")
            file1.close()

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
                    lgp.population[individual_i_1]["cromosome"])
                individual_2_cromo = np.copy(
                    lgp.population[individual_i_2]["cromosome"])

            # Mutation
            individual_1_cromo = lgp.mutation(individual_1_cromo)
            individual_2_cromo = lgp.mutation(individual_2_cromo)

            # Put in next population
            new_population[i] = {'cromosome': individual_1_cromo}
            new_population[i+1] = {'cromosome': individual_2_cromo}

        # Elitism
        for i in range(best_individual_copies):
            new_population[i] = {'cromosome': np.copy(
                lgp.population[lgp.best_individual_i]['cromosome'])}

        lgp.population = new_population
        lgp.generation += 1

    print("")
    print("---------------------------------------------")
    print("LGP Done!")
    print("---------------------------------------------")
    print("")

    print(f"Result generation: {lgp.generation}")
    print(f"----Best fitness: {lgp.best_fitness}")
    print(f"----Error: {lgp.best_error}")
    print(f"----Penalty: {lgp.penalty_done}")
    print(f"----Individual nr: {lgp.best_individual_i}")
    print(f"----Cromosome: {lgp.population[lgp.best_individual_i]['cromosome']}")
    print("----The time difference is :", timeit.default_timer() - lasttime)

    np.save(f"{file_nr}cromosome.npy",
            lgp.population[lgp.best_individual_i]['cromosome'])
    file1 = open(f"{file_nr}result.txt", "w")
    file1.write(f"Generation: {lgp.generation}" + "\n" +\
                f"----Best individual nr: {lgp.best_individual_i}" + "\n" +\
                f"----Error: {lgp.best_error}" + "\n" +\
                f"----Best fitness: {lgp.best_fitness}" + "\n" +\
                f"----Penalty: {lgp.penalty_done}" + "\n")
    file1.close()

if __name__=="__main__":
    main()
