import numpy as np
import random
from fitnessFunctionNumba import FittnessFunction
import timeit
import threading
from queue import Queue


class LGP(FittnessFunction):
    def __init__(self, population_size, nr_variable_registers, constant_registers,
                 nr_datapoints, max_len_cromosome) -> None:

        # Settings
        self.len_instruction = 4 
        self.min_len_cromosome = 8
        self.max_len_cromosome = max_len_cromosome

        # Check
        # 4 genes per operation
        assert(max_len_cromosome % self.len_instruction == 0)
        assert(max_len_cromosome >= self.min_len_cromosome)
        assert(nr_variable_registers >=3) # for 3 angles/ 3 inputs
        assert(population_size%6 ==0)

        # Keep track
        self.generation = 0
        self.best_fitness = 0
        self.best_individual_i = 0

        # Function handler
        super().__init__(
            max_vector=[5,5,5], # meter
            min_vector=[-5, -5, -0.055],  # meter
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

    def calculate_fitness(self, penalty):

        # Decode cromosome and calculate error and fitness
        self.best_fitness = 0
        for i,individual in self.population.items():
            predicted_angles = self.decode_cromosome(individual["cromosome"])
            error = self.get_error(predicted_angles)

            error = np.max(error)
            fitness = 1/error

            if len(individual["cromosome"]) > self.max_len_cromosome:
                fitness = fitness*penalty

            individual["fitness"] = fitness

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual_i = i


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
        mutation_prob = 1/len(cromosome)

        for gene_i in range(0,len(cromosome)-1,4):
            if np.random.uniform() <= mutation_prob:
                cromosome[gene_i] = np.random.randint(0, self.nr_operators) # Operator
            if np.random.uniform() <= mutation_prob:
                cromosome[gene_i+1] = np.random.randint(0, self.nr_variable_registers) # Destination register
            if np.random.uniform() <= mutation_prob:
                cromosome[gene_i+2] = np.random.randint(0, self.nr_registers) # Operand 1
            if np.random.uniform() <= mutation_prob:
                cromosome[gene_i+3] = np.random.randint(0, self.nr_registers) # Operand 2

        return cromosome


def apply_GA(que, population_start, population_end, lgp, tournament_size, tournament_parameter, crossover_probability):
    new_population = {}

    for i in range(population_start, population_end, 2):
        # Tournament
        individual_i_1 = lgp.tournament_selection(tournament_size, tournament_parameter)
        individual_i_2 = lgp.tournament_selection(tournament_size, tournament_parameter)

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

    que.put(new_population)
    

def main():

    nr_generations = 300

    tournament_parameter = 0.75
    tournament_size = 5
    crossover_probability = 0.2
    penalty = 0.1
    best_individual_copies = 2

    lgp = LGP(population_size=240,
            nr_variable_registers=4,
            constant_registers=[1, -1, np.pi],
            max_len_cromosome=144,
            nr_datapoints=10
            )

    print("")
    print("---------------------------------------------")
    print("Starting LGP!")
    print("---------------------------------------------")
    print("")


    lasttime = timeit.default_timer()
    que = Queue()

    while (lgp.generation <= nr_generations):

        lgp.calculate_fitness(penalty)

        if lgp.generation % 10 == 0:
            print(f"Generation: {lgp.generation}")
            print(f"----Best individual nr: {lgp.best_individual_i}")
            print(f"----Error: {1/lgp.best_fitness}")
            print(f"----Best fitness: {lgp.best_fitness}")
            #print("----The time difference is :", timeit.default_timer() - lasttime)
            #lasttime = timeit.default_timer()

 

        t1 = threading.Thread(target=apply_GA, args=(que, 0, int(lgp.population_size/3),
                              lgp, tournament_size, tournament_parameter, crossover_probability))
        t2 = threading.Thread(target=apply_GA, args=(que, int(lgp.population_size/3), int(lgp.population_size/3)*2,
                              lgp, tournament_size, tournament_parameter, crossover_probability))
        t3 = threading.Thread(target=apply_GA, args=(que, int(lgp.population_size/3)*2, lgp.population_size,
                              lgp, tournament_size, tournament_parameter, crossover_probability))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

        new_population = que.get()
        new_population.update(que.get())
        new_population.update(que.get())

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
    print(f"----Error: {1/lgp.best_fitness}")
    print(f"----Individual nr: {lgp.best_individual_i}")
    print(f"----Cromosome: {lgp.population[lgp.best_individual_i]['cromosome']}")
    print("----The time difference is :", timeit.default_timer() - lasttime)

if __name__=="__main__":
    main()
