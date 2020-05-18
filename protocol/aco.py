import config as cf
from protocol.scheduling import Scheduling
import logging
import numpy as np
from network import calculate_distance
from network import calculate_sensing_prob


class ACO(Scheduling):
    """
    Chen, Jiming, et al.
    "Energy-efficient coverage based on probabilistic sensing model in wireless sensor networks."
    IEEE Communications Letters 14.9 (2010): 833-835.
    """

    name = 'ACO' # ant colony optimization

    def __init__(self):
        self.network = None
        self.nodes = None
        self.targets = None
        self.sensing_threshold = None
        self.sensing_matrix = None
        self.stats_matrix = []

        self.pheromones = []  # size : len(ants) x len(nodes)
        self.best_solution = []
        self.best_value = None

        # constants
        self.alpha = 1
        self.beta = 1
        self.rho = 0.1  # p
        self.init_pheromone = 0.5  # initial pheromone value

        # num. of iterations
        self.iteration = 500

        # num. of ants
        self.num_ants = 100

        self.nodes_intensity = []

    def set_mode(self, network, rounds):

        if rounds == 0:
            # get network info
            self.network = network
            self.nodes = network.get_nodes()
            self.targets = network.get_targets()
            self.sensing_threshold = self.network.get_log_threshold()
            self.sensing_matrix = self.network.get_log_matrix()

        for node in self.nodes:  # eta function
            if node.energy > 0:
                node_intensity = sum([calculate_sensing_prob(node, target) for target in self.targets])
            else:
                node_intensity = 0
            self.nodes_intensity.append(node_intensity)

        # logging.info(str(self.nodes_intensity))
        self.initialize_pheromone_values()  # init pheromone

        # main loop start
        for t in range(self.iteration):
            solution_set = []
            values = []
            for ant in range(self.num_ants):

                s = self.construct_solution(ant)

                if self.validate(s):
                    value = self.objective_function(s)
                    if self.best_value is None or self.best_value > value:
                        self.best_solution = np.copy(s)
                        self.best_value = np.copy(value)
                    solution_set.append(s)
                    values.append(value)
                else:
                    solution_set.append([])
                    values.append(-1)
            self.update_pheromone(solution_set, values)
        # main loop end

        # change mode of sensors
        if self.best_solution is None:
            logging.info('cannot find solution')
            self.network.active_mode()
            return np.inf

        else:
            for i, node in enumerate(self.nodes):
                if self.best_solution[i] == cf.ACTIVE and node.energy >= cf.SENSING_ENERGY:
                    node.set_mode(cf.ACTIVE)
                else:
                    node.set_mode(cf.SLEEP)

        return self.best_value

    def initialize_pheromone_values(self):
        self.best_solution = None
        self.best_value = None
        self.pheromones \
            = np.ones((self.num_ants, len(self.nodes))) * self.init_pheromone
        self.stats_matrix = np.random.randint(2, size=(self.num_ants, len(self.nodes)))

    def construct_solution(self, j):
        """
        Create a solution using j-th ant's pheromone
        :param:
                j : ant index ,  i : value index
        eta is a intensity of ant i ; eta[i] = sum(prob of targets) for node i
        """
        prob = [(p ** self.alpha) * ((self.nodes_intensity[i]) ** self.beta)
                for i, p in enumerate(self.pheromones[j])]

        prob /= sum(prob)
        for i, p in enumerate(prob):
            if p > np.random.rand():
                if self.stats_matrix[j][i] == cf.ACTIVE:
                    self.stats_matrix[j][i] = cf.SLEEP
                else:
                    self.stats_matrix[j][i] = cf.ACTIVE

        return self.stats_matrix[j]

    def update_pheromone(self, solutions, values):
        updated_pheromone\
            = np.empty((self.num_ants, len(self.nodes)), dtype=np.float64)
        for j in range(self.num_ants):  # j-th ant
            if values[j] != -1:  # checking whether solution is validate or not
                for i in range(len(self.nodes)):  # i-th value
                    updated_pheromone[j][i] = (1 - self.rho) * self.pheromones[j][i] \
                                              + self.rho * solutions[j][i] * values[j] / self.best_value
            else:  # invalid solution
                for i in range(len(self.nodes)):  #
                    updated_pheromone[j][i] = (1 - self.rho) * self.pheromones[j][i]

        self.pheromones = np.copy(updated_pheromone)

    def validate(self, solution):
        """
         This function uses a log matrix
        """
        nodes = [i for i, a in enumerate(solution) if a != cf.SLEEP
                 and self.network.get_node(i).energy >= cf.SENSING_ENERGY]  # active nodes

        for t in range(cf.NUM_TARGETS):
            no_sense_prob = 0
            for n in nodes:
                no_sense_prob += self.sensing_matrix[t][n]
                if no_sense_prob >= self.sensing_threshold:
                    break

            if no_sense_prob < self.sensing_threshold:
                return False
        return True

    def objective_function(self, x):
        return sum([cf.SENSOR_COST ** self.network.get_node(i).energy
                    for i, mode in enumerate(x) if mode != cf.SLEEP])

