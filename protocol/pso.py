import config as cf
from protocol.scheduling import Scheduling
import logging
import numpy as np
from network import calculate_distance
from network import calculate_sensing_prob


class PSO(Scheduling):
    """
    Chen, Jiming, et al.
    "Energy-efficient coverage based on probabilistic sensing model in wireless sensor networks."
    IEEE Communications Letters 14.9 (2010): 833-835.
    """

    name = 'PSO' # particle swarm optimization

    def __init__(self):
        self.network = None
        self.nodes = None
        self.targets = None
        self.sensing_threshold = None
        self.sensing_matrix = None

        self.velocity = []      # current velocity for each particle
        self.p_local = []       # local best position for each particle
        self.p_current = []     # current position of each particle
        self.p_global = None    # global best position among all the particles

        self.value_p_l = []     # local best value
        self.value_p_g = np.Infinity  # global best value

        # constants
        self.c_1 = 2.5
        self.c_2 = 1.5
        self.w = 0.99

        # num. of iterations
        self.iteration = 2000

        # num of particles
        self.num_particles = 400

    def set_mode(self, network, rounds):
        self.network = network
        self.nodes = network.get_nodes()
        self.targets = network.get_targets()
        self.sensing_threshold = self.network.get_log_threshold()
        self.sensing_matrix = self.network.get_log_matrix()

        self.initialize()  # initialize particles

        self.value_p_l = [self.objective_function(p) if self.validate(p) else np.Infinity
                          for p in self.p_local]
        self.value_p_g, self.p_global = self.evaluate(self.p_local)  # find a best particle

        for t in range(self.iteration):
            # logging.info('PSO iter : '+ str(t))
            for i, p_current in enumerate(self.p_current):

                # calculate velocity
                r_1 = np.random.rand()
                r_2 = np.random.rand()
                self.velocity[i] = self.w * self.velocity[i] \
                                    + self.c_1 * r_1 * (self.p_local[i] - p_current) \
                                    + self.c_2 * r_2 * (self.p_global - p_current)

                # update position by using s-shaped function (sigmoid function)
                sigmoid = 1 / (np.exp(-1 * self.velocity[i]) + 1)
                rand_num = np.random.rand(len(sigmoid))

                for j in range(len(sigmoid)):
                    if sigmoid[j] > rand_num[j] and network.get_node(j).energy >= cf.SENSING_ENERGY:
                        self.p_current[i][j] = cf.ACTIVE  # active
                    else:
                        self.p_current[i][j] = cf.SLEEP  # sleep

                if self.validate(self.p_current[i]):  # validate solution
                    value = self.objective_function(self.p_current[i])

                    # update p_local
                    if self.value_p_l[i] > value:
                        self.p_local[i] = np.copy(self.p_current[i])
                        self.value_p_l[i] = value

                    # update p_global
                    if self.value_p_g > value:
                        self.p_global = np.copy(self.p_current[i])
                        self.value_p_g = value
            # main loop end

        self.network.active_mode()

        # change mode of sensors
        for i, node in enumerate(self.nodes):
            if self.p_global[i] != cf.SLEEP and node.energy >= cf.SENSING_ENERGY:
                node.set_mode(cf.ACTIVE)
            else:
                node.set_mode(cf.SLEEP)

        return self.value_p_g  # return the objective function value of best solution

    def initialize(self):
        self.p_local = np.random.randint(2, size=(self.num_particles, len(self.nodes)))
        self.p_current = np.copy(self.p_local)
        self.velocity = np.zeros((self.num_particles, len(self.nodes)), dtype=float)

    def objective_function(self, x):
        return sum([cf.SENSOR_COST ** self.network.get_node(i).energy
                    for i, mode in enumerate(x) if mode != cf.SLEEP and self.network.get_node(i).energy > 0])

    def evaluate(self, particles):
        """
        Evaluate all particles
        return the best function value and the best particle
        """
        value = [self.objective_function(p) for p in particles]
        return np.min(value), particles[np.argmin(value)]

    def validate(self, particles):
        """
        This function uses a log matrix for minimizing computation time
        """

        nodes = [i for i, a in enumerate(particles) if a != cf.SLEEP
                 and self.network.get_node(i).energy >= cf.SENSING_ENERGY]  # active nodes ID


        for t in range(cf.NUM_TARGETS):
            no_sense_prob = 0
            for n in nodes:
                no_sense_prob += self.sensing_matrix[t][n]
                if no_sense_prob >= self.sensing_threshold:
                    break

            if no_sense_prob < self.sensing_threshold:
                return False
        return True
