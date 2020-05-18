import config as cf
from protocol.scheduling import Scheduling
import logging
import numpy as np
from network import calculate_distance
from network import calculate_sensing_prob
import matplotlib.pyplot as plt
from networkx.utils import pairwise, not_implemented_for
from itertools import combinations, chain
import time


class PROPOSED_SC(Scheduling):
    name = 'PROPOSED_SC' # proposed method only for sensing coverage. This doesn't consider the connectivity between a sink and active sensors.
    # when considering only sensing coverage

    def __init__(self):
        self.network = None
        self.nodes = None
        self.targets = None
        self.sensing_matrix = None
        self.sensing_log_matrix = None
        self.sensing_log_threshold = None
        self.no_cover_targets_node = []
        self.node_influences = None

        self.bats_s = None  # solutions(position vectors)
        self.values_f = None # objective function values for each candidate solutions

        self.best_bat_s = None  # best solution
        self.best_value = -1

        self.velocities = None
        self.frequencies = None

        self.freq_min = 0  # frequency_min
        self.freq_max = 2  # frequency_max

        self.r_p = 0.5  # pulse rate

        self.num_bats = 30  # the number of bats; population size
        self.num_iterations = 1000


    def set_mode(self, network, rounds):

        # get network information
        if rounds == 0: # get a initial network information
            # get network info
            self.network = network
            self.nodes = network.get_nodes()
            self.targets = network.get_targets()
            self.sensing_log_threshold = self.network.get_log_threshold()  # ln(sensing_threshold)
            self.sensing_log_matrix = self.network.get_log_matrix()
            self.sensing_matrix = 1 - self.network.get_matrix()

            self.calculate_influence()

        # check the number of dead nodes
        self.dead_nodes = [node.id for node in self.network.get_nodes()
                           if node.energy < (cf.SENSING_ENERGY)]

        for node in self.dead_nodes:
            for target_id in range(cf.NUM_TARGETS):
                self.node_influences[target_id, node] = 0

        # initialize the frequency and velocity vector to zero vector
        self.frequencies = np.zeros((self.num_bats, len(self.nodes)), dtype=np.float64)
        self.velocities = np.zeros((self.num_bats, len(self.nodes)), dtype=np.float64) # velocity for the sensing nodes

        # initialize populations
        if self.initialize_populations():

            # find the current best
            self.values_f = [self.objective_function(self.bats_s [i][:]) for i in
                             range(self.num_bats)]
            best_arg = np.argsort(self.values_f)[0]
            self.best_bat_s = np.copy(self.bats_s [best_arg])
            self.best_value = self.values_f[best_arg]

            # main loop start
            for i in range(self.num_iterations):

                # update the frequency
                self.frequencies = self.freq_min + (self.freq_max-self.freq_min) * np.random.rand(
                    self.num_bats, len(self.nodes))

                for bat in range(self.num_bats):
                    # adjust velocity of bat_s
                    self.velocities[bat, :] += (self.bats_s[bat] - self.best_bat_s) * self.frequencies[bat,:]

                    # update position vector of bat_s
                    solution = np.copy(self.bats_s[bat])
                    for i in range(len(self.nodes)):
                        if (i not in self.no_cover_targets_node or i not in self.dead_nodes):
                            if np.random.rand() > self.r_p:  # pulse rate
                                solution[i] = self.best_bat_s[i]

                            else:
                                V_shaped_transfer_function = \
                                    abs(np.tanh(self.velocities[bat, i]))
                                if np.random.rand() < V_shaped_transfer_function:
                                    solution[i] = 0 if solution[i] == 1 else 1
                                    self.velocities[bat, i] /= 2  # velocity update


                    value = self.objective_function(solution)
                    if value < self.values_f[bat] \
                             and self.validate(solution):  # check a connectivity and sensing coverage

                        # if the solution is improved, then the local bat and global best bat are updated
                        self.bats_s [bat, :] = np.copy(solution[:])
                        self.values_f[bat] = value

                        # update the global best bat
                        if self.best_value > value:
                            self.best_bat_s = np.copy(solution[:])
                            self.best_value = value
                            # main loop end

        else:  # when failing initializing populations, then all nodes are turn on
            self.best_bat_s = np.ones(len(self.nodes))
            self.best_value = self.objective_function(self.best_bat_s)

        # set the node's state
        for i, node in enumerate(self.nodes): # turn on sensing nodes (active nodes; sensing nodes)
            if self.best_bat_s[i] >= cf.ACTIVE and node.energy >= (cf.COMMUNICATION_ENERGY + cf.SENSING_ENERGY):
                node.set_mode(cf.ACTIVE)
            else: # sleep nodes
                node.set_mode(cf.SLEEP)

        return self.best_value  # cost of the objective function

    def initialize_populations(self):
        """ Initialize the population """
        time_s = time.time()
        self.bats_s  = np.zeros((self.num_bats, len(self.nodes)), dtype=np.int8)

        for bat_id in range(self.num_bats): # initialize the position of all bat_s
            uncovered_list = [target_id for target_id in range(len(self.targets))]

            while len(uncovered_list) != 0:
                for target_id in uncovered_list:
                    if self.validate_t(self.bats_s [bat_id, :], target_id):
                        uncovered_list.remove(target_id)
                        break
                    node_id = self.node_selection(bat_id, target_id)
                    if node_id is not None:
                        self.bats_s [bat_id, node_id] += 1

                    else:  # cannot find nodes for coverage
                        return False

                if len(uncovered_list) == 0:  # if all target is covered, break a loop
                    break

        time_e = time.time()
        # logging.info('initializing time :' + str(time_e - time_s))
        return True

    def calculate_influence(self):
        """
        Calculate the influence of each node for each target.

        influence :  the probability of each sensor node for each target. (Equation (12))

        """
        self.node_influences = np.empty((len(self.targets), len(self.nodes)), dtype=np.float64)

        for t in range(len(self.targets)):
            for n in range(len(self.nodes)):
                self.node_influences[t][n] = self.sensing_matrix[t][n]

    def config_roulette_wheel(self, bat_id, idx):
        """
        Calculates the selection probability for each node.

        :param idx: target id
        :return: the probability list for each node
        """
        influences = [influence if self.bats_s [bat_id][i] == 0 else 0
                      for i, influence in enumerate(self.node_influences[idx][:])]
        sum_score = max(sum(influences), 0.00001)

        prob = []
        for i in range(len(influences)):
            prob.append(sum(influences[0:i + 1]) / sum_score)

        return prob

    def node_selection(self, bat_id, target_id, loop=False):
        """
        Choose the sensing nodes to be turned on.

        :param bat_id: bat's id
        :param target_id:  target's id
        :return:
        """
        prob_list = self.config_roulette_wheel(bat_id, target_id)  # calculate a selection probability for each nodes
        nodes_list = [i for i, value in enumerate(self.node_influences[target_id][:]) if value != 0]
        active = [i for i in self.bats_s [bat_id][nodes_list] if i >= 1]
        if len(nodes_list) == len(active) or len(nodes_list) == 0: # if (all the nodes are turn on) or none of nodes is selected
            return None

        while True:
            rand_val = np.random.rand()
            for i, prob in enumerate(prob_list):
                if prob >= rand_val:
                    if self.bats_s [bat_id][i] >= 1:
                        break # already selected node
                    else:
                        return i

    def objective_function(self, x):
        return sum([cf.SENSOR_COST ** self.network.get_node(i).energy
                    for i, mode in enumerate(x) if mode != cf.SLEEP and self.network.get_node(i).energy >= cf.SENSING_ENERGY])

    def validate(self, s):
        """
        This function uses a log matrix for minimizing computation time
        This function is faster than validate function.

        :return:
            True or False
        """
        nodes = [i for i, a in enumerate(s) if
                 a != cf.SLEEP and self.network.get_node(i).energy >= (cf.SENSING_ENERGY +cf.COMMUNICATION_ENERGY)]  # get list of the active nodes

        for t in range(cf.NUM_TARGETS):
            no_sense_prob = 0
            for n in nodes:
                no_sense_prob += self.sensing_log_matrix[t][n]
                if no_sense_prob >= self.sensing_log_threshold:
                    break

            if no_sense_prob < self.sensing_log_threshold:
                return False

        return True

    def validate_t(self, solution, t):
        """
         check if the target t is covered or not

        :param: solution : a position vector
                 t: target ID
        :return:
          True or  False
        """
        no_sense_prob = 0
        active_nodes = [i for i, a in enumerate(solution) if a != cf.SLEEP and self.network.get_node(i).energy >= (cf.SENSING_ENERGY +cf.COMMUNICATION_ENERGY)]

        for n in active_nodes:
            no_sense_prob += self.sensing_log_matrix[t][n]
            if no_sense_prob >= self.sensing_log_threshold:
                return True
        if no_sense_prob < self.sensing_log_threshold:
            return False
        else:
            return True

    def get_name(self):
        return '%s_%d_%d' % (self.name, self.num_bats,self.num_iterations)


