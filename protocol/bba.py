import config as cf
from protocol.scheduling import Scheduling
import logging
import numpy as np
from network import calculate_distance
from network import calculate_sensing_prob
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.approximation import *
from networkx.utils import pairwise, not_implemented_for
from itertools import combinations, chain
import time


class N_BBA(Scheduling):
    """Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Xin-She Yang.
    "Binary bat algorithm." Neural Computing and Applications 25.3-4 (2014): 663-681.

    This codes is based on 'http://www.mathworks.com.au/matlabcentral/fileexchange/44707'.
    """

    name = 'normal_BBA'

    def __init__(self):
        self.network = None
        self.nodes = None
        self.targets = None
        self.sensing_matrix = None
        self.sensing_log_matrix = None
        self.sensing_log_threshold = None

        self.bats = None  # solutions(position vectors)
        self.values_f = None  # objective function values for each candidate solutions
        self.best_bat = None  # best solution
        self.best_value = -1

        self.velocities = None
        self.frequencies = None

        self.freq_min = 0  # frequency_min
        self.freq_max = 2  # frequency_max

        self.a = 0.25  # loudness
        self.r_p = 0.1  # pulse rate

        self.num_bats = 20  # population size
        self.num_iterations = 1000

        # parameters for connectivity
        self.network_graph = None

        self.prev_node_num = cf.NUM_NODES # the number of alive nodes at the previous timeslot

    def set_mode(self, network, rounds):

        ## get network information
        if rounds == 0: # timeslot 0
            # get network info
            self.network = network
            self.nodes = network.get_nodes()
            self.targets = network.get_targets()
            self.sensing_log_threshold = self.network.get_log_threshold()  # ln(sensing_threshold)
            self.sensing_log_matrix = self.network.get_log_matrix()
            self.sensing_matrix = 1 - self.network.get_matrix()
            self.network_graph = self.network.get_comm_graph().copy()

        # check dead nodes
        self.dead_nodes = [node.id for node in self.network.get_nodes()
                           if node.energy < (cf.SENSING_ENERGY + cf.COMMUNICATION_ENERGY)]

        for node in self.dead_nodes:
            if self.network.get_node(node).energy < (cf.SENSING_ENERGY+cf.COMMUNICATION_ENERGY) and self.network_graph.has_node(node):
                self.network_graph.remove_node(node)


        # initialize the frequency and velocity vector to zero vector
        result =True
        self.frequencies = np.zeros((self.num_bats, 1), dtype=np.float64)
        self.velocities = np.zeros((self.num_bats, len(self.nodes)), dtype=np.float64)

        ##
        init_best = np.ones((1, len(self.nodes))) # best position vector

        if self.validate(init_best[0]): # validate sensing coverage of init best
            self.best_bat = init_best[0]
            self.best_value = self.objective_function(self.best_bat[:])
        else:
            result =False

        # initialize a position vector for each bat
        if self.initialize_populations() and result:  #initializing the position vector for each bat

            # find the current best
            if cf.CONNECTIVITY is True: # if network coverage is considered
                self.values_f = [self.objective_function(self.bats[i][:]) for i in
                                 range(self.num_bats)]
                for i in range(self.num_bats):
                    if self.validate_conn(self.bats[i][:]) and \
                            self.validate(self.bats[i][:]):
                        pass
                    else: # if the initial position vector does not ensure connectivity
                        self.values_f[i] = float('inf')
                if self.init_result: # find best bat position
                    best_arg = np.argsort(self.values_f)[0]
                    self.best_bat = np.copy(self.bats[best_arg])
                    self.best_value = self.values_f[best_arg]

            else:
                self.values_f = [self.objective_function(self.bats[i][:]) for i in
                                 range(self.num_bats)]
                best_arg = np.argsort(self.values_f)[0]
                self.best_bat = np.copy(self.bats[best_arg])
                self.best_value = self.values_f[best_arg]

            # logging.info('best current :' + str(self.best_bat))
            # logging.info('best value :' + str(self.best_value))


            # main loop start
            for i in range(self.num_iterations):

                # update freq
                self.frequencies = self.freq_min + (self.freq_min - self.freq_max) * np.random.rand(self.num_bats, 1)

                for bat in range(self.num_bats): # for each bat
                    # update velocity
                    self.velocities[bat, :] += (self.bats[bat] - self.best_bat) * self.frequencies[bat]

                    solution = np.copy(self.bats[bat])

                    for i in range(len(self.nodes)):
                        V_shaped_transfer_function = \
                            abs(2 / np.pi * np.arctan(2 / np.pi) * self.velocities[bat, i])

                        if np.random.rand() < V_shaped_transfer_function:
                            solution[i] = 0 if solution[i] == 1 else 1

                        if np.random.rand() > self.r_p:  # pulse rate
                            solution[i] = self.best_bat[i]

                    # evaluate new solution
                    value =self.objective_function(solution) # calculate the function value of the new solution
                    if value <= self.values_f[bat] and (np.random.rand()< self.a):
                        if self.validate_conn(solution) and self.validate(
                            solution):  # check a sensing coverage and a connectivity

                            # if the solution improves then updating the local bat and global best bat
                            self.bats[bat, :] = np.copy(solution[:])
                            self.values_f[bat] = value

                            # update the global best bat
                            if self.best_value > value:
                                self.best_bat = np.copy(solution[:])
                                self.best_value = value

                                # main loop end

        else:  # when failing initializing populations, then all nodes are turn on
            self.best_bat = np.ones(len(self.nodes))
            self.best_value = self.objective_function(self.best_bat)


        # set the state of sensor nodes
        for i, node in enumerate(self.nodes):
            if self.best_bat[i] >= cf.ACTIVE and node.energy >= (cf.COMMUNICATION_ENERGY + cf.SENSING_ENERGY):
                node.set_mode(cf.ACTIVE)
            else:
                node.set_mode(cf.SLEEP)


        return self.best_value

    def initialize_populations(self):
        """ Initialize the population """
        time_s = time.time()
        self.bats =  np.random.randint(2, size=(self.num_bats, len(self.nodes)), dtype=np.int8)

        for i in range(self.num_bats):
            if self.validate_conn(self.bats[i][:]) and self.validate(self.bats[i][:]):
                self.init_result = True
                break

        time_e = time.time()
        #logging.info('initializing time :' + str(time_e - time_s))
        return True


    def objective_function(self, bs):
        cost = cf.COMMUNICATION_ENERGY + cf.SENSING_ENERGY  # weight for sensing nodes
        weights = sum([cf.SENSOR_COST ** self.network.get_node(i).energy
                   for i, mode in enumerate(bs) if mode != cf.SLEEP])
        return cost * weights

    def validate(self, s):
        """
        This function uses a log matrix for minimizing computation time

        :return:
            True or False
        """

        nodes = [i for i, a in enumerate(s) if
                 a != cf.SLEEP and self.network.get_node(i).energy >= (cf.COMMUNICATION_ENERGY + cf.SENSING_ENERGY) ]  # get list of the active nodes

        for t in range(cf.NUM_TARGETS):
            no_sense_prob = 0
            for n in nodes:
                no_sense_prob += self.sensing_log_matrix[t][n]
                if no_sense_prob >= self.sensing_log_threshold:
                    break

            if no_sense_prob < self.sensing_log_threshold:
                return False

        return True

    def validate_conn(self, solution):
        """
        Validate connectivity between sink and active sensors
        by using DFS

        :return:
            True or False
        """

        active_nodes = [idx for idx, value in enumerate(solution)  # remove not included nodes in solution
                        if value != 0 and idx not in self.dead_nodes and self.network.get_node(idx).energy >= cf.COMMUNICATION_ENERGY]

        visited = self.DFS(self.network_graph, active_nodes[0], active_nodes)
        if len(visited) == len(active_nodes):
            return True
        else:
            return False

    def DFS(self, graph, start, active_nodes):
        """
        Depth first search

        :param graph: network graph
        :return: visited node list
        """
        visited = []
        stack = [start]
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.append(n)
                adj = [node for node in list(graph.adj[n]) if node in active_nodes]
                stack += set(adj) - set(visited)

        return visited