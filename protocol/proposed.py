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


# Scheduling Using Binary Bat Algorithm for Connected Coverage (based on probabilistic sensing model) in WSN
# if not considering connectivity, you have to use proposed_sc.py

class PROPOSED(Scheduling):
    name = 'PROPOSED'

    def __init__(self):
        self.network = None
        self.nodes = None
        self.targets = None
        self.sensing_matrix = None
        self.sensing_log_matrix = None
        self.sensing_log_threshold = None
        self.no_cover_targets_node = []
        self.node_influences = None
        self.metric_closure = None
        self.sensor_cost = cf.SENSOR_COST

        self.bats_s = None  # solutions(position vectors)
        self.values_f = None # objective function values for each candidate solutions

        self.best_bat_s = None  # best solution
        self.best_value = -1

        self.velocities = None
        self.frequencies = None

        self.freq_min = 0  # frequency_min
        self.freq_max = 2  # frequency_max

        self.r_p = 0.5  # pulse rate

        self.num_bats = 20  # the number of bat couples; population size
        self.num_iterations = 1000

        # parameters for connectivity
        self.network_graph = None
        self.best_bat_r = None # bat_r of the best couple
        self.bats_r = None # bats for relay nodes
        self.velocities_r = None # velocity vectors of bat_r
        self.frequencies_r =None

        self.prev_node_num = cf.NUM_NODES # the number of alive nodes at the previous timeslot

    def set_mode(self, network, rounds):

        ## get network information
        if rounds == 0: # initial conf
            # get network info
            self.network = network
            self.nodes = network.get_nodes()
            self.targets = network.get_targets()
            self.sensing_log_threshold = self.network.get_log_threshold()  # ln(sensing_threshold)
            self.sensing_log_matrix = self.network.get_log_matrix()
            self.sensing_matrix = 1 - self.network.get_matrix()
            self.network_graph = self.network.get_comm_graph().copy()

            # information for initializing position vectors
            self.calculate_influence()
            self.no_cover_targets_node = [i for i in range(cf.NUM_NODES) if sum(self.sensing_matrix[:, i]) == 0]
            self.metric_closure = metric_closure(self.network_graph)

        # check the number of dead nodes
        self.dead_nodes = [node.id for node in self.network.get_nodes()
                           if node.energy < (cf.SENSING_ENERGY + cf.COMMUNICATION_ENERGY)]

        remove = False
        for node in self.dead_nodes:
            if self.network.get_node(node).energy < cf.COMMUNICATION_ENERGY and self.network_graph.has_node(node):
                self.network_graph.remove_node(node)
                self.no_cover_targets_node.append(node)
                remove = True
            for target_id in range(cf.NUM_TARGETS):
                self.node_influences[target_id, node] = 0
        if remove:
            self.metric_closure = metric_closure(self.network_graph)


        # initialize the frequency and velocity vector to zero vector
        self.frequencies = np.zeros((self.num_bats, len(self.nodes)), dtype=np.float64)
        self.frequencies_r = np.zeros((self.num_bats, len(self.nodes)), dtype=np.float64)

        self.velocities = np.zeros((self.num_bats, len(self.nodes)), dtype=np.float64) # velocity for the sensing nodes
        self.velocities_r = np.zeros((self.num_bats, len(self.nodes)), dtype=np.float64)  # velocity for the relay nodes

        # initialize populations
        if self.initialize_populations():

            # find the current best
            if cf.CONNECTIVITY is True: # when considering connectivity
                self.values_f = [self.objective_function(self.bats_s[i][:], self.bats_r[i][:]) for i in
                                 range(self.num_bats)] # calculate the cost of objective function
                best_arg = np.argsort(self.values_f)[0] # find best couple with the least cost
                self.best_bat_s = np.copy(self.bats_s [best_arg])
                self.best_bat_r= np.copy(self.bats_r[best_arg])
                self.best_value = self.values_f[best_arg]

            else:
                self.values_f = [self.objective_function(self.bats_s [i][:], []) for i in
                                 range(self.num_bats)]
                best_arg = np.argsort(self.values_f)[0]
                self.best_bat_s = np.copy(self.bats_s [best_arg])
                self.best_value = self.values_f[best_arg]

            # main loop start
            for i in range(self.num_iterations):

                # update freq
                self.frequencies = self.freq_min + (self.freq_min - self.freq_max) * np.random.rand(
                    self.num_bats, len(self.nodes))
                self.frequencies_r = self.freq_min + (self.freq_min - self.freq_max) * np.random.rand(
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


                    # update velocity of bats_s
                    self.velocities_r[bat, :] += (self.bats_r[bat] - self.best_bat_r) * self.frequencies_r[
                        bat,:]

                    # update position vector of bat_r
                    relay_nodes = np.copy(self.bats_r[bat, :])
                    for i in range(len(self.nodes)):
                        if solution[i] != 1 or (not i in self.dead_nodes):
                            if np.random.rand() > self.r_p:  # pulse rate
                                relay_nodes[i] = self.best_bat_r[i]
                            else:
                                V_shaped_transfer_function = \
                                    abs(np.tanh(self.velocities[bat, i]))
                                if np.random.rand() < V_shaped_transfer_function:
                                    relay_nodes[i] = 0 if relay_nodes[i] == 1 else 1
                                    self.velocities[bat, i] /= 2  # velocity update
                        else:
                            relay_nodes[i] = 0

                    value = self.objective_function(solution, relay_nodes)
                    if value < self.values_f[bat] \
                            and self.validate_conn(solution + relay_nodes) and self.validate2(
                        solution):  # check a connectivity and sensing coverage

                        # if the solution improves, then updating the local bat and global best bat
                        self.bats_s [bat, :] = np.copy(solution[:])
                        self.bats_r[bat, :] = np.copy(relay_nodes[:])
                        self.values_f[bat] = value

                        # update the global best bat
                        if self.best_value > value:
                            self.best_bat_s = np.copy(solution[:])
                            self.best_bat_r= np.copy(relay_nodes[:])
                            self.best_value = value
                            # main loop end

        else:  # when failing initializing populations, then all nodes are turn on
            self.best_bat_s = np.ones(len(self.nodes))
            self.best_value = self.objective_function(self.best_bat_s, [])

        # set the node's state
        if self.best_bat_r is not None:  # turn on relay nodes
            for i, node in enumerate(self.nodes):
                if self.best_bat_r[i] >= 1 and node.energy >= cf.COMMUNICATION_ENERGY:
                    node.set_mode(cf.COMMUNICATION)
                else:
                    node.set_mode(cf.SLEEP)

        for i, node in enumerate(self.nodes): # turn on sensing nodes
            if self.best_bat_s[i] >= cf.ACTIVE and node.energy >= (cf.COMMUNICATION_ENERGY + cf.SENSING_ENERGY):
                node.set_mode(cf.ACTIVE)

        return self.best_value

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

        if cf.CONNECTIVITY is True:  # # initialize the position of all bat_s
            self.bats_r = np.zeros((self.num_bats, len(self.nodes)), dtype=np.int8)

            for b in range(self.num_bats):
                try:
                    node_set = self.add_relay_node(self.bats_s [b, :])
                except:
                    logging.info('can not find relay nodes for connectivity')
                    return False

                for i in node_set:
                    self.bats_r[b, i] = 1 if self.bats_s [b, i] == 0 else 0

        time_e = time.time()
        logging.info('initializing time :' + str(time_e - time_s))
        return True

    def calculate_influence(self):
        """
        Calculate the influence of each node for each target.

        influence :  the probability of each sensor node for each target.

        """
        self.node_influences = np.empty((len(self.targets), len(self.nodes)), dtype=np.float64)

        for t in range(len(self.targets)):
            for n in range(len(self.nodes)):
                self.node_influences[t][n] = self.sensing_matrix[t][n]

    def config_roulette_wheel(self, bat_id, idx):
        """
        Calculates the probability of selection for each node.

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

        prob_list = self.config_roulette_wheel(bat_id, target_id)  # calculate a probability for each nodes
        nodes_list = [i for i, value in enumerate(self.node_influences[target_id][:]) if value != 0]
        active = [i for i in self.bats_s [bat_id][nodes_list] if i >= 1]
        if len(nodes_list) == len(active) or len(nodes_list) == 0:
            return None

        while True:
            rand_val = np.random.rand()
            for i, prob in enumerate(prob_list):
                if prob >= rand_val:
                    if self.bats_s [bat_id][i] >= 1:
                        break
                    else:
                        return i

    def add_relay_node(self, node_set):
        """
        find relay nodes for connectivity by using a Steuner tree algorithm

        :param node_set: active nodes for sensing
        :return: active nodes  ( including nodes for sensing )
        """
        node_set = [i for i, b in enumerate(node_set) if b != 0]
        node_set.append(-1) # add the sink node
        G_star = self.steiner_tree_(self.metric_closure, node_set)
        active_nodes = list(G_star.nodes())
        active_nodes.remove(-1) # remove the sink node
        return active_nodes

    def objective_function(self, bs, br):
        cost_s = cf.COMMUNICATION_ENERGY + cf.SENSING_ENERGY  # cost of the sensing node
        cost_r = cf.COMMUNICATION_ENERGY  # cost of the relay node
        weight_s = sum([self.sensor_cost ** self.network.get_node(i).energy
                   for i, mode in enumerate(bs) if mode != cf.SLEEP])
        weight_r = sum([self.sensor_cost ** self.network.get_node(i).energy
                   for i, mode in enumerate(br) if mode != cf.SLEEP])
        return cost_s * weight_s + cost_r * weight_r


    def validate2(self, s):
        """
        This function uses a log matrix for minimizing computation time
        This function is faster than validate function.

        :return:
            True or False
        """

        nodes = [i for i, a in enumerate(s) if
                 a != cf.SLEEP and self.network.get_node(i).energy >= (cf.SENSING_ENERGY+cf.COMMUNICATION_ENERGY)]  # get list of the active nodes

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
        active_nodes = [idx for idx, value in enumerate(solution)  # remove not included nodes in solution
                        if value != 0 and idx not in self.dead_nodes and self.network.get_node(idx).energy >= cf.COMMUNICATION_ENERGY]

        visited = self.DFS(self.network_graph, active_nodes[0], active_nodes)
        if len(visited) == len(active_nodes):
            return True
        else:
            return False

    def validate_t(self, solution, t):
        """
         validate the sensing coverage for a target t

        :param: t: a target
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

    def DFS(self, graph, start, active_nodes):
        visited = []
        stack = [start]
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.append(n)
                adj = [node for node in list(graph.adj[n]) if node in active_nodes]
                stack += set(adj) - set(visited)

        return visited

    def steiner_tree_(self, M, terminal_nodes):
        H = M.subgraph(terminal_nodes)
        mst_edges = nx.minimum_spanning_edges(H, data=True)
        edges = chain.from_iterable(pairwise(d['path']) for u, v, d in mst_edges)
        T = self.network_graph.edge_subgraph(edges)
        return T

    def get_name(self):
        return '%s_%d_%d' % (self.name, self.num_bats,self.num_iterations)


