import config as cf
from protocol.scheduling import Scheduling
import logging
import copy
import numpy as np
from network import calculate_distance
from network import calculate_sensing_prob
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.approximation import *

class PSCA(Scheduling):
    """
    Shan, Anxing, Xianghua Xu, and Zongmao Cheng.
    "Target coverage in wireless sensor networks with probabilistic sensors."
    Sensors 16.9 (2016): 1372.

    we use networkX package to implement steiner Tree algorithm. This algorithm in networkX is similar to the second paper[Kou81]
    reference:
        Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart,
        “Exploring network structure, dynamics, and function using NetworkX”
        , in Proceedings of the 7th Python in Science Conference (SciPy2008),
        Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008


        Kou, L., George Markowsky, and Leonard Berman.
        "A fast algorithm for Steiner trees."
        Acta informatica 15.2 (1981): 141-145.

    """

    name = 'PSCA'

    def __init__(self):
        self.network = None
        self.nodes = None
        self.targets = None
        self.sensing_threshold = None
        self.sensing_matrix = None
        self.network_graph = None

        self.candidate_cover_set = []
        self.gain_set = None

    def set_mode(self, network, rounds):
        if rounds == 0:
            self.network = network
            self.nodes = network.get_nodes()
            self.targets = network.get_targets()
            self.sensing_threshold = self.network.get_log_threshold()
            self.sensing_matrix = self.network.get_log_matrix()
            self.network_graph = self.network.get_comm_graph().copy()

        self.dead_nodes =[n.id for n in network.get_nodes() if n.energy < cf.COMMUNICATION_ENERGY +cf.SENSING_ENERGY]
        for node in self.dead_nodes:
            if self.network_graph.has_node(node):
                self.network_graph.remove_node(node)


        self.create_candidate_set()
        active_node_set = list(self.psca())

        active_node_set.append(-1)
        try:
            relay_node_set = list(self.add_relay_node(active_node_set))
        except:
            relay_node_set = active_node_set

        active_node_set.remove(-1)
        relay_node_set.remove(-1)
        print(active_node_set)
        print(relay_node_set)

        for node_id in relay_node_set:
            node = self.network.get_node(node_id)
            if node.energy >= cf.COMMUNICATION_ENERGY:
                node.set_mode(cf.COMMUNICATION)

        for node_id in active_node_set:
            node = self.network.get_node(node_id)
            if node.energy >= cf.COMMUNICATION_ENERGY + cf.SENSING_ENERGY:
                node.set_mode(cf.ACTIVE)

        return None

    def create_candidate_set(self):
        self.candidate_cover_set = [[] for _ in range(cf.NUM_TARGETS)]
        self.gain_set = [[] for _ in range(cf.NUM_TARGETS)]
        for t in range(cf.NUM_TARGETS):
            matrix_t = [[node_id, value] for node_id, value in enumerate(self.sensing_matrix[t][:])
                        if value > 0 and
                        self.network.get_node(node_id).energy >= (cf.COMMUNICATION_ENERGY + cf.SENSING_ENERGY)]
            s = sorted(matrix_t, key=lambda x: x[1], reverse=True)
            self.gain_set[t] = s

        for t in range(cf.NUM_TARGETS):
            self.__DFS(0, 0, [], t)  # t-th target

    def psca(self):
        sensor_set = set([])
        uncovered_set = set(range(cf.NUM_TARGETS))

        f_s = np.zeros((cf.NUM_NODES,), dtype=np.int32)

        # find sensors that exist in candidate_cover_set for each target t
        node_candidate_set = [[] for _ in range(cf.NUM_TARGETS)]
        for t, set_t in enumerate(self.candidate_cover_set):
            for s in set_t:
                node_candidate_set[t] += s

            node_candidate_set[t] = list(set(node_candidate_set[t]))

        # calculate f value for each node
        for t in range(cf.NUM_TARGETS):
            for node_id in node_candidate_set[t]:
                f_s[node_id] += 1  # if sensor n exist in C_i

        for t, set_t in enumerate(self.candidate_cover_set):
            # find a set that has minimum number of sensors in set_t
            # select an independent coverage set c from a candidate coverage set C_k, k
            minimum_set = [[i for i in range(cf.NUM_NODES)]]
            s_union_c_length = cf.NUM_NODES

            for s in set_t:
                length = len(sensor_set.union(set(s)))
                if s_union_c_length > length:
                    minimum_set = [s]
                    s_union_c_length = length
                elif s_union_c_length == length:
                    minimum_set.append(s)

            if len(minimum_set) > 1:
                # if the number of elements in minimum_set is more than one, then select the highest weight set
                highest_weight_set = []
                highest_weight = -1

                for i in range(len(minimum_set)):
                    tmp_weight = 0
                    if len(minimum_set[i]) == 1:
                        tmp_weight = f_s[minimum_set[i]]

                    else:
                        for node_id in minimum_set[i]:  # calculate a i-th set's weight
                            tmp_weight += f_s[node_id]

                    if tmp_weight > highest_weight:  # compare value btw the highest weight and the i-th set's weight
                        highest_weight_set = minimum_set[i]
                        highest_weight = tmp_weight
                minimum_set = set(highest_weight_set)

            else:
                minimum_set = set(minimum_set[-1])

            uncovered_set.remove(t)
            sensor_set = sensor_set.union(minimum_set)


        return sensor_set

    def add_relay_node(self, node_set):
        G_star = steiner_tree(self.network_graph, node_set)
        return G_star.nodes()

    def __DFS(self, index_s, total_gains, c_z, index_t):
        c = copy.deepcopy(c_z)

        if index_s >= len(self.gain_set[index_t]):
            return

        if total_gains > self.sensing_threshold:
            self.candidate_cover_set[index_t].append(c)
            return


        self.__DFS(index_s+1, total_gains, c, index_t)
        gain = self.gain_set[index_t][index_s][1]
        c.append(self.gain_set[index_t][index_s][0])

        self.__DFS(index_s+1, total_gains+gain, c, index_t)

    def validate(self, s):
        """
        This function uses a log matrix for minimizing computation time
        """

        nodes = [i for i, a in enumerate(s) if a != cf.SLEEP
                 and self.network.get_node(i).energy > (cf.SENSING_ENERGY+cf.COMMUNICATION_ENERGY)]  # get list of the active nodes


        for t in range(cf.NUM_TARGETS):
            no_sense_prob = 0
            for n in nodes:
                no_sense_prob += self.sensing_matrix[t][n]
                if no_sense_prob > self.sensing_threshold:
                    break

            if no_sense_prob <= self.sensing_threshold:
                return False
        return True
