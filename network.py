from node import Node
from target import Target
import config as cf
import math
import numpy as np
from protocol.scheduling import *
import logging
import result_utils
import time
import networkx as nx

class Network(object):

    def __init__(self, i):
        self.i = i  # i-th run for each scenario
        self.scheduling_protocol = None

        # create targets
        self.targets = [Target(i) for i in range(0, cf.NUM_TARGETS)]

        # create nodes
        self.nodes = [Node(i) for i in range(0, cf.NUM_NODES)]

        # create matrix of sensing probability
        self.matrix = np.zeros((cf.NUM_TARGETS, cf.NUM_NODES), dtype=float)
        self.threshold = 1 - cf.SENSING_THRESHOLD  # 1 -  sensing threshold
        self.create_matrix()  # create matrix and log_matrix

        #
        self.log_matrix = -np.log(self.matrix)
        self.log_threshold = -np.log(self.threshold)  # ln (no sensing threshold)

        # result trace values
        self.num_of_active_nodes_list = []  # the numbers of active nodes at each time slot
        self.costs = []  # objective function value at each time slot
        self.result = []

        # create Communication graph
        if cf.CONNECTIVITY:
            self.sink_node = Node(-1, sink=True)
            self.comm_graph = self.create_comm_graph()

        else:
            cf.COMMUNICATION_ENERGY = 0

    def reset(self):
        for node in self.nodes:
            node.reset()

    def simulate(self):

        self.result = [self.get_protocol_name(), cf.NUM_TARGETS, cf.NUM_NODES]
        first_node_die_round = cf.MAX_ROUNDS # network lifetime
        time_list = []  # computation time for each timeslot
        self.num_of_active_nodes_list = [self.get_protocol_name()]  # Number of active node at each time slots
        self.costs = [self.get_protocol_name()]  # objective function value at each time slot

        end = False
        for rounds in range(cf.MAX_ROUNDS):  # round : timeslot
            logging.info('-------------------------------------')

            # set the mode of each node
            start_time = time.time()
            cost = self.scheduling_protocol.set_mode(self, rounds)
            computation_time = time.time() - start_time

            # output the results at a time slot
            num_of_active_nodes = len(self.get_active_nodes())
            nodes_idx = [n.id for n in self.get_active_nodes()]
            nodes = self.get_alive_nodes()
            nodes_energy = [n.energy for n in self.get_active_nodes()]
            time_list.append(computation_time)

            logging.info('round ' + str(rounds) + ' num. of active sensors :'
                         + str(num_of_active_nodes))
            logging.info('Active nodes\' ID :' + str(nodes_idx))
            logging.info('Active nodes\' remaining energy after sensing :' + str(nodes_energy))
            logging.info('number of alive nodes :' + str(len(nodes)))
            #logging.info('sum remaining energy : ' + str(sum(self.get_remaining_energy())))
            logging.info('compution time :' + str(computation_time))
            logging.info('cost :' + str(cost))

            # validation step
            cover = self.cover_validation() # validate the sensing coverage

            if cf.CONNECTIVITY: # validate the connectivity
                conn_validation = self.conn_validation()

            # consume the energy of sensor nodes
            for node in self.nodes:
                node.consume_energy()
                if node.energy == 0:
                    if rounds < first_node_die_round:  # round that first node dies
                        self.result.append(rounds)
                        first_node_die_round = rounds

            #
            if cover:
                self.num_of_active_nodes_list.append(num_of_active_nodes)
                self.costs.append(cost)

            # create network graph at a timeslot
            if cf.CREATE_GRAPH is True and rounds %5 ==0:
                result_utils.plot_network(self,
                                          filename=str(self.i) + '_' + self.result[0] + '_' + str(rounds),
                                          file_type='pdf', save_dir=cf.SUB_SUB_DIR_NAME)

                if cf.CONNECTIVITY ==True:
                    result_utils.plot_network(self,coverage = 'communication',
                                              filename=str(self.i) + '_' + self.result[0] + '_' + str(rounds),
                                              file_type='pdf', save_dir=cf.SUB_SUB_DIR_NAME)

            # check whether network is over or not
            if end or (not cover):
                # when not ensuring sensing coverage lifetime
                life_time = rounds
                avg_computation_time = np.average(time_list)

                logging.info('network lifetime is ' + str(life_time - 1))
                logging.info('avg computation time is ' + str(avg_computation_time))

                if life_time == 0:
                    self.result.append(0)
                    self.result.append(0)
                else:
                    self.result.append(life_time - 1)
                    self.result.append(avg_computation_time)

                return

            if cf.CONNECTIVITY and (not conn_validation):
                # when not ensuring the connectivity between the sink and the active sensors
                life_time = rounds
                avg_computation_time = np.average(time_list)

                logging.info('network lifetime is ' + str(life_time - 1))
                logging.info('avg computation time is ' + str(avg_computation_time))
                if life_time == 0:
                    self.result.append(0)
                    self.result.append(0)
                else:
                    self.result.append(life_time - 1)
                    self.result.append(avg_computation_time)
                return

            self.sleep_mode()

        logging.info('Exceeded maximum number of iterations')
        self.result.append(cf.MAX_ROUNDS)
        return

    def get_result(self):
        return self.result

    def get_cost_trace(self):
        return self.costs

    def get_num_of_active_nodes_trace(self):
        return self.num_of_active_nodes_list

    def get_remaining_energy(self):
        return [node.energy for node in self.get_nodes()]

    def get_alive_nodes(self):
        nodes_list = [node for node in self.nodes if node.energy > 0]
        return nodes_list

    def get_dead_nodes(self):
        nodes_list = [node for node in self.nodes if node.energy <= 0]
        return nodes_list

    def get_num_alive_nodes(self):
        return len(self.get_alive_nodes())

    def get_targets(self):
        return self.targets

    def get_node(self, idx):
        return [node for node in self.nodes if idx == node.id][0]

    def get_target(self, idx):
        return [target for target in self.targets if idx == target.id][0]

    def get_nodes(self):
        return self.nodes

    def get_active_nodes(self):
        return [node for node in self.nodes if cf.SLEEP != node.mode]

    def get_sleep_nodes(self):
        return [node for node in self.nodes if cf.SLEEP == node.mode]

    def get_num_active_nodes(self):
        return len(self.get_active_nodes())

    def cover_validation(self):
        result = True
        active_nodes = [node for node in self.get_nodes() if
                        node.energy >= (cf.COMMUNICATION_ENERGY+cf.SENSING_ENERGY) and node.mode == cf.ACTIVE]
        prob = []
        target_node_list = []
        for target in self.targets:
            node_list = []
            no_sense_prob = 1
            for node in active_nodes:
                n_prob = 1-calculate_sensing_prob(node, target)
                if n_prob < 1:
                    no_sense_prob *= n_prob
                    node_list.append([node.id, n_prob])

            sense_prob = 1 - no_sense_prob
            if sense_prob < cf.SENSING_THRESHOLD:
                logging.info('not covered target id : ' + str(target.id)+
                             '\n sensing probability :'+str(sense_prob))
                result = False

            target_node_list.append(node_list)
            prob.append(sense_prob)

        logging.info('sensing prob. for each target: ' + str(prob))
        return result

    def conn_validation(self):# 변경
        """
        validate connectivity

        :return: True or False
        """

        G = self.comm_graph.copy()
        sleep_nodes = self.get_sleep_nodes()

        for node in sleep_nodes:  # remove not included nodes in solution
            G.remove_node(node.id)

        # remove active nodes which remaining energy is smaller than COMMUNICATION_ENERGY
        active_nodes = [node for node in self.get_nodes() if
                        node.energy < cf.COMMUNICATION_ENERGY and node.mode == cf.ACTIVE]

        for node in active_nodes:  # remove not included nodes in solution
            G.remove_node(node.id)

        try:
            return nx.is_connected(G)
        except:
            return False

    def create_matrix(self):  # target-node no sensing prob matrix
        for t in range(cf.NUM_TARGETS):
            for n in range(cf.NUM_NODES):
                self.matrix[t][n] = 1 - calculate_sensing_prob(self.get_node(n), self.get_target(t))
        # logging.info(str(self.matrix))

    def create_comm_graph(self):
        """
        Create communication graph

        vertices : nodes
        edges : if two nodes can directly communicate with, then they are connected by edges

        :return:
            graph

        """
        len_nodes = len(self.nodes)
        G = nx.Graph()

        G.add_node(self.sink_node.id, pos=(self.sink_node.pos_x, self.sink_node.pos_y))

        for i in range(len_nodes):
            G.add_node(self.nodes[i].id, pos=(self.nodes[i].pos_x, self.nodes[i].pos_y))
            # logging.info(str(G.node[i]))
        # logging.info(str(G.node))
        for i in range(len_nodes):
            for j in range(i + 1, len_nodes):
                if calculate_distance(self.nodes[i], self.nodes[j]) <= cf.COMMUNICATION_RANGE:
                    G.add_edge(i, j, weight=1)
            #
            if calculate_distance(self.nodes[i], self.sink_node) <= cf.COMMUNICATION_RANGE:
                G.add_edge(i, self.sink_node.id, weight=1)

        return G

    def get_comm_graph(self):
        return self.comm_graph

    def get_matrix(self):
        return self.matrix

    def get_log_matrix(self):
        return self.log_matrix

    def get_threshold(self):
        return self.threshold

    def get_log_threshold(self):
        return self.log_threshold

    def get_protocol_name(self):
        return self.scheduling_protocol.get_name()

    def sleep_mode(self):
        for node in self.nodes:
            node.set_mode(cf.SLEEP)

    def active_mode(self):
        for node in self.get_active_nodes():
            node.set_mode(cf.ACTIVE)


def calculate_distance(node1, node2):
    x1 = node1.pos_x
    y1 = node1.pos_y
    x2 = node2.pos_x
    y2 = node2.pos_y
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_sensing_prob(node, target):
    distance = calculate_distance(node, target)
    if node.sensing_range_s >= distance:
        return 1
    elif node.sensing_range_u > distance:
        return np.exp(-cf.SENSING_K * ((distance - node.sensing_range_s) ** cf.SENSING_M))
    else:
        return 0
