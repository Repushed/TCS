import config as cf
from protocol.scheduling import Scheduling
import logging
import numpy as np
from network import calculate_distance
from network import calculate_sensing_prob


class JOA(Scheduling):
    """
    Lee, Joon-Woo, Joon-Yong Lee, and Ju-Jang Lee.
    "Jenga-inspired optimization algorithm for energy-efficient coverage of unstructured WSNs."
    IEEE Wireless Communications Letters 2.1 (2013): 34-37.
    """

    name = 'JOA' # Jenga-inspired optimization algorithm

    def __init__(self):
        self.network = None
        self.nodes = None
        self.targets = None
        self.sensing_threshold = None
        self.sensing_matrix = None

        self.players_matrix = []
        self.sensors_matrix = []
        self.cover_matrix = []   # PoI (Point of Interest; target)
        self.score_board = []
        self.rank_list = None  # initialized at every new timeslot
        self.cost = []
        self.solution_set = []  # variable C in the paper

        self.num_players = 100
        self.num_turns = 100

    def set_mode(self, network, rounds):
        if rounds == 0:  # get network information
            self.network = network
            self.nodes = network.get_nodes()
            self.targets = network.get_targets()
            self.sensing_threshold = self.network.get_log_threshold()
            self.sensing_matrix = self.network.get_log_matrix()

        self.initialize()

        for n_t in range(self.num_turns):
            # logging.info('turns'+ str(n_t))
            for n_p in range(self.num_players):
                for _ in range(len(self.nodes)): # while player k meets Eq.(5)  --> the maximum number of the selected sensors cannot be exceeded than the number of all the sensors in network

                    sensor_id = self.sensor_selection(n_p)  # by  using the roulette wheel method

                    if sensor_id is None:
                        break

                    self.players_matrix[n_p][sensor_id] = 0

                    if self.validate(self.players_matrix[n_p][:]) is False:
                        self.players_matrix[n_p][sensor_id] = 2
                        break

            # select solution
            self.rank_process()
            self.update_score()


        # set the mode for each sensor
        for i, node in enumerate(self.nodes):
            if self.rank_list[0][i] != cf.SLEEP and node.energy >= cf.SENSING_ENERGY:
                node.set_mode(cf.ACTIVE)
            else:
                node.set_mode(cf.SLEEP)

        return self.cost[0]  # return the objective function value of best solution

    def initialize(self):
        self.players_matrix = np.ones((self.num_players, len(self.nodes)), dtype=np.int32)

        if len(self.nodes) != self.network.get_num_alive_nodes():  # check num of dead nodes
            for p in range(self.num_players):
                for node in self.network.get_dead_nodes():
                    self.players_matrix[p][node.id] = 0

        self.rank_list = None
        self.update_info()
        self.score_board = [self.sensors_matrix[i] * self.cover_matrix[i] for i in range(len(self.nodes))]

    def update_info(self):
        self.sensors_matrix = [node.energy for node in self.nodes]
        self.cover_matrix = []
        for node in self.nodes:
            cover_num = sum([1 for target in self.targets
                             if calculate_sensing_prob(node, target) > 0])
            self.cover_matrix.append(cover_num)

    def sensor_selection(self, player_id):
        prob_list = self.config_roulette_wheel()
        value = None

        if not (1 in self.players_matrix[player_id][:]):
            return value

        for _ in range(cf.NUM_NODES *cf.NUM_TARGETS):
            rand_val = np.random.rand()
            for i, prob in enumerate(prob_list):
                if prob >= rand_val:
                    value = i
                    break  # end for loop

            # logging.info(str(self.players_matrix[player_id][:]))
            if self.players_matrix[player_id][value] != 1:
                continue
            else:
                return value

    def rank_process(self):  # sort and cut rank
        if self.rank_list is not None:
            self.rank_list = np.concatenate((self.rank_list, self.players_matrix), axis=0)
        else: # first turn
            self.rank_list = np.copy(self.players_matrix)

        self.cost = []

        for i in range(len(self.rank_list)):
            self.cost.append(self.objective_function(self.rank_list[i][:]))

        idx = np.argsort(self.cost) # sort suing Eq.(4)

        self.cost = np.array(self.cost)[idx]
        self.rank_list = self.rank_list[idx]
        rank_list_length = len(self.rank_list)

        if rank_list_length > self.num_players: # Cut L_rank by N_P
            self.rank_list = np.delete(self.rank_list, np.arange(self.num_players, rank_list_length, 1), 0) # cut
            self.cost = self.cost[:self.num_players]

    def update_score(self):

        # n_k = [sum(self.rank_list[n_p][:]) for n_p in range(self.num_players)]
        n_k = [sum([1 for v in self.rank_list[n_p][:] if v != 0]) for n_p in range(self.num_players)]
        # logging.info('rank' + str(self.rank_list))
        second_term = sum([self.cost[0] / (n_k[n_p] * self.cost[n_p]) for n_p in range(self.num_players)])

        for i in range(len(self.nodes)):
            self.score_board[i] += second_term  # * n_i[i]

    def config_roulette_wheel(self):
        sum_score = sum(self.score_board)

        prob = []
        for i in range(len(self.score_board)):
            prob.append(sum(self.score_board[0:i+1]) / sum_score)

        return prob

    def objective_function(self, x):
        return sum([cf.SENSOR_COST ** self.network.get_node(i).energy
                    for i, mode in enumerate(x) if mode != cf.SLEEP
                    and self.network.get_node(i).energy > 0])

    def validate(self, s):
        """
        This function validates a sensing coverage by using a log matrix for minimizing computation time
        """

        nodes = [i for i, a in enumerate(s) if a != cf.SLEEP
                 and self.network.get_node(i).energy >=cf.SENSING_ENERGY]  # get list of the active nodes

        for t in range(cf.NUM_TARGETS):
            no_sense_prob = 0
            for n in nodes:
                no_sense_prob += self.sensing_matrix[t][n]
                if no_sense_prob >= self.sensing_threshold:
                    break

            if no_sense_prob < self.sensing_threshold:
                return False
        return True

