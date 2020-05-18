
import numpy as np
import config as cf


class Node(object):

    def __init__(self, id, sink = False):
        self.id = id

        self.mode = cf.SLEEP #  state of the node

        # If nodes are heterogeneous nodes, you must change the values below.
        # self.data_size = cf.DATA_SIZE
        self.sensing_range_s = cf.SENSING_RANGE_S
        self.sensing_range_u = cf.SENSING_RANGE_U
        self.communication_range = cf.COMMUNICATION_RANGE

        self.energy = cf.INIT_ENERGY  # current energy
        self.sensing_energy = cf.SENSING_ENERGY
        self.sleep_energy = cf.SLEEP_ENERGY
        self.communication_energy = cf.COMMUNICATION_ENERGY

        if sink:
            self.pos_x = cf.SINK_POS_X
            self.pos_y = cf.SINK_POS_Y
            self.energy = float('inf') # infinite energy
            self.mode= cf.ACTIVE
        else:
            self.pos_x = np.random.uniform(0, cf.AREA_WIDTH)
            self.pos_y = np.random.uniform(0, cf.AREA_LENGTH)


    def set_mode(self, value):
        self.mode = value

    def get_mode(self):
        return True if self.mode else False

    def consume_energy(self):
        if cf.CONNECTIVITY is False:
            if self.mode == cf.ACTIVE: # active node (sesning node)
                self.energy -= self.sensing_energy
            else: # sleep node
                self.energy -= self.sleep_energy

        else:
            if self.mode == cf.ACTIVE: # sensing node
                self.energy -= (self.sensing_energy + self.communication_energy)

            elif self.mode == cf.COMMUNICATION: # relay node
                self.energy -= self.communication_energy

            else: # sleep node
                self.energy -= self.sleep_energy

    def reset(self):
        self.energy = cf.INIT_ENERGY
        self.mode = cf.SLEEP

    def change_position(self):  # for mobile node.....  but it is not implemented yet
        self.pos_x = np.random.uniform(0, cf.AREA_WIDTH)
        self.pos_y = np.random.uniform(0, cf.AREA_LENGTH)