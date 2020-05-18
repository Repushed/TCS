
import numpy as np
import config as cf


class Target(object):

    def __init__(self, id):
        self.id = id
        self.pos_x = round(np.random.uniform(0, cf.AREA_WIDTH), 1)
        self.pos_y = round(np.random.uniform(0, cf.AREA_LENGTH), 1)

    def change_position(self):
        self.pos_x = round(np.random.uniform(0, cf.AREA_WIDTH), 1)
        self.pos_y = round(np.random.uniform(0, cf.AREA_LENGTH), 1)
