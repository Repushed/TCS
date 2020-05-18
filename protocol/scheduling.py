import config as cf
import numpy as np

class Scheduling(object):
    name = 'None'
    def set_mode(self, network, rounds):
        nodes = network.get_alive_nodes()

        for node in nodes:
            node.set_mode(cf.ACTIVE)

        return np.inf  # return the objective function value of best solution

    def get_name(self):
        return self.name


    def replace(self,var_name, value):
        setattr(self,var_name,value)


