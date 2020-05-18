import os
import config as cf
import logging, sys
from network import Network
from protocol.pso import *
from protocol.aco import *
from protocol.joa import *
from protocol.proposed_sc import *

from protocol.psca import *
from protocol.proposed import *
from protocol.proposed_rand_init import *
from protocol.bba import *

from result_utils import *



def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    protocols = cf.PROTOCOLS

    # draw_sensing_model()
    if not os.path.exists(cf.DIR_NAME):  # create dir
        os.makedirs(cf.DIR_NAME)

    result_columns = ['protocol', 'num_of_targets', 'num_of_nodes',
                      'first_nodes_die', 'lifetime', 'avg_computation_time', 'run']
    save_csv(result_columns)

    for scenario in cf.SCENARIO_LIST:
        cf.NUM_NODES = scenario[0]
        cf.NUM_TARGETS = scenario[1]

        cf.SUB_DIR_NAME = cf.DIR_NAME + str(cf.NUM_TARGETS) + '_' + str(cf.NUM_NODES) + '/'
        if not os.path.exists(cf.SUB_DIR_NAME):  # create sub dir for each scenario
            os.makedirs(cf.SUB_DIR_NAME)

        for i in range(cf.NUM_OF_RUN):
            costs = []
            active_node_trace = []
            cf.SUB_SUB_DIR_NAME = cf.DIR_NAME + str(cf.NUM_TARGETS) + '_' + str(cf.NUM_NODES) + '/' + str(i) + '/'
            if not os.path.exists(cf.SUB_SUB_DIR_NAME):  # create sub sub dir for one run of the scenario
                os.makedirs(cf.SUB_SUB_DIR_NAME)

            network = Network(i)
            plot_network(network, filename='net_model_' + str(i),  file_type='png', save_dir=cf.SUB_SUB_DIR_NAME,)

            # simulation start
            for p in protocols:
                logging.info(str(p) + ' simulation start----------------')
                protocol_class = eval(p[0])
                network.scheduling_protocol = protocol_class()

                if p[1] is not None:
                    for var in p[1:]:
                        network.scheduling_protocol.replace(var[0],var[1])


                network.simulate()

                result = network.get_result()
                result.append(i)
                save_csv(result)

                active_node_trace.append(network.get_num_of_active_nodes_trace())
                costs.append(network.get_cost_trace())

                logging.info('simulation end----------------')
                network.reset()

            # save trace results
            for p in range(len(protocols)):
                save_csv(costs[p], file_name='cost_' + str(i), save_dir=cf.SUB_DIR_NAME)
                save_csv(active_node_trace[p], file_name='active_node_trace_' + str(i), save_dir=cf.SUB_DIR_NAME)


if __name__ == '__main__':
    main()
