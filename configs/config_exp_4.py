import time

# result directory
DATE_TIME = time.strftime("%m%d%H%M")
DIR_NAME = './result/' + DATE_TIME + '/'
SUB_DIR_NAME = None
SUB_SUB_DIR_NAME = None
CREATE_GRAPH = False

PROTOCOLS = [
    ['PROPOSED_SC',['num_bats',20,],['num_iterations',1000]],   # Proposed Method
    ['PROPOSED_SC',['num_bats',30,],['num_iterations',1000]],   # Proposed Method
    ['ACO',None],                                                   # Ant Colony Optimization
    ['JOA',None],                                                   # Jenga-inspired optimization algorithm
    ['PSO',None],                                                   # Particle Swarm Optimization


]



SCENARIO_LIST = [  # [num of nodes, num of targets]

    [100, 10],
    [150, 10],
    [200, 10],
    [100, 30],
    [150, 30],
    [200, 30],

]

NUM_OF_RUN = 10 # number of run for each scenario

# sensor area
AREA_WIDTH = 10
AREA_LENGTH = 10


NUM_NODES = 0
NUM_TARGETS = 0

FIX_POS_TARGET = True
FIX_POS_NODE = True

# mode
SLEEP = 0           # sleep mode
ACTIVE = 1          # communication + sensing mode
COMMUNICATION = 2   # relay mode

CONNECTIVITY = False


# DATA_SIZE = 100  # bytes

# energy
INIT_ENERGY = 10  #
SENSING_ENERGY = 1  #
COMMUNICATION_ENERGY =0  #
SLEEP_ENERGY = 0


## When considering connectivity
SINK_POS_X = 0
SINK_POS_Y = 0


# max number of iterations (time slot; lifetime)
MAX_ROUNDS = 10000

# prob sensing model parameter
SENSING_RANGE_S = 1.5   # R_s = 1.5 m
SENSING_RANGE_U = 6.0 # R_u = 6.0 m
COMMUNICATION_RANGE = 2 * SENSING_RANGE_U

SENSING_K = 0.5
SENSING_M = 0.5
SENSING_THRESHOLD = 0.999
SENSOR_COST = 0.9   # range (0,1] ; SENSOR_COST ** remaining_energy

