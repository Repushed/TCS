import time

# result directory
DATE_TIME = time.strftime("%m%d%H%M")
DIR_NAME = './result/' + DATE_TIME + '/'
SUB_DIR_NAME = None
SUB_SUB_DIR_NAME = None
CREATE_GRAPH = True

# experiment 1 :Finding Optimal Numbers of Parameters
PROTOCOLS = [ # protocol_name , [ variable_name1, value1], [ variable_name2, value2], [ variable_name3, value3] ....]

    ['PROPOSED', ['num_bats', 5, ], ['num_iterations', 100], ],
    ['PROPOSED', ['num_bats', 10, ], ['num_iterations',100], ],
    ['PROPOSED', ['num_bats', 20, ], ['num_iterations',100], ],
    ['PROPOSED', ['num_bats', 30, ], ['num_iterations',100], ],
    ['PROPOSED', ['num_bats', 40, ], ['num_iterations',100], ],

    ['PROPOSED', ['num_bats', 5, ], ['num_iterations', 250], ],
    ['PROPOSED', ['num_bats', 10, ], ['num_iterations', 250], ],
    ['PROPOSED', ['num_bats', 20, ], ['num_iterations', 250], ],
    ['PROPOSED', ['num_bats', 30, ], ['num_iterations', 250], ],
    ['PROPOSED', ['num_bats', 40, ], ['num_iterations', 250], ],

    ['PROPOSED', ['num_bats', 5, ], ['num_iterations', 500], ],
    ['PROPOSED', ['num_bats', 10, ], ['num_iterations', 500], ],
    ['PROPOSED', ['num_bats', 20, ], ['num_iterations', 500], ],
    ['PROPOSED', ['num_bats', 30, ], ['num_iterations', 500], ],
    ['PROPOSED', ['num_bats', 40, ], ['num_iterations', 500], ],

    ['PROPOSED', ['num_bats', 5, ], ['num_iterations', 1000], ],
    ['PROPOSED', ['num_bats', 10, ], ['num_iterations', 1000], ],
    ['PROPOSED', ['num_bats', 20, ], ['num_iterations', 1000], ],
    ['PROPOSED', ['num_bats', 30, ], ['num_iterations', 1000], ],
    ['PROPOSED', ['num_bats', 40, ], ['num_iterations', 1000], ],

    ['PROPOSED', ['num_bats', 5, ], ['num_iterations', 2000], ],
    ['PROPOSED', ['num_bats', 10, ], ['num_iterations', 2000], ],
    ['PROPOSED', ['num_bats', 20, ], ['num_iterations', 2000], ],
    ['PROPOSED', ['num_bats', 30, ], ['num_iterations', 2000], ],
    ['PROPOSED', ['num_bats', 40, ], ['num_iterations', 2000], ],

]


SCENARIO_LIST = [  # [num of nodes, num of targets]

    [300, 20],

]

NUM_OF_RUN = 10 # number of run for each scenario

# sensor area
AREA_WIDTH = 75
AREA_LENGTH = 75


NUM_NODES = 0
NUM_TARGETS = 0

FIX_POS_TARGET = True
FIX_POS_NODE = True

# mode
SLEEP = 0           # sleep mode
ACTIVE = 1          # communication + sensing mode
COMMUNICATION = 2   # relay mode

CONNECTIVITY = True


# DATA_SIZE = 100  # bytes

# energy
INIT_ENERGY = 30  #
SENSING_ENERGY = 1  #
COMMUNICATION_ENERGY = 2  #
SLEEP_ENERGY = 0


## When considering connectivity
SINK_POS_X = 0
SINK_POS_Y = 0


# max number of iteration (time slot; lifetime)
MAX_ROUNDS = 10000

# prob sensing model parameter
SENSING_RANGE_S = 10   # R_s
SENSING_RANGE_U = 16.5 # R_u
COMMUNICATION_RANGE = 2 * SENSING_RANGE_U

SENSING_K = 0.5
SENSING_M = 0.5
SENSING_THRESHOLD = 0.90
SENSOR_COST = 0.9   # range (0,1] ; SENSOR_COST ** remaining_energy
