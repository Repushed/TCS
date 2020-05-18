import matplotlib.pyplot as plt
from matplotlib.pyplot import text
from node import *
from target import *
from network import *
import config as cf
import numpy as np
import csv
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

def plot_network(network, coverage='sensing', mode='save', filename='result', file_type = 'pdf', save_dir=cf.DIR_NAME):
    nodes = [node for node in network.get_alive_nodes()]
    targets = network.get_targets()
    dead_nodes = [node for node in network.get_dead_nodes()]

    X_a = [node.pos_x for node in nodes if node.mode == cf.ACTIVE]
    Y_a = [node.pos_y for node in nodes if node.mode == cf.ACTIVE]

    X_s = [node.pos_x for node in nodes if node.mode == cf.SLEEP]
    Y_s = [node.pos_y for node in nodes if node.mode == cf.SLEEP]

    X_d = [node.pos_x for node in dead_nodes]
    Y_d = [node.pos_y for node in dead_nodes]

    X_t = [target.pos_x for target in targets]
    Y_t = [target.pos_y for target in targets]

    plt.figure()
    ax = plt.gca()

    plt.xlim([0, cf.AREA_WIDTH])
    plt.ylim([0, cf.AREA_LENGTH])
    ax.set_aspect(1.0)

    plt.grid()
    plt.scatter(X_a, Y_a, marker='o', label='sensing node', color='k', s=15, zorder=2 )
    plt.scatter(X_s, Y_s, marker='^', label='sleep node', color='y', alpha=0.5, s=5, zorder=2 )
    plt.scatter(X_d, Y_d, marker='x', label='dead node', color='dimgray', alpha=0.5, s=5, zorder=2 )
    plt.scatter(X_t, Y_t, marker='*', label='PoI', color='m', s=15, zorder=2 )

    if len(X_a) == 0:
        for node in network.get_nodes():
            ax.annotate(node.id, (node.pos_x, node.pos_y), color='k', fontsize=8, zorder=3)

    for node in network.get_active_nodes():
        ax.annotate(node.id, (node.pos_x, node.pos_y), color='r', fontsize=10, zorder=3)

    for target in network.get_targets():
        ax.annotate(target.id, (target.pos_x, target.pos_y), color='m', fontsize=10, zorder=3)

    if coverage == 'sensing':
        # plot sensing coverage

        for i in range(len(X_a)):
            circle1 = plt.Circle((X_a[i], Y_a[i]), cf.SENSING_RANGE_S, color='b', linestyle=(0,(1,5)),fill=False, alpha=0.5 ,zorder=2 )
            ax.add_artist(circle1)

        for i in range(len(X_a)):
            cnt = 0
            fis = reversed(np.linspace(0, cf.SENSING_RANGE_U, 200, endpoint=True))
            prev = 0
            for r in fis:
                cnt += 1
                a = __calculate_sensing_prob(cf.SENSING_RANGE_S, cf.SENSING_RANGE_U, r)

                if prev == 1:
                    circle = plt.Circle((X_a[i], Y_a[i]), r, color='deepskyblue', alpha=1)
                    ax.add_artist(circle)
                    break
                else:
                    x = 1 - (a - 1) / (prev - 1)
                    circle = plt.Circle((X_a[i], Y_a[i]), r, color='deepskyblue', alpha=x)
                    ax.add_artist(circle)
                    prev = a
        
        if len(X_a) != 0:
                # create color bar 
                colors = ['w', 'deepskyblue']

                cm = LinearSegmentedColormap.from_list('custom', colors, N=256)
                cmap = cm
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ticks=np.linspace(0, 1, 11),
                     boundaries=np.arange(0, 1, .001))


        # plot legend 
        if cf.CONNECTIVITY is True:
            # sink node
            sink =plt.scatter(0., 0., marker='P', label='sink node', color='r', s=100 ,zorder=3 )
            sink.set_clip_on(False)
            # relay nodes
            X_r = [node.pos_x for node in nodes if node.mode == cf.COMMUNICATION]
            Y_r = [node.pos_y for node in nodes if node.mode == cf.COMMUNICATION]
            plt.scatter(X_r, Y_r, marker='D', label='relay node', color='g', s=15, zorder=3)
            plt.legend(bbox_to_anchor=(0., 1.00, 1.0, 0.1), loc='upper center', ncol=6, fontsize='xx-small')
        else:
	    
            plt.legend(bbox_to_anchor=(0., 1.00, 1.0, 0.1), loc='upper center', ncol=4, fontsize='xx-small')

    elif coverage =='communication':
        # plot communication range for each active node
        if cf.CONNECTIVITY is True:
            # sink node
            sink =plt.scatter(0., 0., marker='P', label='sink node', color='r', s=100 ,zorder=3 )
            sink.set_clip_on(False)
            # relay nodes
            X_r = [node.pos_x for node in nodes if node.mode == cf.COMMUNICATION]
            Y_r = [node.pos_y for node in nodes if node.mode == cf.COMMUNICATION]
            plt.scatter(X_r, Y_r, marker='D', label='relay node', color='g', s=15, zorder=3)
            plt.legend(bbox_to_anchor=(0., 1.00, 1.0, 0.1), loc='upper center', ncol=6, fontsize='xx-small')

            for i in range(len(X_a)):
                circle1 = plt.Circle((X_a[i], Y_a[i]), cf.COMMUNICATION_RANGE,
                                     color='y', fill=False, linestyle=(0, (5, 5)), alpha=.7, linewidth=0.5, zorder=1)
                ax.add_artist(circle1)

            for i in range(len(X_r)):
                circle2 = plt.Circle((X_r[i], Y_r[i]), cf.COMMUNICATION_RANGE,
                                     color='r', fill=False, linestyle=(0, (5, 5)), alpha=.7, linewidth=0.5, zorder=1)
                ax.add_artist(circle2)

            filename = filename+'-c'

    if mode == 'save':
        file_name = save_dir + filename
        plt.savefig(file_name +'.' +file_type)  # bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()



def save_csv(result, file_name='result', save_dir=cf.DIR_NAME):
    f = open(save_dir + file_name + '.csv', 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(result)
    f.close()


def draw_sensing_model():
    s = cf.SENSING_RANGE_S
    u = cf.SENSING_RANGE_U
    xs = np.arange(0.0, u, 0.02)
    xs2 = np.arange(u, u+1, 0.02)
    y = [1 if x < s else __calculate_sensing_prob(s, u, x) for x in xs ]
    plt.plot(xs, y, color='k')
    plt.plot(xs2, [0] * len(xs2), color='k')
    plt.axvline(x=cf.SENSING_RANGE_S, ymax=1, ymin=0.0, color='r', linestyle='--', label='$r_s$')
    plt.axvline(x=cf.SENSING_RANGE_U, ymax=__calculate_sensing_prob(s, u, u),
                ymin=0.0, color='b', linestyle=':', label='$r_u$')

    plt.xlim(0.0, u+1)
    plt.ylim(0.0, 1)

    plt.ylabel('probability')
    plt.xlabel('distance')
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()

def draw_v_transfer_function():

    v = np.arange(-6, 6, 0.02)
    # t_v = [abs((2 / np.pi) * np.arctan((np.pi / 2) * x)) for x in v]
    t_v = [abs(np.tanh(x)) for x in v]

    plt.plot(v, t_v, color='k')
    plt.ylim(0.0, 1)
    plt.xlim(-6, 6)

    plt.ylabel('T(V)') # T : transfer function
    plt.xlabel('V')
    plt.grid()
    plt.show()
    plt.close()

def draw_s_transfer_function():

    v = np.arange(-8, 8, 0.02)
    t_s = [1/(1+np.exp(-x)) for x in v]
    plt.plot(v, t_s, color='k')
    plt.ylim(0.0, 1)
    plt.xlim(-8, 8)

    plt.ylabel('T(V)') # T : transfer function
    plt.xlabel('V')
    plt.grid()
    plt.show()
    plt.close()

def draw_transfer_function():

    v = np.arange(-8, 8, 0.02)
    # t_v = [abs((2 / np.pi) * np.arctan((np.pi / 2) * x)) for x in v]
    t_v = [abs(np.tanh(x)) for x in v]
    plt.plot(v, t_v, linestyle='-',color='k')
    v = np.arange(-8, 8, 0.02)
    t_s = [1/(1+np.exp(-x)) for x in v]


    plt.plot(v, t_s, linestyle='-.',color='k')
    plt.ylim(0.0, 1)
    plt.xlim(-8, 8)

    plt.ylabel('T(V)') # T : transfer function
    plt.xlabel('V')
    plt.grid()
    plt.show()
    plt.close()

def __calculate_sensing_prob(s, u, distance):  # for draw_sensing_model function
    if s > distance:
        return 1
    elif u >= distance:
        return np.exp(-cf.SENSING_K * ((distance - s) ** cf.SENSING_M))
    else:
        return 0

