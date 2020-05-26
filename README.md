# TCS (Target coverage simulator in WSN).
TCS simulates the target coverage (TC) with connectivity between nodes and sink. \  
It is implemented in Python 3.

# How to run it
1. pip3 install -r requirements.txt

2. python3 run.py

&nbsp; or 

2. python3 main.py



# Code structure 


- **config.py** contains network environment confiurations such as area size, the number of nodes,  a sink node position, etc.

- **protocols/** contains classes that implement sensor activation methods for TC. They set the mode of sensors in a timeslot.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **pso.py** : particle swarm optimization \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **aco.py** : Ant Colony Optimization \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **joa.py** : jenga optimization algorithm \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **psca.py** : PSCA (probabilistic sensor coverage algorithm)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **bba.py** : binary bat algrothm (this algorithm considers the network connectivity between active sensors and sink.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **proposed.py** : the method based on binary bat algorithm\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **proposed_rand_init.py** : the method based on binary bat algorithm with random initialization\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **proposed_sc.py** : the method based on binary bat algorithm only for tartget coverage. it doesn't condider the connectivity.\
 
 - **network.py** create a network based on config.py. In each timeslot, it calls the method in / protocols / to determine the mode of the sensor. And the energy for each sensor is reduced according to the mode of the sensor. 
 
 - **result_utils.py** contain functions used for modeling the areas in a timeslot and saving simulation results.
 
 
 # Reference 
 
 - Chen, Jiming, et al. "Energy-efficient coverage based on probabilistic sensing model in wireless sensor networks." IEEE Communications Letters 14.9 (2010): 833-835.
 - Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Xin-She Yang. "Binary bat algorithm." Neural Computing and Applications 25.3-4 (2014): 663-681.
 - Lee, Joon-Woo, Joon-Yong Lee, and Ju-Jang Lee. "Jenga-inspired optimization algorithm for energy-efficient coverage of unstructured WSNs." IEEE Wireless Communications Letters 2.1 (2013): 34-37.
 - Shan, Anxing, Xianghua Xu, and Zongmao Cheng. "Target coverage in wireless sensor networks with probabilistic sensors." Sensors 16.9 (2016): 1372.
 - Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008
 - Kou, L., George Markowsky, and Leonard Berman. "A fast algorithm for Steiner trees." Acta informatica 15.2 (1981): 141-145.
