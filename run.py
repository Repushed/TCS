from shutil import *
from time import sleep  
exp = 4

if exp ==1 :
	# experimnet 1 
	copyfile('./configs/config_exp_1.py', 'config.py')

elif exp == 2:
	# experimnet 2 
	copyfile('./configs/config_exp_2.py', 'config.py')

elif exp == 3:
	# experimnet 3 
	copyfile('./configs/config_exp_3.py', 'config.py')

elif exp == 4:
	# experimnet 4
	copyfile('./configs/config_exp_4.py', 'config.py')

elif exp == 5:
	# experimnet 5 
	copyfile('./configs/config_exp_5.py', 'config.py')

	# 50 * 50 m^2 network area
	#cf.AREA_WIDTH = 50
	#cf.AREA_LENGTH = 50


sleep(5)
from main import * 
main()

