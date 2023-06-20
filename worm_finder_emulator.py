# # http://docs.enthought.com/python-for-LabVIEW/guide/start.html

import numpy as np
import sys

def worm_finder(image, avg_chance_of_worms=0.25):
    # do stuff
    random_number = np.random.uniform()

    if random_number < avg_chance_of_worms:
        success = 1
    else:
        success = 0
    print(success)    
    return(success)

if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    # print(type(np.array(args[1])))
    # print(type(float(*args[2:])))
    worm_finder(np.array(args[1]), float(*args[2:]))
    