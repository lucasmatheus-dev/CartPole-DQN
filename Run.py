import random
import tensorflow as tf
import numpy as np

def seed(value):
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)

def change_seed(i_seed):
    list_seeds = [100, 573, 982, 588, 576, 123, 1337, 1212, 1050, 1989]
    seed(list_seeds[i_seed])

def run():
    pass

if __name__ == "__main__":
    run()