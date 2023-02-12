from random_generator import *
from step_generator import *
from util import *

if __name__ == '__main__':

    filepath = '../dataset/synthetic_step.csv'

    fieldnames = ['no', 'gender', 'age',  'height', 'weight', 'bmi', 'season', 'ta', 'hr', 'pmv']

    write_header(filepath, fieldnames)

    step_generator(filepath)

    print("done")
