# -*- coding: utf-8 -*
import sys
import os

DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')

if __name__ == "__main__":
    print(MODEL_PATH)