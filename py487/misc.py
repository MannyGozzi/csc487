import numpy as np
import pandas as pd

def exercise_1():
    a = np.ones([6, 4]) * 2
    return a

def exercise_2():
    b = np.eye(6, 4) * 2 + 1
    return b

def exercise_3(c):
    return c.mean()

def exercise_4(c):
    row_means,col_means = None,None
    row_means = c.mean(axis=1)
    col_means = c.mean(axis=0)
    return row_means,col_means

def exercise_5(a):
    ones = 0
    for row in a:
      for element in row:
        if element == 1:
          ones += 1
    otherOnes = len(np.where(a.flat == 1)[0])
    return ones, otherOnes

def exercise_6():
    a = pd.DataFrame(np.ones([6, 4]))
    return a

def exercise_7():
    b = pd.DataFrame(np.eye(6, 4))
    return b