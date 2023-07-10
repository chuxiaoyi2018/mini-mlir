import numpy as np
import math

true_result = np.loadtxt('true_result.txt')
inference_result = np.loadtxt('inference_result.txt')

def compare():
    for x,y in zip(true_result, inference_result):
        if math.fabs(x - y) > 0.001:
            print("False")
            return
    print("Compare All Success")

compare()
