
import numpy as np
import math

true_result = np.loadtxt('true_result.txt')
inference_result = np.loadtxt('inference_result.txt')

def compare():
    total_err = 0
    for x,y in zip(true_result, inference_result):
        total_err += math.fabs(x - y)
    return total_err/np.max(inference_result.shape)

def cos_sim(a, b):
  """计算两个向量a和b的余弦相似度"""
  
  a = np.array(a) 
  b = np.array(b)

  inner_product = np.dot(a, b)
  # 内积
  norm_a = np.linalg.norm(a)  
  norm_b = np.linalg.norm(b)
  # 模长
  cos_sim = inner_product / (norm_a * norm_b)

  return cos_sim


total_err = compare()
cos_sim = cos_sim(true_result.flatten(), inference_result.flatten())
print(f"total_err : {total_err}")
print(f"cos_sim : {cos_sim}")
