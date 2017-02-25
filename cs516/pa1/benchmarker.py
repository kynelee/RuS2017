#! /usr/bin/env python3

import sys
import os
import subprocess


matrix_names = ["cant.mtx", "consph.mtx", "FullChip.mtx",
                "mac_econ_fwd500.mtx", "mc2depi.mtx", "pdb1HYS.mtx", "pwtk.mtx",
                "rma10.mtx", "scircuit.mtx", "shipsec1.mtx", "turon_m.mtx", "webbase-1M.mtx"]

vector_names = [i[0:-4] + "_vec" + ".txt" for i in matrix_names]
matrix_names = [os.path.join(os.path.dirname(__file__), i) for i in matrix_names]
vector_names = [os.path.join(os.path.dirname(__file__), i) for i in vector_names]

matrix_names = ["matrices/" + i for i in matrix_names]
vector_names = ["matrices/" + i for i in vector_names]

binary_name = sys.argv[1]
blocksize = sys.argv[2]
blocknum = sys.argv[3]



def ctr(base):
    c = base
    while True:
        yield c
        c = c + 1

def get_diff(x, y):
    den = 1 if float(y) == 0 else abs(float(y))
    return abs(float(x) - float(y))/den
  
TOLERANCE = 10 ** -2

exe_args_list = []

for alg in ("atomic" ,"segment", "design"):
  for i in range(0, len(matrix_names)):
    matrix_name = matrix_names[i]
    vector_name = vector_names[i]
    exe_args_list.append([sys.argv[1], "-mat", matrix_name, "-ivec", vector_name, "-alg", alg, "-blockSize", sys.argv[2], "-blockNum", sys.argv[3]])
      

try: 
  with open("benchmark_results.txt", 'w') as f:
    for exe in exe_args_list:
      print(exe)

      compl_time = subprocess.check_output(exe, stderr = subprocess.STDOUT).decode('utf-8') 
      time = float(compl_time[:compl_time.find("milli-seconds")].split(' ')[-2])
      f.write("Algorithm: {} Matrix: {} Time: {}".format(exe[6], exe[2], str(time)))
      

except subprocess.CalledProcessError as e:
    print("Command '" + str(e.cmd)+ "'failed!")
    print("Output:")
    print(e.output)
