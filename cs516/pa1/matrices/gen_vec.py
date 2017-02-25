import sys
import random 
import os




matrix_names = ["cant.mtx", "consph.mtx", "FullChip.mtx",
                "mac_econ_fwd500.mtx", "mc2depi.mtx", "pdb1HYS.mtx", "pwtk.mtx",
                "rail4284.mtx", "rma10_b.mtx", "rma10.mtx", "scircuit_b.mtx",
                "scircuit.mtx", "shipsec1.mtx", "turon_m.mtx", "webbase-1M.mtx"]

vector_names = [i[0:-4] + "_vec" + ".txt" for i in matrix_names]


for i in range(len(vector_names)):
  vector = vector_names[i]
  matrix = matrix_names[i]
  with open(vector, 'w') as f:
    with open(matrix, 'r') as x:
      s = x.readline()
      while s[0] == "%":
        s = x.readline()
      size = s.split(" ")[0]
      print size
    f.write(str(size) + '\n')
    for i in xrange(0, int(size)):
      f.write(str(random.random()) + '\n')
