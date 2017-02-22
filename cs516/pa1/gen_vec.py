import sys
import random 

matrix= sys.argv[1]
vector = sys.argv[2]

dim = int(sys.argv[3])

"""
with open(vector, 'w') as f:
  with open(matrix, 'r') as x:
    s = x.readline()
    while s[0] == "%":
      s = x.readline()
    size = s.split(" ")[0]
    print size
  f.write(str(size) + '\n')
  for i in xrange(0, int(size)):
    f.write(str(1) + '\n')
"""


with open(vector, 'w') as v:
  with open(matrix, 'w') as m:
    m.write(str(dim) + " " + str(dim) + " " +  str(dim*dim) + "\n")
    w.write(str(dim) + "\n")
    for i in range(1, dim + 1):
      v.write(str(1) + '\n')
      for j in range(1, dim+ 1):
        m.write("{} {} 1.0\n".format(j, i))
