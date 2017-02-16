import sys
import random 

matrix= sys.argv[1]
vector = sys.argv[2]


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
