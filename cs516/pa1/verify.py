import sys

matrix_file = sys.argv[1]
vector_file = sys.argv[2]
output_file = sys.argv[3]

def verify(matrix_file, vector_file, output_file):
  vector = []
  result = []

  with open(vector_file, 'r') as vec:
    vec_vals = vec.readlines()[1:]

  vector = [float(x) for x in vec_vals] 

  with open(matrix_file, 'r') as x:
    s = x.readline()
    while s[0] == "%":
      s = x.readline()

    dims = s.split(" ")
    print(dims)
    matrix = [[0 for i in range(int(dims[1]))] for i in range(int(dims[0]))]

    for line in x:
      point = [l.strip() for l in line.split(" ")]
      matrix[point[0]][point[1]] = point[2]

  print len(matrix)
  print matrix[0]


verify(matrix_file, vector_file, output_file)
