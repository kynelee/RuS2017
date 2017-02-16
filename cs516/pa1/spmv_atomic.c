#include "genresult.cuh"
#include <sys/time.h>

__global__ void getMulAtomic_kernel(const int nnz, 
                                    const float * A,
                                    const float * x, 
                                    const float * y)
{
  /*
  int thread_id = blockDim.x  * blockIdx.x + threadIdx.x;
  int thread_num = blockDim.x *  gridDim.x;
  int iter = nnz % thread_num ? nnz/thread.num + 1: nnz/thread_num;

  for (int i = 0; i < iter; i ++){
    int dataid = thread_id + thread_num + i;
    if(dataid < nnz) {
      float data = A[dataid];
      float temp = data * x[col];
      atomicAdd(&y[row], temp);
    }
  }
  */
}

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate here...*/

		/* Sample timing code */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernels...*/

    
    printf("%d by %d total %d", mat->M, mat->N, mat->nz);


// getMulAtomic_kernel<<<blockNum, blocksize>>>(nnz, coord_row, coord_col, mat, vec, );

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);
    /*Deallocate.*/
}
