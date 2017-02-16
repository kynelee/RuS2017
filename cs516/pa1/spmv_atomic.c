#include "genresult.cuh"
#include <sys/time.h>

void verify(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * result){
  /* Verifies the result of SPMV done on GPU */

  float * product = (float *) calloc(vec->nz, sizeof(float));
  for(int i = 0; i < mat->nz; i ++){
    float val = mat->val[i];
    int row = mat->rIndex[i];
    int col = mat->cIndex[i];
    float temp = vec->val[col] * val;
    product[row] += temp;
  }

  printf("result size %d\n", result->nz);
  printf("result first %f\n", result->val[0]);

  float error = 0;

  for(int i = 0; i < result->nz; i++){
    float product_val = product[i];
    float result_val = result->val[i];

    if (product_val - result_val > .001){
      printf("product %f result %f", product_val, result_val);
      printf("Found error at vector line %d", i);
      break;
    }

    if(product_val - result_val > error){
      error = product_val - result_val;
    }
  }

  printf("All good, biggest error found was %f", error);
}


__global__ void getMulAtomic_kernel(const int nnz, 
                                    const int * coord_row,
                                    const int * coord_col, 
                                    float * A,
                                    float * x, 
                                    float * y)
{
  int thread_id = blockDim.x  * blockIdx.x + threadIdx.x;
  int thread_num = blockDim.x * gridDim.x;
  int iter = nnz % thread_num ? nnz/thread_num + 1: nnz/thread_num;

  for (int i = 0; i < iter; i ++){
    int dataid = thread_id + thread_num * i;
    if(dataid < nnz) {
      float data = A[dataid];
      int row = coord_row[dataid];
      int col = coord_col[dataid];
      float temp = data * x[col];
      atomicAdd(&y[row], temp);
    }
  }
}

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate here...*/
    float * A;
    float * x;
    float * y;
    int * coord_row;
    int * coord_col;

    const unsigned int matrix_bytes = mat->nz * sizeof(float);
    const unsigned int vector_bytes = vec->nz * sizeof(float);

    cudaMalloc((float**)&A, matrix_bytes);
    cudaMemset(A, 0, matrix_bytes);
    cudaMemcpy(A, mat->val, matrix_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((int**)&coord_row, matrix_bytes);
    cudaMemset(coord_row, 0, matrix_bytes);
    cudaMemcpy(coord_row, mat->rIndex, matrix_bytes, cudaMemcpyHostToDevice);
    
    cudaMalloc((int**)&coord_col, matrix_bytes);
    cudaMemset(coord_col, 0, matrix_bytes);
    cudaMemcpy(coord_col, mat->cIndex, matrix_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((float**)&x, vector_bytes);
    cudaMemset(x, 0, vector_bytes);
    cudaMemcpy(x, vec->val, vector_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((float**)&y, vector_bytes);
    cudaMemset(y, 0, vector_bytes);
    

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);


    getMulAtomic_kernel<<<blockNum, blockSize>>>(mat->nz, coord_row, coord_col, A, x, y);
    

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);


    cudaMemcpy(res->val, y, vector_bytes, cudaMemcpyDeviceToHost);

    verify(mat, vec, res);

    cudaFree(A);
    cudaFree(x);
    cudaFree(y);
    cudaFree(coord_row);
    cudaFree(coord_col);
    /*Deallocate.*/
}
