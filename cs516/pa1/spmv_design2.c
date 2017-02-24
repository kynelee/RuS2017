#include "genresult.cuh"
#include <sys/time.h>


__global__ void getMulDesign_kernel(const int nnz, 
                                    float * A,
                                    float * x, 
                                    float * y)
{
  int thread_id = blockDim.x  * blockIdx.x + threadIdx.x; 
  int thread_num = blockDim.x * gridDim.x;
  int iter = nnz % thread_num ? nnz/thread_num + 1: nnz/thread_num;
  
  for (int i = 0; i < iter; i ++){ //column sorted
    int dataid = 3*(threadIdx.x + blockIdx.x * iter * blockDim.x + i * blockDim.x);
    //int dataid = 3*(thread_id + thread_num*i);
    if(dataid < nnz * 3) {
      int row = (int) A[dataid];
      int col = (int) A[dataid + 1];
      float data = A[dataid + 2];
      float temp = data * x[col];
      atomicAdd(&y[row], temp);
    }
  }
}


/* Ax = x1[col1] + x2[col2] .. n */

/* Ax = x1[col1] + x2[col2] .. n
 * Want SHM access for the vector 
 * SHM for results
 * Group results -> write to vector 
 * Each block accesses a contiguous chunk of vector coefficeints, as well as 
 * cutdown on writes to global memory by writing writes to shared memory 
 * and then aggreagating as the last warp
 */
/*
__global__ void getMulDesign_kernel(const int nnz, 
                                    const int * coord_row,
                                    const int * coord_col, 
                                    float * A,
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
      float vec_data = tex1Dfetch(tex, col); // Re use the same vector 
      float temp = data * vec_data;
      atomicAdd(&y[row], temp);
    }
  }
}
*/

void get_MATT_format(MatrixInfo * mat, float * matt_matrix){
  for(int i = 0; i < mat->nz; i ++){
    int idx = i * 3;
    matt_matrix[idx] = (float) mat->rIndex[i];
    matt_matrix[idx + 1] = (float) mat->cIndex[i];
    matt_matrix[idx + 2] = (float) mat->val[i];
//    matt_matrix[idx+3] = 0;
  }
}

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate here...*/

    const unsigned int matrix_bytes = (size_t) mat->nz * sizeof(float) * 3;
    const unsigned int vector_bytes = (size_t) vec->nz * sizeof(float);

    float * A;
    float * x;
    float * y;

    float * matt_matrix = (float *) calloc(mat->nz * 3, sizeof(float));

    get_MATT_format(mat, matt_matrix);

    cudaMalloc((float**)&A, matrix_bytes);
    cudaMemset(A, 0, matrix_bytes);
    cudaMemcpy(A, matt_matrix, matrix_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((float**)&x, vector_bytes);
    cudaMemset(x, 0, vector_bytes);
    cudaMemcpy(x, vec->val, vector_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((float**)&y, vector_bytes);
    cudaMemset(y, 0, vector_bytes);
    

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);


    getMulDesign_kernel<<<blockNum, blockSize>>>(mat->nz, A, x, y);
    

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    cudaMemcpy(res->val, y, vector_bytes, cudaMemcpyDeviceToHost);

    printf("Running my own sequential verification on results\n");
    verify(mat, vec, res);

    cudaFree(A);
    cudaFree(x);
    cudaFree(y);
//    cudaFree(tex_cache);
//    cudaUnbindTexture(tex);
    /*Deallocate.*/
}
