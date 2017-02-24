#include "genresult.cuh"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

__device__ float warp_scan(const int lane, const int *rows, float *vals, float * y){
  if ( lane >= 1 && rows[threadIdx.x] == rows[threadIdx.x - 1] )
    vals[threadIdx.x] += vals[threadIdx.x - 1];
  if ( lane >= 2 && rows[threadIdx.x] == rows[threadIdx.x - 2] )
    vals[threadIdx.x] += vals[threadIdx.x - 2];
  if ( lane >=4 && rows[threadIdx.x] == rows[threadIdx.x - 4] )
    vals[threadIdx.x] += vals[threadIdx.x - 4];
  if ( lane >= 8 && rows[threadIdx.x] == rows[threadIdx.x - 8] )
    vals[threadIdx.x] += vals[threadIdx.x - 8];
  if ( lane >= 16 && rows[threadIdx.x] == rows[threadIdx.x - 16] )
    vals[threadIdx.x] += vals[threadIdx.x - 16];
  return vals[threadIdx.x];
}

__global__ void scan_matrix(const int nnz, const int* coord_row, const int* coord_col, const float* A, const float* x, float* y){
  /* This kernel implements segment scan on a warp basis. Each warp is
   * responsible for a summing a certain amount of contiguous data - at each iteration
   * of the warp, a thread can write to the result vector if it's the thread
   * which is computing the last entry for the row, or it's the last thread
   * in a warp and needs to write the results of its last carry*/
    __shared__ int rows[1024];
    __shared__ float vals[1024];

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_num = blockDim.x * gridDim.x;
    int iter = nnz % thread_num ? nnz/thread_num + 1: nnz/thread_num;
    int entries_per_block = iter * blockDim.x;


    /* This is the code for the carry logic, if you don't want to atomically
     * write for every warp's execution */
    int lane = threadIdx.x & 31; //powers of 2 are great
    int warp_id = threadIdx.x >> 5;
    int abs_warp_id = thread_id >> 5; 

    __shared__ float carry[32];
    __shared__ int carry_rows[32];

    if(lane == 0){
      carry[warp_id] = 0;
      carry_rows[warp_id] = coord_row[thread_id];
    }

    for(int i = 0; i < iter; i++){
      int dataid = thread_id + i * thread_num;
      //int dataid = lane + abs_warp_id * iter * 32 + 32*i;
      //int dataid = threadIdx.x + blockIdx.x * entries_per_block + i * blockDim.x; 
      if(dataid < nnz){
        float data = A[dataid];
        int col = coord_col[dataid];
        int row = coord_row[dataid];
        rows[threadIdx.x] = row;
        vals[threadIdx.x] = data * x[col]; 
         
        // Perform segment scan warp routine
        // The following commented code is an alternative version which carries the values 
        // by atomic add instead of utilizing shared memory.
        
        /*
        float val= warp_scan(lane, rows, vals, y);
        if((lane!= 31 && row != rows[threadIdx.x + 1]) || lane == 31){ // end of row or last thread in warp, so carry results over
          atomicAdd(&y[row], val);   
        }
        */

        if(lane == 0){
          if(carry_rows[warp_id] != row){ //If you can't carry over, add it to the row
            atomicAdd(&y[carry_rows[warp_id]], carry[warp_id]);
          }
          else{ // Carry over the value from prev warp iteration
            vals[threadIdx.x] += carry[warp_id];
          }
        } 

        float partial_sum = warp_scan(lane, rows, vals, y);

        if(lane != 31 && row != rows[threadIdx.x + 1]){ // end of row, and not the last thread in warp
                                                        // so add the values
            atomicAdd(&y[row], partial_sum);   
        }

        if(lane == 31){ // Set the new carry if you're the last thread in the warp
          carry[warp_id] = partial_sum;
          carry_rows[warp_id] = row;
        }
      }
    }
    
    if(lane == 31){ //The last carries haven't been added yet, so add them 
      atomicAdd(&y[carry_rows[warp_id]], carry[warp_id]);
    }
    
}

int cmpfunc (const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}


void sort_matrix(MatrixInfo * mat, int * sorted_rows, float * sorted_vals, int * sorted_cols){ // sorts the matrix by row in O(n) where n 
  // is the number of non zero entries
  memcpy(sorted_rows, mat->rIndex, mat->nz * sizeof(int));
  qsort(sorted_rows, mat->nz, sizeof(int), cmpfunc);
  int * row_start = (int *) calloc(mat->nz, sizeof(int));
  for(int i = 1; i < mat->nz; i++){
    if(sorted_rows[i] != sorted_rows[i - 1]){
      row_start[sorted_rows[i]] = i;
    }
  }
  row_start[sorted_rows[0]] = 0;
  for(int i = 0; i < mat->nz; i++){
    int row = mat->rIndex[i];
    int insert_index = row_start[row];
    sorted_vals[insert_index] = mat->val[i];
    sorted_cols[insert_index] = mat->cIndex[i];
    row_start[row] +=1;
  }
  mat->rIndex = sorted_rows;
  mat->cIndex = sorted_cols;
  mat->val = sorted_vals;
}


void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    float * A;
    float * x;
    float * y;
    int * coord_row;
    int * coord_col;

    const unsigned int matrix_bytes = (size_t) mat->nz * sizeof(float);
    const unsigned int vector_bytes = (size_t) vec->nz * sizeof(float);

    int * sorted_rows = (int *) calloc(mat->nz, sizeof(int));
    float * sorted_vals= (float*) calloc(mat->nz, sizeof(float));
    int * sorted_cols= (int *)calloc(mat->nz, sizeof(int));
    
    sort_matrix(mat, sorted_rows, sorted_vals, sorted_cols);

    cudaMalloc((float**)&A, matrix_bytes);
    cudaMemset(A, 0, matrix_bytes);
    cudaMemcpy(A, mat->val, matrix_bytes, cudaMemcpyHostToDevice);

    
    cudaMalloc((int**)&coord_row, matrix_bytes);
    cudaMemset(coord_row, 0, matrix_bytes);
    cudaMemcpy(coord_row, mat->rIndex,  matrix_bytes, cudaMemcpyHostToDevice);
    
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

    scan_matrix<<<blockNum, blockSize>>>(mat->nz, coord_row, coord_col, A, x, y);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);
    cudaMemcpy(res->val, y, vector_bytes, cudaMemcpyDeviceToHost);
    
    verify(mat, vec, res);

    cudaFree(A);
    cudaFree(x);
    cudaFree(y);
    cudaFree(coord_row);
    cudaFree(coord_col);
    /*Deallocate.*/
}
