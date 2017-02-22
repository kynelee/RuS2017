#include "genresult.cuh"
#include <sys/time.h>

void verify2(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * result){
  /* Verifies the result of SPMV done on GPU by sequentially doing it on the
   * CPU */
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

  printf("result size%d\n", result->nz);
  printf("result first %f\n", result->val[0]);

  for(int i = 0; i < result->nz; i++){
    float product_val = product[i];
    float result_val = result->val[i];

    if(product_val - result_val > error){
      error = product_val - result_val;
    }
  }

  printf("All good, biggest error found was %f", error);
}

__device__ void warp_scan(const int lane, const int * rows, float * vals) 
{
  if (lane >=1 && rows[threadIdx.x] == rows[threadIdx.x - 1]){
    vals[threadIdx.x] += vals[threadIdx.x - 1];
  }
  if (lane >=2 && rows[threadIdx.x] == rows[threadIdx.x - 2]){
    vals[threadIdx.x] += vals[threadIdx.x - 2];
  }
  if (lane >=4 && rows[threadIdx.x] == rows[threadIdx.x - 4]){
    vals[threadIdx.x] += vals[threadIdx.x - 4];
  }
  if (lane >=8 && rows[threadIdx.x] == rows[threadIdx.x - 8]){
    vals[threadIdx.x] += vals[threadIdx.x - 8];
  }
  if (lane >=16 && rows[threadIdx.x] == rows[threadIdx.x - 16]){
    vals[threadIdx.x] += vals[threadIdx.x - 16];
  }
}

__global__ void scan_matrix(const int nnz, 
                            const int * coord_row,
                            const int * coord_col, 
                            float * A,
                            float * x, 
                            float * y)
{
  /* Each block consists of multiple warps. Each block can process multiple
   * rows, and each warp is assigned a number of row-contiguous nonzero entries, and can
   * iterate multiple times. 
   * First, we scan for the results of an individual warp. If the warp
   * iterates, we carry the result to the next iteration. 
   * We store the last entry of each warp in warp_vals, and it's
   * corresponding row index in warp_rows
   * Then, we use the first warp to accumulate the results of all the warps as
   * long as they belong to the same row. 
   * Finally, if you're the last thread for a row, you can add your partial
   * result to the corresponding result vector. Note that the last thread
   * may have a partial result for the row, since a row can be processed by
   * multiple blocks */

  __shared__ int rows[1024]; // map from threadIdx -> row
  __shared__ float vals[1024]; // map from threadIdx -> value

  int thread_id = blockDim.x  * blockIdx.x + threadIdx.x; 
  int thread_num = blockDim.x * gridDim.x;
  int iter = nnz % thread_num ? nnz/thread_num + 1: nnz/thread_num;

  for (int i = 0; i < iter; i ++){ //compute multiplications for all entries in matrix
    int dataid = thread_id + thread_num * i;
    if(dataid < nnz) {
      float data = A[dataid];
      int col = coord_col[dataid];
      A[dataid] = data * x[col];
    }
  }
  __syncthreads(); //sync after all multiplication is complete
  
  /*__shared__ float warp_vals[32]; // map from warp_id -> warp result
  __shared__ float warp_rows[32]; // map from warp_id -> warp row */
  
  int lane = threadIdx.x % 32;
  int warp_id = thread_id >> 5;
//  __shared__ float carry[32]; // keeps track of carry for each warp
//  __shared__ float carry_rows[32];

  for(int i = 0; i < iter; i ++){ //compute result for each warp, if a warp overlaps rows, then warp_result is only the result for the last row
    int dataid = thread_id + thread_num * i;
    if (dataid < nnz){
      int row = coord_row[dataid];
      rows[threadIdx.x] = row; 
      vals[threadIdx.x] = A[dataid];
      __syncthreads();

      warp_scan(lane, rows, vals); 
      __syncthreads();

      if(lane == 31 && rows[threadIdx.x] == rows[threadIdx.x + 1]
          && i!= iter - 1){ //last thread for warp, carry value, as long as the last 
        vals[threadIdx.x + 1] += vals[threadIdx.x];
 //       carry[warp_id] = vals[threadIdx.x];
      }

      __syncthreads();
      if(rows[threadIdx.x] != rows[threadIdx.x + 1]){ //last thread for row, so write result
        float temp = vals[threadIdx.x];
        atomicAdd(&y[row], temp);
      }
  	}
  }

  /*
  __syncthreads();

  if(lane == 31){ //record results for results for each warp in warp_results
    warp_vals[warp_id] = vals[threadIdx.x];
    warp_rows[warp_id] = rows[threadIdx.x];
  }
  __syncthreads();

  // Use the first warp to scan the results (combines warp results if they
  // belong to the same row)
  if(warp_id == 0){
    warp_scan(lane, warp_rows, warp_vals);
  }
  __syncthreads();

  if(rows[threadIdx.x] != rows[threadIdx.x + 1]){ //last thread for row
    if(warp_id == 0){ 
      atomicAdd(&y[rows[threadIdx.x]], warp_result);
    } 
    else{
      if warp_row[warp_id - 1] != warp_row[warp_id]{
        atomicAdd(&y[rows[threadIdx.x]], warp_result; 
      }
      else{
        atomicAdd(&y[coord_row[dataid]], warp_results[warp_id] + warp_result);
      }
    }
  }
  */
}


void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    printf("WTF!!");

    /* Sort and allocate the initial matrix and corresponding indice vectors into 
     * row order s.t Row(A[x]) > Row(A[n]) for all n < x
     */

    float * A;
    float * x;
    float * y;
    int * coord_row;
    int * coord_col;

    const unsigned int matrix_bytes = (size_t) mat->nz * sizeof(float);
    const unsigned int vector_bytes = (size_t) vec->nz * sizeof(float);
    

    
    int * sorted_row = (int *) calloc(mat->nz, sizeof(int));
    int * sorted_col= (int *) calloc(mat->nz, sizeof(int));
    float * sorted_val = (float *) calloc(mat->nz, sizeof(float));

    const unsigned int matrix_bytes = (size_t) mat->nz * sizeof(float);
    const unsigned int vector_bytes = (size_t) vec->nz * sizeof(float);

    // sort based on value of row, stable
    // then map new indexes 
  


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

    verify2(mat, vec, res);

    cudaFree(A);
    cudaFree(x);
    cudaFree(y);
    cudaFree(coord_row);
    cudaFree(coord_col);
    /*Deallocate.*/
}
