#include "genresult.cuh"
#include <sys/time.h>


__global__ void getMulDesign_kernel(const int nnz, 
                                    int * coord_row,
                                    float * A,
                                    float * x, 
                                    float * y)
{
  int thread_id = blockDim.x  * blockIdx.x + threadIdx.x; 
  int thread_num = blockDim.x * gridDim.x;
  int iter = nnz % thread_num ? nnz/thread_num + 1: nnz/thread_num;
  
  for (int i = 0; i < iter; i ++){
    int dataid = threadIdx.x + blockIdx.x * iter * blockDim.x + i * blockDim.x;
    //int dataid = (thread_id + thread_num*i);
    if(dataid < nnz) {
      int row = coord_row[dataid];
      float data = A[dataid];
      float temp = data * x[dataid];

      atomicAdd(&y[row], temp);
    }
  }
}

    
void get_MATT_vector(int nz, int * cols, float * vec, float * matt_vector){
  for(int i = 0; i < nz; i ++){
    matt_vector[i] =  vec[cols[i]];
  }
}

int reorder_matrix(MatrixInfo * mat, int * sorted_rows, int * sorted_cols, 
    float * sorted_vals, int *ord_rows, int * ord_cols, float * ord_vals){ // sorts the matrix by row in O(n) where n 
  // is the number of non zero entries
  memcpy(sorted_rows, mat->rIndex, mat->nz * sizeof(int));
  qsort(sorted_rows, mat->nz, sizeof(int), cmpfunc);
  int * row_start = (int *) calloc(mat->M, sizeof(int));
  
  int unique_row_count = 0;
  for(int i = 1; i < mat->nz; i++){
    if(sorted_rows[i] != sorted_rows[i - 1]){
      unique_row_count +=1;
      row_start[sorted_rows[i]] = i;
    }
  }
  row_start[sorted_rows[0]] = 0;
  if(sorted_rows[0] != sorted_rows[1]){
    unique_row_count+=1;
  }
  for(int i = 0; i < mat->nz; i++){
    int row = mat->rIndex[i];
    int insert_index = row_start[row];
    sorted_vals[insert_index] = mat->val[i];
    sorted_cols[insert_index] = mat->cIndex[i];
    row_start[row] +=1;
  }

  for(int i = 0; i < mat->nz; i++){
    ord_rows[i] = sorted_rows[i];
    ord_cols[i] = sorted_cols[i];
    ord_vals[i] = sorted_vals[i];
  }


  int unique_count = 1; 
  int not_unique_count = unique_row_count;
  
  for(int i = 1; i < mat->nz; i++){
    int idx;
    if(sorted_rows[i] != sorted_rows[i-1]){
      idx = unique_count;
      unique_count +=1;
    }
    else{
      idx = not_unique_count; 
      not_unique_count = not_unique_count+1;
    }
    ord_rows[idx] = sorted_rows[i]; 
    ord_cols[idx] = sorted_cols[i];
    ord_vals[idx] = sorted_vals[i]; 
  }
  return unique_row_count;
}

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate here...*/

    const unsigned int matrix_bytes = (size_t) mat->nz * sizeof(float);
    const unsigned int vector_bytes = (size_t) vec->nz * sizeof(float);

    int * coord_row;
    int * coord_col;

    float * A;
    float * x;
    float * y;

    float * matt_vector = (float *) calloc(mat->nz, sizeof(float));

    int * sorted_rows = (int *) calloc(mat->nz, sizeof(int));
    float * sorted_vals= (float*) calloc(mat->nz, sizeof(float));
    int * sorted_cols= (int *)calloc(mat->nz, sizeof(int));

    int * ord_rows = (int *) calloc(mat->nz, sizeof(int));
    float * ord_vals = (float*) calloc(mat->nz, sizeof(float));
    int * ord_cols = (int *)calloc(mat->nz, sizeof(int));

    

    reorder_matrix(mat, sorted_rows, sorted_cols, sorted_vals, ord_rows, ord_cols, ord_vals);
    get_MATT_vector(mat->nz, ord_cols, vec->val,  matt_vector); //change to ord_cols and ord_vals once done

   
    cudaMalloc((float**)&A, matrix_bytes);
    cudaMemset(A, 0, matrix_bytes);
    cudaMemcpy(A, ord_vals, matrix_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((int**)&coord_row, matrix_bytes);
    cudaMemset(coord_row, 0, matrix_bytes);
    cudaMemcpy(coord_row, ord_rows, matrix_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((float**)&x, matrix_bytes);
    cudaMemset(x, 0, matrix_bytes);
    cudaMemcpy(x, matt_vector, matrix_bytes, cudaMemcpyHostToDevice);


    /*
    cudaMalloc((float**)&x, matrix_bytes);
    cudaMemset(x, 0, matrix_bytes);
    cudaMemcpy(x, matt_vector, matrix_bytes, cudaMemcpyHostToDevice);
    */

    /*
    cudaMalloc((float**)&A, matrix_bytes);
    cudaMemset(A, 0, matrix_bytes);
    cudaMemcpy(A, ord_vals, matrix_bytes, cudaMemcpyHostToDevice);
    */

    
    
    

    /* 
    cudaMalloc((float**)&x, vector_bytes);
    cudaMemset(x, 0, vector_bytes);
    cudaMemcpy(x, vec->val, vector_bytes, cudaMemcpyHostToDevice);
    */
    

    cudaMalloc((float**)&y, vector_bytes);
    cudaMemset(y, 0, vector_bytes);
    

    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);


    getMulDesign_kernel<<<blockNum, blockSize>>>(mat->nz, coord_row, A, x, y);
    

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
