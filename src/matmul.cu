#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define MATRIX_DIM 1024
#define BLOCK_SIZE 32
#define GRID_SIZE 32

__global__ void matmul(int *a, int *b, int *c){
    int threadRow = threadIdx.x;
    int threadCol = threadIdx.y;

    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;

    int outputRow = (blockRow * BLOCK_SIZE) + threadRow;
    int outputCol = (blockCol * BLOCK_SIZE) + threadCol;
    int local_c = 0;
    __shared__ int A[BLOCK_SIZE][BLOCK_SIZE], B[BLOCK_SIZE][BLOCK_SIZE];
    
    for(int tileIndex = 0; tileIndex < GRID_SIZE; tileIndex ++){
        A[threadRow][threadCol] = a[(MATRIX_DIM * outputRow) + ((tileIndex * BLOCK_SIZE) + threadCol)];
        B[threadRow][threadCol] = b[((tileIndex * BLOCK_SIZE + threadRow) * MATRIX_DIM) + (outputCol)];
        __syncthreads();
        for (size_t i = 0; i < BLOCK_SIZE; i++)
        {
            local_c += A[threadRow][i] * B[i][threadCol];
        }
        __syncthreads();
         
    }
    c[outputRow * MATRIX_DIM + outputCol] = local_c;
}



int main(){
    int i;
    const auto size = MATRIX_DIM * MATRIX_DIM;
    int *a = (int*)malloc(sizeof(int) * size);          
    int *b = (int*)malloc(sizeof(int) * size);          
    int *c = (int*)malloc(sizeof(int) * size);

    for(i=0; i<size; i++){
        a[i]=1;
        b[i]=2;
  	}
    int *gpu_a, *gpu_b, *gpu_c;

    cudaMalloc((void**)&gpu_a, sizeof(int)*size); 
    cudaMalloc((void**)&gpu_b, sizeof(int)*size);
    cudaMalloc((void**)&gpu_c, sizeof(int)*size);
    struct timespec start, stop; 
    double time;


    cudaMemcpy(gpu_a, a, sizeof(int)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, sizeof(int)*size, cudaMemcpyHostToDevice);

    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

    matmul<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);				
    cudaMemcpy(c, gpu_c, sizeof(int)*size, cudaMemcpyDeviceToHost);
    
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
		time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("time is %f ns\n", time*1e9);

    printf("c[451][451] = %d\n", c[451 * MATRIX_DIM + 451]);
  	
    free(a);
    free(b);
    free(c);
    cudaFree(gpu_a);  
    cudaFree(gpu_b);  
    cudaFree(gpu_c);
    return 0;
}