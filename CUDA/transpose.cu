#include <stdlib.h>
#include <stdio.h>

#include "cuda_utils.h"
#include "timer.c"

#define MAX_THREADS 1024
#define BLOCK_DIM 16 // MAX: 32 since we allocate threads to blocks as BLOCK_DIM * BLOCK_DIM

/* Notes:
 - max number of blocks is 2560, or 160 if a full block of 32x32 threads is used
 - max number of threads for 1 NVIDIA V100 (has 80 SMs) is 163840
*/

typedef float dtype;

// UNUSED. reused from reduction, only for number of threads
unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

// UNUSED. reused from reduction, only to dynamically get number of threads and blocks
void getNumBlocksAndThreads(int n, int maxThreads, int &blocks, int &threads)
{
     threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
     blocks = (n + threads - 1) / threads;
}

// UNUSED! Efficient version below.
__global__ 
void matTrans_naive(dtype* AT, dtype* A, int N)  {
     /* Naive approach */
     //int end = N*N - 1;
     unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
     // unused for 1D: unsigned int yIdx = xIdx * (N * (blockIdx.x + 1)) % end;     
     //printf("N: %d, end: %d\n", N*N, end);     

     if (xIdx < N && yIdx < N) {
	unsigned int idx_in = xIdx + N * yIdx;
	unsigned int idx_out = yIdx + N * xIdx;
       // printf("in: %d, out: %d\n", idx_in, idx_out);     
	AT[idx_out] = A[idx_in];

        // unused: the very last element needs to be filled in since indices cover only up end - 2.
        //if (threadIdx.x == end)
	//   AT[end] = A[idx_in];

        //printf("AT[%d]: %f, taking in A[%d] = %f\n", idx_out, AT[idx_out], idx_in, A[idx_in]);
     }
}

// Efficient approach using shared memory and padding to avoid bank conflicts
__global__ 
void matTrans(dtype* AT, dtype* A, int N)  {
     __shared__ dtype temp[BLOCK_DIM][BLOCK_DIM+1];

     unsigned int xIdx = blockIdx.x * BLOCK_DIM + threadIdx.x;
     unsigned int yIdx = blockIdx.y * BLOCK_DIM + threadIdx.y;

     if ((xIdx < N) && (yIdx < N)) {
	unsigned int idx_in = xIdx + N * yIdx;
	temp[threadIdx.y][threadIdx.x] = A[idx_in];
     }

     __syncthreads();
     
     // transposed indices
     xIdx = blockIdx.y * BLOCK_DIM + threadIdx.x;
     yIdx = blockIdx.x * BLOCK_DIM + threadIdx.y;
     
     if ((xIdx < N) && (yIdx < N)) {
	unsigned int idx_out = xIdx + N * yIdx; // optional since idx_out = idx_in for N-by-N matrix
	AT[idx_out] = temp[threadIdx.x][threadIdx.y];
     }
}

void
parseArg (int argc, char** argv, int* N)
{
	if(argc == 2) {
		*N = atoi (argv[1]);
		assert (*N > 0);
	} else {
		fprintf (stderr, "usage: %s <N>\n", argv[0]);
		exit (EXIT_FAILURE);
	}
}

void
initArr (dtype* in, int N)
{
	int i;

	for(i = 0; i < N; i++) {
		in[i] = (dtype) rand () / RAND_MAX;
	}
}

void
cpuTranspose (dtype* A, dtype* AT, int N)
{
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			AT[j * N + i] = A[i * N + j];
		}
	}
}

int
cmpArr (dtype* a, dtype* b, int N)
{
	int cnt, i;

	cnt = 0;
	for(i = 0; i < N; i++) {
 		//printf("GPU: %f, CPU: %f\n", a[i], b[i]); 
		if(abs(a[i] - b[i]) > 1e-6) cnt++;
	}

	return cnt;
}

void
gpuTranspose (dtype* A, dtype* AT, int N)
{
  struct stopwatch_t* timer = NULL;
  long double t_gpu;
  const int mem_size = N * N * sizeof(dtype);  

  // Added: Allocate device memory
  dtype *d_A, *d_AT; // device arrays; A and AT are in host!   
  int threads, blocks;
  CUDA_CHECK_ERROR (cudaMalloc (&d_A, mem_size));
  CUDA_CHECK_ERROR (cudaMalloc (&d_AT, mem_size));
  CUDA_CHECK_ERROR (cudaMemcpy (d_A, A, mem_size, cudaMemcpyHostToDevice));
  
  // Added: Block and thread initialization	
  //getNumBlocksAndThreads(N*N, MAX_THREADS, blocks, threads);
  //dim3 gblocks(blocks, 1, 1);
  //dim3 tb(threads, 1, 1);
  blocks = N / BLOCK_DIM;
  threads = BLOCK_DIM;
  printf("2D allocation. Number of blocks: %d; number of threads: %d\n", blocks, threads); 

  dim3 gblocks(blocks, blocks, 1);
  dim3 tb(threads, threads, 1);

  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();
  
  // Added: warm up kernel
  matTrans <<<gblocks, tb>>> (d_AT, d_A, N);
  cudaDeviceSynchronize();  

  stopwatch_start (timer);

  /* run your kernel here */
  matTrans <<<gblocks, tb>>> (d_AT, d_A, N);

  /* end kernel */   
  cudaDeviceSynchronize ();
  t_gpu = stopwatch_stop (timer);
  fprintf (stdout, "GPU transpose: %Lg secs ==> %Lg billion elements/second\n",
           t_gpu, (N * N) / t_gpu * 1e-9 );
  
  // Added: Copy result back to host, then free the device arrays
  CUDA_CHECK_ERROR (cudaMemcpy (AT, d_AT, mem_size, cudaMemcpyDeviceToHost)); 
  //CUDA_CHECK_ERROR (cudaFree(d_A));
  //CUDA_CHECK_ERROR (cudaFree(d_AT));
}

int 
main(int argc, char** argv)
{
  /* variables */
	dtype *A, *ATgpu, *ATcpu;
  int err;

	int N;

  struct stopwatch_t* timer = NULL;
  long double t_cpu;

	N = -1;
	parseArg (argc, argv, &N);

  /* input and output matrices on host */
  /* output */
  ATcpu = (dtype*) malloc (N * N * sizeof (dtype));
  ATgpu = (dtype*) malloc (N * N * sizeof (dtype));

  /* input */
  A = (dtype*) malloc (N * N * sizeof (dtype));

	initArr (A, N * N);

	/* GPU transpose kernel */
	gpuTranspose (A, ATgpu, N);

  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

	stopwatch_start (timer);
  /* compute reference array */
	cpuTranspose (A, ATcpu, N);
  t_cpu = stopwatch_stop (timer);
  fprintf (stdout, "Time to execute CPU transpose kernel: %Lg secs\n",
           t_cpu);

  /* check correctness */
	err = cmpArr (ATgpu, ATcpu, N * N);
	if(err) {
		fprintf (stderr, "Transpose failed: %d\n", err);
	} else {
		fprintf (stdout, "Transpose successful\n");
	}

	free (A);
	free (ATgpu);
	free (ATcpu);

  return 0;
}
