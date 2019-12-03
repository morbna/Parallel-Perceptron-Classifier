#include "header.h"
#define MAX_BLOCK_THREADS 1024


__global__ void addKernel(double *dev_a, double *dev_b, double *dev_c, int n, int dim, double t)
{
    int i = threadIdx.x;
	int b = blockIdx.x;
	int d = blockDim.x;

	int idx = b * d + i;

	if (idx < n * dim) 
		dev_c[idx] = dev_a[idx] + dev_b[idx] * t;
}

// Helper function for using CUDA to set points in parallel.
cudaError_t setPointsWithCuda(Point *points, int n, int dim, double t, double *dev_vecInitloc, double *dev_vecV, double *dev_vecCurrentloc)
{
	char errorBuffer[100];
	cudaError_t cudaStatus;
	int numBlocks;

    // Choose which GPU to run on
    cudaStatus = cudaSetDevice(0);
	checkError(cudaStatus, dev_vecInitloc, dev_vecV, dev_vecCurrentloc,
	"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");

	numBlocks = (n*dim) / MAX_BLOCK_THREADS;
	if ( (n*dim) % MAX_BLOCK_THREADS)
		numBlocks++;

    // Launch a kernel on the GPU 
    addKernel<<<numBlocks, MAX_BLOCK_THREADS >>>(dev_vecInitloc, dev_vecV, dev_vecCurrentloc, n, dim, t);

    // Check for any errors launching the kernel

    cudaStatus = cudaGetLastError();
	sprintf(errorBuffer, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	checkError(cudaStatus, dev_vecInitloc, dev_vecV, dev_vecCurrentloc, errorBuffer);
    

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.

    cudaStatus = cudaDeviceSynchronize();
	sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	checkError(cudaStatus, dev_vecInitloc, dev_vecV, dev_vecCurrentloc, errorBuffer);


	// Copy output vector from GPU buffer to host memory.

	cudaStatus = cudaMemcpy(points->vecCurrentloc, dev_vecCurrentloc, dim * sizeof(double) * n , cudaMemcpyDeviceToHost);
	checkError(cudaStatus, dev_vecInitloc, dev_vecV, dev_vecCurrentloc, errorBuffer);

    return cudaStatus;
}

cudaError_t initCudaMemory(Point *points, int n, int dim, double **dev_a, double **dev_b, double **dev_c)
{
	cudaError_t cudaStatus;
	char errorBuffer[100];

	// Choose which GPU to run on
	cudaStatus = cudaSetDevice(0);
	checkError(cudaStatus, *dev_a, *dev_b, *dev_c,
		"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");

	// Allocate GPU buffers for three vectors 

	cudaStatus = cudaMalloc((void**)dev_a, n * dim * sizeof(double));
	checkError(cudaStatus, *dev_a, *dev_b, *dev_c, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)dev_b, n * dim * sizeof(double));
	checkError(cudaStatus, *dev_a, *dev_b, *dev_c, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)dev_c, n * dim * sizeof(double));
	checkError(cudaStatus, *dev_a, *dev_b, *dev_c, "cudaMalloc failed!");

	// Copy input vectors from host memory to GPU buffers.

	cudaStatus = cudaMemcpy(*dev_a, points->vecInitloc, dim * sizeof(double) * n, cudaMemcpyHostToDevice);
	checkError(cudaStatus, *dev_a, *dev_b, *dev_c, "cudaMemcpy failed!\n");

	cudaStatus = cudaMemcpy(*dev_b, points->vecV, dim * sizeof(double) * n, cudaMemcpyHostToDevice);
	checkError(cudaStatus, *dev_a, *dev_b, *dev_c, "cudaMemcpy failed!\n");

	cudaStatus = cudaGetLastError();
	sprintf(errorBuffer, "initCudaMemory failed: %s\n", cudaGetErrorString(cudaStatus));
	checkError(cudaStatus, *dev_a, *dev_b, *dev_c, "cudaMemcpy failed!\n");

	return cudaStatus;
}

void freeCudaMemory(double *dev_a, double *dev_b, double *dev_c)
{
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

void checkError(cudaError_t cudaStatus, double *dev_a, double *dev_b, double *dev_c, const char* errorMessage)
{
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, errorMessage);
		fprintf(stderr, "\n");
		freeCudaMemory(dev_a, dev_b, dev_c);
	}
}

// unused below

__global__ 	void fKernel(int *dev_f, double *dev_w, double *dev_points,int n,int  dim) {
	int i = threadIdx.x;
	int b = blockIdx.x;
	int d = blockDim.x;

	int idx = b * d + i;

	if (idx < n) {
		double res = dev_w[0];
		for (int k = 0; k < dim; k++)
			res += dev_w[k + 1] * dev_points[idx*dim+k];

		dev_f[idx]=(res >= 0) ? A : B;
	}
}

void markWithCuda(Classifier C, double *weights, int *fArray, double *dev_points) {

	int *dev_f;
	double *dev_w;
	int numBlocks;

	// Choose which GPU to run on
	cudaSetDevice(0);


	cudaMalloc((void**)&dev_f, C.N  * sizeof(int));
	cudaMalloc((void**)&dev_w, (C.K+1) * sizeof(double));
	cudaMemcpy(dev_f, fArray, sizeof(int) * C.N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_w, weights, (C.K+1) * sizeof(double), cudaMemcpyHostToDevice);

	numBlocks = (C.N) / MAX_BLOCK_THREADS;
	if ((C.N) % MAX_BLOCK_THREADS)
		numBlocks++;

	// Launch a kernel on the GPU 
	fKernel <<<numBlocks, MAX_BLOCK_THREADS >>> (dev_f, dev_w, dev_points,C.N, C.K);

	cudaMemcpy(fArray, dev_f,C.N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_f);
	cudaFree(dev_w);
}
