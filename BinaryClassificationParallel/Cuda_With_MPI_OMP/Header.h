#pragma once

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

constexpr auto PATH= "C:/Users/cudauser/Desktop/input.txt";
constexpr auto OUT= "C:/Users/cudauser/Desktop/output.txt";

constexpr auto SUCCESS= 1;
constexpr auto FAILURE = 0;

constexpr auto IGNORE = -1;

constexpr auto TRUE= 1;
constexpr auto FALSE= 0;

constexpr auto MASTER= 0;

constexpr auto TAG_HALT= 0xABAB;
constexpr auto TAG_CONTINUE= 0xCDCD;
constexpr auto TAG_RESTART = 0xEFEF;

enum Classes { A = 1, B = -1 };

typedef struct {
	int N;			// number of points
	int K;			// number of coordinates of points
	double dT;		// increment value of t
	double tMax;	// maximum value of t
	double alpha;	// conversion ratio
	int LIMIT;		// the maximum number of iterations
	double QC;		// Quality of Classifier to be reached 
}Classifier;

typedef struct {
	double *vecInitloc; // ptr to vector of initial locations
	double *vecV;       // ptr to vector of velocities
	double *vecCurrentloc;       // ptr to vector of current locations
	int sign;
}Point;


// MPI
void MPI_createPointType(MPI_Datatype* MPI_Point);
void MPI_createClassifierType(MPI_Datatype* MPI_Classifier);
void packPoints(Point *points, int n, char* buffer, int bufferSize, int dim, MPI_Datatype MPI_Point);
void unpackPoints(Point **points, int n, char* buffer, int bufferSize, int dim, MPI_Datatype MPI_Point);

// UTILITY
int readFile(FILE **F, Classifier * C, Point ** P);
void readPoints(FILE *F, Point **P, int dim, int n);
void allocatePoints(Point **P, int dim, int n, int includeBase, int includeVectors);
void freePoints(Point *P, int n);
void checkAllocation(void *p);
void checkCudaError(cudaError_t cudaStatus);

// WEIGHTS
void initWeights(double *weights, int dim);
void updateWeights(Point X, int sign, double *weights, double alpha, int dim);
int writeWeights(double *weights, int dim, double time, double q, int solved);

// HELPER
void perceptronRoutine(Point *points, Classifier classifier, double *Weights, int *solved, double *q, double t,
	double *dev_a, double *dev_b, double *dev_c);
void initClassifier(Point* points, Classifier classifier, int myId, int numprocs, double time0);
void classify(Classifier C, Point* points, double *weights);
void classifyWithCuda(Classifier C, Point *points, double *weights, double *dev_points);
void classifyParallel(Classifier C, Point* points, double *weights, int* threadFirstMis);

int countMis(int n, int dim, Point *X, double *weights);
int sign_discriminant(double *x, double *w, int dim);
void setPoints(Point *points, int n, int dim, double t);

// CUDA
cudaError_t initCudaMemory(Point *points, int n, int dim, double **dev_a, double **dev_b, double **dev_c);
void freeCudaMemory(double *dev_a, double *dev_b, double *dev_c);
cudaError_t setPointsWithCuda(Point *points, int n, int dim, double t, double *dev_a, double *dev_b, double *dev_c);
__global__ void addKernel(double *dev_a, double *dev_b, double *dev_c, int n, int dim, double t);
void checkError(cudaError_t cudaStatus, double *dev_a, double *dev_b, double *dev_c, const char* errorMessage);
void markWithCuda(Classifier C, double *weights, int *fArray, double *dev_points);