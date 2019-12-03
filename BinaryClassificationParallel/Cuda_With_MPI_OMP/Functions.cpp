#include "Header.h"

void checkAllocation(void *p) {

	if (!p) {
		printf("MEMORY ERROR!\n");
		fflush(stdout);
		exit(1);
	}
}

void readPoints(FILE *F, Point **P, int dim, int n) {
	int i, j;

	allocatePoints(P, dim, n, TRUE, TRUE);
	Point *points = (*P);

	for (i = 0; i < n; i++) {
		for (j = 0; j < dim; j++)
			fscanf_s(F, "%lf", &points[i].vecInitloc[j]);
		for (j = 0; j < dim; j++)
			fscanf_s(F, "%lf", &points[i].vecV[j]);

		fscanf_s(F, "%d", &points[i].sign);
	}
	printf("Found %d points\n", n);
	fflush(stdout);

}

int readFile(FILE **F, Classifier * C, Point ** P) {

	printf("attemtping to read file!\n");
	fflush(stdout);
	fopen_s(F, PATH, "r+");
	
	if (*F == NULL) {
		perror(PATH);
		fflush(stdout);
		return FAILURE;
	}
	else {
		printf("File read successfully!\n");
		fflush(stdout);
	}
	fscanf_s(*F, "%d %d %lf %lf %lf %d %lf", &C->N, &C->K, &C->dT, &C->tMax, &C->alpha, &C->LIMIT, &C->QC);
	readPoints(*F, P, C->K, C->N);

	fclose(*F);
	fflush(stdout);
	return SUCCESS;
}

void freePoints(Point *P, int n) {

	// vectors are sequentially allocated
	P->vecInitloc;
	P->vecV;
	P->vecCurrentloc;

	free(P);
}

int sign_discriminant(double *x, double *w, int dim)
{
	double res = w[0];
	for (int i = 0; i < dim; i++)
		res += w[i + 1] * x[i];

	return (res >= 0) ? A : B;
}

void updateWeights(Point X, int sign, double *weights, double alpha, int dim) {

	weights[0] += alpha * sign; // bias

//#pragma omp parallel for  // slowdown if used
	for (int i = 0; i < dim; i++) {
		weights[i + 1] += alpha * sign * X.vecCurrentloc[i];
	}
}

void classify(Classifier C, Point* points, double *weights) 
{
	int f, i, j, iter = 0, counter = 0;

	initWeights(weights, C.K);

	for (j = 0; j < C.LIMIT; j++) { // total iterations

		for (i = 0; i < C.N; i++, counter++) { // iterate points

			f = sign_discriminant(points[i].vecCurrentloc, weights, C.K);

			if (f != points[i].sign) { // misclassified, update weights
				counter = 0;
				updateWeights(points[i], points[i].sign, weights, C.alpha, C.K);
			}

			if (counter >= C.N) // all points classified correctly
				return;
		}
	} // LIMIT reached
}

int countMis(int n, int  dim, Point *X, double *weights) {
	int count=0;

	#pragma omp parallel for reduction(+:count)
	for (int i = 0; i < n; ++i) {

		int f = sign_discriminant(X[i].vecCurrentloc, weights, dim);

		if (f!=X[i].sign)
			count++;
	}

	return count;
}

void initWeights(double *weights, int dim) {
	int i;
	for (i = 0; i <= dim; i++)
		weights[i] = 0;
}

int writeWeights(double *weights, int dim, double time, double q, int solved) {
	FILE *F;
	fopen_s(&F, OUT, "w");

	if (F == NULL) {
		printf("Error writing to file!\n");
		fflush(stdout);
		return FAILURE;
	}

	if (solved)
		fprintf(F, "time minimum = %lf	q= %lf\n\n", time, q);
	else
		fprintf(F, "time was not found\n\n");

	for (int i = 0; i <= dim; i++)
		fprintf(F, "%lf\n", weights[i]);

	fclose(F);
	printf("Weights written to file!\n");
	fflush(stdout);
	return SUCCESS;
}

void MPI_createPointType(MPI_Datatype* MPI_Point) {
	int blockLengths[4] = { 1, 1, 1, 1 };
	MPI_Aint disp[4];
	MPI_Datatype types[4] = { MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE, MPI_INT};

	disp[0] = offsetof(Point, vecInitloc);
	disp[1] = offsetof(Point, vecV);
	disp[2] = offsetof(Point, vecCurrentloc);
	disp[3] = offsetof(Point, sign);

	MPI_Type_create_struct(4, blockLengths, disp, types, MPI_Point);
	MPI_Type_commit(MPI_Point);
}

void MPI_createClassifierType(MPI_Datatype* MPI_Classifier) {
	int blockLengths[7] = { 1, 1, 1, 1, 1, 1, 1 };
	MPI_Aint disp[7];
	MPI_Datatype types[74] = { MPI_INT, MPI_INT, MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_DOUBLE};

	disp[0] = offsetof(Classifier, N);
	disp[1] = offsetof(Classifier, K);
	disp[2] = offsetof(Classifier, dT);
	disp[3] = offsetof(Classifier, tMax);
	disp[4] = offsetof(Classifier, alpha);
	disp[5] = offsetof(Classifier, LIMIT);
	disp[6] = offsetof(Classifier, QC);

	MPI_Type_create_struct(7, blockLengths, disp, types, MPI_Classifier);
	MPI_Type_commit(MPI_Classifier);
}

void allocatePoints(Point **P, int dim, int n, int includeBase,int includeVectors) {
	int i;
	Point *points;
	double *vec_a, *vec_b ,*vec_c;

	if (includeBase) { // allocate Points
		*P = (Point*)malloc(sizeof(Point)*n);
		checkAllocation(*P);
	}

	points = *P;

	if (includeVectors) { // allocate vectors

		// allocate sequential memory blocks for three vectors

		vec_a = (double*)malloc(sizeof(double) * n * dim);
		checkAllocation(vec_a);

		vec_b = (double*)malloc(sizeof(double) * n * dim);
		checkAllocation(vec_b);

		vec_c = (double*)malloc(sizeof(double) * n * dim);
		checkAllocation(vec_c);

		// assign addrs to vectors

		for (i = 0; i < n; i++) {

			points[i].vecInitloc = vec_a + i * dim;
			checkAllocation(points[i].vecInitloc);

			points[i].vecV = vec_b + i * dim;
			checkAllocation(points[i].vecV);

			points[i].vecCurrentloc = vec_c + i * dim;
			checkAllocation(points[i].vecCurrentloc);
		}
	}
}

void packPoints(Point *points, int n, char* buffer,int bufferSize,int dim, MPI_Datatype MPI_Point) {
	int position=0;

	MPI_Pack(points, n, MPI_Point, buffer, bufferSize, &position, MPI_COMM_WORLD);

	// all vectors are sequential...
	MPI_Pack(points->vecInitloc, dim * n, MPI_DOUBLE, buffer, bufferSize, &position, MPI_COMM_WORLD);
	MPI_Pack(points->vecV, dim * n, MPI_DOUBLE, buffer, bufferSize, &position, MPI_COMM_WORLD);

}

void unpackPoints(Point **P, int n, char* buffer, int bufferSize, int dim, MPI_Datatype MPI_Point) {
	int position=0;
	Point *points = *P;

	MPI_Unpack(buffer, bufferSize, &position, points, n, MPI_Point, MPI_COMM_WORLD);

	// allocate vectors only
	allocatePoints(P, dim, n, FALSE, TRUE);

	// all vectors are sequential...
	MPI_Unpack(buffer, bufferSize, &position, points->vecInitloc, dim * n, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, points->vecV, dim * n, MPI_DOUBLE, MPI_COMM_WORLD);

}

void initClassifier(Point *points, Classifier classifier, int myId, int numprocs, double time0) 
{
	cudaError_t cudaStatus;
	MPI_Status status;
	int i, dim = classifier.K, n = classifier.N, tag = TAG_CONTINUE, solved = FALSE;
	double  q, t, *qArray ,*Weights, *dev_a, *dev_b, *dev_c;

	if (myId == MASTER) {
		qArray = (double*)malloc(sizeof(double)* numprocs);
		checkAllocation(qArray);
	}

	Weights = (double*)calloc(classifier.K + 1, sizeof(double));
	checkAllocation(Weights);

	// init cuda pointers for later use
	cudaStatus = initCudaMemory(points, n, dim, &dev_a, &dev_b, &dev_c);
	checkCudaError(cudaStatus);
 
	// init process time offset
	t = myId * classifier.dT;

	while (tag == TAG_CONTINUE && !solved) {

		// stop if t > tMax, or last time to check reached
		if ( (t + classifier.dT) > classifier.tMax){
			tag = TAG_HALT;
			q = IGNORE;
		}
		else { // perceptron alogorithm routine for set time
			perceptronRoutine(points, classifier, Weights, &solved, &q, t,dev_a, dev_b, dev_c);
		}

		// gather q results from processes
		MPI_Gather(&q, 1, MPI_DOUBLE, qArray, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

		if (myId == MASTER) {
			// check if any q is within desired QC range
			for (i = 0; i < numprocs; i++) {
				q = qArray[i];

				if (q != IGNORE) {
					// solved time
					t = t + i * classifier.dT;
					solved = TRUE;
					tag = TAG_HALT;
					break;
				}
			}
		}

		// advance time to check
		if (!solved)
			t += classifier.dT * numprocs;

		// process directive
		MPI_Bcast(&tag, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	}

	// get solving weights
	// only MASTER and solving process will have this flag set to TRUE
	if (solved)
	{
		if (myId == MASTER && MASTER !=i) // master is not solving process
			MPI_Recv(Weights, dim + 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		else if (myId !=MASTER) // solving slave
			MPI_Send(Weights, dim + 1, MPI_DOUBLE, MASTER, TAG_HALT, MPI_COMM_WORLD);
	}
	else if (myId == MASTER) { // MASTER always handles last time to check, if needed

		// move to last time to check
		t = 0;
		while ( (t+ classifier.dT) <= classifier.tMax)
			t += classifier.dT;

		perceptronRoutine(points, classifier, Weights, &solved, &q, t, dev_a, dev_b, dev_c);
	}

	// write results to file
	if ( myId == MASTER) {

		printf("\nRuntime: %lf seconds\nt= %lf, q= %lf\nWeights: ", MPI_Wtime() - time0, t, q);
		for (int z = 0; z <= dim; z++)
			printf("%lf, ", Weights[z]);
		printf("\n\n");
		fflush(stdout);

		// write to file
		if (!writeWeights(Weights, classifier.K, t, q, solved))
			exit(1);

		free(qArray);
	}

	free(Weights);
	freeCudaMemory(dev_a, dev_b, dev_c);
}                                        

void checkCudaError(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess) {
		printf("Cuda failed!");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, FAILURE);
		exit(1);
	}
}

void perceptronRoutine(Point *points, Classifier classifier, double *Weights, int *solved, double *q, double t,
	double *dev_a,double *dev_b,double *dev_c) {

	cudaError_t cudaStatus;
	int mis, n = classifier.N, dim = classifier.K;

	// set points according to time, with Cuda
	cudaStatus = setPointsWithCuda(points, n, dim, t, dev_a, dev_b, dev_c);
	checkCudaError(cudaStatus);

	//setPoints(points, n, dim, t);

	// calculate weights
	classify(classifier, points, Weights);

	// classifyParallel(classifier, points, Weights, threadFirstMis);
	// classifyWithCuda(classifier, points, Weights,dev_c);

	// count misclassifications
	mis = countMis(n, dim, points, Weights);

	// calculate quality of classifier

	*q = mis / (double)n;

	if (*q < classifier.QC)
		*solved = TRUE;
	else
		*q = IGNORE;
}

// unused below

void setPoints(Point *points, int n, int dim, double t) {
#pragma omp parallel for
	for (int i = 0; i < n; i++)
		for (int j = 0; j < dim; j++)
			points[i].vecCurrentloc[j] = points[i].vecInitloc[j] + points[i].vecV[j] * t;
}

void classifyWithCuda(Classifier C, Point *points, double *weights, double *dev_points) {
	int f, i, j, iter = 0, counter = 0;
	int *fArray = (int*)malloc(sizeof(int)*C.N);
	initWeights(weights, C.K);

	markWithCuda(C, weights, fArray, dev_points);

	for (j = 0; j < C.LIMIT; j++) { // total iterations

		for (i = 0; i < C.N; i++, counter++) { // iterate points

			if (fArray[i] != points[i].sign) { // misclassified, update weights
				f = sign_discriminant(points[i].vecCurrentloc, weights, C.K);
				updateWeights(points[i], points[i].sign, weights, C.alpha, C.K);
				markWithCuda(C, weights, fArray, dev_points);
				counter = 0;
			}

			if (counter >= C.N) { // all points classified correctly
				return;
				free(fArray);
			}
		}
	} // LIMIT reached
	free(fArray);
}

void classifyParallel(Classifier C, Point* points, double *weights, int* threadFirstMis)
{
	int f, i, j, remainderMis, iterMis = FALSE, firstMis = -1, threadHit;
	int maxT = omp_get_max_threads(), n = C.N;
	Point p;

	// iterMis        // current iteration misclassified point detection
	// remainderMis   // current iteration remainder misclassified point detection
	// firstMis       // first misclassified point index in current iteration

	initWeights(weights, C.K);

	for (j = 0; j < C.LIMIT; j++) { // total iterations

		remainderMis = FALSE;
		threadHit = maxT;

#pragma omp parallel private(f, i) shared(remainderMis,threadFirstMis, threadHit)
		{
			int tid = omp_get_thread_num();
			threadFirstMis[tid] = n;

#pragma omp for
			for (i = firstMis + 1; i < n; i++) {

				if (threadHit < tid) // earlier thread found mis
					break;

				f = sign_discriminant(points[i].vecCurrentloc, weights, C.K);

				if (f != points[i].sign) { // misclassified
					remainderMis = TRUE;
					threadHit = tid; // negligible critical section
					threadFirstMis[tid] = i;
					break;
				}

			}
		}

		// get first mis index
		firstMis = threadFirstMis[0];
		for (int t = 1; t < maxT; t++)
			if (threadFirstMis[t] < firstMis)
				firstMis = threadFirstMis[t];

		if (remainderMis) { // misclassified point found in remainder. Restart iteration

			iterMis = TRUE;
			p = points[firstMis]; // mis point
			f = sign_discriminant(p.vecCurrentloc, weights, C.K);
			updateWeights(p, f, weights, C.alpha, C.K);
			j--;
		}
		else if (iterMis) {// no mis points in remainder, cycle not complete, continue to next iteration
			iterMis = FALSE;
			firstMis = -1;
		}
		else // no mis points for entire cycle, max quality reached
			break;
	}
}
