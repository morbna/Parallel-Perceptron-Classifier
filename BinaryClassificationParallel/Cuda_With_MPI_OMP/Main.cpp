#include "Header.h"

int main(int argc, char *argv[])
{
	Classifier classifier;
	Point *points;
	FILE *F;
	double time0;
	int bufferSize;
	char *buffer;

	int numprocs, myId;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Datatype MPI_Point, MPI_Classifier;
	MPI_createPointType(&MPI_Point);
	MPI_createClassifierType(&MPI_Classifier);

	time0 = MPI_Wtime();

	if (numprocs < 2) {
		printf("at least 2 processes are required!");
		MPI_Abort(MPI_COMM_WORLD, FAILURE);
		exit(1);
	}

	// Init points
	if (myId == MASTER) {
		if (!readFile(&F, &classifier, &points)) {
			MPI_Abort(MPI_COMM_WORLD, FAILURE);
			exit(1);
		}
	}

	// Bcast classifier info
	MPI_Bcast(&classifier, 1, MPI_Classifier, MASTER, MPI_COMM_WORLD);

	// Allocation for my points, not including vectors
	if (myId != MASTER)
		allocatePoints(&points, classifier.K, classifier.N, TRUE, FALSE);

	bufferSize = (sizeof(Point) + (2 * sizeof(double)*classifier.K)) * classifier.N;
	buffer = (char*)malloc(bufferSize);

	if (myId == MASTER) // Pack points 
		packPoints(points, classifier.N, buffer, bufferSize, classifier.K, MPI_Point);

	// Bcast points
	MPI_Bcast(buffer, bufferSize, MPI_PACKED, MASTER, MPI_COMM_WORLD);

	if (myId != MASTER) // Unpack points
		unpackPoints(&points, classifier.N, buffer, bufferSize, classifier.K, MPI_Point);
	
	free(buffer);

	// start numprocs classifiers
	initClassifier(points, classifier, myId, numprocs, time0);

	// free
	freePoints(points, classifier.N);

	MPI_Finalize();
	return 0;
}