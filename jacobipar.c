#include "linearSystem.h"

JacobiRet jacobipar(double** A, double* B, int matrixSize, int numberOfThreads) {
    int iterationCounter;
    double* prevX = (double*)calloc(matrixSize, sizeof(double));
    double* currX = (double*)malloc(sizeof(double) * matrixSize);

    double deviation = INFINITY;
    for (iterationCounter = 0; deviation >= MIN_DEVIATION && iterationCounter <= MAX_ITERATIONS; iterationCounter++) {
        deviation = 0;
        #pragma omp parallel for num_threads(numberOfThreads) shared(matrixSize, A, B, prevX, currX) reduction(max: deviation)
        for (int i = 0; i < matrixSize; i++) {
            // Estimates new values for the ith value
            double sum = 0.0;
            for (int j = 0; j < matrixSize; j++) {
                if (i != j) {
                    sum += A[i][j] * prevX[j];
                }
            }
            currX[i] = (B[i] - sum) / A[i][i];

            // calculates new deviation
            double diff = fabs(currX[i] - prevX[i]);
            deviation = (diff > deviation) ? diff : deviation;

            // Overrides previous iteration
            prevX[i] = currX[i];
        }
    }
    free(prevX);

    JacobiRet ret = {currX, iterationCounter};
    return ret;
}

int main(int argc, char** args) {
    if (argc != 3) {
        printf("Error: usage ./jacobipar <N> <T>, where N is the square matrix dimensions and T is the desired number of threads\n");
        return 0;
    }
    // Fixes a random seed and gets the desired matrixSize and numberOfThreads
    srand(69420);
    int matrixSize = atoi(args[1]);
    int numberOfThreads = atoi(args[2]);

    double** A = randomDiagonallyDominantMatrix(matrixSize);
    double* B = randomVector(matrixSize);

    // Determining the solution to the system of linear equations
    double startTime = omp_get_wtime();
    JacobiRet jacobiRet = jacobipar(A, B, matrixSize, numberOfThreads);
    double endTime = omp_get_wtime();

    printf("Solved %dx%d linear system in %.3lf seconds after %d iterations using %d threads\n", matrixSize, matrixSize, endTime - startTime, jacobiRet.iterationsTaken, numberOfThreads);
    if (matrixSize <= 3) {
        showLinearSystem(A, B, jacobiRet.solution, matrixSize);
        showWolframAlphaInput(A, B, matrixSize);
    }

    for (int i = 0; i < matrixSize; i++)
        free(A[i]);
    free(A);
    free(B);
    free(jacobiRet.solution);
}