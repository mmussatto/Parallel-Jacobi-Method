#include "linearSystem.h"

JacobiRet jacobiseq(double** A, double* B, int matrixSize) {
    int iterationCounter;
    double* prevX = (double*)calloc(matrixSize, sizeof(double));
    double* currX = (double*)malloc(sizeof(double) * matrixSize);

    double deviation = INFINITY;
    for (iterationCounter = 0; deviation >= MIN_DEVIATION && iterationCounter <= MAX_ITERATIONS; iterationCounter++) {
        for (int i = 0; i < matrixSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < matrixSize; j++) {
                if (i != j) {
                    sum += A[i][j] * prevX[j];
                }
            }
            currX[i] = (B[i] - sum) / A[i][i];
        }
        deviation = maxDiff(prevX, currX, matrixSize);
        copyVector(prevX, currX, matrixSize);
    }
    free(prevX);

    JacobiRet ret = {currX, iterationCounter};
    return ret;
}

int main(int argc, char** args) {
    if (argc != 2) {
        printf("Error: usage ./jacobiseq <N>, where N is the square matrix dimensions\n");
        return 0;
    }

    // Fixes a random seed and gets the desired matrixSize
    srand(69420);
    int matrixSize = atoi(args[1]);

    double** A = randomDiagonallyDominantMatrix(matrixSize);
    double* B = randomVector(matrixSize);

    // Determining the solution to the system of linear equations
    double startTime = omp_get_wtime();
    JacobiRet jacobiRet = jacobiseq(A, B, matrixSize);
    double endTime = omp_get_wtime();

    printf("Solved %dx%d linear system in %.3lf seconds after %d iterations\n", matrixSize, matrixSize, endTime - startTime, jacobiRet.iterationsTaken);
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