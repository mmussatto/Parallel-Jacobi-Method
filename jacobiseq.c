#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN_DEVIATION 1e-7
#define MAX_ITERATIONS 100

typedef struct jacobiRet {
    double* solution;
    int iterationsTaken;
} JacobiRet;

double randDouble() {
    return rand() / (0.5 * RAND_MAX) - 1.0;
}

/** Generates a random A matrix such that Jacobi Method converges
 * Note that a sufficient condition for convergence is that
 * the matrix is diagonally dominant, i.e., for each row we garantee that
 * the sum of the absolute value of every term in that row besides the diagonal one
 * is less than the absolute value of the diagonal term.
 *
 * As such for row k, if we assume that rowSum is the sum of every term in that
 * row besides the diagonal one, then rowSum/X = A[k][k] then dividing every
 * element besides the diagonal one by X = rowSum/A[k][k] + c, where c is
 * a positive constant transforms any matrix into a diagonally dominant matrix
 */
double** randomDiagonallyDominantMatrix(int matrixSize) {
    // Allocates memory for the A matrix
    double** matrix = (double**)malloc(sizeof(double*) * matrixSize);
    for (int i = 0; i < matrixSize; i++) {
        matrix[i] = (double*)malloc(sizeof(double) * matrixSize);
    }

    for (int i = 0; i < matrixSize; i++) {
        matrix[i][i] = randDouble();
        double rowSum = 0.0;
        for (int j = 0; j < matrixSize; j++) {
            if (i == j) continue;
            matrix[i][j] = randDouble();
            rowSum += fabs(matrix[i][j]);
        }
        double X = rowSum / fabs(matrix[i][i]) + fabs(randDouble());
        for (int j = 0; j < matrixSize; j++) {
            if (i == j) continue;
            matrix[i][j] /= X;
        }
    }

    return matrix;
}

double* randomVector(int vectorSize) {
    // Creating a random B matrix
    double* vector = (double*)malloc(sizeof(double) * vectorSize);
    for (int i = 0; i < vectorSize; i++)
        vector[i] = randDouble();

    return vector;
}

void copyVector(double* dest, double* source, int vectorSize) {
    for (int i = 0; i < vectorSize; i++)
        dest[i] = source[i];
}

double maxDiff(double* curr, double* prev, int vectorSize) {
    double maxDiff = 0;
    for (int i = 0; i < vectorSize; i++) {
        double diff = fabs(curr[i] - prev[i]);
        maxDiff = (diff > maxDiff) ? diff : maxDiff;
    }
    return maxDiff;
}

JacobiRet jacobiseq(double** A, double* B, int matrixSize) {
    int iterationCounter;
    double* prevX = (double*)malloc(sizeof(double) * matrixSize);
    double* currX = (double*)malloc(sizeof(double) * matrixSize);
    for (int i = 0; i < matrixSize; i++)
        prevX[i] = 0;

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

void showLinearSystem(double** A, double* B, double* X, int matrixSize) {
    for (int i = 0; i < matrixSize; i++) {
        printf("[");
        for (int j = 0; j < matrixSize - 1; j++) {
            printf("%9.6lf ", A[i][j]);
        }
        if (i == (matrixSize - 1) / 2)
            printf("%9.6lf] [%9.6lf]  =  [%9.6lf]\n", A[i][matrixSize - 1], X[i], B[i]);
        else
            printf("%9.6lf] [%9.6lf]     [%9.6lf]\n", A[i][matrixSize - 1], X[i], B[i]);
    }
}

void showWolframAlphaInput(double** A, double* B, int matrixSize) {
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize - 1; j++) {
            char letter = 'A' + j;
            printf("(%lf * %c) + ", A[i][j], letter);
        }
        printf("(%lf * %c) = %lf", A[i][matrixSize - 1], 'A' + matrixSize - 1, B[i]);

        if (i != matrixSize - 1)
            printf(" and ");
        else
            printf("\n");
    }
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