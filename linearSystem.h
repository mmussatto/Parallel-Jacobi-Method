#ifndef LINEARSYSTEM_H
#define LINEARSYSTEM_H

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
        matrix[i][i] = randDouble() * matrixSize;
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
#endif