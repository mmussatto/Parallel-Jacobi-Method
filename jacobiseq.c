#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN_DEVIATION 1e-7
#define MAX_ITERATIONS 100

double randDouble() {
    return rand() / (1.0 * RAND_MAX);
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

double* jacobiseq(double** A, double* B, int matrixSize) {
    double* prevX = (double*)malloc(sizeof(double) * matrixSize);
    double* currX = (double*)malloc(sizeof(double) * matrixSize);
    for (int i = 0; i < matrixSize; i++)
        prevX[i] = 0;

    double deviation = 100;
    int iterationCounter = 0;
    for (int iterationCounter = 0; deviation >= MIN_DEVIATION && iterationCounter <= MAX_ITERATIONS; iterationCounter++) {
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

    return currX;
}

int main(int argc, char** args) {
    if (argc != 2) {
        printf("Error: usage ./jacobiseq <N>, where N is the square matrix dimensions\n");
        return 0;
    }

    // Fixes a random seed and gets the desired matrixSize
    srand(69420);
    int matrixSize = atoi(args[1]);

    // Allocates memory for the A matrix
    double** A = (double**)malloc(sizeof(double*) * matrixSize);
    for (int i = 0; i < matrixSize; i++) {
        A[i] = (double*)malloc(sizeof(double) * matrixSize);
    }

    /** Generates a random A matrix such that Jacobi Method converges
     * Note that a sufficient condition for convergence is that
     * the matrix is diagonally dominant, i.e., for each row we garantee that
     * the sum of the absolute value of every term in that row besides the diagonal one
     * is less than the absolute value of the diagonal term.
     *
     * Here, we are generating A matrices that every elements is positive. As such
     * for row k, if we assume that rowSum is the sum of every term in that row
     * besides the diagonal one, then rowSum/X = A[k][k] then dividing every
     * element besides the diagonal one by X = rowSum/A[k][k] + c, where c is
     * a positive constant transforms any matrix into a diagonally dominant matrix
     */
    for (int i = 0; i < matrixSize; i++) {
        A[i][i] = randDouble();
        double rowSum = 0.0;
        for (int j = 0; j < matrixSize; j++) {
            if (i == j) continue;
            A[i][j] = randDouble();
            rowSum += A[i][j];
        }
        double X = rowSum / A[i][i] + randDouble();
        for (int j = 0; j < matrixSize; j++) {
            if (i == j) continue;
            A[i][j] /= X;
        }
    }

    // Creating a random B matrix
    double* B = (double*)malloc(sizeof(double) * matrixSize);
    for (int i = 0; i < matrixSize; i++)
        B[i] = (double)rand() / (1.0 * RAND_MAX);

    // Determining the solution to the system of linear equations
    double* X = jacobiseq(A, B, matrixSize);

    // Showing everything in a clean way
    for (int i = 0; i < matrixSize; i++) {
        printf("[");
        for (int j = 0; j < matrixSize - 1; j++) {
            printf("%lf ", A[i][j]);
        }
        if (i == (matrixSize - 1) / 2)
            printf("%lf] [%lf]  =  [%lf]\n", A[i][matrixSize - 1], X[i], B[i]);
        else
            printf("%lf] [%lf]     [%lf]\n", A[i][matrixSize - 1], X[i], B[i]);
    }
}