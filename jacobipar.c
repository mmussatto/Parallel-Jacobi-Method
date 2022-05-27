#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN_DEVIATION 1e-7
#define MAX_ITERATIONS 100
#pragma GCC diagnostic ignored "-Wunused-result"

typedef struct jacobiRet {
    double* solution;
    int iterationsTaken;
} JacobiRet;

/**
 * Creates a random double using the rand() function. 
 * The range of the number created is: -1 < n < 1.
 * 
 * @return double 
 */
double randDouble() {
    return rand() / (0.5 * RAND_MAX) - 1.0;
}

/** 
 * Generates a random A matrix such that Jacobi Method converges
 * Note that a sufficient condition for convergence is that
 * the matrix is diagonally dominant, i.e., for each row we garantee that
 * the sum of the absolute value of every term in that row besides the diagonal one
 * is less than the absolute value of the diagonal term.
 *
 * As such for row k, if we assume that rowSum is the sum of every term in that
 * row besides the diagonal one, then rowSum/X = A[k][k] then dividing every
 * element besides the diagonal one by X = rowSum/A[k][k] + c, where c is
 * a positive constant transforms any matrix into a diagonally dominant matrix
 * 
 * @param matrixSize    size of the matrix   
 * @return double**     a diagonally dominant matrix 
 */
double** randomDiagonallyDominantMatrix(int matrixSize) {
    // Allocates memory for the matrix A
    double** matrix = (double**)malloc(sizeof(double*) * matrixSize);
    for (int i = 0; i < matrixSize; i++) {
        matrix[i] = (double*)malloc(sizeof(double) * matrixSize);
    }

    // Fill the matrix in such way that it is diogonally dominant
    for (int i = 0; i < matrixSize; i++) {
        // Fill the matrix
        matrix[i][i] = randDouble();
        double rowSum = 0.0;
        for (int j = 0; j < matrixSize; j++) {
            if (i == j) continue;
            matrix[i][j] = randDouble();
            rowSum += fabs(matrix[i][j]);
        }

        // Making the matrix diagonally dominant
        double X = rowSum / fabs(matrix[i][i]) + fabs(randDouble());
        for (int j = 0; j < matrixSize; j++) {
            if (i == j) continue;
            matrix[i][j] /= X;
        }
    }

    return matrix;
}

/**
 *  Creates a random vector with size designated by the user.
 *  The type of the elements inside the vector is double.
 * 
 * @param vectorSize   
 * @return double* 
 */
double* randomVector(int vectorSize) {
    // Creating a random B matrix
    double* vector = (double*)malloc(sizeof(double) * vectorSize);
    for (int i = 0; i < vectorSize; i++)
        vector[i] = randDouble();

    return vector;
}

/**
 * @brief Copies a vector to another one of the same size.
 * 
 * @param dest          destination of the copied vector
 * @param source        vector to be copied
 * @param vectorSize    size of the vector  
 */
void copyVector(double* dest, double* source, int vectorSize) {
    for (int i = 0; i < vectorSize; i++)
        dest[i] = source[i];
}

/**
 * @brief Calculates the maximum difference between the current iteration and the last one. 
 *  It receives two vectors, the current and the previous one.
 *  For each element of the vector it calculates the absolute diference between 
 *  the current and previous vector.
 * 
 * @param curr          current iteration vector
 * @param prev          previous iteration vector
 * @param vectorSize    size of the vectors
 * @return double 
 */
double maxDiff(double* curr, double* prev, int vectorSize) {
    double maxDiff = 0;
    for (int i = 0; i < vectorSize; i++) {
        double diff = fabs(curr[i] - prev[i]);
        maxDiff = (diff > maxDiff) ? diff : maxDiff;
    }
    return maxDiff;
}

/**
 * @brief This function prints the Linear System (AX = B) in the terminal.
 *
 *  The Matrix A is a square matrix. 
 *  The vectors have the dimentions of the rows of the matrix A.
 * 
 * @param A             matrix A
 * @param B             vector B
 * @param X             vector X
 * @param matrixSize    size of the matrix
 */
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

/**
 * @brief Ask user an equation line and compare the result with expected B matrix value
 * 
 * @param A             matrix A
 * @param B             vector B
 * @param X             vector X
 * @param matrixSize    size of the matrix
 */
void compareResult(double** A, double* B, double* X, int matrixSize) {
    int equation;
    printf("Choose an equation to evaluate: ");
    scanf("%d", &equation);

    // Calculate solution and compare to B[]
    if(equation >= 0 && equation < matrixSize) {
        double sum = 0;
        for (int i = 0; i < matrixSize; i++) {
            sum += A[equation][i] * X[i];
        }

        printf("Compare of results:\n");
        printf(" Calculated result = %.8lf\n", sum);
        printf(" Expected result = %.8lf\n", B[equation]);
        printf(" Deviation = %.8lf\n", sum-B[equation]);
    }
    else {
        printf("Equation number out of range\n");
    }
}

/**
 * @brief Implementation of the parallel Jacobi method.
 * 
 * @param A                 diagonally dominant matrix (A) of the linear system
 * @param B                 vector of constant terms (B) of the linear system
 * @param matrixSize        size of the matrix A
 * @param numberOfThreads   number of threads
 * @return JacobiRet        structure containing vector of solutions and number of iterations taken
 */
JacobiRet jacobipar(double** A, double* B, int matrixSize, int numberOfThreads) {
    // Previous and current values of the solution (x^k and x^{k+1})
    int iterationCounter;
    double* prevX = (double*)calloc(matrixSize, sizeof(double)); // Init with zeros
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

    //Create Matrix A
    double** A = randomDiagonallyDominantMatrix(matrixSize);

    //Create Vector B
    double* B = randomVector(matrixSize);

    // Determining the solution to the system of linear equations
    double startTime = omp_get_wtime();
    JacobiRet jacobiRet = jacobipar(A, B, matrixSize, numberOfThreads);
    double endTime = omp_get_wtime();

    //Print the time needed to solve the Linear System, the number of iterations and the number of threads
    printf("Solved %dx%d linear system in %.7lf seconds after %d iterations using %d threads\n", matrixSize, matrixSize, endTime - startTime, jacobiRet.iterationsTaken, numberOfThreads);
    
    //If matriz has order lower than 4, print the Linear System in the terminal
    if (matrixSize <= 20) {
        showLinearSystem(A, B, jacobiRet.solution, matrixSize);
    }

    // Ask user to choose an equation
    compareResult(A, B, jacobiRet.solution, matrixSize);

    //Deallocate memory
    for (int i = 0; i < matrixSize; i++)
        free(A[i]);
    free(A);
    free(B);
    free(jacobiRet.solution);

    //Program Finished
    return 0;
}