#include <stdio.h>
#include <stdlib.h>

#define MIN_DEVIATION 1e-3
#define MAX_ITERATIONS 1000

double randDouble() {
    return rand() / (1.0 * RAND_MAX);
}

void copyVector(double* dest, double* source, int vectorSize) {
    for(int i = 0; i < vectorSize; i++)
        dest[i] = source[i];
}

double* jacobiseq(double** A, double* B, int matrixSize) {
	double* x0 = (double*) malloc(sizeof(double)*matrixSize);
    for(int i = 0; i < matrixSize; i++)
        x0[i] = randDouble();
    
	double* x = (double*) malloc(sizeof(double)*matrixSize);
    copyVector(x, x0, matrixSize);

	double deviation = 100;
	int iterationCounter = 0;      //A quantidade de iterações feitas

	for(int iterationCounter = 0; deviation >= MIN_DEVIATION && iterationCounter <= MAX_ITERATIONS; iterationCounter++) {
		for (int i = 0; i < matrixSize; i++) {
			double soma = 0.0;
			for (int j = 0; j < matrixSize; j++) {
				if (i != j) {
					soma += (A[i][j] * x0[j]) / A[i][i];
				}
			}
            x[i] = (B[i] / A[i][i]) - soma;
		}
		//deviation = calcErro(x, x0, matrixSize);
		copyVector(x, x0, matrixSize);
	}
	
	return x;
	
}
	

/*
//Faz o cálculo do erro
private static double calcErro(double[] a, double[] b) {
	double result[] = new double[a.length];
	for (int i = 0; i < a.length; i++) {
		result[i] = Math.abs(a[i] - b[i]);
	}
	int cont = 0;            //um contador
	double maior = 0;       //o maior número contido no vetor
	while (cont < a.length) {
		maior = Math.max(maior, result[cont]);
		cont++;
	}
	return maior;
}*/

int main(int argc, char** args) {
    if(argc != 2) {
        printf("Error: usage ./jacobiseq <N>, where N is the square matrix dimensions");
        return 0;
    }

    srand(69420);
    int matrixSize = atoi(args[1]);

    double** A = (double**) malloc(sizeof(double*)*matrixSize);
    for(int i = 0; i < matrixSize; i++) {
        A[i] = (double*) malloc(sizeof(double)*matrixSize);
    }

    for(int i = 0; i < matrixSize; i++)
        for(int j = 0; j < matrixSize; j++)
            A[i][j] = randDouble();
    
    double* B = (double*) malloc(sizeof(double)*matrixSize);

    for(int i = 0; i < matrixSize; i++)
        B[i] = (double) rand() / (1.0 * RAND_MAX);
    
    //double* X = jacobi(A, B, matrixSize);
    double* X = jacobiseq(A, B, matrixSize);

    for(int i = 0; i < matrixSize; i++) {
        printf("[");
        for(int j = 0; j < matrixSize - 1; j++) {
            printf("%lf ", A[i][j]);
        }
        if(i == (matrixSize-1)/2)
            printf("%lf] [%lf]  =  [%lf]\n", A[i][matrixSize-1], X[i], B[i]);
        else
            printf("%lf] [%lf]     [%lf]\n", A[i][matrixSize-1], X[i], B[i]);
    }

}