#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** args) {
    if (argc != 3) {
        printf("Error: usage ./jacobipar <N> <T>, where N is the square matrix dimensions and T is the desired number of threads\n");
        return 0;
    }
    int matrixSize = atoi(args[1]);
    int numberOfThreads = atoi(args[2]);

    printf("%d %d\n", matrixSize, numberOfThreads);
}