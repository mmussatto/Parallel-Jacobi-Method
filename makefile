#***********************************************************
# Computacao de Alto Desempenho - SSC0903                  *
#                                                          *
# Alessandro de Freitas Guerreiro   No USP 11233891        *
# Bruno Alvarenga Colturato         No USP 11200251        *
# Murilo Mussatto                   No USP 11234245        *
#                                                          *
# Sao Carlos - SP                                          *
# 2022                                                     *
# **********************************************************

PROJ_NAME = jacobiseq jacobi-omp
# Compiler
CC=gcc

# default run program
PROG = -1

# Flags for compiler
CC_FLAGS=-I						\
		 -Wall					\
		 -Wextra				\
		 -Wshadow				\
		 -Wundef				\
		 -Wformat=2				\
		 -Wfloat-equal			\
		 -Wcast-align			\
		 -std=c17				\
		 -march=native			\
		 -O3					\
		 -Ofast

# Libraries
LIBS=-lm -fopenmp

# Compilation
all: clean $(PROJ_NAME) jacobimpi halley
	@echo "Done"

$(PROJ_NAME):
	@echo "Compiling $@"
	@$(CC) -o $@ $@.c $(CC_FLAGS) $(LIBS)

jacobimpi:
	@echo "Compiling jacobi-mpi"
	@mpicc -o jacobipar jacobi-mpi.c $(CC_FLAGS) $(LIBS)

halley:
	@touch halley.txt
	@for h in $$(seq 2 9); do\
		echo "hal0$$h slots=1" >> halley.txt;\
	done
	@for h in $$(seq 10 3 13); do\
		echo "hal$$h slots=1" >> halley.txt;\
	done

.PHONY: run
run:
	@if [ $(PROG) = jacobiseq ]; then\
        read -p "Enter the matrix's order: " ORDER \
		&& ./$(PROG) $${ORDER};\
	elif [ $(PROG) = jacobipar ]; then\
        read -p "Enter the matrix's order, the number of processors and the desired number of threads: " ARGS \
		&& mpirun -np 1 --hostfile halley.txt $(PROG) $${ARGS};\
	elif [ $(PROG) = jacobi-omp ]; then\
		read -p "Enter the matrix's order and the desired number of threads: " ARGS \
		&& ./$(PROG) $${ARGS};\
	else \
        echo "Usage: make run PROG=<prog>, where <prog> can be either jacobiseq, jacobipar or jacobi-omp.";\
	fi

.PHONY: clean
clean:
	@echo "Cleaning"
	@rm -rf $(PROJ_NAME) jacobipar halley.txt