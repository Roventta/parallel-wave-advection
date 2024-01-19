# Makefile for COMP4300/8300 Assignment 2 (OpenMP)
# Peter Strazdins, SOCO ANU, Apr 21     
.SUFFIXES:
.PRECIOUS: %.o

HDRS=serAdvect.h parAdvect.h
OBJS=serAdvect.o parAdvect.o
PROG=testAdvect
CCFLAGS=-O3 -fopenmp

all: $(PROG) 

%: %.o $(OBJS)
	gcc -o $* $*.o $(OBJS) -lgomp -lm
%.o: %.c $(HDRS)
	gcc -Wall $(CCFLAGS) -c $*.c
clean:
	rm -f *.o $(PROG)

ori:
	gcc -Wall -O3 -fopenmp -c parAdvect_ori.c
	gcc -Wall -O3 -fopenmp -c testAdvect_ori.c
	gcc -o testAdvect_ori testAdvect_ori.o serAdvect.o parAdvect_ori.o -lgomp -lm
bad:
	gcc -Wall -O3 -fopenmp -c parAdvect_bad.c
	gcc -Wall -O3 -fopenmp -c testAdvect_bad.c
	gcc -o testAdvect_bad testAdvect_bad.o serAdvect.o parAdvect_bad.o -lgomp -lm
