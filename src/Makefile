.DEFAULT_GOAL := all

main.o: main.cu
	nvcc -ccbin mpic++ -c main.cu -I. -arch native

args.o: args.cc
	g++ -c args.cc -I.
	
all: main.o args.o
	nvcc -ccbin mpic++ -o main -arch native main.o args.o

.PHONY: clean
clean:
	rm -rf main *.o
