.DEFAULT_GOAL := all

SRC_DIR := src
BUILD_DIR := build

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/validate.o: $(SRC_DIR)/validate.cc | $(BUILD_DIR)
	g++ -c $(SRC_DIR)/validate.cc -I. -o $(BUILD_DIR)/validate.o 

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu | $(BUILD_DIR)
	nvcc -ccbin mpic++ -c $(SRC_DIR)/main.cu -I. -arch native -o $(BUILD_DIR)/main.o

$(BUILD_DIR)/args.o: $(SRC_DIR)/args.cc | $(BUILD_DIR)
	g++ -c $(SRC_DIR)/args.cc -I. -o $(BUILD_DIR)/args.o

$(BUILD_DIR)/walltime.o: $(SRC_DIR)/walltime.cc | $(BUILD_DIR)
	g++ -c $(SRC_DIR)/walltime.cc -I. -o $(BUILD_DIR)/walltime.o

all: $(BUILD_DIR)/main.o $(BUILD_DIR)/args.o $(BUILD_DIR)/walltime.o
	nvcc -ccbin mpic++ -o $(BUILD_DIR)/main -arch native $(BUILD_DIR)/main.o $(BUILD_DIR)/args.o $(BUILD_DIR)/walltime.o

validate: $(BUILD_DIR)/validate.o $(BUILD_DIR)/args.o 
	g++ -o $(BUILD_DIR)/validate $(BUILD_DIR)/validate.o $(BUILD_DIR)/args.o

.PHONY: clean
clean:
	rm -rf main $(BUILD_DIR)
