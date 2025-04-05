#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "args.h"
#include <cstdint>
#include <mpi.h>

#define DEBUG
#ifdef DEBUG
    #define PRINT(...) printf(__VA_ARGS__)
    #define MPI_PRINT(rank, ...) if(rank == 0) printf(__VA_ARGS__);
#else
    #define PRINT(...)
    #define MPI_PRINT(rank, ...)
#endif

#define MAX_ITERAION 255
#define INT_TYPE uint8_t // Note we use a smaller int type to save on memory

__global__
void mandelbrot(INT_TYPE *p, int length, int offset, int tasks_per_thread, int img_width, int img_height, float start_x, float start_y, float step) {
   
    // Thread Index 
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
   

    for(int i = 0; i < tasks_per_thread; i++) {
        // if(threadIdx.x == 0) {
        //     printf("Block: %d index: %d\n", blockIdx.x, idx);
        // }

        int task_idx = idx * tasks_per_thread + i;

        // Mask out threads that are out of range
        if(task_idx >= length) {
            return;
        }

        // Pixel Index
        int pixel_idx = task_idx + offset;

        // Convert Pixel Index to Image Coordinates
        int img_x = pixel_idx % img_width;
        int img_y = pixel_idx / img_width;

        // Convert Image Coordinates to Cartesian Coordinates
        int cart_x = img_x;
        int cart_y = img_height - img_y - 1;

        float real = cart_x * step + start_x;
        float im = cart_y * step + start_y;
        
        float x = 0;
        float y = 0;


        INT_TYPE iteration = 0;

        while(iteration < MAX_ITERAION && x * x + y * y <= 4) {
            float temp = x*x - y*y + real;
            y = 2 * x * y + im;
            x = temp;
            iteration++;
        }

        p[task_idx] = iteration;
    }
}

int main(int argc, char* argv[]) {
  
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);



  
    Args args;
    if(!Args::parse(argc, argv, args)) {
        PRINT("Failed to parse args!\nExpected: %s --start <float> <float> --dim <int> <int> --step <float> --file <string> --tasks-per-thread <int>", argv[0]);
        MPI_Finalize();
        return 1; 
    }

      // Total number of pixel that need to be computed 
    int problem_size = args.width * args.height; 


    // Print on rank 0;
    MPI_PRINT(rank, "Calculating Mandelbrot Set!\nstart_x: %f\nstart_y: %f\nend_x: %f\nend_y: %f\ndelta: %f\nimg_width: %d\nimg_height: %d\ntasks_per_thread: %d\nProblem_size: %d\nNodes: %d\n", 
                args.start_x, 
                args.start_y, 
                args.start_x + args.width * args.step, 
                args.start_y + args.height * args.step,
                args.step,
                args.width,
                args.height,
                args.tasks_per_thread,
                problem_size,
                world_size
        );
   

    // Amount of pixels designated for each node 
    int node_length = problem_size / world_size; 

    // Update the amount of work for the last node to take into account excess;
    int remainder_length = problem_size % world_size;
    int curr_node_length = node_length + (rank == world_size - 1? remainder_length : 0); 
    
    // Number of cuda threads requires for current node
    int curr_node_required_threads = curr_node_length / args.tasks_per_thread + (curr_node_length % args.tasks_per_thread != 0);

    // Number of blocks needed for current node
    int curr_node_required_blocks = (curr_node_required_threads / 1024) + (curr_node_required_threads % 1024 != 0);

    int curr_node_offset = rank * node_length;

    PRINT("Rank %d/%d: Threads: %d Blocks: %d Tasks: %d Offset: %d\n", 
            rank, 
            world_size, 
            curr_node_required_threads, 
            curr_node_required_blocks, 
            curr_node_length,
            curr_node_offset
    );

    INT_TYPE *host_buffer, *device_buffer, *global_buffer;

    // Declare a buffer on the host which will be used to export data;
    if(!(host_buffer = (INT_TYPE*) calloc(curr_node_length, sizeof(INT_TYPE)))) {
        MPI_PRINT(rank, "Failed to allocate buffer on host\n");
        MPI_Abort(MPI_COMM_WORLD, 2); 
    }

    // Declare a buffer on the device which will be used for computation;
    cudaError_t cudaStatus = cudaMalloc((void**) &device_buffer, curr_node_length * sizeof(INT_TYPE));
    if(cudaStatus != cudaSuccess) {
        MPI_PRINT(rank, "Failed to allocate buffer on device: %s\n", cudaGetErrorString(cudaStatus));
        MPI_Abort(MPI_COMM_WORLD, 3);
    };

    // cudaMemset(device_buffer, 0, node_length * sizeof(INT_TYPE));

    
    mandelbrot<<<curr_node_required_blocks, 1024>>>(
        device_buffer, 
        curr_node_length, 
        curr_node_offset, 
        args.tasks_per_thread, 
        args.width, 
        args.height, 
        args.start_x, 
        args.start_y, 
        args.step
    );

    cudaDeviceSynchronize();

    cudaMemcpy(host_buffer, device_buffer, node_length * sizeof(INT_TYPE), cudaMemcpyDeviceToHost);

    

    if(!rank) {
        if(!(global_buffer = (INT_TYPE*) malloc(problem_size * sizeof(INT_TYPE)))) {
            MPI_PRINT(rank, "Failed to allocate global buffer on device");
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }

    int *recvcounts = (int*) malloc(world_size * sizeof(int));
    int *recvoffset = (int*) malloc(world_size * sizeof(int));
    int offset = 0;

    for(int i = 0; i < world_size; i++) {
        recvcounts[i] = node_length; 
        if(i == world_size - 1) {
            recvcounts[i] += remainder_length;
        }

        recvoffset[i] = offset;
        offset += node_length;

        MPI_PRINT(rank, "%d offset: %d count: %d\n", i, recvoffset[i], recvcounts[i]);
    }

    MPI_Gatherv(host_buffer, curr_node_length, MPI_UINT8_T, global_buffer, recvcounts, recvoffset, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    if(!rank) {

        PRINT("Writing to %s\n", args.file.c_str());

        std::ofstream file;
        file.open(args.file);

        file << "Dim: " << args.width << " " << args.height << "\n";

        for(int i = 0; i < problem_size; i++) {
            //printf("%d = %d\n", i, host_buffer[i]);
            file << +global_buffer[i] << "\n";
        }

        file.close();
    }

    MPI_Finalize();
}
