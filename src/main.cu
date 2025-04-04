#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "args.h"
#include <cstdint>

#ifdef DEBUG
    #define PRINT(...) printf(__VA_ARGS__)
#else
    #define PRINT(...)
#endif

#define MAX_ITERAION 255
#define INT_TYPE uint8_t

__global__
void mandelbrot(INT_TYPE *p, int offset, int length, int img_width, int img_height, float start_x, float start_y, float step) {
   
    // Thread Index 
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
   
    // if(threadIdx.x == 0) {
    //     printf("Block: %d index: %d\n", blockIdx.x, idx);
    // }

    // Mask out threads that are out of range
    if(idx >= length) {
        return;
    }

    // Pixel Index
    int pixel_idx = idx + offset;

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

    p[idx] = iteration;
}

int main(int argc, char* argv[]) {

    Args args;

    if(!Args::parse(argc, argv, args)) return 1;
    
    fprintf(stderr, "Calculating Mandelbrot Set!\nstart_x: %f\nstart_y: %f\nend_x: %f\nend_y: %f\ndelta: %f\nwidth: %d\nheight: %d\n", 
            args.start_x, 
            args.start_y, 
            args.start_x + args.width * args.step, 
            args.start_y + args.height * args.step,
            args.step,
            args.width,
            args.height
    );
    
    INT_TYPE *host_buffer, *device_buffer;

    int length = args.width * args.height;

    // Declare a buffer on the host which will be used to export data;
    if(!(host_buffer = (INT_TYPE*) calloc(length, sizeof(INT_TYPE)))) {
        fprintf(stderr, "Failed to allocate buffer on host\n");
        return 1;
    }

    // Declare a buffer on the device which will be used for computation;
    cudaError_t cudaStatus = cudaMalloc((void**) &device_buffer, length * sizeof(INT_TYPE));
    if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate buffer on device: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    };

    cudaMemset(device_buffer, 0, length);

    int numblocks = (length / 1024) + (length % 1024 == 0? 0 : 1);

    fprintf(stderr, "Using %d Block with 1024 threads per block\n", numblocks);

    mandelbrot<<<numblocks,1024>>>(device_buffer, 0, length, args.width, args.height, args.start_x, args.start_y, args.step);

    cudaDeviceSynchronize();

    cudaMemcpy(host_buffer, device_buffer, length * sizeof(INT_TYPE), cudaMemcpyDeviceToHost);

    std::cerr << "Writing to " << args.file << "\n";

    std::ofstream file;
    file.open(args.file);

    file << "Dim: " << args.width << " " << args.height << "\n";

    for(int i = 0; i < length; i++) {
        //printf("%d = %d\n", i, host_buffer[i]);
        file << +host_buffer[i] << "\n";
    }

    file.close();
    return 0;
}
