#include "args.h"
#include <cstdint>
#include <iostream>
#include <fstream>

#define INT_TYPE uint8_t
#define MAX_ITERAION 255

int main(int argc, char* argv[]) {

    Args args;
    if(!Args::parse(argc, argv, args)) {
        printf("Failed to parse args!\nExpected: %s --start <float> <float> --dim <int> <int> --step <float> --file <string> --tasks-per-thread <int>", argv[0]);
    }

    INT_TYPE* p;

    // Declare a buffer on the host which will be used to export data;
    if(!(p = (INT_TYPE*) malloc(args.width * args.height * sizeof(INT_TYPE)))) {
        printf("Failed to allocate buffer\n");
    }


    // Do it serially
    for(int img_y = 0; img_y < args.height; img_y++) {
        for(int img_x = 0; img_x < args.width; img_x++) {
            
            int cart_x = img_x;
            int cart_y = args.height - img_y - 1;

            float real = cart_x * args.step + args.start_x;
            float im = cart_y * args.step + args.start_y;
        
            float x = 0;
            float y = 0;


            INT_TYPE iteration = 0;

            while(iteration < MAX_ITERAION && x * x + y * y <= 4) {
                float temp = x*x - y*y + real;
                y = 2 * x * y + im;
                x = temp;
                iteration++;
            }

            p[img_x + img_y * args.width] = iteration;
        }
    }

    std::ofstream file;
    file.open(args.file);

    file << "Dim: " << args.width << " " << args.height << "\n";

    for(int i = 0; i < args.height * args.width; i++) {
        //printf("%d = %d\n", i, host_buffer[i]);
        file << +p[i] << "\n";
    }

    file.close();

}
