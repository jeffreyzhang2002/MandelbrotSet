#ifndef ARGS_H
#define ARGS_H

#include <string>

class Args {
public:
    float start_x;        //Starting offset x
    float start_y;        //Starting offset y
    int width;            // width in pixels of the image 
    int height;           // height in pixels of the image
    float step;           // size of each pixel
    int max_iteration;    // Max iteration 
    int tasks_per_thread; // Number of tasks per thread 
    std::string file;     // File Path

    Args();

    static bool parse(int argc, char* argv[], Args& args);

};

#endif
