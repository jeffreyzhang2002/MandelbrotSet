#include "args.h"
#include <cstring>

bool check_float(char* str, float& set) {
    char *endptr;

    set = strtod(str, &endptr);
            
    if(str == endptr) {
        return false;
    }
    return true;
}

Args::Args() {
    this->start_x = 0;
    this->start_y = 0;
    this->width = 0;
    this->height = 0;
    this->step = 0;
    this->tasks_per_thread = 0;
    this->file = std::string("");
}

bool Args::parse(int argc, char* argv[], Args& args) {
    if(argc <= 3) {
        return false;
    }
    
    int i = 1;
    char *endptr;

    while(i < argc) {
        if(std::strcmp(argv[i], "--start") == 0) {
            if(i + 2 >= argc) {
                return false;
            }

            if(!check_float(argv[i+1], args.start_x) || !check_float(argv[i+2], args.start_y)) {
                return false;
            }
            i += 2;
        } else if (std::strcmp(argv[i], "--dim") == 0) {
            if(i + 2 >= argc) {
                return false;
            }

            args.width = atoi(argv[i+1]);
            args.height = atoi(argv[i+2]);

            i += 2;
        } else if (std::strcmp(argv[i], "--step") == 0) {
            if(i + 1 > argc) {
                return false;
            }
            
            if(!check_float(argv[i+1], args.step)) {
                return false;
            }
            i += 1;
        } else if (std::strcmp(argv[i], "--file") == 0) {
       
            args.file = std::string(argv[i + 1]);

            i += 1;
        } else if (std::strcmp(argv[i], "--tasks-per-thread") == 0) {
            args.tasks_per_thread = atoi(argv[i+1]);

            i += 1;
        } else {
            return false;
        }

        i += 1;
    }

    if(args.step <= 0 || args.width <= 0 || args.height <= 0) {
        return false;
    }

    if(args.tasks_per_thread < 1) {
        return false;
    }

    if(args.file == "") {
        return false;
    }

    return true;
};

