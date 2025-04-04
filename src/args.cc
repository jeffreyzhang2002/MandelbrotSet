#include "args.h"
#include <cstring>

bool check_float(char* str, float& set) {
    char *endptr;

    set = strtod(str, &endptr);
            
    if(str == endptr) {
        printf("%s is not a float\n", str);
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
    this->file = std::string("");
}

bool Args::parse(int argc, char* argv[], Args& args) {
    if(argc <= 3) {
        printf("Usage: %s --start <float> <float> --dim <float> <float> --step <int> --file <path>\n", argv[0]);
        return 0;
    }
    
    int i = 1;
    char *endptr;

    while(i < argc) {
        if(std::strcmp(argv[i], "--start") == 0) {
            if(i + 2 >= argc) {
                printf("--start flag is missing two arguments\n");
                return false;
            }

            if(!check_float(argv[i+1], args.start_x) || !check_float(argv[i+2], args.start_y)) {
                printf("Failed to parse float for --start argument\n");
                return false;
            }
            i += 2;
        } else if (std::strcmp(argv[i], "--dim") == 0) {
            if(i + 2 >= argc) {
                printf("--dim flag is missing two arguments\n");
                return false;
            }

            args.width = atoi(argv[i+1]);
            args.height = atoi(argv[i+2]);

            i += 2;
        } else if (std::strcmp(argv[i], "--step") == 0) {
            if(i + 1 > argc) {
                printf("--step flag is missing argument\n");
                return false;
            }
            
            if(!check_float(argv[i+1], args.step)) {
                printf("Failed to parse float for --step argument\n");
                return false;
            }
            i += 1;
        } else if (std::strcmp(argv[i], "--file") == 0) {
       
            args.file = std::string(argv[i + 1]);

            i += 1;
        } else {
            printf("%s is not a valid flag\n", argv[i]);
            return false;
        }

        i += 1;
    }

    if(args.step <= 0 || args.width <= 0 || args.height <= 0) {
        printf("step and dimensions must be positive values\n");
        return false;
    }

    if(args.file == "") {
        printf("path must not be empty");
        return false;
    }

    return true;
};

