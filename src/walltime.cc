#include <stddef.h>
#include <sys/time.h>
#include "walltime.h"

void get_walltime(double* wcTime) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  *wcTime = (double)(tp.tv_sec + tp.tv_usec/1000000.0);
}

