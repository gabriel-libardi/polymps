#include "timing.h"
#include <sys/time.h>

namespace mps {

double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) * 0.000001;
}

}  // namespace mps
