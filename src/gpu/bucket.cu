#include "bucket.h"
#include "constants.h"

namespace mps {

__global__ void InitializeDoubleArray(int n, double* d_array, double value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_array[i] = value;
    }
}

__global__ void InitializeIntArray(int n, int* d_array, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_array[i] = value;
    }
}

__global__ void MakeBucket(int d_nP, int d_nBx, int d_nBxy, int d_nBxyz, double d_DBinv,
                           int* d_bfst, int* d_blst, int* d_nxt,
                           int* d_Typ, double* d_Pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_nP) {
        if (d_Typ[i] != mps::kGST) {
            int ix = static_cast<int>((d_Pos[i * 3] - mps::kMinX) * d_DBinv) + 1;
            int iy = static_cast<int>((d_Pos[i * 3 + 1] - mps::kMinY) * d_DBinv) + 1;
            int iz = static_cast<int>((d_Pos[i * 3 + 2] - mps::kMinZ) * d_DBinv) + 1;

            int ib = iz * d_nBxy + iy * d_nBx + ix;
            int j = atomicExch(&d_blst[ib], i);
            if (j == -1) {
                d_bfst[ib] = i;
            } else {
                d_nxt[j] = i;
            }
        }
    }
}

}  // namespace mps
