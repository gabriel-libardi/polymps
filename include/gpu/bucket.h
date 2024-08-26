#ifndef MPS_PROJECT_BUCKET_H_
#define MPS_PROJECT_BUCKET_H_

#include <cuda_runtime.h>

namespace mps {

/**
 * @brief CUDA kernel to initialize a double array on the device.
 *
 * @param[in] n      The size of the array.
 * @param[in, out] d_array  Pointer to the device array to be initialized.
 * @param[in] value  The value to initialize each element of the array with.
 */
__global__ void InitializeDoubleArray(int n, double* d_array, double value);

/**
 * @brief CUDA kernel to initialize an int array on the device.
 *
 * @param[in] n      The size of the array.
 * @param[in, out] d_array  Pointer to the device array to be initialized.
 * @param[in] value  The value to initialize each element of the array with.
 */
__global__ void InitializeIntArray(int n, int* d_array, int value);

/**
 * @brief CUDA kernel to create buckets for particles based on their positions.
 *
 * This kernel assigns each particle to a bucket based on its position in the simulation space.
 *
 * @param[in] d_nP       The total number of particles.
 * @param[in] d_nBx      The number of buckets along the x-axis.
 * @param[in] d_nBxy     The product of the number of buckets along the x and y axes.
 * @param[in] d_nBxyz    The total number of buckets in the 3D space.
 * @param[in] d_DBinv    The inverse of the bucket size.
 * @param[in, out] d_bfst  Pointer to the first particle index in each bucket.
 * @param[in, out] d_blst  Pointer to the last particle index in each bucket.
 * @param[in, out] d_nxt   Pointer to the next particle index in each bucket.
 * @param[in] d_Typ      Pointer to the particle types array.
 * @param[in] d_Pos      Pointer to the particle positions array.
 */
__global__ void MakeBucket(int d_nP, int d_nBx, int d_nBxy, int d_nBxyz, double d_DBinv,
                           int* d_bfst, int* d_blst, int* d_nxt,
                           int* d_Typ, double* d_Pos);

}  // namespace mps

#endif  // MPS_PROJECT_BUCKET_H_
