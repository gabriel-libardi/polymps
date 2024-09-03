#ifndef MPS_PROJECT_GRID_OPERATIONS_H_
#define MPS_PROJECT_GRID_OPERATIONS_H_

#include <cuda_runtime.h>

namespace mps {

/**
 * @brief CUDA kernel to compute the viscosity term for each particle.
 *
 * @param[in] d_nP      Number of particles.
 * @param[in] d_nBx     Number of buckets along the x-axis.
 * @param[in] d_nBxy    Product of the number of buckets along the x and y axes.
 * @param[in] d_nBxyz   Total number of buckets in the 3D space.
 * @param[in] d_DBinv   Inverse of the bucket size.
 * @param[in, out] d_bfst  Pointer to the first particle index in each bucket.
 * @param[in, out] d_blst  Pointer to the last particle index in each bucket.
 * @param[in, out] d_nxt   Pointer to the next particle index in each bucket.
 * @param[in] d_Typ     Pointer to the particle types array.
 * @param[in] d_Pos     Pointer to the particle positions array.
 * @param[in] d_Vel     Pointer to the particle velocities array.
 * @param[in, out] d_Acc  Pointer to the particle accelerations array.
 * @param[in] d_r       Interaction radius.
 * @param[in] d_A1      Viscosity coefficient.
 */
__global__ void ComputeViscosityTerm(int d_nP, int d_nBx, int d_nBxy, int d_nBxyz, double d_DBinv,
                                     int* d_bfst, int* d_blst, int* d_nxt,
                                     int* d_Typ, double* d_Pos, double* d_Vel, double* d_Acc, double d_r, double d_A1);

/**
 * @brief CUDA kernel to update the position and velocity of each particle.
 *
 * @param[in] d_nP      Number of particles.
 * @param[in, out] d_Typ  Pointer to the particle types array.
 * @param[in, out] d_Pos  Pointer to the particle positions array.
 * @param[in, out] d_Vel  Pointer to the particle velocities array.
 * @param[in, out] d_Acc  Pointer to the particle accelerations array.
 * @param[in, out] d_Prs  Pointer to the particle pressures array.
 */
__global__ void UpdateParticles(int d_nP, int* d_Typ, double* d_Pos, double* d_Vel, double* d_Acc, double* d_Prs);

/**
 * @brief CUDA kernel to check for particle collisions and adjust velocities.
 *
 * @param[in] d_nP      Number of particles.
 * @param[in] d_nBx     Number of buckets along the x-axis.
 * @param[in] d_nBxy    Product of the number of buckets along the x and y axes.
 * @param[in] d_nBxyz   Total number of buckets in the 3D space.
 * @param[in] d_DBinv   Inverse of the bucket size.
 * @param[in, out] d_bfst  Pointer to the first particle index in each bucket.
 * @param[in, out] d_blst  Pointer to the last particle index in each bucket.
 * @param[in, out] d_nxt   Pointer to the next particle index in each bucket.
 * @param[in] d_Typ     Pointer to the particle types array.
 * @param[in] d_Pos     Pointer to the particle positions array.
 * @param[in, out] d_Vel  Pointer to the particle velocities array.
 * @param[in, out] d_Acc  Pointer to the particle accelerations array.
 * @param[in] d_Dns     Pointer to the particle densities array.
 * @param[in] d_rlim2   Squared collision distance limit.
 * @param[in] d_COL     Collision coefficient.
 */
__global__ void CheckCollision(int d_nP, int d_nBx, int d_nBxy, int d_nBxyz, double d_DBinv,
                               int* d_bfst, int* d_blst, int* d_nxt,
                               int* d_Typ, double* d_Pos, double* d_Vel, double* d_Acc, double* d_Dns, double d_rlim2, double d_COL);

/**
 * @brief CUDA kernel to compute the pressure of each particle.
 *
 * @param[in] d_nP      Number of particles.
 * @param[in] d_nBx     Number of buckets along the x-axis.
 * @param[in] d_nBxy    Product of the number of buckets along the x and y axes.
 * @param[in] d_nBxyz   Total number of buckets in the 3D space.
 * @param[in] d_DBinv   Inverse of the bucket size.
 * @param[in, out] d_bfst  Pointer to the first particle index in each bucket.
 * @param[in, out] d_blst  Pointer to the last particle index in each bucket.
 * @param[in, out] d_nxt   Pointer to the next particle index in each bucket.
 * @param[in] d_Typ     Pointer to the particle types array.
 * @param[in] d_Pos     Pointer to the particle positions array.
 * @param[in, out] d_Prs  Pointer to the particle pressures array.
 * @param[in] d_Dns     Pointer to the particle densities array.
 * @param[in] d_r       Interaction radius.
 * @param[in] d_n0      Reference number density.
 * @param[in] d_A2      Pressure coefficient.
 */
__global__ void ComputePressure(int d_nP, int d_nBx, int d_nBxy, int d_nBxyz, double d_DBinv,
                                int* d_bfst, int* d_blst, int* d_nxt,
                                int* d_Typ, double* d_Pos, double* d_Prs, double* d_Dns, double d_r, double d_n0, double d_A2);

/**
 * @brief CUDA kernel to compute the pressure gradient for each particle.
 *
 * @param[in] d_nP      Number of particles.
 * @param[in] d_nBx     Number of buckets along the x-axis.
 * @param[in] d_nBxy    Product of the number of buckets along the x and y axes.
 * @param[in] d_nBxyz   Total number of buckets in the 3D space.
 * @param[in] d_DBinv   Inverse of the bucket size.
 * @param[in, out] d_bfst  Pointer to the first particle index in each bucket.
 * @param[in, out] d_blst  Pointer to the last particle index in each bucket.
 * @param[in, out] d_nxt   Pointer to the next particle index in each bucket.
 * @param[in] d_Typ     Pointer to the particle types array.
 * @param[in] d_Pos     Pointer to the particle positions array.
 * @param[in, out] d_Acc  Pointer to the particle accelerations array.
 * @param[in] d_Prs     Pointer to the particle pressures array.
 * @param[in] d_invDns  Pointer to the inverse of the particle densities array.
 * @param[in] d_r       Interaction radius.
 * @param[in] d_A3      Pressure gradient coefficient.
 */
__global__ void ComputePressureGradient(int d_nP, int d_nBx, int d_nBxy, int d_nBxyz, double d_DBinv,
                                        int* d_bfst, int* d_blst, int* d_nxt,
                                        int* d_Typ, double* d_Pos, double* d_Acc, double* d_Prs, double* d_invDns, double d_r, double d_A3);

/**
 * @brief CUDA kernel to perform the second update of particle positions and velocities.
 *
 * @param[in] d_nP      Number of particles.
 * @param[in, out] d_Typ  Pointer to the particle types array.
 * @param[in, out] d_Pos  Pointer to the particle positions array.
 * @param[in, out] d_Vel  Pointer to the particle velocities array.
 * @param[in, out] d_Acc  Pointer to the particle accelerations array.
 * @param[in, out] d_Prs  Pointer to the particle pressures array.
 */
__global__ void UpdateParticles2(int d_nP, int* d_Typ, double* d_Pos, double* d_Vel, double* d_Acc, double* d_Prs);

/**
 * @brief CUDA kernel to accumulate pressure values for each particle.
 *
 * @param[in] d_nP      Number of particles.
 * @param[in, out] d_pav  Pointer to the accumulated pressure array.
 * @param[in] d_Prs     Pointer to the particle pressures array.
 */
__global__ void AddPressureArray(int d_nP, double* d_pav, double* d_Prs);

}  // namespace mps

#endif  // MPS_PROJECT_GRID_OPERATIONS_H_
