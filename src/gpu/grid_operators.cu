#include "grid_operations.h"
#include "constants.h"

namespace mps {

__global__ void ComputeViscosityTerm(int d_nP, int d_nBx, int d_nBxy, int d_nBxyz, double d_DBinv,
                                     int* d_bfst, int* d_blst, int* d_nxt,
                                     int* d_Typ, double* d_Pos, double* d_Vel, double* d_Acc, double d_r, double d_A1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_nP) {
        if (d_Typ[i] == mps::kFLD) {
            double acc_x = 0.0;
            double acc_y = 0.0;
            double acc_z = 0.0;
            double pos_ix = d_Pos[i * 3];
            double pos_iy = d_Pos[i * 3 + 1];
            double pos_iz = d_Pos[i * 3 + 2];
            double vel_ix = d_Vel[i * 3];
            double vel_iy = d_Vel[i * 3 + 1];
            double vel_iz = d_Vel[i * 3 + 2];

            int ix = static_cast<int>((pos_ix - mps::kMinX) * d_DBinv) + 1;
            int iy = static_cast<int>((pos_iy - mps::kMinY) * d_DBinv) + 1;
            int iz = static_cast<int>((pos_iz - mps::kMinZ) * d_DBinv) + 1;

            for (int jz = iz - 1; jz <= iz + 1; jz++) {
                for (int jy = iy - 1; jy <= iy + 1; jy++) {
                    for (int jx = ix - 1; jx <= ix + 1; jx++) {
                        int jb = jz * d_nBxy + jy * d_nBx + jx;
                        int j = d_bfst[jb];
                        while (j != -1) {
                            if (j != i && d_Typ[j] != mps::kGST) {
                                double v0 = d_Pos[j * 3] - pos_ix;
                                double v1 = d_Pos[j * 3 + 1] - pos_iy;
                                double v2 = d_Pos[j * 3 + 2] - pos_iz;
                                double dst2 = v0 * v0 + v1 * v1 + v2 * v2;

                                if (dst2 < d_r * d_r) {
                                    double dst = sqrt(dst2);
                                    double w = (d_r / dst) - 1.0;
                                    acc_x += (d_Vel[j * 3] - vel_ix) * w;
                                    acc_y += (d_Vel[j * 3 + 1] - vel_iy) * w;
                                    acc_z += (d_Vel[j * 3 + 2] - vel_iz) * w;
                                }
                            }
                            j = d_nxt[j];
                        }
                    }
                }
            }
            d_Acc[i * 3] = acc_x * d_A1 + mps::kGx;
            d_Acc[i * 3 + 1] = acc_y * d_A1 + mps::kGy;
            d_Acc[i * 3 + 2] = acc_z * d_A1 + mps::kGz;
        }
    }
}

__global__ void UpdateParticles(int d_nP, int* d_Typ, double* d_Pos, double* d_Vel, double* d_Acc, double* d_Prs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_nP) {
        if (d_Typ[i] == mps::kFLD) {
            d_Vel[i * 3] += d_Acc[i * 3] * mps::kDt;
            d_Vel[i * 3 + 1] += d_Acc[i * 3 + 1] * mps::kDt;
            d_Vel[i * 3 + 2] += d_Acc[i * 3 + 2] * mps::kDt;

            d_Pos[i * 3] += d_Vel[i * 3] * mps::kDt;
            d_Pos[i * 3 + 1] += d_Vel[i * 3 + 1] * mps::kDt;
            d_Pos[i * 3 + 2] += d_Vel[i * 3 + 2] * mps::kDt;

            d_Acc[i * 3] = 0.0;
            d_Acc[i * 3 + 1] = 0.0;
            d_Acc[i * 3 + 2] = 0.0;

            // Check particle status (e.g., mark as ghost if out of bounds)
            if (d_Typ[i] != mps::kGST) {
                if (d_Pos[i * 3] > mps::kMaxX || d_Pos[i * 3] < mps::kMinX ||
                    d_Pos[i * 3 + 1] > mps::kMaxY || d_Pos[i * 3 + 1] < mps::kMinY ||
                    d_Pos[i * 3 + 2] > mps::kMaxZ || d_Pos[i * 3 + 2] < mps::kMinZ) {
                    d_Typ[i] = mps::kGST;
                    d_Prs[i] = d_Vel[i * 3] = d_Vel[i * 3 + 1] = d_Vel[i * 3 + 2] = 0.0;
                }
            }
        }
    }
}

__global__ void CheckCollision(int d_nP, int d_nBx, int d_nBxy, int d_nBxyz, double d_DBinv,
                               int* d_bfst, int* d_blst, int* d_nxt,
                               int* d_Typ, double* d_Pos, double* d_Vel, double* d_Acc, double* d_Dns, double d_rlim2, double d_COL) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_nP) {
        if (d_Typ[i] == mps::kFLD) {
            double mi = d_Dns[d_Typ[i]];
            double pos_ix = d_Pos[i * 3];
            double pos_iy = d_Pos[i * 3 + 1];
            double pos_iz = d_Pos[i * 3 + 2];
            double vel_ix = d_Vel[i * 3];
            double vel_iy = d_Vel[i * 3 + 1];
            double vel_iz = d_Vel[i * 3 + 2];

            int ix = static_cast<int>((pos_ix - mps::kMinX) * d_DBinv) + 1;
            int iy = static_cast<int>((pos_iy - mps::kMinY) * d_DBinv) + 1;
            int iz = static_cast<int>((pos_iz - mps::kMinZ) * d_DBinv) + 1;

            double vel_ix2 = vel_ix;
            double vel_iy2 = vel_iy;
            double vel_iz2 = vel_iz;

            for (int jz = iz - 1; jz <= iz + 1; jz++) {
                for (int jy = iy - 1; jy <= iy + 1; jy++) {
                    for (int jx = ix - 1; jx <= ix + 1; jx++) {
                        int jb = jz * d_nBxy + jy * d_nBx + jx;
                        int j = d_bfst[jb];
                        while (j != -1) {
                            if (j != i && d_Typ[j] != mps::kGST) {
                                double v0 = d_Pos[j * 3] - pos_ix;
                                double v1 = d_Pos[j * 3 + 1] - pos_iy;
                                double v2 = d_Pos[j * 3 + 2] - pos_iz;
                                double dst2 = v0 * v0 + v1 * v1 + v2 * v2;

                                if (dst2 < d_rlim2) {
                                    double fDT = (vel_ix - d_Vel[j * 3]) * v0 + (vel_iy - d_Vel[j * 3 + 1]) * v1 + (vel_iz - d_Vel[j * 3 + 2]) * v2;
                                    if (fDT > 0.0) {
                                        double mj = d_Dns[d_Typ[j]];
                                        fDT *= d_COL * mj / (mi + mj) / dst2;
                                        vel_ix2 -= v0 * fDT;
                                        vel_iy2 -= v1 * fDT;
                                        vel_iz2 -= v2 * fDT;
                                    }
                                }
                            }
                            j = d_nxt[j];
                        }
                    }
                }
            }
            d_Acc[i * 3] = vel_ix2;
            d_Acc[i * 3 + 1] = vel_iy2;
            d_Acc[i * 3 + 2] = vel_iz2;
        }
    }
}

__global__ void ComputePressure(int d_nP, int d_nBx, int d_nBxy, int d_nBxyz, double d_DBinv,
                                int* d_bfst, int* d_blst, int* d_nxt,
                                int* d_Typ, double* d_Pos, double* d_Prs, double* d_Dns, double d_r, double d_n0, double d_A2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_nP) {
        if (d_Typ[i] != mps::kGST) {
            double pos_ix = d_Pos[i * 3];
            double pos_iy = d_Pos[i * 3 + 1];
            double pos_iz = d_Pos[i * 3 + 2];
            double ni = 0.0;

            int ix = static_cast<int>((pos_ix - mps::kMinX) * d_DBinv) + 1;
            int iy = static_cast<int>((pos_iy - mps::kMinY) * d_DBinv) + 1;
            int iz = static_cast<int>((pos_iz - mps::kMinZ) * d_DBinv) + 1;

            for (int jz = iz - 1; jz <= iz + 1; jz++) {
                for (int jy = iy - 1; jy <= iy + 1; jy++) {
                    for (int jx = ix - 1; jx <= ix + 1; jx++) {
                        int jb = jz * d_nBxy + jy * d_nBx + jx;
                        int j = d_bfst[jb];
                        while (j != -1) {
                            if (j != i && d_Typ[j] != mps::kGST) {
                                double v0 = d_Pos[j * 3] - pos_ix;
                                double v1 = d_Pos[j * 3 + 1] - pos_iy;
                                double v2 = d_Pos[j * 3 + 2] - pos_iz;
                                double dst2 = v0 * v0 + v1 * v1 + v2 * v2;

                                if (dst2 < d_r * d_r) {
                                    double dst = sqrt(dst2);
                                    double w = (d_r / dst) - 1.0;
                                    ni += w;
                                }
                            }
                            j = d_nxt[j];
                        }
                    }
                }
            }
            double mi = d_Dns[d_Typ[i]];
            d_Prs[i] = (ni > d_n0) ? (ni - d_n0) * d_A2 * mi : 0.0;
        }
    }
}

__global__ void ComputePressureGradient(int d_nP, int d_nBx, int d_nBxy, int d_nBxyz, double d_DBinv,
                                        int* d_bfst, int* d_blst, int* d_nxt,
                                        int* d_Typ, double* d_Pos, double* d_Acc, double* d_Prs, double* d_invDns, double d_r, double d_A3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_nP) {
        if (d_Typ[i] == mps::kFLD) {
            double acc_x = 0.0;
            double acc_y = 0.0;
            double acc_z = 0.0;
            double pos_ix = d_Pos[i * 3];
            double pos_iy = d_Pos[i * 3 + 1];
            double pos_iz = d_Pos[i * 3 + 2];

            int ix = static_cast<int>((pos_ix - mps::kMinX) * d_DBinv) + 1;
            int iy = static_cast<int>((pos_iy - mps::kMinY) * d_DBinv) + 1;
            int iz = static_cast<int>((pos_iz - mps::kMinZ) * d_DBinv) + 1;

            double pre_min = d_Prs[i];

            for (int jz = iz - 1; jz <= iz + 1; jz++) {
                for (int jy = iy - 1; jy <= iy + 1; jy++) {
                    for (int jx = ix - 1; jx <= ix + 1; jx++) {
                        int jb = jz * d_nBxy + jy * d_nBx + jx;
                        int j = d_bfst[jb];
                        while (j != -1) {
                            if (j != i && d_Typ[j] != mps::kGST) {
                                double v0 = d_Pos[j * 3] - pos_ix;
                                double v1 = d_Pos[j * 3 + 1] - pos_iy;
                                double v2 = d_Pos[j * 3 + 2] - pos_iz;
                                double dst2 = v0 * v0 + v1 * v1 + v2 * v2;

                                if (dst2 < d_r * d_r) {
                                    if (pre_min > d_Prs[j]) pre_min = d_Prs[j];
                                }
                            }
                            j = d_nxt[j];
                        }
                    }
                }
            }

            for (int jz = iz - 1; jz <= iz + 1; jz++) {
                for (int jy = iy - 1; jy <= iy + 1; jy++) {
                    for (int jx = ix - 1; jx <= ix + 1; jx++) {
                        int jb = jz * d_nBxy + jy * d_nBx + jx;
                        int j = d_bfst[jb];
                        while (j != -1) {
                            if (j != i && d_Typ[j] != mps::kGST) {
                                double v0 = d_Pos[j * 3] - pos_ix;
                                double v1 = d_Pos[j * 3 + 1] - pos_iy;
                                double v2 = d_Pos[j * 3 + 2] - pos_iz;
                                double dst2 = v0 * v0 + v1 * v1 + v2 * v2;

                                if (dst2 < d_r * d_r) {
                                    double dst = sqrt(dst2);
                                    double w = (d_r / dst) - 1.0;
                                    w *= (d_Prs[j] - pre_min) / dst2;
                                    acc_x += v0 * w;
                                    acc_y += v1 * w;
                                    acc_z += v2 * w;
                                }
                            }
                            j = d_nxt[j];
                        }
                    }
                }
            }
            d_Acc[i * 3] = acc_x * d_invDns[mps::kFLD] * d_A3;
            d_Acc[i * 3 + 1] = acc_y * d_invDns[mps::kFLD] * d_A3;
            d_Acc[i * 3 + 2] = acc_z * d_invDns[mps::kFLD] * d_A3;
        }
    }
}

__global__ void UpdateParticles2(int d_nP, int* d_Typ, double* d_Pos, double* d_Vel, double* d_Acc, double* d_Prs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_nP) {
        if (d_Typ[i] == mps::kFLD) {
            d_Vel[i * 3] += d_Acc[i * 3] * mps::kDt;
            d_Vel[i * 3 + 1] += d_Acc[i * 3 + 1] * mps::kDt;
            d_Vel[i * 3 + 2] += d_Acc[i * 3 + 2] * mps::kDt;

            d_Pos[i * 3] += d_Acc[i * 3] * mps::kDt * mps::kDt;
            d_Pos[i * 3 + 1] += d_Acc[i * 3 + 1] * mps::kDt * mps::kDt;
            d_Pos[i * 3 + 2] += d_Acc[i * 3 + 2] * mps::kDt * mps::kDt;

            d_Acc[i * 3] = 0.0;
            d_Acc[i * 3 + 1] = 0.0;
            d_Acc[i * 3 + 2] = 0.0;

            // Check particle status (e.g., mark as ghost if out of bounds)
            if (d_Typ[i] != mps::kGST) {
                if (d_Pos[i * 3] > mps::kMaxX || d_Pos[i * 3] < mps::kMinX ||
                    d_Pos[i * 3 + 1] > mps::kMaxY || d_Pos[i * 3 + 1] < mps::kMinY ||
                    d_Pos[i * 3 + 2] > mps::kMaxZ || d_Pos[i * 3 + 2] < mps::kMinZ) {
                    d_Typ[i] = mps::kGST;
                    d_Prs[i] = d_Vel[i * 3] = d_Vel[i * 3 + 1] = d_Vel[i * 3 + 2] = 0.0;
                }
            }
        }
    }
}

__global__ void AddPressureArray(int d_nP, double* d_pav, double* d_Prs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_nP) {
        d_pav[i] += d_Prs[i];
    }
}

}  // namespace mps
