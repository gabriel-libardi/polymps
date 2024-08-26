#include "data_io.h"
#include "parameters.h"
#include "cuda_helpers.h"
#include "timing.h"
#include "bucket.h"
#include "grid_operations.h"


int main(int argc, char** argv) {
    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(0));

    // Read data
    mps::ReadData();

    // Allocate and set simulation parameters
    mps::SimulationParameters params;
    params.SetParameters();
    params.AllocateBucket();

    // CUDA memory allocations
    double *d_acc;
    double *d_pos;
    double *d_vel;
    double *d_prs;
    double *d_pav;
    int *d_typ;
    int *d_bfst;
    int *d_blst;
    int *d_nxt;

    CUDA_CHECK(cudaMalloc((void**)&d_typ, sizeof(int) * nP));
    CUDA_CHECK(cudaMalloc((void**)&d_acc, sizeof(double) * nP * 3));
    CUDA_CHECK(cudaMalloc((void**)&d_pos, sizeof(double) * nP * 3));
    CUDA_CHECK(cudaMalloc((void**)&d_vel, sizeof(double) * nP * 3));
    CUDA_CHECK(cudaMalloc((void**)&d_prs, sizeof(double) * nP));
    CUDA_CHECK(cudaMalloc((void**)&d_pav, sizeof(double) * nP));
    CUDA_CHECK(cudaMalloc((void**)&d_bfst, sizeof(int) * params.n_bxyz));
    CUDA_CHECK(cudaMalloc((void**)&d_blst, sizeof(int) * params.n_bxyz));
    CUDA_CHECK(cudaMalloc((void**)&d_nxt, sizeof(int) * nP));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_typ, typ, sizeof(int) * nP, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_acc, acc, sizeof(double) * nP * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, pos, sizeof(double) * nP * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, vel, sizeof(double) * nP * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prs, prs, sizeof(double) * nP, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pav, pav, sizeof(double) * nP, cudaMemcpyHostToDevice));

    // Start timing
    double start_time;
    start_time = mps::GetTime();

    int total_threads;
    int blocks;

    while (TIM < params.kFinTim) {
        if (iLP % params.kOptFqc == 0) {
            CUDA_CHECK(cudaMemcpy(typ, d_typ, sizeof(int) * nP, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pos, d_pos, sizeof(double) * nP * 3, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vel, d_vel, sizeof(double) * nP * 3, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(prs, d_prs, sizeof(double) * nP, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pav, d_pav, sizeof(double) * nP, cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemset(d_pav, 0, sizeof(double) * nP));

            mps::WriteData();

            int p_num = 0;
            for (int i = 0; i < nP; i++) {
                if (typ[i] != mps::kGST) {
                    p_num++;
                }
            }
            printf("%5d th TIM: %lf / p_num: %d\n", iLP, TIM, p_num);

            if (TIM >= params.kFinTim) {
                break;
            }
        }

        total_threads = params.n_bxyz;
        blocks = total_threads / THREADS + 1;
        CUDA_CHECK(cudaMemset(d_bfst, -1, sizeof(int) * params.n_bxyz));
        CUDA_CHECK(cudaMemset(d_blst, -1, sizeof(int) * params.n_bxyz));
        CUDA_CHECK(cudaMemset(d_nxt, -1, sizeof(int) * nP));
        mps::MakeBucket<<<blocks, THREADS>>>(nP, params.n_bx, params.n_bxy, params.n_bxyz, params.db_inv, d_bfst, d_blst, d_nxt, d_typ, d_pos);
        CUDA_CHECK(cudaDeviceSynchronize());

        total_threads = nP;
        blocks = total_threads / THREADS + 1;
        mps::ComputeViscosityTerm<<<blocks, THREADS>>>(nP, params.n_bx, params.n_bxy, params.n_bxyz, params.db_inv, d_bfst, d_blst, d_nxt, d_typ, d_pos, d_vel, d_acc, params.r, params.a1);
        CUDA_CHECK(cudaDeviceSynchronize());

        mps::UpdateParticles<<<blocks, THREADS>>>(nP, d_typ, d_pos, d_vel, d_acc, d_prs);
        CUDA_CHECK(cudaDeviceSynchronize());

        mps::CheckCollision<<<blocks, THREADS>>>(nP, params.n_bx, params.n_bxy, params.n_bxyz, params.db_inv, d_bfst, d_blst, d_nxt, d_typ, d_pos, d_vel, d_acc, params.dns, params.rlim2, params.col);
        CUDA_CHECK(cudaMemcpy(d_vel, d_acc, sizeof(double) * nP * 3, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        mps::ComputePressure<<<blocks, THREADS>>>(nP, params.n_bx, params.n_bxy, params.n_bxyz, params.db_inv, d_bfst, d_blst, d_nxt, d_typ, d_pos, d_prs, params.dns, params.r, params.n0, params.a2);
        CUDA_CHECK(cudaDeviceSynchronize());

        mps::ComputePressureGradient<<<blocks, THREADS>>>(nP, params.n_bx, params.n_bxy, params.n_bxyz, params.db_inv, d_bfst, d_blst, d_nxt, d_typ, d_pos, d_acc, d_prs, params.inv_dns, params.r, params.a3);
        CUDA_CHECK(cudaDeviceSynchronize());

        mps::UpdateParticles2<<<blocks, THREADS>>>(nP, d_typ, d_pos, d_vel, d_acc, d_prs);
        CUDA_CHECK(cudaDeviceSynchronize());

        mps::ComputePressure<<<blocks, THREADS>>>(nP, params.n_bx, params.n_bxy, params.n_bxyz, params.db_inv, d_bfst, d_blst, d_nxt, d_typ, d_pos, d_prs, params.dns, params.r, params.n0, params.a2);
        CUDA_CHECK(cudaDeviceSynchronize());

        mps::AddPressureArray<<<blocks, THREADS>>>(nP, d_pav, d_prs);
        CUDA_CHECK(cudaDeviceSynchronize());

        iLP++;
        TIM += params.kDt;
    }

    // End timing
    double end_time;
    end_time = mps::GetTime();
    printf("Total: %13.6lf sec\n", end_time - start_time);

    // Free memory
    CUDA_CHECK(cudaFree(d_typ));
    CUDA_CHECK(cudaFree(d_acc));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_prs));
    CUDA_CHECK(cudaFree(d_pav));
    CUDA_CHECK(cudaFree(d_bfst));
    CUDA_CHECK(cudaFree(d_blst));
    CUDA_CHECK(cudaFree(d_nxt));

    return 0;
}

