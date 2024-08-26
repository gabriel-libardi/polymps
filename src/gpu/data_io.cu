#include "data_io.h"
#include "constants.h"
#include <cstdio>
#include <cstdlib>

namespace mps {

// Global variables for particle data
extern double *Acc, *Pos, *Vel, *Prs, *pav;
extern int *Typ;
extern int nP;
extern FILE* fp;
extern int iF;

/**
 * @brief Reads the particle data from the input file.
 */
void ReadData() {
    fp = fopen(mps::kInFile, "r");
    if (fp == nullptr) {
        fprintf(stderr, "Error opening input file: %s\n", mps::kInFile);
        exit(EXIT_FAILURE);
    }

    fscanf(fp, "%d", &nP);
    printf("Number of particles: %d\n", nP);

    Acc = (double*)malloc(sizeof(double) * nP * 3);
    Pos = (double*)malloc(sizeof(double) * nP * 3);
    Vel = (double*)malloc(sizeof(double) * nP * 3);
    Prs = (double*)malloc(sizeof(double) * nP);
    pav = (double*)malloc(sizeof(double) * nP);
    Typ = (int*)malloc(sizeof(int) * nP);

    for (int i = 0; i < nP; i++) {
        int a[2];
        double b[8];
        fscanf(fp, "%d %d %lf %lf %lf %lf %lf %lf %lf %lf",
               &a[0], &a[1], &b[0], &b[1], &b[2], &b[3], &b[4], &b[5], &b[6], &b[7]);
        Typ[i] = a[1];
        Pos[i * 3] = b[0];
        Pos[i * 3 + 1] = b[1];
        Pos[i * 3 + 2] = b[2];
        Vel[i * 3] = b[3];
        Vel[i * 3 + 1] = b[4];
        Vel[i * 3 + 2] = b[5];
        Prs[i] = b[6];
        pav[i] = b[7];
    }

    fclose(fp);

    for (int i = 0; i < nP; i++) {
        if (Typ[i] != mps::kGST) {
            if (Pos[i * 3] > mps::kMaxX || Pos[i * 3] < mps::kMinX ||
                Pos[i * 3 + 1] > mps::kMaxY || Pos[i * 3 + 1] < mps::kMinY ||
                Pos[i * 3 + 2] > mps::kMaxZ || Pos[i * 3 + 2] < mps::kMinZ) {
                Typ[i] = mps::kGST;
                Prs[i] = Vel[i * 3] = Vel[i * 3 + 1] = Vel[i * 3 + 2] = 0.0;
            }
        }
    }

    for (int i = 0; i < nP * 3; i++) {
        Acc[i] = 0.0;
    }
}

/**
 * @brief Writes the current state of the particle data to an output file.
 */
void WriteData() {
    char output_filename[256];
    sprintf(output_filename, "output%05d.prof", iF);
    fp = fopen(output_filename, "w");
    if (fp == nullptr) {
        fprintf(stderr, "Error opening output file: %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "%d\n", nP);
    for (int i = 0; i < nP; i++) {
        int a[2] = { i, Typ[i] };
        double b[8] = {
            Pos[i * 3], Pos[i * 3 + 1], Pos[i * 3 + 2],
            Vel[i * 3], Vel[i * 3 + 1], Vel[i * 3 + 2],
            Prs[i], pav[i] / mps::kOptFqc
        };
        fprintf(fp, "%d %d %lf %lf %lf %lf %lf %lf %lf %lf\n",
                a[0], a[1], b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
        pav[i] = 0.0;
    }

    fclose(fp);
    iF++;
}

}  // namespace mps
