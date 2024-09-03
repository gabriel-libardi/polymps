#include "parameters.h"
#include "constants.h"
#include <cstdio>

namespace mps {

void SimulationParameters::SetParameters() {
    // Set initial values for simulation parameters
    r = mps::kPclDst * 2.1;
    r2 = r * r;
    db = r * (1.0 + mps::kCrtNum);
    db2 = db * db;
    db_inv = 1.0 / db;

    Dns[mps::kFLD] = mps::kDnsFld;
    Dns[mps::kWLL] = mps::kDnsWll;
    invDns[mps::kFLD] = 1.0 / mps::kDnsFld;
    invDns[mps::kWLL] = 1.0 / mps::kDnsWll;

    rlim = mps::kPclDst * mps::kDstLmtRat;
    rlim2 = rlim * rlim;
    col = 1.0 + mps::kColRat;

    n0 = 0.0;
    lambda = 0.0;

    // Calculate n0 and lambda
    for (int ix = -4; ix < 5; ix++) {
        for (int iy = -4; iy < 5; iy++) {
            for (int iz = -4; iz < 5; iz++) {
                double x = mps::kPclDst * static_cast<double>(ix);
                double y = mps::kPclDst * static_cast<double>(iy);
                double z = mps::kPclDst * static_cast<double>(iz);
                double dst2 = x * x + y * y + z * z;
                if (dst2 <= r2) {
                    if (dst2 == 0.0) continue;
                    double dst = sqrt(dst2);
                    n0 += (r / dst) - 1.0;
                    lambda += dst2 * ((r / dst) - 1.0);
                }
            }
        }
    }

    lambda = lambda / n0;
    A1 = 2.0 * mps::kKnmVsc * mps::kDim / n0 / lambda;
    A2 = mps::kSnd * mps::kSnd / n0;
    A3 = -mps::kDim / n0;
}

void SimulationParameters::AllocateBucket() {
    // Calculate the number of buckets along each axis
    n_bx = static_cast<int>((mps::kMaxX - mps::kMinX) * db_inv) + 3;
    n_by = static_cast<int>((mps::kMaxY - mps::kMinY) * db_inv) + 3;
    n_bz = static_cast<int>((mps::kMaxZ - mps::kMinZ) * db_inv) + 3;

    n_bxy = n_bx * n_by;
    n_bxyz = n_bx * n_by * n_bz;

    printf("nBx: %d  nBy: %d  nBz: %d  nBxy: %d  nBxyz: %d\n", n_bx, n_by, n_bz, n_bxy, n_bxyz);
}

}  // namespace mps
