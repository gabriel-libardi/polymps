#ifndef MPS_PROJECT_PARAMETERS_H_
#define MPS_PROJECT_PARAMETERS_H_

namespace mps {

const int kNumType = 2;

/**
 * @brief Holds and manages the parameters used in the MPS simulation.
 */
class SimulationParameters {
public:
    double r;          ///< Interaction radius
    double r2;         ///< Interaction radius squared
    double db;         ///< Bucket size
    double db2;        ///< Bucket size squared
    double db_inv;     ///< Inverse of bucket size
    double Dns[kNumTyp];  ///< Densities for different types of particles
    double invDns[kNumTyp]; ///< Inverse densities for different types of particles
    double rlim;       ///< Collision distance limit
    double rlim2;      ///< Squared collision distance limit
    double col;        ///< Collision coefficient
    double n0;         ///< Reference number density
    double lambda;     ///< Lambda for computing viscosity term
    double A1, A2, A3; ///< Coefficients for viscosity, pressure, and pressure gradient terms

    int n_bx, n_by, n_bz;  ///< Number of buckets along each axis
    int n_bxy, n_bxyz;     ///< Total number of buckets in 2D and 3D space

    /**
     * @brief Initializes the simulation parameters.
     */
    void SetParameters();

    /**
     * @brief Allocates and sets up the bucket structure for the simulation.
     */
    void AllocateBucket();
};

}  // namespace mps

#endif  // MPS_PROJECT_PARAMETERS_H_
