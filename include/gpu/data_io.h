#ifndef MPS_PROJECT_DATA_IO_H_
#define MPS_PROJECT_DATA_IO_H_

#include <string>

namespace mps {

/**
 * @brief Reads the particle data from the input file.
 *
 * This function reads the initial particle positions, velocities, pressures, and types from a specified input file.
 * It allocates the necessary memory and initializes the particle data arrays.
 */
void ReadData();

/**
 * @brief Writes the current state of the particle data to an output file.
 *
 * This function writes the positions, velocities, pressures, and other relevant data of the particles to an output file.
 * It is typically called at regular intervals during the simulation to save the state for post-processing or visualization.
 */
void WriteData();

}  // namespace mps

#endif  // MPS_PROJECT_DATA_IO_H_
