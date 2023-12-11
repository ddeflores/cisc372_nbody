#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__ void computeAccelKernel(double* d_mass, vector3* d_hPos, vector3* d_accels) {
    // get thread identifiers
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // make sure the threads are in bounds
    if (i < NUMENTITIES && j < NUMENTITIES) {
        if (i != j) {
            // calculate and store distances
            vector3 distance;
            for (int k = 0; k < 3; k++) {
                distance[k] = d_hPos[i][k] - d_hPos[j][k];
            }
            // calculate and store force magnitude
            double magnitude_sq = distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2];
            double magnitude = sqrt(magnitude_sq);
            double force = (-GRAV_CONSTANT * d_mass[j]) / magnitude_sq;
            // calculate and store acceleration
            for (int k = 0; k < 3; k++) {
                d_accels[i * NUMENTITIES + j][k] = force * distance[k] / magnitude;
            }
        } else {
            // 0 acceleration for self
            for (int k = 0; k < 3; k++) {
                d_accels[i * NUMENTITIES + j][k] = 0.0;
            }
        }
    }
}

__global__ void updatePosVelKernel(vector3* d_hPos, vector3* d_hVel, vector3* d_accels) {
    // get identifier
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUMENTITIES) {
        // sum the accelerations
        vector3 totalAccel = {0.0, 0.0, 0.0};
        for (int j = 0; j < NUMENTITIES; j++) {
            for (int k = 0; k < 3; k++) {
                totalAccel[k] += d_accels[idx * NUMENTITIES + j][k];
            }
        }
        // update the velocity and position for device
        for (int k = 0; k < 3; k++) {
            d_hVel[idx][k] += totalAccel[k] * INTERVAL;
            d_hPos[idx][k] += d_hVel[idx][k] * INTERVAL;
        }
    }
}

void compute() {
    // allocate memory on GPU and transfer data from CPU to GPU
    vector3 *d_hPos, *d_hVel, *d_accels;
    double* d_mass;

    cudaMalloc(&d_hPos, NUMENTITIES * sizeof(vector3));
    cudaMalloc(&d_hVel, NUMENTITIES * sizeof(vector3));
    cudaMalloc(&d_accels,NUMENTITIES * NUMENTITIES * sizeof(vector3));
    cudaMalloc(&d_mass, NUMENTITIES * sizeof(double));

    cudaMemcpy(d_hPos, hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, NUMENTITIES  * sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((NUMENTITIES + dimBlock.x - 1) / dimBlock.x, (NUMENTITIES + dimBlock.y - 1) / dimBlock.y);
    // launch kernel
    computeAccelKernel<<<dimGrid, dimBlock>>>(d_mass, d_hPos, d_accels);
    cudaDeviceSynchronize();
    
    dim3 dimBlockSingle(256);
    dim3 dimGridSingle((NUMENTITIES + dimBlockSingle.x - 1) / dimBlockSingle.x);
    // launch kernel
    updatePosVelKernel<<<dimBlockSingle, dimGridSingle>>>(d_hPos, d_hVel, d_accels);
    cudaDeviceSynchronize();

    // transfer updated data back to CPU and free GPU memory
    cudaMemcpy(hPos, d_hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);
    
    // free allocated memory
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
    cudaFree(d_accels);
}
