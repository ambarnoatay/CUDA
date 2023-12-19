/*
Author: Ambar Noatay

Last Date Modified: 11/07/2023
Description: 
To simulate a 2D random walk. A random walk is a mathematical process that describes a path consisting of a sequence of random steps. 
This code simulates a large number of walkers taking steps either north, south, east, or west on a grid, and calculate the 
average distance they travel from the origin.
The Calculation is done using 1) Normal CUDA Memory Allocation  2) Pinned CUDA memory Allocation and 3) Managed CUDA memory Allocation

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include<chrono>
#include<iostream>
#include<string.h>





__global__ void randomWalkKernel(float *xval, float *yval,float *out, int numWalkers, int numSteps,int seed) {
    
    // Each thread simulates the random walk for a single walker
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, threadId, 0, &state);
    if (threadId < numWalkers) 
    {
        for (int i = 0; i < numSteps; i++) 
        {
            // Generate a random direction for the walker
            float y = curand_uniform(&state);
            int x = y/=0.25;
            int randomDirection = x%4;
            
            

            // Update the walker's position based on the random direction
            switch (randomDirection) 
            {
                case 0:
                    xval[threadId] += 1.0f; //+x direction
                    break;
                case 1:
                    xval[threadId] -= 1.0f; //-x direction
                    break;
                case 2:
                    yval[threadId + 1] += 1.0f; //+y direction
                    break;
                case 3:
                    yval[threadId + 1] -= 1.0f; //-y direction
                    break;
            }
            
        }
        // Calculate Distance from the Origin
        out[threadId] = sqrt( (xval[threadId]*xval[threadId]) + (yval[threadId]*yval[threadId]) );
        
    }
}
int main(int argc, char **argv)
{
    
    
    
    
    if(argc != 5 || (strcmp(argv[1],"-W")!=0) || (strcmp(argv[3],"-I")!=0)) 
    {
        printf("Usage: %s -W <numWalkers> -I <numSteps>\n", argv[0]);
        printf("Bye \n");
        exit(0);
    }
    // find number of walkers and number of steps
    int numWalkers = atoi(argv[2]);
    int numSteps = atoi(argv[4]);
    if(numWalkers<1 || numSteps<1)
    {
        printf("Error: No. of Walkers and Steps should be greater than 0\n");
        printf("Bye\n");
        exit(0);
    }
    
    float *xval,*yval,*out;
    float *dxval,*dyval,*dout; 
    int N = numWalkers;

    // Allocate host memory

    out = new float[N];
    xval = new float[N];
    yval = new float[N];


    // Initialize host arrays
    for(int i = 0; i < N; i++)
    {
        xval[i] = 1.0f;
        yval[i] = 2.0f;
       
    }

    // Allocate device memory 
    cudaMalloc((void**)&dyval, sizeof(float) * N);
    cudaMalloc((void**)&dxval, sizeof(float) * N);
    cudaMalloc((void**)&dout, sizeof(float) * N);
    
   


    // Executing kernel 
    int blockSize = 10;
    int gridSize = ((N + blockSize) / blockSize);
    auto start_time = std::chrono::high_resolution_clock::now(); // Calculate Start Time
    randomWalkKernel<<<gridSize, blockSize>>>(dxval, dyval,dout, N,numSteps,sizeof(float)*numWalkers);
    
    // Transfer data back to host memory
    cudaMemcpy(out, dout, sizeof(float) * numWalkers, cudaMemcpyDeviceToHost);
    
    

    //Calculate Average Distance of walkers:
    float avgDistance =0;

    for(int i=0;i<numWalkers;i++)
    {
        avgDistance+=out[i];
        

    }
    avgDistance = avgDistance/numWalkers;
    
    auto end_time = std::chrono::high_resolution_clock::now();// Calculate End Time
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(); // Calculate duration
    printf("Normal CUDA memory Allocation:\n");
    std::cout<<"    Time to calculate(microsec): "<<duration<<std::endl;
    printf("    Average distance from origin: %.3f\n", avgDistance);
    






    // Deallocate device memory

    cudaFree(dxval);
    cudaFree(dyval);
    cudaFree(dout);
    
    // Deallocate host memory
    free(xval); 
    free(yval); 
    free(out);
    
    // Instantiate variables for Pinnd Memory Calculations
    float *pxval,*pyval,*pout;
    float *dpxval,*dpyval,*dpout; 

    // Allocate device memory and deviceHost Memory
    cudaMalloc((void**)&dpyval, sizeof(float) * N);
    cudaMalloc((void**)&dpxval, sizeof(float) * N);
    cudaMalloc((void**)&dpout, sizeof(float) * N);
    cudaMallocHost((void**)&pyval, sizeof(float) * N);
    cudaMallocHost((void**)&pxval, sizeof(float) * N);
    cudaMallocHost((void**)&pout, sizeof(float) * N);

    auto start_timep = std::chrono::high_resolution_clock::now();// Calculate Start Time

    // Transfer data from host to device memory
    cudaMemcpy(dpxval, pxval, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dpyval, pyval, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dpout, pout, sizeof(float) * N, cudaMemcpyHostToDevice);


    // Executing kernel 
    randomWalkKernel<<<gridSize, blockSize>>>(dpxval, dpyval,dpout, numWalkers,numSteps,sizeof(float)*numWalkers);

    // Transfer data back to host memory
    cudaMemcpy(pout, dpout, sizeof(float) * numWalkers, cudaMemcpyDeviceToHost);

    //Calculate Average Distance
    float avgDistancep =0;

    for(int i=0;i<numWalkers;i++)
    {
        avgDistancep+=pout[i];
        

    }
    avgDistancep = avgDistancep/numWalkers;

    
    auto end_timep = std::chrono::high_resolution_clock::now();// Calculate End Time
    auto durationp = std::chrono::duration_cast<std::chrono::microseconds>(end_timep - start_timep).count();// Calculate duration
    printf("Pinned CUDA memory Allocation:\n");
    std::cout<<"    Time to calculate(microsec): "<<durationp<<std::endl;
    printf("    Average distance from origin: %.3f\n", avgDistancep);

    // Deallocate Device memory
    cudaFree(dpxval);
    cudaFree(dpyval);
    cudaFree(dpout);   
    cudaFree(pxval); 
    cudaFree(pyval); 
    cudaFree(pout);




    float *mxval,*myval,*mout;
    
    //Allocate Memory in GPU
    cudaMallocManaged((void**)&myval, sizeof(float) * N);
    cudaMallocManaged((void**)&mxval, sizeof(float) * N);
    cudaMallocManaged((void**)&mout, sizeof(float) * N);
    auto start_timem = std::chrono::high_resolution_clock::now();// Calculate Start Time

    // Executing kernel 
    randomWalkKernel<<<gridSize, blockSize>>>(mxval, myval,mout, numWalkers,numSteps,sizeof(float)*numWalkers);
    cudaDeviceSynchronize();

    //Calculate Average Distance
    float avgDistancem =0;

    for(int i=0;i<numWalkers;i++)
    {
        avgDistancem+=mout[i];
        

    }
    avgDistancem = avgDistancem/numWalkers;

    auto end_timem = std::chrono::high_resolution_clock::now();// Calculate End Time
    auto durationm = std::chrono::duration_cast<std::chrono::microseconds>(end_timem - start_timem).count(); // Calculate duration
    
    printf("Managed CUDA memory Allocation:\n");
    std::cout<<"    Time to calculate(microsec): "<<durationm<<std::endl;
    printf("    Average distance from origin: %.3f\n", avgDistancem);
    printf("Bye\n");

    // Deallocate Device memory
    cudaFree(mxval); 
    cudaFree(myval); 
    cudaFree(mout);

}
