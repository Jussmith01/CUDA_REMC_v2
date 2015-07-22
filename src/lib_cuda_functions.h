#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sstream>
#include <string>
#include <string.h>
#include <fstream>

using namespace std;

// ********************************************************************* //
// ***************************DEFINE CONSTANTS************************** //
// ********************************************************************* //
__constant__ double Kb=1.3806488e-23; // Boltzmans Constant eV/K.
__constant__ unsigned int d_ModB[1]; //N value, set on device.


// ********************************************************************* //
// ****************************C++ FUNCTIONS**************************** //
// ********************************************************************* //
void Get_threads_blocks(int &threads, int &blocks,int Tinc,int N)
{
        //*************************************************************//
        //*****************SCALABLE CUDA PROPERTIES SET****************//
        //*************************************************************//
        int device, MaxThreads;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaError_t (cudaGetDeviceProperties(&props,device));
        cudaError_t devpcode = cudaGetLastError();
        if (devpcode != cudaSuccess)
                printf("Cuda device setting error -- %s\n",cudaGetErrorString(devpcode));
        MaxThreads =  props.maxThreadsPerBlock;

        if (N * N <= MaxThreads)
        {
                threads = N * N;
                blocks = Tinc;
        } else {
                threads = N;
                blocks = Tinc * N; //This allows computation of spin glasses containing MaxThreads * MaxThreads number of particles.
        }
}

//*************************************************************//
//*****************PRINT CUDA DEVICE PROPERTIES****************//
//*************************************************************//
void Print_CUDA_Device_Props (ofstream &file1)
{
	int dcount,device;
        cudaGetDevice(&device);
        cudaGetDeviceCount(&dcount);
        cudaDeviceProp props;
        cudaError_t (cudaGetDeviceProperties(&props,device));
        cudaError_t devpcode = cudaGetLastError();
        if (devpcode != cudaSuccess)
                printf("Cuda device setting error -- %s\n",cudaGetErrorString(devpcode));
        file1 << "|--------------------DEVICE PROPERTIES-------------------|"<< "\n";
        file1 << "CUDA version:   v" << CUDART_VERSION << "\n";
        file1 << "Number of Devices: " <<  dcount << "\n";
        file1 << "On Device: (" << device << ") " << props.name <<  "\n";
        file1 << "Clock Rate: " << props.clockRate/1000 << "Mhz\n";
        file1 << "Max Threads per Block: " << props.maxThreadsPerBlock << "\n";
        file1 << "Max Threads per MultiProcessor: " << props.maxThreadsPerMultiProcessor << "\n";
        file1 << "MultiProcessor: " << props.multiProcessorCount << "\n";
        file1 << "Total Memory: " << props.totalGlobalMem/(1024*1024) << "Mb" << "\n";
        file1 << "Memory Clock Rate: " << props.memoryClockRate/1000 << "Mhz\n";
        file1 << "|--------------------------------------------------------|"<< "\n\n";
}
// ********************************************************************* //
// ***************************CUDA FUNCTIONS**************************** //
// ********************************************************************* //
//CALCULATE a MOD B = c, return value.
__device__ unsigned int modb(unsigned int a,unsigned int b) {
        float tmp1, tmp2 ;
        unsigned int t1,c;
        float eps = 0.5f ;
        tmp1 = __uint2float_rz( a ) ;
        tmp2 = __uint2float_rz( b ) ;
        tmp1 = floorf( (tmp1 + eps ) / tmp2 ) ;
        t1 = __float2uint_rz( tmp1 ) ;
        c = a - b*t1 ;
        return c;
        }
//CALCULATE a MOD B = c, return value.
__device__ unsigned int mod(unsigned int a) {
        float tmp1, tmp2 ;
        unsigned int t1,c,b;
        float eps = 0.5f ;
        b = (unsigned int) d_ModB[0];
        tmp1 = __uint2float_rz( a ) ;
        tmp2 = __uint2float_rz( b ) ;
        tmp1 = floorf( (tmp1 + eps ) / tmp2 ) ;
        t1 = __float2uint_rz( tmp1 ) ;
        c = a - b*t1 ;
        return c;
        }
//CALCULATE THE "x coordinate" NUMBER OF Sa ARRAY
__device__ unsigned int mod_rev(unsigned int a) {
        float x;
        unsigned int b;

        b = (unsigned int) d_ModB[0];
        x = floorf( a / b ) ;
        x = mod(x);
        return x;
        }

// ********************************************************************* //
// **********************CUDA CALC ENERGY OF STATE********************** //
// ********************************************************************* //
//NOTE: USING SINGLE SPIN FLIP DYNAMICS

//*******sub energy calculator using the ising spin hamiltonian********
__device__ double calc_E_sub(int *Sarr) //First val of Sarr is elem to check, following 4 vals is the values of its neighbors. 
{
        double x = 0,H = 0,mu = 0,J=-1.0;
        int i,k=4; //k is the number of nearest neighbors to current S ele'm.
        for (i = 1; i <= k; ++i){
                x += Sarr[i]*Sarr[0];
        }
        x = -(x*J);
        x -= Sarr[0] * H * mu; //currently unused
        return x;
}

//**************main energy calculator kernal****************
__global__ void __CalcE_array_cuda__(int *Sa,double *EaT)// contains only the position to calc the energy of. 
{
        int tId = threadIdx.x + blockIdx.x * blockDim.x;
        float x,y,energy;
        int SE[5],Sarr[5];
        unsigned int tmp,N;
        int i;

        //Define block value
        N = (unsigned int) d_ModB[0];
        int Bloc = floorf(blockIdx.x/N) * N * N ;

        //
        SE[0] = (int) tId;
        tmp = (unsigned int) tId;
        x = mod_rev(tmp);
        y = mod(tmp);
        //****PERIODIC BOUNDARY CONDITIONS*******//
        SE[1] = (int) Bloc + (mod(x+(N-1)) * N + y);
        SE[2] = (int) Bloc + (mod(x+(N+1)) * N + y);
        SE[3] = (int) Bloc + (mod(y+(N-1)) + x * N);
        SE[4] = (int) Bloc + (mod(y+(N+1)) + x * N);

        for (i = 0; i < 5; ++i)
        {
                Sarr[i] = (int) Sa[SE[i]];
        }
        energy = calc_E_sub(Sarr);
        EaT[tId] = energy;
        //EaT[blockIdx.x] = SE[3];
}

//*************SUM ENERGY ARRAY*******************
__global__ void __sumE_array_cuda__(double *Ea,double *EaT)// Sums the energies into the energy array. 
{
        int Id = threadIdx.x + blockDim.x * blockIdx.x,i;
        Ea[Id] = 0;
        int N = d_ModB[0] * d_ModB[0];
        for (i = 0; i < N; ++i)
        {
                Ea[Id] += EaT[i + N  * Id];
        }
        //Ea[Id] = Id;
}

// ----------------------------------------------------------------- //
// -------BUILD TEMP AND INITIAL STATES ARRAY(DEVICE SIDE)---------- //
// ----------------------------------------------------------------- //
__global__ void __temper_array_cuda__(double *Ta, double *Tmin, double *Tincsize) {
        int id = blockIdx.x;

        double tmp;
	//double a = 1;
        tmp = Tmin[0] + Tincsize[0] * id;
	Ta[id] = tmp;
        //Ta[id] = exp(a*tmp);
}

__global__ void __iState_array_cuda__(int *Sa, int *Ra) {
        int tId = threadIdx.x + blockIdx.x * blockDim.x;
        Sa[tId] = powf(-1,Ra[tId]);
}

