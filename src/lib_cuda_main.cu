#include <stdio.h>
#include <time.h>
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
#include "lib_cuda_main.h"
#include "lib_mpi.h"
#include "lib_classes.h"
#include "lib_cuda_functions.h"

using namespace std;

//___________________________________________________________________//
// ----------------------------------------------------------------- //
// ------------------MC EVOLUTION STEP(EACH REPLICA)---------------- //
// ----------------------------------------------------------------- //
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
extern void build_iState_array_cuda(int *Sa,double *Ta,double *Ea,double Tmax,double Tmin,int Tinc,int N,float *gpu_t)
{
        int *d_Sa,*Ra,*d_Ra;
        unsigned int *ModB;
        double *Tincsize,*Tl,*d_Ta,*d_Ea,*EaT,*d_EaT;
        double *d_Tincsize,*d_Tmin;

        //cudaSetDevice(1);

        //******DEFINE THREADS AND BLOCK SIZES*******
	int blocks, threads;
	Get_threads_blocks(threads,blocks,Tinc,N);

        //*****MEMORY ALLOCATIONS CPU******
        Tincsize = (double *)malloc(sizeof(double));
        Tl = (double *)malloc(sizeof(double));
        int RaSize = N * N * Tinc; Ra = (int *)malloc(RaSize * sizeof(int));
        ModB = (unsigned int *)malloc(sizeof(unsigned int)); ModB[0] = (unsigned int)N;
        EaT = (double *)malloc(N * N * Tinc * sizeof(double));

        //*****MEMORY ALLOCATIONS GPU*******    
        cudaError_t (cudaMalloc ((void **) &d_Tincsize, sizeof(double)));
        cudaError_t (cudaMalloc ((void **) &d_Sa, Tinc * N * N * sizeof(int)));
        cudaError_t (cudaMalloc ((void **) &d_Ra, Tinc * N * N * sizeof(int)));
        cudaError_t (cudaMalloc ((void **) &d_Ta, Tinc * sizeof(double)));
        cudaError_t (cudaMalloc ((void **) &d_Ea, Tinc * sizeof(double)));
        cudaError_t (cudaMalloc ((void **) &d_Tmin, sizeof(double)));
        cudaError_t (cudaMalloc ((void **) &d_EaT, N * N  * Tinc * sizeof(double)));

        cudaMemcpyToSymbol (d_ModB, ModB, sizeof(unsigned int));
        cudaError_t codeMemA1 = cudaGetLastError();
        if (codeMemA1 != cudaSuccess)
                printf("Cuda error INITIAL MEM ALLOCATION COPY -- %s\n",cudaGetErrorString(codeMemA1));

        //********DEFINE NEEDED VALUES*******
	//double a = 1;
        //Tincsize[0] = (log(Tmax)/a - log(Tmin)/a) / Tinc;
	Tincsize[0] = (Tmax - Tmin) / Tinc;
        //Tl[0] = log(Tmin)/a;
	Tl[0] = Tmin;
        random_ints(Ra,RaSize,2,1);

        //*****Create events for GPU timer*****
        cudaEvent_t event1, event2;
        cudaError_t(cudaEventCreate(&event1));
        cudaError_t(cudaEventCreate(&event2));

        /* ... Load CPU data into GPU buffers  */
        cudaError_t (cudaMemcpy(d_Tincsize, Tincsize, sizeof(double), cudaMemcpyHostToDevice));
        cudaError_t (cudaMemcpy(d_Tmin, Tl, sizeof(double), cudaMemcpyHostToDevice));
        cudaError_t (cudaMemcpy(d_Ra, Ra,RaSize * sizeof(int), cudaMemcpyHostToDevice));
        cudaError_t (cudaMemcpy(d_Ea,Ea,Tinc * sizeof(double),cudaMemcpyHostToDevice));
        cudaError_t (cudaMemcpy(d_EaT,EaT, N * N * Tinc * sizeof(double),cudaMemcpyHostToDevice));
        cudaError_t codeMem1 = cudaGetLastError();
        if (codeMem1 != cudaSuccess)
                printf("Cuda error INITIAL MEM COPY -- %s\n",cudaGetErrorString(codeMem1));

        //*****RUN PROGRAMS(EVENTS TIME GPU CALCULATIONS)****
        cudaEventRecord(event1,0);

        __temper_array_cuda__ <<<Tinc,1>>> (d_Ta,d_Tmin,d_Tincsize);
        cudaThreadSynchronize();
        cudaError_t code = cudaGetLastError();
        if (code != cudaSuccess)
                printf("Cuda BUILD INITIAL TEMPER error -- %s\n",cudaGetErrorString(code));

        __iState_array_cuda__ <<<blocks,threads>>> (d_Sa,d_Ra);
        cudaThreadSynchronize();
        cudaError_t code2 = cudaGetLastError();
        if (code2 != cudaSuccess)
                printf("Cuda BUILD ISTATE  error -- %s\n",cudaGetErrorString(code2));

        __CalcE_array_cuda__ <<<blocks,threads>>> (d_Sa,d_EaT);
        cudaThreadSynchronize();
        cudaError_t code3 = cudaGetLastError();
        if (code3 != cudaSuccess)
                printf("Cuda INITIAL E CALC error -- %s\n",cudaGetErrorString(code3));

        __sumE_array_cuda__ <<<1,Tinc>>> (d_Ea,d_EaT);
        cudaThreadSynchronize();
        cudaError_t code4 = cudaGetLastError();
        if (code != cudaSuccess)
                printf("Cuda SUM E ERROR  error -- %s\n",cudaGetErrorString(code4));

        cudaError_t (cudaEventRecord(event2,0));

        /* ... Transfer data from GPU to CPU */
        cudaError_t (cudaMemcpy(Sa,d_Sa,Tinc * N * N * sizeof(int),cudaMemcpyDeviceToHost));
        cudaError_t (cudaMemcpy(Ta,d_Ta,Tinc * sizeof(double),cudaMemcpyDeviceToHost));
        cudaError_t (cudaMemcpy(Ea,d_Ea,Tinc * sizeof(double),cudaMemcpyDeviceToHost));

        cudaError_t (cudaMemcpy(Tincsize, d_Tincsize, sizeof(double), cudaMemcpyDeviceToHost));
        cudaError_t (cudaMemcpy(Tl, d_Tmin, sizeof(double), cudaMemcpyDeviceToHost));
        cudaError_t (cudaMemcpy(Ra, d_Ra,RaSize * sizeof(int), cudaMemcpyDeviceToHost));
        cudaError_t (cudaMemcpy(EaT,d_EaT, N * N * Tinc * sizeof(double),cudaMemcpyDeviceToHost));

	
        cudaError_t codeMemcp1 = cudaGetLastError();
        if (codeMemcp1 != cudaSuccess)
                printf("Cuda MEMORY 1 error -- %s\n",cudaGetErrorString(codeMemcp1));
	
        //Sync the events so that 2 doesnt finish before the end of computations.
        cudaError_t (cudaEventSynchronize(event1));
        cudaError_t (cudaEventSynchronize(event2));

        //char* cudaGetErrorString(cudaError_t error);
        float dt_ms;
        cudaError_t (cudaEventElapsedTime(&dt_ms, event1, event2));
        cudaError_t codetime = cudaGetLastError();
        if (codetime != cudaSuccess)
                printf("Cuda TIMER 1 error -- %s\n",cudaGetErrorString(codetime));
	for (int i = 0; i < Tinc; ++i)
	{
		cout << "TEMP(" << i << ")= " << Ta[i] << "\n";
	}

        gpu_t[0] += (float) dt_ms;
        //************FREE MEMORY***********
        free(Tincsize); free(Tl); free(Ra); free(ModB); free(EaT);
        cudaFree(d_Tincsize); cudaFree(d_Sa); cudaFree(d_Ta); cudaFree(d_Tmin); cudaFree(d_Ra); cudaFree(d_Ea); cudaFree(d_EaT);
}

//___________________________________________________________________//
// ----------------------------------------------------------------- //
// ------------------MC EVOLUTION STEP(EACH REPLICA)---------------- //
// ----------------------------------------------------------------- //
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
extern void MCEvo_States_cuda(int *Sa,double *Ta,double *Ea,int N,int Tinc,ofstream &file1,int *REMC_parm,double convc,float *gpu_t,char *dir)
{
        //NOTE THAT REMC_parm contains the parameters for the run:
	//REMC_parm[i]: i = 0 -> MC cycles per RE, i = 1 -> Replica exchanges, i = 2 ->energy sample every x MC cycles
	
	int i,j,k,*perm_func,*Reps_Ex,pId_loc,*d_Sa;
	double *d_Ea,*d_EaT;
	struct  timespec timerN,timer0;
	histo_data *hist_dat;
	
	//Define run parameters -- #MCcyclesTotal = MCc*ESs*REs 
	int MCc = REMC_parm[0]; //Define MC cycles per RE
	int REs = REMC_parm[1]; //Define number of replica exchanges
	int ESs = REMC_parm[2]; //Define number of energy samples per cycle 
	int RepSw = REMC_parm[3]; //Replica exchange switch 0 turns it on 1 turns it off
	//Set time
	clock_gettime(CLOCK_REALTIME,&timer0);

	cout << "!!!!!REMC BEGINS!!!!!\n";
	//*************************************************************//
        //*******************MEMORY ALLOCATIONS (HOST)*****************//
	//*************************************************************//

        Reps_Ex = (int *)malloc(Tinc * sizeof(int));
	hist_dat = (histo_data *)malloc(Tinc * sizeof(histo_data));
        perm_func = (int *)malloc(Tinc * sizeof(int));

        //*****MEMORY ALLOCATIONS GPU*******    
        cudaError_t(cudaMalloc ((void **) &d_Sa, Tinc * N * N * sizeof(int)));
        cudaError_t(cudaMalloc ((void **) &d_Ea, Tinc * sizeof(double)));
        cudaError_t(cudaMalloc ((void **) &d_EaT, Tinc * N * N * sizeof(double)));
        cudaError_t codeMemAlloc = cudaGetLastError();
        if (codeMemAlloc != cudaSuccess)
                printf("Cuda error ALLOCATIONS -- %s\n",cudaGetErrorString(codeMemAlloc));


        //******PRINT CUDA DEVICE PROPERTIES*********//
 	Print_CUDA_Device_Props (file1);
	
        //******DEFINE THREADS AND BLOCK SIZES*******//
        int blocks, threads;
	Get_threads_blocks(threads,blocks,Tinc,N);
	file1 << "\nSetting CUDA block/thread count:\n";
        file1 << "Blocks: " << blocks << " Threads: " << threads  << "\n\n";
 	
	//*****Create events for GPU timer*****
	float dt_ms;
        cudaEvent_t event3, event4;
        cudaEventCreate(&event3);
        cudaEventCreate(&event4);

	//SETUP CLASSES OBJECTS FOR HISTOGRAM DATA LOGGING AND OUTPUT FILES
	file1 << "Replica Temperatures:\n";
	for (i = 0; i < Tinc;++i)
	{
		hist_dat[i].reset_probs();
 		hist_dat[i].set_RE_prob();
		hist_dat[i].Temp = Ta[i];
		hist_dat[i].array_size = (REs * ESs);
        	hist_dat[i].set_Ealloc();
		perm_func[i] = i;
		Reps_Ex[i] = 0;
                file1 << "Temp("  << i << ")= " << hist_dat[i].Temp << "\n";
	}
	
//----------------------------------------------------------------------------//
        //*************************************************************//
        //*******************BEGIN REPLICA EXCHANGE MC*****************//
        //*************************************************************//
	file1 << "\n|------------------REMC STARTING------------------|\n";
        for (j = 0; j < REs; ++j)
        {
                cout << "|******BEGIN CYCLE " << j << "********|" << "\n";
		file1 << "|************Cycle " << j  << "************|" << "\n";	
                //*************RUN REPLICA EXCHANGE CODE*************
		switch(RepSw)
		{
			case 0:
			switch(j)
			{
				case 0:
				cout << "replica exchange skipped" << "\n";
				break;
			
				default:
                		Replic_Ex_omp(Ea,Ta,Reps_Ex,Tinc,hist_dat,perm_func,j,N,file1);
              			break;
			}
			break;

			default:
			file1 << "Replica Exchange Disabled\n";
			break;
                }
		//*****RUN MC PROGRAMS(EVENTS TIME GPU CALCULATIONS)****
			for (k = 0; k < ESs ; ++k)
			{ 
                        	CalcE_MCEvo_OMP(Sa, Ta, Tinc, MCc, N, hist_dat, perm_func, j);

				//***********SAMPLE THE ENERGY STATES**************
				cudaEventRecord(event3,0);
				
				//-------Load states into the GPU buffers---------//
				cudaError_t(cudaMemcpy(d_Sa,Sa,Tinc * N * N * sizeof(int),cudaMemcpyHostToDevice));				
				cudaError_t codeMem = cudaGetLastError();
                                if (codeMem != cudaSuccess)
                                        printf("Cuda error Mem -- %s\n",cudaGetErrorString(codeMem));
				
				//------CALCULATE THE ENERGY FOR HISTO DATA-------//
			        __CalcE_array_cuda__ <<<blocks,threads>>> (d_Sa,d_EaT);
        			cudaThreadSynchronize();
        			cudaError_t code = cudaGetLastError();
        			if (code != cudaSuccess)
                			printf("Cuda error -- %s\n",cudaGetErrorString(code));

			        __sumE_array_cuda__ <<<Tinc,1>>> (d_Ea,d_EaT);
        			cudaThreadSynchronize();
        			cudaError_t code1 = cudaGetLastError();
        			if (code1 != cudaSuccess)
                		printf("Cuda error -- %s\n",cudaGetErrorString(code1));
 				
				//--------END ENERGY CALC FOR HIST DATA-----------//	
			 	
				//------Load data from GPU to HOST------//
				 cudaError_t(cudaMemcpy(Ea,d_Ea,Tinc * sizeof(double),cudaMemcpyDeviceToHost));				
				
				//-----------SAVE GPU TIMER DATA------------//
				cudaEventRecord(event4,0);
				cudaEventSynchronize(event3);
			        cudaEventSynchronize(event4);
				cudaEventElapsedTime(&dt_ms, event3, event4);
			        gpu_t[1] += (float) dt_ms;


				clock_gettime(CLOCK_REALTIME,&timerN);
				//-------------Save energies------------//
				for (i = 0; i < Tinc; ++i)
				{
					pId_loc = perm_search(perm_func, Tinc, i);
					hist_dat[i].energies[k+j*ESs] = Ea[pId_loc];
					hist_dat[i].replica_array[k+j*ESs] = pId_loc;
					hist_dat[i].timestamp_array[k+j*ESs] = (double)((timerN.tv_nsec*0.000000001 + timerN.tv_sec) - (timer0.tv_nsec*0.000000001 + timer0.tv_sec));
                		}

			}
		//Run streaming output
		streaming_output(file1,Ea,Reps_Ex,hist_dat,Tinc,j,ESs,RepSw);
	}

// ----------------------------------------------------------------------------//
	//*************************************************************//
        //**************PRINT HISTOGRAM DATA TO FILES******************//
        //*************************************************************//
	for (i = 0; i < Tinc; ++i)
	{
	hist_dat[i].print_data(dir);
        }
	//*************************************************************//
        //**************CALCULATE FINAL ENERGIES AND END***************//
        //*************************************************************//

       CalcSa_Energy_cuda(Sa,Ea,N,Tinc,blocks,threads,gpu_t);//Calc energy of new states
// ----------------------------------------------------------------------------//

        //************FREE MEMORY***********//

	hist_dat[0].freeEalloc();
        free(perm_func);
	free(Reps_Ex); free(hist_dat);
	cudaFree(d_Sa); cudaFree(d_Ea); cudaFree(d_EaT); 
	cout << "|---------END REMC RUN----------|" << "\n";
	file1 << "\n|------------------REMC FINISHED------------------|\n";
}

//___________________________________________________________________//
// ----------------------------------------------------------------- //
// -----------Calculate Enery Of State(EACH REPLICA)---------------- //
// ----------------------------------------------------------------- //
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
extern void CalcSa_Energy_cuda(int *Sa,double *Ea,int N,int Tinc,int blocks,int threads,float *gpu_t)
{
        int *d_Sa;
        double *d_Ea,*EaT,*d_EaT;
	//cudaSetDevice(1);

        //*****MEMORY ALLOCATIONS CPU******
        EaT = (double *)malloc(Tinc * N * N *sizeof(double));

        //*****MEMORY ALLOCATIONS GPU*******    
        cudaError_t(cudaMalloc ((void **) &d_Sa, Tinc * N * N * sizeof(int)));
        cudaError_t(cudaMalloc ((void **) &d_Ea, Tinc * sizeof(double)));
        cudaError_t(cudaMalloc ((void **) &d_EaT, Tinc * N * N * sizeof(double)));
	cudaError_t codeMemAlloc = cudaGetLastError();
        if (codeMemAlloc != cudaSuccess)
                printf("Cuda error ALLOCATIONS -- %s\n",cudaGetErrorString(codeMemAlloc));

	//*****Create events for GPU timer*****
        cudaEvent_t event5, event6;
        cudaEventCreate(&event5);
        cudaEventCreate(&event6);

        /* ... Load CPU data into GPU buffers  */
        cudaError_t(cudaMemcpy(d_Sa,Sa,Tinc * N * N * sizeof(int),cudaMemcpyHostToDevice));
        cudaError_t codeMemA2 = cudaGetLastError();
        if (codeMemA2 != cudaSuccess)
                printf("Cuda error INITIAL MEM 1 ALLOCATION COPY -- %s\n",cudaGetErrorString(codeMemA2));
        
	cudaError_t(cudaMemcpy(d_Ea,Ea,Tinc * sizeof(double),cudaMemcpyHostToDevice));
        cudaError_t codeMemA3 = cudaGetLastError();
        if (codeMemA3 != cudaSuccess)
                printf("Cuda error INITIAL MEM 2 ALLOCATION COPY -- %s\n",cudaGetErrorString(codeMemA3));

	cudaError_t(cudaMemcpy(d_EaT,EaT,Tinc * N * N * sizeof(double),cudaMemcpyHostToDevice));
        cudaError_t codeMemA1 = cudaGetLastError();
        if (codeMemA1 != cudaSuccess)
                printf("Cuda error INITIAL MEM 3 ALLOCATION COPY -- %s\n",cudaGetErrorString(codeMemA1));

        //*****RUN PROGRAMS(EVENTS TIME GPU CALCULATIONS)****
        cudaEventRecord(event5,0);
        __CalcE_array_cuda__ <<<blocks,threads>>> (d_Sa,d_EaT);
	cudaThreadSynchronize();
        cudaError_t code = cudaGetLastError();
        if (code != cudaSuccess)
                printf("Cuda error -- %s\n",cudaGetErrorString(code));


        __sumE_array_cuda__ <<<Tinc,1>>> (d_Ea,d_EaT);
        cudaThreadSynchronize();
	cudaError_t code1 = cudaGetLastError();
        if (code1 != cudaSuccess)
                printf("Cuda error -- %s\n",cudaGetErrorString(code1));

        cudaEventRecord(event6,0);

        /* ... Transfer data from GPU to CPU */
        cudaMemcpy(Ea,d_Ea,Tinc * sizeof(double),cudaMemcpyDeviceToHost);

        //Sync the events so that 2 doesnt finish before the end of computations.
        cudaEventSynchronize(event5);
        cudaEventSynchronize(event6);

        float dt_ms;
        cudaEventElapsedTime(&dt_ms, event5, event6);
        gpu_t[2] += (float) dt_ms;

        //************FREE MEMORY***********
        free(EaT);
        cudaFree(d_Sa); cudaFree(d_Ea); cudaFree(d_EaT);
}

