#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <random>
#include <iostream>
#include <fstream>
//#include "lib_objs.h"
#include "lib_mpi.h"
#include "lib_cuda_main.h"

using namespace std;

// ********************************************************************* //
// *************************MAIN PROGRAM******************************** //
// **********************(CPU SIDE openMPI)***************************** //
// --------------------------(CALLS CUDA)------------------------------- //
// ********************************************************************* //

int main (int argc, char *argv[]) 
{
//*************DELCLARATIONS***************//
        int N,j,i,k,Tinc,*Sa, REMC_parm[4];
	double sum=0,Tmax,Tmin,*Ta,*Ea,sumd,convc;
	char *inputf = argv[1],*outputf = argv[2],dir[40];
	time_t w1;
	clock_t t;
	float gpu_t[4];
cout << "|------Program Running------|\n";
//*********END DECLARATIONS****************//
//					   //
//************OPEN OUTPUT FILE*************//
	ofstream file1;
        file1.open (outputf);
//**********END OPEN OPFILE*****************//
//					    //
//**********RUN PREAMBLE FUNCTIONS**********//
	input_pars(9,inputf,outputf,file1,Tmax,Tmin,Tinc,N,REMC_parm,convc,dir); // read inputvalues into memory
	time_stamp(0,file1,w1,t,0); //Begin time monitoring	
//**********END PREAMBLE FUNCTIONS**********//	
//					    //
//********HOST MEMORY ALLOCATIONS***********//
	//Allocate the memory for each state object
        Sa = (int *)malloc(N*N*Tinc*sizeof(int));
        Ta = (double *)malloc(Tinc*sizeof(double));
        Ea = (double *)malloc(Tinc*sizeof(double));
//********END HOST MEMORY ALLOCATION********//
//					    //
//***********MAIN SEQUENCE CODE*************//
	build_iState_array_cuda(Sa,Ta,Ea,Tmax,Tmin,Tinc,N,gpu_t);
	MCEvo_States_cuda(Sa,Ta,Ea,N,Tinc,file1,REMC_parm,convc,gpu_t,dir);//Execute REMC Cycles on the systems.
//**********END MAIN SEQUENCE CODE**********//
//					    //
//************SET ENDING TIME STAMP*********//
	time_stamp(1,file1,w1,t,gpu_t);//end time monitoring and write data
//******END COMPILE AND PRINT RESULTS*******//
//					    //
//*********FREE MEMORY ALLOCATIONS**********//
	free(Sa);free(Ta);free(Ea); 
//********END FREE MEMORY ALLOCATIONS*******//
}

