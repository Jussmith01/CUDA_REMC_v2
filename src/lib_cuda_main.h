#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "lib_mpi.h"

void build_iState_array_cuda(int *Sa,double *Ta,double *Ea,double Tmax,double Tmin,int Tinc,int N,float *gpu_t);

void MCEvo_States_cuda(int *Sa,double *Ta,double *Ea,int N,int Tinc,ofstream &file1,int *REMC_parm,double convc,float *gpu_t,char *dir);

void CalcSa_Energy_cuda(int *Sa,double *Ea,int N,int Tinc,int blocks,int threads,float *gpu_t);

