#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>
#include "lib_classes.h"

using namespace std;

//Program Operation Functions
void input_pars(int c,char *inputf,char *outputf,ofstream &file1,double &Tmax,double &Tmin,int &Tinc,int &N, int *REMC_parm,double &convc,char *dir);
void time_stamp(int param,ofstream &file1,time_t &w1,clock_t &t,float *gpu_t);
void streaming_output(ofstream &file1, double *Ea,int *reps_ex,histo_data *hist_dat,int Tinc,int reId,int ESs,int RepSw);
int perm_search(int *array,int array_size, int N);

//MPI Math Functions
void random_real(double* rval, int N,double mval, double lval);
void random_ints(int* rval, int N,int mval,int lval);


//oMP Replica Exchange Function
void Replic_Ex_omp(double *Ea,double *Ta,int *Reps_Ex,int Tinc,histo_data *hist_dat,int *perm_func,int reId,int N,ofstream &file1);

//oMP MC cycles Function
void CalcE_MCEvo_OMP(int *Sa,double *Ta,int Tinc, int MCcycles,int N,histo_data *hist_dat,int *perm_func,int reId);
