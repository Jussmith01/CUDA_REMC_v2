#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string.h>
#include <string>
#include <iostream>
#include <math.h>
#include <time.h>
#include <ctime>
#include <random>
#include <fstream>
#include "lib_cuda_main.h"
#include "lib_mpi.h"
#include "lib_classes.h"

using namespace std;
// ********************************************************************* //
// ********************RANDOM REAL GENERATOR**************************** //
// ***[random_real(array to output to, size of array,topval,lowval)]**** //
// ********************************************************************* //
extern void random_real(double* rval, int N,double mval,double lval) //CPD SIDE, THREADED FOR MULI-CPU
{

        std::uniform_real_distribution<double> distribution(lval,mval);

        #pragma omp parallel for firstprivate(N)
                for (int j = 0; j < N; ++j)
                {
		int tid = omp_get_thread_num();
                int seed = (j+1)*(tid+1)*time(0);
                std::minstd_rand0 generator (seed);
                rval[j] =  distribution(generator);
                }

}
// ********************************************************************* //
// *****************RANDOM INTEGER GENERATOR**************************** //
// **********[random_ints(array to output to, size of array)]*********** //
// ********************************************************************* //
extern void random_ints(int* rval, int N,int mval,int lval) //CPD SIDE, THREADED FOR MULI-CPU
{
        std::uniform_int_distribution<int> distribution(lval,mval);

	//Set OMP environments
        #pragma omp parallel for firstprivate(N)
                for (int j = 0; j < N; ++j)
                {
                int tid = omp_get_thread_num();
                int seed = (j+1)*(tid+1)*time(0);
                std::minstd_rand0 generator (seed);
                rval[j] =  distribution(generator);
                }
}

extern int perm_search(int *array,int array_size, int N)
{
	int position;
	for (int k = 0; k < array_size; ++k)
	{
		if (array[k] == N)
		{
			position = k;
			break;
		}
		
	}
	return position;
}
// ********************************************************************* //
// *****************REPLICA EXCHANGE USING oMP************************** //
// *******[Replic_Ex_omp(array to output to, size of array)]************ //
// ********************************************************************* //
extern void Replic_Ex_omp(double *Ea,double *Ta,int *Reps_Ex,int Tinc,histo_data *hist_dat, int *perm_func,int reId,int N,ofstream &file1) //CPD SIDE, THREADED FOR MULI-CPU
{
        int j,i,k,pId,pIdn,pId_loc,pIdn_loc;
	double *rval_R;

	//Memory Allocations	
	rval_R = (double *)malloc(Tinc * sizeof(double));

	//Definitions
	double Kb = 1;//1.3806488e-23;
	random_real(rval_R,Tinc,1,0);
//***********************************************************************//
//*************************Start Replica Exchange************************//
//***********************************************************************//
        double *Dtmp,dE,Delta,dB;
	int *tPa;
        Dtmp = (double *)malloc(Tinc * sizeof(double));

	for (int k = 0; k < 2; ++k)
	{
		for (i = 0; i < Tinc/2; ++i)
		{

			//Define permutation IDs
			pId = i*2+k;
			pIdn = (pId + 1) % Tinc;		
			file1 << "(T" << pId << " -> T" << pIdn << ")\n";
			//Find permutation location
			pId_loc = perm_search(perm_func,Tinc, pId);
			pIdn_loc = perm_search(perm_func,Tinc, pIdn);
		
			//Record RE run
			hist_dat[pId].RE_prob_dat[1] += 1;

			//Calculate dE
               		dE = (Ea[pIdn_loc] - Ea[pId_loc]);
                	dB = (1 / (double)(Kb * Ta[pId])) - (1 / (double)(Kb * Ta[pIdn]));
                	Delta =  dE * dB;
			file1 << "Delta: " << Delta << " dE: " << dE << " dB: " << dB << " Exp(-Delta): " <<  exp(-1 * Delta); 	
			//Begin Metropolis-Hastings
			if (Delta > 0)
			{ 
        	        	Dtmp[i] = exp(-1 * Delta);

				//Acceptance Criterion
        	        	if (rval_R[i] < Dtmp[i]) //accept the exchange, update the permutation function
        	        	{
					perm_func[pId_loc] = pIdn;
					perm_func[pIdn_loc] = pId;
			        	Reps_Ex[pId] = 1;
                	        	hist_dat[pId].RE_prob_dat[0] += 1;//Record move accepted
					file1 << " ACCEPTED (Delta > 0)\n";
                		} else { //decline the exchange
               				Reps_Ex[pId] = 0;
					file1 << " DECLINED\n";
                		}
			} else {//accept dE < 0
         	        	perm_func[pIdn_loc] = pId;
	                	perm_func[pId_loc] = pIdn;
                        	Reps_Ex[pId] = 1;
                        	hist_dat[pId].RE_prob_dat[0] += 1;//Record move accepted
				file1 << " ACCEPTED (Delta < 0)\n";
			}
		}
	}
	//FREE MEMORY
        free(rval_R); free(Dtmp);
}
// ********************************************************************* //
// ***********************CALCULATE ENERGY SUB************************** //
// ********************************************************************* //
//*******sub energy calculator using the ising spin hamiltonian********
double calc_E_sub_OMP(int *Sarr) //First val of Sarr is elem to check, following 4 vals is the values of its neighbors. 
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

// ********************************************************************* //
// ****************************MODULAR MATH***************************** //
// ********************************************************************* //
int mod_rev(int a, int N) {
        int b,x;

        b =  N;
        x = floor( a / b ) ;
        x = x % N;
        return x;
        }

// ********************************************************************* //
// **************************CPU MC CYCLES*******************************//
// ********************************************************************* //
extern void CalcE_MCEvo_OMP(int *Sa,double *Ta,int Tinc, int MCcycles,int N,histo_data *hist_dat, int *perm_func, int reId)
{
        int j,k,*rval_I;
        double *rval_R;

	//Memory Allocations    
        rval_R = (double *)malloc(Tinc * MCcycles * sizeof(double));
        rval_I = (int *)malloc(Tinc * MCcycles * sizeof(int));

	//Set OMP environments
	omp_set_nested(1);

        //Calculate random arrays
        random_real(rval_R,Tinc * MCcycles,1,0);
        random_ints(rval_I,Tinc * MCcycles,(N * N)-1,0);
	
	//Begin primary parallelization
	//Start cycles
	#pragma omp parallel for schedule(dynamic) firstprivate(Tinc,N)
	for (k = 0; k < MCcycles; ++k)
	{
		for (int l = 0; l < Tinc; ++l)
		{
			int tId = l;
			double Kb = 1;
			int pId = perm_func[tId];
			int pId_loc = perm_search(perm_func,Tinc, pId); //pId_loc can be replaced by tId... they give the same information.
			double energy_new,energy_old;
		        int SE[5],Sarr[5];
        		int tmp1 = tId;
			double B = (1/(Kb * Ta[pId]));
			int i,x,y;
			int Bloc = tId * N * N;
        		SE[0] = (int) Bloc + rval_I[tId+k*Tinc];
        		int tmp = SE[0];
        		x = mod_rev(tmp,N);
        		y = tmp % N;
        			//****PERIODIC BOUNDARY CONDITIONS*******//
				SE[1] = (int) Bloc + (((x+(N-1)) % N) * N + y);
				SE[2] = (int) Bloc + (((x+(N+1)) % N) * N + y);
				SE[3] = (int) Bloc + (((y+(N-1)) % N) + x * N);
				SE[4] = (int) Bloc + (((y+(N+1)) % N) + x * N);
			for (i = 0; i < 5; ++i)
        		{
               			Sarr[i] =  Sa[SE[i]];
        		}
			//-------------------------------------------//
        		//************METROPOLIS ALGORITHM***********//
        		energy_old = calc_E_sub_OMP(Sarr);
        		Sarr[0] = (int) Sarr[0] * (-1);
        		energy_new = calc_E_sub_OMP(Sarr);
       			double dE = energy_new - energy_old;
			double tmpnum = exp(-1 * B * dE);
                        //cout << "HELLO FROM 4: " << l << "\n";

			//cout << "\n MONTECARLO dE: " << dE << "\n";
			if (dE > 0)// 
        		{
				//cout << "rval: " << rval_R[tId] << " Exchange PROB: " << tmpnum << "\n";
               			if (rval_R[tId+k*Tinc] < tmpnum)
               			{
					Sa[SE[0]] = Sa[SE[0]] * -1;//ACCEPTANCE
					hist_dat[pId_loc].prob_a[1] += 1;
               	 		} else {
					//cout << "MOVE DECLINED: " << "\n";
				}
        		} else {
				Sa[SE[0]] = Sa[SE[0]] * -1;//ACCEPTANCE
				hist_dat[pId_loc].prob_a[0] += 1;
				//cout << "MOVE ACCEPTED dE <0: " << "\n";
			}
			hist_dat[pId_loc].prob_a[2] += 1;
                        //cout << "HELLO FROM 5: " << l << "\n";

		}	//************END MC CYCLE*************//
	}
	free(rval_R);
	free(rval_I);
}


// ********************************************************************* //
// *****************COMPUTE WALL AND CPU TIMES************************** //
// ********[time_stamp(0 for start 1 for end, output file)]************* //
// ********************************************************************* //

string mk_time_string(double time_val)
{
	int days, hours, minutes;
	float seconds;
	
	days = floor(time_val/86400);
	hours = floor((time_val-(days*86400))/3600);
	minutes = floor((time_val - (days * 86400)-(hours * 3600))/60);
	seconds = (float)(time_val - (days * 86400) - (hours * 3600) - (minutes * 60));

        ostringstream time_str;
        time_str << days << " days " << hours  << " hours " << minutes << " minutes " << seconds  << " seconds";
        return time_str.str();
}

extern void time_stamp(int param,ofstream &file1,time_t &w1,clock_t &t,float *gpu_t)
{
        if (param == 0) { //This sets to initial clock timer
                t = clock();
                time(&w1);
                time_t now = time(0);
                char* dt = ctime(&now);
                file1 << "Start Time: " << dt << "\n";
        } else { //This writes the end clock results to output file
                time_t w2;
                double seconds,clocktime;
		double tgtime;
                clocktime = ((clock() - t)/CLOCKS_PER_SEC);
                time(&w2);
                seconds = difftime(w2,w1);

                file1 << "\n" << "|********Ending Time Stamp********|" << "\n";
                file1 << "Computation Time: " << mk_time_string(clocktime).c_str() <<  "\n";
                file1 << "GPU Real Time (Build Starting System): " << mk_time_string((double)(gpu_t[0]*0.001)).c_str() << "\n";
                file1 << "GPU Real Time (Calculating Energies): " <<mk_time_string((double)((gpu_t[1] + gpu_t[2])*0.001)).c_str() <<  "\n";
		tgtime = (gpu_t[0] + gpu_t[1] + gpu_t[2]) * 0.001;
		file1 << "Total GPU Real Time: " << mk_time_string((double)tgtime).c_str() << "\n";
		file1 << "Wall Time: "<< mk_time_string(seconds).c_str() << "\n";
                time_t now = time(0);
                char* dt = ctime(&now);
                file1 << "\n" << "End Time: " << dt << "\n";

        }
}
// ********************************************************************* //
// ***********************INPUT FILE PARSING**************************** //
// ****[input_pars(# of inputs, inputF, outputF, Fstream, [INPUTS])]**** //
// ********************************************************************* //
// LIST OF INPUTS: 1) N val
extern void input_pars(int c,char *inputf,char *outputf,ofstream &file1,double &Tmax,double &Tmin,int &Tinc,int &N, int *REMC_parm,double &convc,char *dir)
{

        double input[c];int i=0;

        ifstream myReadFile;
        myReadFile.open(inputf);

        if (myReadFile.is_open())
        {
                while (!myReadFile.eof())
                {
			if (i <= 8)
			{
                        	myReadFile >> input[i];//read input file
                        	++i;
			} else {
				myReadFile >> dir;
			}
                }
        }
        myReadFile.close();
        file1 << "|************INPUT PARAMETERS************|" <<"\n";
        file1 << "Input File: " << inputf << "\n";
        file1 << "Output File: " << outputf << "\n";
	file1 << "T Max: " << input[0] << "\n"; Tmax = (double)input[0];
	file1 << "T Min: " << input[1] << "\n"; Tmin = (double)input[1];
	file1 << "Replicas: " << input[2] << "\n"; Tinc = (int)input[2];
        file1 << "Size of Spin 2D Glass: " << input[3] << "\n"; N = (int)input[3];
        file1 << "MC cycles per energy sample: " << input[4] << "\n"; REMC_parm[0] = (int)input[4];
        file1 << "Number of REs: " << input[5] << "\n"; REMC_parm[1] = (int)input[5];
        file1 << "Energy samples per RE: " << input[6] << "\n"; REMC_parm[2]  = (int)input[6];
	file1 << "Replica Exchange Switch: " << input[7] << "\n"; REMC_parm[3]  = (int)input[7];
        file1 << "NULL SETTING: " << input[8] << "\n"; convc = (double)input[8];
	int status = mkdir(dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (status == -1)
	{
		file1 << "Existing Save Directory: " << dir << "\n\n";
	} else {
		file1 << "Created Save Directory: " << dir << "\n\n";
	}
}
// ********************************************************************* //
// **********************STREAMING OUTPUT DATA************************** //
// ********************************************************************* //
// LIST OF INPUTS: 1) N val
extern void streaming_output(ofstream &file1, double *Ea,int *Reps_Ex,histo_data *hist_dat,int Tinc,int reId,int ESs,int RepSw)
{
		file1.precision(8);	
		cout.precision(8);
		file1 << "\n";
                int sumd = 0,i,k;
		long int sum = 0;
		float testval = 6/5;
                int  sum2 = 0,sum3 = 0;
                double result;
		//Print replicas exchanged
		switch(RepSw)
		{
			case(0):
                	switch(reId)
                	{
                        	case 0:
                        	break;

                        	default:
                        	for (i = 0; i < Tinc; ++i)
                        	{
                        	        sum += Reps_Ex[i];
                        	        Reps_Ex[i] = 0;
				}
                       		break;
                	}
                        break;

                        default:
                        break;
                }

                file1 << "Replicas Exchanged: " << sum << "\n\n";
                cout << "Replicas Exchanged: " << sum << "\n";
                sumd = 0;
		//Print Average Ensemble Energy
                switch(reId)
                {
                        case 0:
                        break;

                	default:
                	for (i = 0; i < Tinc; ++i)
                	{
				sum = 0;
				file1 << "Ensemble Temp: " << hist_dat[i].Temp << "\n";
				cout << "Ensemble Temp: " << hist_dat[i].Temp << "\n";
				for (k = 0; k < reId*ESs;++k)
				{
					sum += hist_dat[i].energies[k];
				}
				file1 << "Average Ensemble Energy: " << sum / (double)(reId*ESs)  << "\n"; 
                		cout << "Average Ensemble Energy: " << sum / (double)(reId*ESs)  << "\n";
			}
                        break;
                }

		//Print Probabilities
		sumd = 0;
                for (i = 0; i < Tinc; ++i)
                {
                        sumd += (int)hist_dat[i].prob_a[0];
                        sum2 += (int)hist_dat[i].prob_a[1];
                        sum3 += (int)hist_dat[i].prob_a[2];

                        hist_dat[i].prob_a[0] = 0;
                        hist_dat[i].prob_a[1] = 0;
                        hist_dat[i].prob_a[2] = 0;
                }
		
		//cout << "SUMD: " << sumd << " SUM2: " << sum2 << " SUM3: " << sum3 << "\n";
                result = sum2 / (double)(sum3 - sumd);
                file1 << "\nProbability of move acceptance during MC cycle (dE>0): " << result << "\n";
                cout << "Probability of move acceptance during MC cycle (dE>0): " << result << "\n";
		
		result = (sum2 + sumd) / (double)sum3;
                file1 << "Probability of move acceptance during MC cycle: " << result << "\n";
		cout << "Probability of move acceptance during MC cycle: " << result << "\n";
		sumd = 0; sum3 = 0; result = 0;
                switch(RepSw)
                {
                        case(0):
                	switch(reId)
                	{
                        	case 0:
                        	break;

                        	default:
				file1 << "\nProbability of move acceptance during replica exchange: " << "\n";
               			for (i = 0; i < Tinc; ++i)
                		{
                        		sumd = (int)hist_dat[i].RE_prob_dat[0];
                        		sum3 = (int)hist_dat[i].RE_prob_dat[1];
                        		result = sumd / (double)sum3;
                        		file1 << "P(T" << i << " -> T" << (i+1) % Tinc << ")= " << result << "\n";
                        		cout << "P(T" << i << " -> T" << (i+1) % Tinc << ")= " << result << " Numer: " << sumd << " Denum " << sum3 <<"\n";
                		}
				break;
			}
			break;

			default:
			break;
		}
		file1 << "|----PRINTER COMPLETE----|" << "\n\n";
		cout << "PRINTER COMPLETE" << "\n\n";
}


