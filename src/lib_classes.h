#ifndef histo_data_h
#define histo_data_h

#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <string.h>
#include <fstream>

using namespace std;

// ********************************************************************* //
// ***************************DEFINE CLASSES**************************** //
// ********************************************************************* //
class histo_data
{
        public:
        double Temp;
        double *energies;
	int *replica_array;
	double *timestamp_array;
        int array_size; //size of the histogram array for storing the data
        int prob_a[3]; //holds data of accepted MC cycles for probability calculations
	int RE_prob_dat[2]; //index 0 holds numerator, index 1 holds denominator

        //Member functions
        void set_Ealloc(void); //Allocate energy data array
        string fname_histo(char *dir); //save the file names for the histogram data
	string fname_replicas(char *dir); //save the file names for the replica data
	string fname_totdat(char *dir); //save the file names for the total data
        void freeEalloc(void); //free the allocated data for energies
        void print_data(char *dir); //Printing the data in the array to the filename
        void reset_probs(void); //Reset the porbability loggers for the next MC cycle
	void set_RE_prob (void); //Resets the probability loggers for RE cycles
};
#endif
