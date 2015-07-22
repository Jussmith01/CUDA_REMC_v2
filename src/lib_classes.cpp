#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <sstream>
#include <string>
#include <string.h>
#include <fstream>
#include "lib_classes.h"

using namespace std;

void histo_data::reset_probs ()
{
        prob_a[0] = 0;
        prob_a[1] = 0;
        prob_a[2] = 0;
}

void histo_data::set_RE_prob ()
{
        RE_prob_dat[0] = 0;
        RE_prob_dat[1] = 1;
}

void histo_data::set_Ealloc ()
{
        energies = (double *)malloc(array_size * sizeof(double));
	replica_array = (int *)malloc(array_size * sizeof(int));
	timestamp_array = (double *)malloc(array_size * sizeof(double));
}

void histo_data::freeEalloc ()
{
        free(energies);
	free(replica_array);
	free(timestamp_array);
}

string histo_data::fname_histo (char *dir)
{
                string basename = "HistoData-", endname = "K.dat";
                ostringstream fullname;
                fullname << dir << basename << Temp << endname;
                //cout << fullname.str() << " - " << fullname.str().size() << "\n";
                return fullname.str();
}

string histo_data::fname_replicas (char *dir)
{
                string basename = "ReplicaData-", endname = "K.dat";
                ostringstream fullname;
                fullname << dir << basename << Temp << endname;
                //cout << fullname.str() << " - " << fullname.str().size() << "\n";
                return fullname.str();
}

string histo_data::fname_totdat (char *dir)
{
                string basename = "TotalData-", endname = "K.dat";
                ostringstream fullname;
                fullname << dir << basename << Temp << endname;
                //cout << fullname.str() << " - " << fullname.str().size() << "\n";
                return fullname.str();
}

void histo_data::print_data (char *dir)
{
                int i;
		//Save Histogram Data
                ofstream outputfile;
                outputfile.open(fname_histo(dir).c_str());
                for (i = 0; i < array_size; ++i)
                {
                      outputfile <<  energies[i]  << "\n";
                }
                outputfile.close();
                outputfile.clear();

		//Save Replica Data
                ofstream outputfile2;
                outputfile2.open(fname_replicas(dir).c_str());
                for (i = 0; i < array_size; ++i)
                {
                      outputfile2 <<  replica_array[i]  << "\n";
                }
                outputfile2.close();
                outputfile2.clear();

                //Total Data Out Put
                ofstream outputfile3;
		outputfile3.precision(15);
                outputfile3.open(fname_totdat(dir).c_str());
                for (i = 0; i < array_size; ++i)
                {
                      outputfile3 << "Sample(" << i << ") Time: " << timestamp_array[i] << "s Replica: " << replica_array[i] << " Energy: " << energies[i] << "\n";
                }
                outputfile3.close();
                outputfile3.clear();

}

