#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <random>
#include <iostream>
#include <fstream>

using namespace std;

// ********************************************************************* //
// *************************MAIN PROGRAM******************************** //
// ********************************************************************* //

int main (int argc, char *argv[]) 
{
	cout << "Program Working...\n";
//*************DELCLARATIONS***************//
        int i=0;
	double sum=0,sumd=0,result;
	char *inputf = argv[1];
//*********END DECLARATIONS****************//

//***********MAIN SEQUENCE CODE*************//
        double input;

        ifstream myReadFile;
        myReadFile.open(inputf);

        if (myReadFile.is_open())
        {
                while (!myReadFile.eof())
                {
                        myReadFile >> input;//read input file
			sum += input;
			sumd += 1;
                        ++i;
                }
        }
        myReadFile.close();
//**********END MAIN SEQUENCE CODE**********//

//**********COMPILE AND PRINT RESULTS*******//
	result = sum / sumd;
        cout << "|************Results************|" << "\n";
        cout.precision(10);
	cout << "Read File: " << inputf << "\n";
	cout << "Average: " << result << "\n";
        cout << "\n\n";
//******END COMPILE AND PRINT RESULTS*******//
}

