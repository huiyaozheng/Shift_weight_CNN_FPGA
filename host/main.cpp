#ifndef IOSTREAM
#define IOSTREAM
#include<iostream>
#endif
#ifndef VECTOR
#define VECTOR
#include<vector>
#endif
#include"readMnist.h"
#include"clControl.h"
#ifndef DEFINES
#define DEFINES
#include "defines.h"
#endif
int main() {
	std::cout << "Read and convert input data" << std::endl;
	std::string filename = "../model/data/t10k-images.idx3-ubyte";
	std::vector<std::vector<DTYPE>> imageData;
	read_Mnist(filename, imageData);

	filename = "../model/data/t10k-labels.idx1-ubyte";
	std::vector<char> imageLabels(NUM_OF_IMAGES);
	read_Mnist_Label(filename, imageLabels);
	
	execute_batch(imageData, imageLabels);
	return 0;
}
