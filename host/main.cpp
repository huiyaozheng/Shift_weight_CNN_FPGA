// Copyright (C) 2018 Huiyao Zheng
// 
// This file is part of Shift_weight_CNN_FPGA.
// 
// Shift_weight_CNN_FPGA is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// Shift_weight_CNN_FPGA is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with Shift_weight_CNN_FPGA.  If not, see <http://www.gnu.org/licenses/>.

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
