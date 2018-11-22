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

#ifndef VECTOR
#define VECTOR
#include<vector>
#endif
#ifndef DEFINES
#define DEFINES
#include "defines.h"
#endif
void read_Mnist(const std::string& filename, std::vector<std::vector<DTYPE>>& vec);
void read_Mnist_Label(const std::string& filename, std::vector<char>& vec);