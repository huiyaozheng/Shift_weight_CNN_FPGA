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

#define NUM_OF_IMAGES 10000
#define DTYPE short

#define CONV_1_CHANNELS 10
#define CONV_1_IN_ROWS 28
#define CONV_1_IN_COLS 28
#define CONV_1_FILTER_SIZE 5  // assume all filters are square
#define CONV_1_STRIDE 1       // strides not used in code 
#define CONV_1_OUT_ROWS 24
#define CONV_1_OUT_COLS 24
#define CONV_1_VECTOR_LENGTH ((CONV_1_FILTER_SIZE - 1) * CONV_1_IN_COLS + CONV_1_FILTER_SIZE) // 117
#define CONV_1_OUT_LENGTH ((CONV_1_IN_ROWS - CONV_1_FILTER_SIZE + 1) * (CONV_1_IN_COLS - CONV_1_FILTER_SIZE + 1)) // 576

#define POOL_1_IN_ROWS 24
#define POOL_1_IN_COLS 24
#define POOL_1_FILTER_SIZE 2
#define POOL_1_OUT_LENGTH ((POOL_1_IN_ROWS / POOL_1_FILTER_SIZE) * (POOL_1_IN_COLS / POOL_1_FILTER_SIZE)) // 144

#define CONV_2_CHANNELS 20
#define CONV_2_IN_ROWS 12
#define CONV_2_IN_COLS 12
#define CONV_2_FILTER_SIZE 5
#define CONV_2_STRIDE 1
#define CONV_2_OUT_ROWS 8
#define CONV_2_OUT_COLS 8
#define CONV_2_VECTOR_LENGTH ((CONV_2_FILTER_SIZE - 1) * CONV_2_IN_COLS * CONV_1_CHANNELS + CONV_2_FILTER_SIZE * CONV_1_CHANNELS) // 530
#define CONV_2_OUT_LENGTH ((CONV_2_IN_ROWS - CONV_2_FILTER_SIZE + 1) * (CONV_2_IN_COLS - CONV_2_FILTER_SIZE + 1)) // 64

#define POOL_2_IN_ROWS 8
#define POOL_2_IN_COLS 8
#define POOL_2_FILTER_SIZE 2
#define POOL_2_OUT_LENGTH ((POOL_2_IN_ROWS / POOL_2_FILTER_SIZE) * (POOL_2_IN_COLS / POOL_2_FILTER_SIZE)) //16

#define FC_1_IN_ROWS 4
#define FC_1_IN_COLS 4
#define FC_1_IN_CHANNELS 20
#define FC_1_CHANNELS 50

#define FC_2_INPUT_SIZE 50
#define FC_2_OUTPUT_SIZE 10


