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

#ifndef FSTREAM
#define FSTREAM
#include<fstream>
#endif
#include"readMnist.h"
#include<cmath>
#include<string>
#ifndef BITSET
#define BITSET
#include<bitset>
#endif
#ifndef IOSTREAM
#define IOSTREAM
#include<iostream>
#endif
#ifndef MAP
#define MAP
#include<unordered_map>
#endif

std::unordered_map<unsigned char, DTYPE> conversionTable;

std::string convertToFixedPoint(float value, int int_bits, int frac_bits) {
	/* 
	    Convert a floating point value into a (1+int_bits+frac_bits)-bit
	    fixed point number.
	*/
	const int word_bits = int_bits + frac_bits;
	bool is_neg = false;
	if (value < 0) {
		is_neg = true;
		value = -value;
	}

	// truncate the value if it exceeds the upper bound of representation
	const int max_exp = static_cast<int>(floor(sqrt(value)));
	for (int i = max_exp; i > int_bits - 1; --i) {
		if (value > pow(2, i)) { value -= pow(2, i); }
	}

	// truncate the value if it exceeds the lower bound of representation
	const float min_exp = pow(2, -frac_bits);
	if (value <= min_exp) {
		std::string fixed_point;
		for (int i = 0; i < word_bits + 1; ++i) {
			fixed_point.insert(0, "0");
		}
		return fixed_point;
	}

	// convert the value into string
	std::string fixed_point;
	for (int i = 0; i < word_bits; ++i) {
		const float stride = pow(2, word_bits - frac_bits - i - 1);
		if (value >= stride) {
			value = value - stride;
			fixed_point.append("1");
		}
		else {
			fixed_point.append("0");
		}
	}

	// apply 2's complement if the value is negative
	if (is_neg) {
		const int MAX_LENGTH = 128;
		const int v = std::stoi(fixed_point, nullptr, 2) - (1 << (word_bits + 1));
		fixed_point = std::bitset<MAX_LENGTH>(~v + 1).to_string().substr(MAX_LENGTH - (word_bits + 1), word_bits + 1);
	}
	else {
		fixed_point.insert(0, "0");
	}
	return fixed_point;
}

// Generate a look-up table to speed up the conversion process
void preprocess() {
	const float mean = 0.1307;
	const float var = 0.3081;
	for (int i = 0; i <= 255; ++i) {
		DTYPE converted = static_cast<DTYPE>(std::stoi(convertToFixedPoint(((i / 255.0) - mean) / var, 5, 10), nullptr, 2));
		conversionTable[i] = converted;
	}
}

void read_Mnist(const std::string& filename, std::vector<std::vector<DTYPE>>& vec) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
		file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number)); // number of images
		file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number)); // number of rows
		file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number)); // number of cols

		const int n_rows = 28;
		const int n_cols = 28;

		preprocess();

		const float mean = 0.1307;
		const float var = 0.3081;
		for (int i = 0; i < NUM_OF_IMAGES; ++i) {
			std::vector<DTYPE> tp;
			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					unsigned char temp = 0;
					file.read(reinterpret_cast<char *>(&temp), sizeof(temp));
					tp.push_back(conversionTable[temp]);
				}
			}
			vec.push_back(tp);
		}
	}
}

void read_Mnist_Label(const std::string& filename, std::vector<char>& vec) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
		file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number)); // number of images
		for (int i = 0; i < NUM_OF_IMAGES; ++i) {
			unsigned char temp = 0;
			file.read(reinterpret_cast<char *>(&temp), sizeof(temp));
			vec[i] = temp;
		}
	}
}
