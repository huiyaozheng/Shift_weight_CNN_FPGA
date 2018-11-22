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