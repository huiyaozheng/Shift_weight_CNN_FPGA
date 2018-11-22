#include <CL/opencl.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "defines.h"
#include<cassert>
#ifndef FSTREAM
#define FSTREAM
#include<fstream>
#endif
#ifndef IOSTREAM
#define IOSTREAM
#include<iostream>
#endif
#ifndef VECTOR
#define VECTOR
#include<vector>
#endif
#include"clControl.h"
#include <CL/cl.hpp>
#ifndef BITSET
#define BITSET
#include<bitset>
#endif
#include <malloc.h>
#define AOCL_ALIGNMENT 64

void checkStatus(cl_int status, const char* where) {
	if (status != CL_SUCCESS) {
		std::cout << "Error code: " << status << " at " << where << std::endl;
	}
}

void cleanup(DTYPE* ptrArray) {
	_aligned_free(ptrArray);
}

void execute_batch(std::vector<std::vector<DTYPE>> data, std::vector<char> labels) {
	cl_int err;
	// Get platform and device
	// Assume default
	cl_platform_id pid;
	cl_int status = clGetPlatformIDs(1, &pid, nullptr);
	checkStatus(status, "get platform");
	cl_device_id did;
	status = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 1, &did, nullptr);
	checkStatus(status, "get device");
	cl_context context = clCreateContext(nullptr, 1, &did, nullptr, nullptr, &status);
	checkStatus(status, "create context");

	// Load FPGA image
	const char* fileName = "mnist.aocx";
	FILE* fp;
	fopen_s(&fp, fileName, "rb");
	// Get the size of the file
	fseek(fp, 0, SEEK_END);
	size_t binarySize = ftell(fp);
	// Allocate space for the binary
	const unsigned char* binary = new unsigned char[binarySize];
	// Go back to the file start
	rewind(fp);
	// Read the file into the binary
	if (fread((void*)binary, binarySize, 1, fp) == 0) {
		delete[] binary;
		fclose(fp);
		return;
	}

	// Create program
	cl_program program = clCreateProgramWithBinary(context, 1, &did, &binarySize, &binary, &err, &status);
	checkStatus(status, "create program");
	status = clBuildProgram(program, 0, nullptr, "", nullptr, nullptr);
	checkStatus(status, "build program");

	cl_command_queue queue0 = clCreateCommandQueue(context, did, NULL, &status);
	checkStatus(status, "create queue 0");
	cl_command_queue queue1 = clCreateCommandQueue(context, did, NULL, &status);
	checkStatus(status, "create queue 1");
	cl_command_queue queue2 = clCreateCommandQueue(context, did, NULL, &status);
	checkStatus(status, "create queue 2");
	cl_command_queue queue3 = clCreateCommandQueue(context, did, NULL, &status);
	checkStatus(status, "create queue 3");
	cl_command_queue queue4 = clCreateCommandQueue(context, did, NULL, &status);
	checkStatus(status, "create queue 4");
	cl_command_queue queue5 = clCreateCommandQueue(context, did, NULL, &status);
	checkStatus(status, "create queue 5");
	cl_command_queue queue6 = clCreateCommandQueue(context, did, NULL, &status);
	checkStatus(status, "create queue 6");

	cl_kernel input_kernel = clCreateKernel(program, "data0", &status);
	checkStatus(status, "create kernel input");
	cl_kernel conv_1_kernel = clCreateKernel(program, "conv1", &status);
	checkStatus(status, "create kernel 1");
	cl_kernel conv_2_kernel = clCreateKernel(program, "conv2", &status);
	checkStatus(status, "create kernel 3");
	cl_kernel pool_1_kernel = clCreateKernel(program, "pool_relu1", &status);
	checkStatus(status, "create kernel 2");
	cl_kernel pool_2_kernel = clCreateKernel(program, "pool_relu2", &status);
	checkStatus(status, "create kernel 4");
	cl_kernel fc_1_kernel = clCreateKernel(program, "fc1", &status);
	checkStatus(status, "create kernel 5");
	cl_kernel fc_2_kernel = clCreateKernel(program, "fc2", &status);
	checkStatus(status, "create kernel 6");

	cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
	                                    sizeof(DTYPE) * CONV_1_IN_COLS * CONV_1_IN_ROWS * NUM_OF_IMAGES, nullptr,
	                                    &status);
	checkStatus(status, "create input buffer");
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * FC_2_OUTPUT_SIZE * NUM_OF_IMAGES,
	                                     nullptr, &status);
	checkStatus(status, "create output buffer");

	status = clSetKernelArg(input_kernel, 0, sizeof(cl_mem), &inputBuffer);
	checkStatus(status, "set input buffer");
	status = clSetKernelArg(fc_2_kernel, 0, sizeof(cl_mem), &outputBuffer);
	checkStatus(status, "set output buffer");

	// Use aligned malloc to allow DMA
	DTYPE * ptrs = static_cast<DTYPE*>(_aligned_malloc(sizeof(DTYPE) * CONV_1_IN_COLS * CONV_1_IN_ROWS * NUM_OF_IMAGES,
	                                              AOCL_ALIGNMENT));
	if (ptrs == nullptr) {
		cleanup(ptrs);
		return;
	}
	for (int j = 0; j < NUM_OF_IMAGES; ++j) {
		for (int i = 0; i < CONV_1_IN_COLS * CONV_1_IN_ROWS; ++i) {
			ptrs[j * CONV_1_IN_COLS * CONV_1_IN_ROWS + i] = data[j][i];
		}
	}

	status = clEnqueueWriteBuffer(queue0, inputBuffer, CL_TRUE, 0,
	                              sizeof(DTYPE) * CONV_1_IN_COLS * CONV_1_IN_ROWS * NUM_OF_IMAGES, ptrs, 0, nullptr,
	                              nullptr);
	checkStatus(status, "write input data");

	std::cout << "\nLaunching...\n";

	cl_event start, finish;

	status = clEnqueueTask(queue0, input_kernel, 0, nullptr, nullptr);
	checkStatus(status, "enqueue kernel 1");
	status = clEnqueueTask(queue1, conv_1_kernel, 0, nullptr, &start);
	checkStatus(status, "enqueue kernel 1");
	status = clEnqueueTask(queue2, pool_1_kernel, 0, nullptr, nullptr);
	checkStatus(status, "enqueue kernel 2");
	status = clEnqueueTask(queue3, conv_2_kernel, 0, nullptr, nullptr);
	checkStatus(status, "enqueue kernel 3");
	status = clEnqueueTask(queue4, pool_2_kernel, 0, nullptr, nullptr);
	checkStatus(status, "enqueue kernel 4");
	status = clEnqueueTask(queue5, fc_1_kernel, 0, nullptr, nullptr);
	checkStatus(status, "enqueue kernel 5");
	status = clEnqueueTask(queue6, fc_2_kernel, 0, nullptr, &finish);
	checkStatus(status, "enqueue kernel 6");

	int result[FC_2_OUTPUT_SIZE * NUM_OF_IMAGES];
	status = clEnqueueReadBuffer(queue6, outputBuffer, CL_TRUE, 0, sizeof(int) * FC_2_OUTPUT_SIZE * NUM_OF_IMAGES,
	                             result, 0, nullptr,
	                             nullptr);
	checkStatus(status, "read output data");
	std::cout << "\nFinished...\n";

	cl_ulong startNanoSecond, finishNanoSecond;
	clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startNanoSecond, nullptr);
	clGetEventProfilingInfo(finish, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &finishNanoSecond, nullptr);
	std::cout << (finishNanoSecond - startNanoSecond) / 1e9f << std::endl;

	// Check correctness
	int correct = 0;
	for (int j = 0; j < NUM_OF_IMAGES; ++j) {
		int max = result[j * FC_2_OUTPUT_SIZE];
		int max_digit = 0;
		for (int i = 0; i < FC_2_OUTPUT_SIZE; ++i) {
			if (result[i + j * FC_2_OUTPUT_SIZE] > max) {
				max = result[i + j * FC_2_OUTPUT_SIZE];
				max_digit = i;
			}
		}
		if (max_digit == labels[j]) correct++;
	}
	std::cout << 100.0 * correct / NUM_OF_IMAGES;

	// Clean up
	clReleaseKernel(input_kernel);
	clReleaseKernel(conv_1_kernel);
	clReleaseKernel(pool_1_kernel);
	clReleaseKernel(conv_2_kernel);
	clReleaseKernel(pool_2_kernel);
	clReleaseKernel(fc_1_kernel);
	clReleaseKernel(fc_2_kernel);
	clReleaseCommandQueue(queue0);
	clReleaseCommandQueue(queue1);
	clReleaseCommandQueue(queue2);
	clReleaseCommandQueue(queue3);
	clReleaseCommandQueue(queue4);
	clReleaseCommandQueue(queue5);
	clReleaseCommandQueue(queue6);
	clReleaseMemObject(inputBuffer);
	clReleaseMemObject(outputBuffer);
	clReleaseProgram(program);
	clReleaseContext(context);
	cleanup(ptrs);
}
