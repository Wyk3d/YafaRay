#include <cassert>
#include <cmath>
#include <cstring>

#include <CL/cl.h>

#define KERNEL_SOURCE \
        "__kernel void vectorAdd(int n, __global const float* a, __global const float* b, __global float* c)\n" \
        "{\n"                                                           \
        "    int gid = get_global_id(0);\n"                             \
        "\n"                                                            \
        "    if (gid >= n)\n"                                           \
        "        return; \n"                                            \
        "\n"                                                            \
        "    c[gid] = a[gid] + b[gid];\n"                               \
        "}\n"

#define N 1024
#define LOCAL_WORK_SIZE 256

int
main(void)
{
        const char* source = KERNEL_SOURCE;
        const int n = N;
        size_t size;
        cl_int error;

        float* a = new float[n];
        for (int i = 0; i < n; ++i)
                a[i] = i;

        float* b = new float[n];
        for (int i = 0; i < n; ++i)
                b[i] = n - i;

        float* c = new float[n];

        cl_context context = clCreateContextFromType(0, CL_DEVICE_TYPE_CPU, 0, 0, &error);
        assert(error == CL_SUCCESS);

        error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &size);
        assert(error == CL_SUCCESS);
        char* buffer = new char[size];
        cl_device_id* devices = reinterpret_cast<cl_device_id*>(buffer);
        error = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, 0);
        assert(error == CL_SUCCESS);
        
        cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &error);
        assert(error == CL_SUCCESS);

        cl_mem ad = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(cl_float), 0, &error);
        assert(error == CL_SUCCESS);
        cl_mem bd = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(cl_float), 0, &error);
        assert(error == CL_SUCCESS);
        cl_mem cd = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(cl_float), 0, &error);
        assert(error == CL_SUCCESS);
       
        size = strlen(source);
        cl_program program = clCreateProgramWithSource(context, 1, &source, &size, &error);
        assert(error == CL_SUCCESS);
        error = clBuildProgram(program, 0, 0, 0, 0, 0);
        assert(error == CL_SUCCESS);
        
        cl_kernel kernel = clCreateKernel(program, "vectorAdd", &error);
        assert(error == CL_SUCCESS);

        error = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &n);
        assert(error == CL_SUCCESS);
        error = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &ad);
        assert(error == CL_SUCCESS);
        error = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &bd);
        assert(error == CL_SUCCESS);
        error = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &cd);
        assert(error == CL_SUCCESS);

        error = clEnqueueWriteBuffer(queue, ad, CL_FALSE, 0, n * sizeof(cl_float), a, 0, 0, 0);
        assert(error == CL_SUCCESS);
        error = clEnqueueWriteBuffer(queue, bd, CL_FALSE, 0, n * sizeof(cl_float), b, 0, 0, 0);
        assert(error == CL_SUCCESS);

        size_t local_work_size = LOCAL_WORK_SIZE;
        size_t global_work_size = (size_t) ceil((double) n / local_work_size) * local_work_size;
        error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global_work_size, &local_work_size, 0, 0, 0);
        assert(error == CL_SUCCESS);

        error = clEnqueueReadBuffer(queue, cd, CL_TRUE, 0, n * sizeof(cl_float), c, 0, 0, 0);
        assert(error == CL_SUCCESS);

        for (int i = 0; i < n; ++i)
                assert(c[i] == n);

        delete [] a;
        delete [] b;
        delete [] c;
        delete [] buffer;

        error = clReleaseMemObject(ad);
        assert(error == CL_SUCCESS);
        error = clReleaseMemObject(bd);
        assert(error == CL_SUCCESS);
        error = clReleaseMemObject(cd);
        assert(error == CL_SUCCESS);
        error = clReleaseKernel(kernel);
        assert(error == CL_SUCCESS);
        error = clReleaseProgram(program);
        assert(error == CL_SUCCESS);
        error = clReleaseCommandQueue(queue);
        assert(error == CL_SUCCESS);
        error = clReleaseContext(context);
        assert(error == CL_SUCCESS);
}
