#include <yafray_config.h>

#include "cl_util.h"
#include <time.h>
#include <math.h>

const char *kernel_source = CL_SRC(
  __kernel void vectorAdd(int n,
                          __global const float* a,
                          __global const float* b,
                          __global float* c)
  {
    int gid = get_global_id(0);
    if (gid >= n)
      return;
    c[gid] = a[gid] + b[gid];
  }
);

/*const char *kernel_source = CL_SRC(
  #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
  __constant char hw[] = "Hello World\n";
  __kernel void hello(__global char * out)
  {
    size_t tid = get_global_id(0);
    out[tid] = hw[tid];
  }
);*/



#define N 1024
#define LOCAL_WORK_SIZE 256

#include <iostream>

void printPlatformInfo(CLPlatform platform) {
	std::cout << "Platform " << platform.getName() << " (" << platform.getId() << ")" << std::endl;
	std::cout << "vendor: " << platform.getVendor() << std::endl;
	std::cout << "version: " << platform.getVersion() << std::endl;
	std::cout << "profile: " << platform.getProfile() << std::endl;
	std::cout << "extensions: " << platform.getExtensions() << std::endl;
	std::cout << std::endl;
}

void printDeviceInfo(CLDevice device) {
	CLError err;
	std::cout << "Device " << device.getName(&err) << " (" << device.getId() << ")" << std::endl;
	checkErr(err);
	std::cout << "type: " << device.getType(&err) << std::endl;
	checkErr(err);
	std::cout << "vendor: " << device.getVendor(&err) << " (" << device.getVendorId(&err) << ")" << std::endl;
	checkErr(err);
	std::cout << "version: " << device.getVersion(&err) << std::endl;
	checkErr(err);
	std::cout << "driver version: " << device.getDriverVersion(&err) << std::endl;
	checkErr(err);
	std::cout << "profile: " << device.getProfile(&err) << std::endl;
	checkErr(err);
	std::cout << "extensions: " << device.getExtensions(&err) << std::endl;
	checkErr(err);
	std::cout << "max compute units: " << device.getMaxComputeUnits(&err) << std::endl;
	checkErr(err);
	int dim =  device.getMaxWorkItemDimensions(&err);
	checkErr(err);
	std::cout << "max work item dimensions: " << dim << std::endl;
	checkErr(err);
	std::cout << "max work item sizes: " << std::endl;
	checkErr(err);
	std::list<size_t> sizes = device.getMaxWorkItemSizes(&err);
	checkErr(err);
	int i = 0;
	for(std::list<size_t>::iterator itr = sizes.begin(); itr != sizes.end(); ++itr)
		std::cout << " " << i++ << " - " << *itr << std::endl;
	std::cout << "max work group size: " << device.getMaxWorkGroupSize(&err) << std::endl;
	checkErr(err);
	std::cout << "preferred char vector width: " << device.getPreferredVectorWidthChar(&err) << std::endl;
	checkErr(err);
	std::cout << "preferred short vector width: " << device.getPreferredVectorWidthShort(&err) << std::endl;
	checkErr(err);
	std::cout << "preferred int vector width: " << device.getPreferredVectorWidthInt(&err) << std::endl;
	checkErr(err);
	std::cout << "preferred long vector width: " << device.getPreferredVectorWidthLong(&err) << std::endl;
	checkErr(err);
	std::cout << "preferred float vector width: " << device.getPreferredVectorWidthFloat(&err) << std::endl;
	checkErr(err);
	std::cout << "preferred double vector width: " << device.getPreferredVectorWidthDouble(&err) << std::endl;
	checkErr(err);
	std::cout << "max clock frequency: " << device.getMaxClockFrequency(&err) << std::endl;
	checkErr(err);
	std::cout << "address bits: " << device.getAddressBits(&err) << std::endl;
	checkErr(err);
	std::cout << "max mem alloc size: " << device.getMaxMemAllocSize(&err) << " bytes" << std::endl;
	checkErr(err);
	std::cout << "supports images: " << (device.hasImageSupport(&err) ? "yes" : "no") << std::endl;
	checkErr(err);
	if(device.hasImageSupport(&err)) {
		std::cout << "max read image args: " << device.getMaxReadImageArgs(&err) << std::endl;
		checkErr(err);
		std::cout << "max write image args: " << device.getMaxWriteImageArgs(&err) << std::endl;
		checkErr(err);
		std::cout << "image 2d width: " << device.getImage2DMaxWidth(&err) << std::endl;
		checkErr(err);
		std::cout << "image 2d height: " << device.getImage2DMaxHeight(&err) << std::endl;
		checkErr(err);
		std::cout << "image 3d width: " << device.getImage3DMaxWidth(&err) << std::endl;
		checkErr(err);
		std::cout << "image 3d height: " << device.getImage3DMaxHeight(&err) << std::endl;
		checkErr(err);
		std::cout << "image 3d depth: " << device.getImage3DMaxDepth(&err) << std::endl;
		checkErr(err);
	}
	std::cout << "max samplers: " << device.getMaxSamplers(&err) << std::endl;
	checkErr(err);
	std::cout << "max parameter size: " << device.getMaxParameterSize(&err) << " bytes" << std::endl;
	checkErr(err);
	std::cout << "mem base address align: " << device.getMemBaseAddrAlign(&err) << " bits" << std::endl;
	checkErr(err);
	std::cout << "single precision FP properties: " << std::endl;
	cl_device_fp_config fp_config = device.getSingleFPConfig(&err);
	checkErr(err);
	if(fp_config & CL_FP_DENORM)
		std::cout << " denorms supported" << std::endl;
	if(fp_config & CL_FP_INF_NAN)
		std::cout << " INF and quiet NANs are supported" << std::endl;
	if(fp_config & CL_FP_ROUND_TO_NEAREST)
		std::cout << " round to nearest even rounding mode supported" << std::endl;
	if(fp_config & CL_FP_ROUND_TO_ZERO)
		std::cout << " round to zero rounding mode supported" << std::endl;
	if(fp_config & CL_FP_ROUND_TO_INF)
		std::cout << " round to +ve and -ve infinity rounding modes supported" << std::endl;
	if(fp_config & CL_FP_FMA)
		std::cout << " IEEE754-2008 fused multiply-add is supported" << std::endl;
	std::cout << "global cache type: ";
	cl_device_mem_cache_type cache_type = device.getGlobalMemCacheType(&err);
	checkErr(err);
	switch(cache_type) {
		case CL_NONE:
			std::cout << "none" << std::endl;
			break;
		case CL_READ_ONLY_CACHE:
			std::cout << "read only" << std::endl;
			break;
		case CL_READ_WRITE_CACHE:
			std::cout << "read write" << std::endl;
			break;
	}
	if(cache_type != CL_NONE) {
		 std::cout << "global cacheline size: " << device.getGlobalMemCacheLineSize(&err) << " bytes" << std::endl;
		 checkErr(err);
		 std::cout << "global cache size: " << device.getGlobalMemCacheSize(&err) << " bytes" << std::endl;
		 checkErr(err);
	}
	std::cout << "global memory size: " << device.getGlobalMemSize(&err) << " bytes" << std::endl;
	checkErr(err);
	std::cout << "max constant buffer size: " << device.getMaxConstantBufferSize(&err) << " bytes" << std::endl;
	checkErr(err);
	std::cout << "max constant args: " << device.getMaxConstantArgs(&err) << std::endl;
	checkErr(err);
	cl_device_local_mem_type local_mem_type = device.getLocalMemType(&err);
	checkErr(err);
	std::cout << "local mem type: ";
	switch(local_mem_type) {
		case CL_LOCAL:
			std::cout << "local" << std::endl;
			break;
		case CL_GLOBAL:
			std::cout << "global" << std::endl;
			break;
	}
	std::cout << "local mem size: " << device.getLocalMemSize(&err) << " bytes" << std::endl;
	checkErr(err);
	std::cout << "supports error correction: " << (device.hasErrorCorrectionSupport(&err) ? "yes" : "no") << std::endl;
	checkErr(err);
	std::cout << "profiling timer resolution: " << device.getType(&err) << " ns" << std::endl;
	checkErr(err);
	std::cout << "little endian: " << (device.isLittleEndian(&err) ? "yes" : "no") << std::endl;
	checkErr(err);
	std::cout << "compiler available: " << (device.hasCompiler(&err) ? "yes" : "no") << std::endl;
	checkErr(err);
	std::cout << "execution capabilities: " << std::endl;
	cl_device_exec_capabilities exec_cap = device.getExecutionCapabilities(&err);
	checkErr(err);
	if(exec_cap & CL_EXEC_KERNEL)
		std::cout << " can execute OpenCL kernels" << std::endl;
	if(exec_cap & CL_EXEC_NATIVE_KERNEL)
		std::cout << " can execute native kernels" << std::endl;
	std::cout << "command queue properties: " << std::endl;
	cl_command_queue_properties queue_props = device.getQueueProperties(&err);
	checkErr(err);
	if(queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
		std::cout << " supports out of order execution";
	if(queue_props & CL_QUEUE_PROFILING_ENABLE)
		std::cout << " supports profiling";

	std::cout << std::endl;
}

int
main(void)
{
        //const char* source = kernel_source;
        const int n = N;


        float* a = new float[n];
        for (int i = 0; i < n; ++i)
                a[i] = i;

        float* b = new float[n];
        for (int i = 0; i < n; ++i)
                b[i] = n - i;

        float* c = new float[n];

		CLMain cl;
		CLError err;

		std::list<CLPlatform> platforms = cl.getPlatforms(&err);
		checkErr(err || platforms.empty(), "failed to find platforms\n");
		CLPlatform platform = *platforms.begin();

		for(std::list<CLPlatform>::iterator itr = platforms.begin(); itr != platforms.end(); ++itr)
			printPlatformInfo(*itr);

		std::list<CLDevice> devices = platform.getDevices(&err);
		checkErr(err || devices.empty(), "failed to find devices for the chosen platform");
		CLDevice device = *devices.begin();

		for(std::list<CLDevice>::iterator itr = devices.begin(); itr != devices.end(); ++itr) {
			// if there are is more than one device to choose from, override with a GPU
			if(itr->getType() == CL_DEVICE_TYPE_GPU)
				device = *itr;
			printDeviceInfo(*itr);
		}

		CLContext* context;
		context = platform.createContext(device, &err);
		checkErr(err || context == NULL, "failed to get a context for the chosen device\n");

		CLCommandQueue *queue = context->createCommandQueue(device, &err);
		checkErr(err || queue == NULL, "failed to create command queue");

		CLBuffer *buf_a = context->createBuffer(CL_MEM_READ_ONLY, n * sizeof(cl_float), NULL, &err);
		checkErr(err || buf_a == NULL, "failed to create buffer");
		CLBuffer *buf_b = context->createBuffer(CL_MEM_READ_ONLY, n * sizeof(cl_float), NULL, &err);
		checkErr(err || buf_b == NULL, "failed to create buffer");
		CLBuffer *buf_c = context->createBuffer(CL_MEM_WRITE_ONLY, n * sizeof(cl_float), NULL, &err);
		checkErr(err || buf_c == NULL, "failed to create buffer");

		CLProgram *program = buildCLProgram(kernel_source, context, device);

		CLKernel *kernel = program->createKernel("vectorAdd", &err);
		checkErr(err || !kernel, "failed to create kernel");

		kernel->setArgs(n, buf_a, buf_b, buf_c, &err);
		checkErr(err, "failed to set args");

		queue->writeBuffer(buf_a, 0, n*sizeof(cl_float), a, &err);
		checkErr(err, "failed to write buf a");
		queue->writeBuffer(buf_b, 0, n*sizeof(cl_float), b, &err);
		checkErr(err, "failed to write buf b");

		size_t local_work_size = LOCAL_WORK_SIZE;
		size_t global_work_size = (size_t) ceil((double) n / local_work_size) * local_work_size;

		queue->runKernel(kernel, Range1D(global_work_size, local_work_size), &err);
		checkErr(err, "failed to run kernel");

		queue->readBuffer(buf_c, 0, n*sizeof(cl_float), c, &err);
		checkErr(err, "failed to read buf c");

		for (int i = 0; i < n; ++i)
			checkErr(c[i] == n, "result not computed correctly");

		delete [] a;
		delete [] b;
		delete [] c;

		kernel->free(&err);
		checkErr(err, "failed to release kernel");
		buf_c->free(&err);
		checkErr(err, "failed to release buffer c");
		buf_b->free(&err);
		checkErr(err, "failed to release buffer b");
		buf_a->free(&err);
		checkErr(err, "failed to release buffer a");
		queue->free(&err);
		checkErr(err, "failed to release queue");
		context->free(&err);
		checkErr(err, "failed to release context");
}
