#include <core_api/environment.h>
#include <opencl_wrapper/cl_util.h>

CLProgram * buildCLProgram(const char *kernel_source, CLContext *context, CLDevice device, const char *options) {
	CLError err;
	CLProgram *program = context->createProgram(kernel_source, &err);
	checkErr(err || program == NULL, "failed to create program");

	std::cout << "building program ...";
	CLError build_error;

	clock_t start = clock();
	program->build(&build_error, options);
	clock_t end = clock();
	std::cout << "[" << (float)(end-start)/(float)CLOCKS_PER_SEC << "s]: ";

	std::string log = program->getBuildLog(device, &err);

	if(build_error) std::cout << "FAILED: " << build_error.getString() << " ("
		<< build_error.getCode() << ")" << std::endl;
	else std::cout << "SUCCESS." << std::endl;

	checkErr(err, "failed to get build log");
	if(log != "") std::cout << log << std::endl;
	if(build_error)
		exit(EXIT_FAILURE);

	return program;
}

CLApplication::CLApplication()
{
	CLError err;
	CLMain cl;
	std::list<CLPlatform> platforms = cl.getPlatforms(&err);
	checkErr(err || platforms.empty(), "failed to find platforms\n");
	bool first = true;
	for(std::list<CLPlatform>::iterator p_itr = platforms.begin(); p_itr != platforms.end(); ++p_itr)
	{
		std::list<CLDevice> devices = p_itr->getDevices(&err);
		checkErr(err || devices.empty(), "failed to find devices for the chosen platform");
		for(std::list<CLDevice>::iterator d_itr = devices.begin(); d_itr != devices.end(); ++d_itr) {
			if(first || d_itr->getType() == CL_DEVICE_TYPE_GPU) {
				platform = *p_itr;
				device = *d_itr;
			}
			first = false;
		}
	}

	context = platform.createContext(device, &err);
	checkErr(err || context == NULL, "failed to get a context for the chosen device\n");

	queue = context->createCommandQueue(device, &err);
	checkErr(err || queue == NULL, "failed to create command queue");

	cl_uint vendor_id = device.getVendorId(&err);
	checkErr(err, "failed to get device vendor");

	if(vendor_id == CL_VENDOR_NVIDIA) {
		cl_build_options += " -cl-nv-verbose";
	} else if(vendor_id == CL_VENDOR_AMD) {
		putenv("GPU_DUMP_DEVICE_KERNEL=3");
	}
}

CLApplication::~CLApplication()
{
	CLError err;
	if(queue) {
		queue->free(&err);
		checkErr(err, "failed to free queue");
	}
	if(context) {
		context->free(&err);
		checkErr(err, "failed to free context");
	}
}