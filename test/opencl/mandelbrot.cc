#include <core_api/environment.h>
#include <core_api/imagehandler.h>
#include <string>
#include <time.h>
#include <stdio.h>
#include "cl_util.h"

using namespace yafaray;

renderEnvironment_t *env;
imageHandler_t *handler;

unsigned int w = 256;
unsigned int h = 192;
//unsigned int w = 512;
//unsigned int h = 384;
//unsigned int w = 1024;
//unsigned int h = 768;
float MinRe = -2.0;
float MaxRe = 1.0;
float MinIm = -1.2;
float MaxIm = MinIm+(MaxRe-MinRe)*h/w;
float Re_factor = (MaxRe-MinRe)/(w-1);
float Im_factor = (MaxIm-MinIm)/(h-1);
unsigned MaxIterations = 60000;

void putN(int x, int y, unsigned n) {
	if(n == MaxIterations) {
		handler->putPixel(x,y, color_t(0,0,0));
	} else {
		if(n < MaxIterations/2) {
			float c = n*255/(float)(MaxIterations/2-1);
			handler->putPixel(x,y, color_t(c, 0,0));
		} else {
			float c = (n-MaxIterations/2)*255/(float)(MaxIterations/2);
			handler->putPixel(x, y, color_t(255, c, c));
		}
	}
}

unsigned nCPU[2048][2048], nGPU[2048][2048];

void doCPU() {
	// from http://warp.povusers.org/Mandelbrot/
	for(unsigned y=0; y<h; ++y)
	{
		float c_im = MaxIm - y*Im_factor;
		for(unsigned x=0; x<w; ++x)
		{
			float c_re = MinRe + x*Re_factor;

			float Z_re = c_re, Z_im = c_im;
			unsigned n;
			for(n=0; n<MaxIterations; ++n)
			{
				float Z_re2 = Z_re*Z_re, Z_im2 = Z_im*Z_im;
				if(Z_re2 + Z_im2 > 4)
					break;
				Z_im = 2*Z_re*Z_im + c_im;
				Z_re = Z_re2 - Z_im2 + c_re;
			}
			putN(x,y,n);
			nCPU[y][x] = n;
		}
		printf("+");
	}
}

void runOCL( void (*cb)(CLDevice device, CLContext *context, CLCommandQueue *queue) ) {
	CLError err;
	CLMain cl;
	std::list<CLPlatform> platforms = cl.getPlatforms(&err);
	checkErr(err || platforms.empty(), "failed to find platforms\n");
	CLPlatform platform = *platforms.begin();

	std::list<CLDevice> devices = platform.getDevices(&err);
	checkErr(err || devices.empty(), "failed to find devices for the chosen platform");
	CLDevice device = *devices.begin();

	CLContext *context = platform.createContext(device, &err);
	checkErr(err || context == NULL, "failed to get a context for the chosen device\n");

	CLCommandQueue *queue = context->createCommandQueue(device, &err);
	checkErr(err || queue == NULL, "failed to create command queue");

	cb(device, context, queue);
	
	queue->free(&err);
	checkErr(err, "failed to free queue");
	context->free(&err);
	checkErr(err, "failed to free context");
}

const char *kernel_source = CL_SRC(
   __kernel void mandelbrot(
		__global unsigned* a
	)
	{
		PARAM_PRIVATE_MANDELBROT(ct);

		int x = get_global_id(0);
		int y = get_global_id(1);
		if (x >= ct.w || y >= ct.h)
			return;

		float c_im = ct.MaxIm - y * ct.Im_factor;
		float c_re = ct.MinRe + x * ct.Re_factor;

		float Z_re = c_re;
		float Z_im = c_im;

		unsigned n;
		for(n = 0; n < ct.MaxIterations; ++n)
		{
			float Z_re2 = Z_re*Z_re, Z_im2 = Z_im*Z_im;
			if(Z_re2 + Z_im2 > 4)
				break;
			Z_im = 2*Z_re*Z_im + c_im;
			Z_re = Z_re2 - Z_im2 + c_re;
		}

		buffer(CLK_LOCAL_MEM_FENCE);

		a[x + y * ct.w] = n;
	}
);

void doOpenCL(CLDevice device, CLContext *context, CLCommandQueue *queue) {
	CLError err;
	unsigned *a = new cl_uint[w*h];
	memset(a, 0, w*h*sizeof(cl_uint));

	CLBuffer *buf_a = context->createBuffer(CL_MEM_READ_WRITE, w*h*sizeof(cl_uint), NULL, &err);
	checkErr(err, "failed to create buffer");

	// the memory on the GPU must be initialized because it may even retain correct results
	// from previous runs of the kernel when its latest version does not run properly
	queue->writeBuffer(buf_a, 0, w*h*sizeof(cl_uint), buf_a, &err);
	checkErr(err, "failed to initialize buffer");

	CLSrcParamMap ct("MANDELBROT");
	ct["w"] = w;
	ct["h"] = h;
	ct["MaxIm"] = MaxIm;
	ct["MinRe"] = MinRe;
	ct["Re_factor"] = Re_factor;
	ct["Im_factor"] = Im_factor;
	ct["MaxIterations"] = MaxIterations;

	std::string src = (std::string)ct + kernel_source;
	//printf("%s", src.c_str());

	CLProgram *program = buildCLProgram(src.c_str(), context, device);

	CLKernel *kernel = program->createKernel("mandelbrot", &err);
	checkErr(err, "failed to create kernel");
	
	kernel->setArgs(buf_a, &err);
	checkErr(err, "failed to set args");

	queue->runKernel(kernel, Range2D(w,h,16,16), &err);
	checkErr(err, "failed to run kernel");

	queue->readBuffer(buf_a, 0, w*h*sizeof(int), a, &err);
	checkErr(err, "failed to read buffer a");

	for(unsigned y = 0; y < h; y++) {
		for(unsigned x = 0; x < w; x++) {
			unsigned n = a[y*w + x];
			putN(x,y,n);
			nGPU[y][x] = n;
		}
	}

	kernel->free(&err);
	checkErr(err, "failed to free kernel");
	
	program->free(&err);
	checkErr(err, "failed to free program");

	buf_a->free(&err);
	checkErr(err, "failed to free buf a");
	delete a;
}

void doOpenCL() {
	runOCL( &doOpenCL );
}

imageHandler_t *createHandler(std::string name, std::string type, int width, int height)
{
	paraMap_t map;
	map["type"] = type;
	map["width"] = width;
	map["height"] = height;
	return env->createImageHandler(name, map);
}

void checkDiff() {
	for(unsigned i = 0; i < h; i++) {
		for(unsigned j = 0; j < w; j++) {
			if(nCPU[i][j] != nGPU[i][j]) {
				printf("different values (%d/%d) at (%d,%d)\n", nCPU[i][j], nGPU[i][j], i, j);
			}
		}
	}
}

int main() {
	env = new renderEnvironment_t();

	std::string ppath = "";

	if (env->getPluginPath(ppath))
	{
		Y_DEBUG(1) << "The plugin path is: " << ppath << yendl;
		env->loadPlugins(ppath);
	}
	else
	{
		Y_ERROR << "Getting plugin path from render environment failed!" << yendl;
		return 1;
	}

	handler = createHandler("mandelbrot", "jpg", w, h);

	clock_t start, end;

	start = clock();
	doCPU();
	end = clock();
	printf("CPU time: %f\n", (end-start)/(float)CLOCKS_PER_SEC);

	handler->saveToFile("mandelbrotCPU.jpg");

	start = clock();
	doOpenCL();
	end = clock();
	printf("OpenCL time: %f\n", (end-start)/(float)CLOCKS_PER_SEC);

	//checkDiff();

	handler->saveToFile("mandelbrotOCL.jpg");

	delete handler;
	delete env;

	return 0;
}
