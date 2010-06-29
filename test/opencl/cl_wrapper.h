#ifndef _CL_WRAPPER_H
#define _CL_WRAPPER_H

#if defined(__APPLE__) || defined(__MACOSX)
#include <cl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <list>
#include <string.h>
#include <cassert>

#include "cl_error.h"
#include "cl_base.h"
#include "cl_source.h"
#include "cl_mem.h"
#include "cl_kernel.h"
#include "cl_device.h"
#include "cl_program.h"
#include "cl_buffer.h"
#include "cl_image.h"
#include "cl_queue.h"
#include "cl_context.h"
#include "cl_platform.h"
#include "cl_main.h"



#endif //_CL_WRAPPER_H