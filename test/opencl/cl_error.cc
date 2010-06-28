#include <yafray_config.h>
#include "cl_wrapper.h"

CLCombinedError CLError::operator||(bool error_condition)
{
	if(error_condition) {
		return CLCombinedError(!error_condition, *this);
	} else
		return *this;
}