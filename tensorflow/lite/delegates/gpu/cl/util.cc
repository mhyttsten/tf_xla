#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSutilDTcc {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSutilDTcc() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/cl/util.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

std::string CLErrorCodeToString(cl_int error_code) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSutilDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/gpu/cl/util.cc", "CLErrorCodeToString");

  switch (error_code) {
    case CL_SUCCESS:
      return "Success";
    case CL_DEVICE_NOT_FOUND:
      return "Device not found";
    case CL_DEVICE_NOT_AVAILABLE:
      return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:
      return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:
      return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:
      return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:
      return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:
      return "Build program failure";
    case CL_MAP_FAILURE:
      return "Mapping failure";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "Misaligned sub-buffer offset";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "Execution status error for events in wait list";
    case CL_COMPILE_PROGRAM_FAILURE:
      return "Compile program failure";
    case CL_LINKER_NOT_AVAILABLE:
      return "Linker not available";
    case CL_LINK_PROGRAM_FAILURE:
      return "Link program failure";
    case CL_DEVICE_PARTITION_FAILED:
      return "Device partition failed";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "Kernel argument information not available";

    case CL_INVALID_VALUE:
      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:
      return "Invalid device type";
    case CL_INVALID_PLATFORM:
      return "Invalid platform";
    case CL_INVALID_DEVICE:
      return "Invalid device";
    case CL_INVALID_CONTEXT:
      return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:
      return "Invalid command queue";
    case CL_INVALID_HOST_PTR:
      return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:
      return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:
      return "Invalid image size";
    case CL_INVALID_SAMPLER:
      return "Invalid sampler";
    case CL_INVALID_BINARY:
      return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:
      return "Invalid build options";
    case CL_INVALID_PROGRAM:
      return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:
      return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:
      return "Invalid kernel definition";
    case CL_INVALID_KERNEL:
      return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:
      return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:
      return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:
      return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:
      return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:
      return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:
      return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "Invalid event wait list";
    case CL_INVALID_EVENT:
      return "Invalid event";
    case CL_INVALID_OPERATION:
      return "Invalid operation";
    case CL_INVALID_GL_OBJECT:
      return "Invalid GL object";
    case CL_INVALID_BUFFER_SIZE:
      return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:
      return "Invalid mip-level";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "Invalid global work size";
    case CL_INVALID_PROPERTY:
      return "Invalid property";
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "Invalid image descriptor";
    case CL_INVALID_COMPILER_OPTIONS:
      return "Invalid compiler options";
    case CL_INVALID_LINKER_OPTIONS:
      return "Invalid linker options";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "Invalid device partition count";
    case CL_INVALID_PIPE_SIZE:
      return "Invalid pipe size";
    case CL_INVALID_DEVICE_QUEUE:
      return "Invalid device queue";
    case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
      return "Invalid GL sharegroup reference KHR";

    default:
      return absl::StrCat("Unknown OpenCL error code - ", error_code);
  }
}

int ChannelTypeToSizeInBytes(cl_channel_type type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSutilDTcc mht_1(mht_1_v, 331, "", "./tensorflow/lite/delegates/gpu/cl/util.cc", "ChannelTypeToSizeInBytes");

  switch (type) {
    case CL_FLOAT:
      return 4;
    case CL_HALF_FLOAT:
      return 2;
    default:
      return 0;
  }
}

bool OpenCLSupported() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSutilDTcc mht_2(mht_2_v, 345, "", "./tensorflow/lite/delegates/gpu/cl/util.cc", "OpenCLSupported");
 return LoadOpenCL().ok(); }

absl::Status CreateCLBuffer(cl_context context, int size_in_bytes,
                            bool read_only, void* data, cl_mem* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSutilDTcc mht_3(mht_3_v, 351, "", "./tensorflow/lite/delegates/gpu/cl/util.cc", "CreateCLBuffer");

  cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
  if (data) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }
  cl_int error_code;
  *result = clCreateBuffer(context, flags, size_in_bytes, data, &error_code);
  if (!*result) {
    return absl::UnknownError(
        absl::StrCat("Failed to allocate device memory (clCreateBuffer): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateCLSubBuffer(cl_context context, cl_mem parent,
                               size_t origin_in_bytes, size_t size_in_bytes,
                               bool read_only, cl_mem* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSutilDTcc mht_4(mht_4_v, 371, "", "./tensorflow/lite/delegates/gpu/cl/util.cc", "CreateCLSubBuffer");

  cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;

  cl_buffer_region region{};
  region.origin = origin_in_bytes;
  region.size = size_in_bytes;

  cl_int error_code;
  if (!clCreateSubBuffer) {
    return absl::InternalError("clCreateSubBuffer is not supported.");
  }
  *result = clCreateSubBuffer(parent, flags, CL_BUFFER_CREATE_TYPE_REGION,
                              &region, &error_code);

  if (!*result) {
    return absl::UnknownError(
        absl::StrCat("Failed to allocate device memory (clCreateSubBuffer): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateRGBAImage2D(cl_context context, int width, int height,
                               cl_channel_type channel_type, void* data,
                               cl_mem* result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSutilDTcc mht_5(mht_5_v, 398, "", "./tensorflow/lite/delegates/gpu/cl/util.cc", "CreateRGBAImage2D");

  cl_image_desc desc;
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = width;
  desc.image_height = height;
  desc.image_depth = 0;
  desc.image_row_pitch = 0;
  desc.image_slice_pitch = 0;
  desc.num_mip_levels = 0;
  desc.num_samples = 0;
  desc.buffer = nullptr;

  cl_image_format format;
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = channel_type;

  cl_mem_flags flags = CL_MEM_READ_WRITE;
  if (data) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }

  cl_int error_code;
  *result =
      CreateImage2DLegacy(context, flags, &format, &desc, data, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create 2D texture (clCreateImage): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
