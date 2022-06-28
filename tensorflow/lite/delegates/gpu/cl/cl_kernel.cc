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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_program.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

absl::Status GetKernelMaxWorkGroupSize(cl_kernel kernel, cl_device_id device_id,
                                       int* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "GetKernelMaxWorkGroupSize");

  size_t max_work_group_size;
  cl_int error_code =
      clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                               sizeof(size_t), &max_work_group_size, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to get info CL_KERNEL_WORK_GROUP_SIZE ",
                     CLErrorCodeToString(error_code)));
  }
  *result = static_cast<int>(max_work_group_size);
  return absl::OkStatus();
}

absl::Status GetKernelPrivateMemorySize(cl_kernel kernel,
                                        cl_device_id device_id, int* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "GetKernelPrivateMemorySize");

  cl_ulong private_mem_size;
  cl_int error_code =
      clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PRIVATE_MEM_SIZE,
                               sizeof(cl_ulong), &private_mem_size, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to get info CL_KERNEL_PRIVATE_MEM_SIZE ",
                     CLErrorCodeToString(error_code)));
  }
  *result = static_cast<int>(private_mem_size);
  return absl::OkStatus();
}

}  // namespace

CLKernel::CLKernel(CLKernel&& kernel)
    : info_(kernel.info_),
      binding_counter_(kernel.binding_counter_),
      function_name_(std::move(kernel.function_name_)),
      program_(kernel.program_),
      kernel_(kernel.kernel_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_2(mht_2_v, 240, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "CLKernel::CLKernel");

  kernel.kernel_ = nullptr;
}

CLKernel& CLKernel::operator=(CLKernel&& kernel) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_3(mht_3_v, 247, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "=");

  if (this != &kernel) {
    Release();
    std::swap(info_, kernel.info_);
    std::swap(binding_counter_, kernel.binding_counter_);
    function_name_ = std::move(kernel.function_name_);
    std::swap(program_, kernel.program_);
    std::swap(kernel_, kernel.kernel_);
  }
  return *this;
}

CLKernel::~CLKernel() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_4(mht_4_v, 262, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "CLKernel::~CLKernel");
 Release(); }

absl::Status CLKernel::ReInit() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_5(mht_5_v, 267, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "CLKernel::ReInit");

  clReleaseKernel(kernel_);
  cl_kernel* kern_ptr = const_cast<cl_kernel*>(&kernel_);
  int error_code;
  *kern_ptr = clCreateKernel(program_, function_name_.c_str(), &error_code);
  if (!kernel_ || error_code != CL_SUCCESS) {
    *kern_ptr = nullptr;
    return absl::UnknownError(absl::StrCat("Failed to create ", function_name_,
                                           CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

void CLKernel::Release() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_6(mht_6_v, 283, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "CLKernel::Release");

  if (kernel_) {
    clReleaseKernel(kernel_);
    clReleaseProgram(program_);
    kernel_ = nullptr;
  }
}

absl::Status CLKernel::CreateFromProgram(const CLProgram& program,
                                         const std::string& function_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_7(mht_7_v, 296, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "CLKernel::CreateFromProgram");

  int error_code;
  function_name_ = function_name;
  kernel_ =
      clCreateKernel(program.program(), function_name.c_str(), &error_code);
  if (!kernel_ || error_code != CL_SUCCESS) {
    kernel_ = nullptr;
    return absl::UnknownError(absl::StrCat("Failed to create ", function_name,
                                           CLErrorCodeToString(error_code)));
  }

  program_ = program.program();
  clRetainProgram(program_);

  RETURN_IF_ERROR(GetKernelPrivateMemorySize(kernel_, program.GetDeviceId(),
                                             &info_.private_memory_size));
  RETURN_IF_ERROR(GetKernelMaxWorkGroupSize(kernel_, program.GetDeviceId(),
                                            &info_.max_work_group_size));
  return absl::OkStatus();
}

absl::Status CLKernel::SetMemory(int index, cl_mem memory) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_8(mht_8_v, 320, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "CLKernel::SetMemory");

  return SetBytes(index, &memory, sizeof(cl_mem));
}

absl::Status CLKernel::SetMemoryAuto(cl_mem memory) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_9(mht_9_v, 327, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "CLKernel::SetMemoryAuto");

  return SetBytesAuto(&memory, sizeof(cl_mem));
}

absl::Status CLKernel::SetBytes(int index, const void* ptr, int length) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_10(mht_10_v, 334, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "CLKernel::SetBytes");

  const int error_code = clSetKernelArg(kernel_, index, length, ptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to set kernel arguments - ",
                                           CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CLKernel::SetBytesAuto(const void* ptr, int length) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_kernelDTcc mht_11(mht_11_v, 346, "", "./tensorflow/lite/delegates/gpu/cl/cl_kernel.cc", "CLKernel::SetBytesAuto");

  const int error_code = clSetKernelArg(kernel_, binding_counter_, length, ptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat(
        "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
        "(at index - ", binding_counter_, ")"));
  }
  binding_counter_++;
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
