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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/cl_program.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetProgramBuildInfo(cl_program program, cl_device_id id,
                                cl_program_build_info info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "GetProgramBuildInfo");

  size_t size;
  cl_int error_code =
      clGetProgramBuildInfo(program, id, info, 0, nullptr, &size);
  if (error_code != CL_SUCCESS) {
    return absl::StrCat("Failed to GetProgramBuildInfo - ",
                        CLErrorCodeToString(error_code));
  }

  std::string result(size - 1, 0);
  error_code =
      clGetProgramBuildInfo(program, id, info, size, &result[0], nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::StrCat("Failed to GetProgramBuildInfo - ",
                        CLErrorCodeToString(error_code));
  }
  return result;
}

absl::Status GetBinarySize(cl_program program, size_t* binary_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_1(mht_1_v, 224, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "GetBinarySize");

  cl_int error_code = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                                       sizeof(size_t), binary_size, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to get program binary size - ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status BuildProgram(cl_program program, const CLDevice& device,
                          const std::string& compiler_options) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("compiler_options: \"" + compiler_options + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_2(mht_2_v, 240, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "BuildProgram");

  const int error_code = clBuildProgram(
      program, 0, nullptr, compiler_options.c_str(), nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat(
        "Failed to build program executable - ",
        CLErrorCodeToString(error_code),
        GetProgramBuildInfo(program, device.id(), CL_PROGRAM_BUILD_LOG)));
  }

  return absl::OkStatus();
}

std::string CompilerOptionToString(const GpuInfo& gpu_info,
                                   CompilerOptions option) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_3(mht_3_v, 257, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "CompilerOptionToString");

  switch (option) {
    case CompilerOptions::kAdrenoFullSimd:
      if (gpu_info.IsAdreno()) {
        if (gpu_info.adreno_info.IsAdreno3xx() ||
            gpu_info.adreno_info.IsAdreno4xx()) {
          return "-qcom-accelerate-16-bit";
        } else {
          return "-qcom-accelerate-16-bit=true";
        }
      } else {
        return "unsupported";
      }
    case CompilerOptions::kAdrenoMoreWaves:
      if (gpu_info.IsAdreno()) {
        if (!(gpu_info.adreno_info.IsAdreno3xx() ||
              gpu_info.adreno_info.IsAdreno4xx())) {
          return "-qcom-accelerate-16-bit=false";
        } else {
          return "";
        }
      } else {
        return "unsupported";
      }
    case CompilerOptions::kClFastRelaxedMath:
      return "-cl-fast-relaxed-math";
    case CompilerOptions::kClDisableOptimizations:
      return "-cl-opt-disable";
    case CompilerOptions::kCl20:
      return "-cl-std=CL2.0";
    case CompilerOptions::kCl30:
      return "-cl-std=CL3.0";
  }
}

}  // namespace

std::string CompilerOptionsToString(
    const GpuInfo& gpu_info,
    const std::vector<CompilerOptions>& compiler_options) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_4(mht_4_v, 299, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "CompilerOptionsToString");

  std::string result;
  for (auto option : compiler_options) {
    absl::StrAppend(&result, CompilerOptionToString(gpu_info, option), " ");
  }
  return result;
}

CLProgram::CLProgram(cl_program program, cl_device_id device_id)
    : program_(program), device_id_(device_id) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_5(mht_5_v, 311, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "CLProgram::CLProgram");
}

CLProgram::CLProgram(CLProgram&& program)
    : program_(program.program_), device_id_(program.device_id_) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_6(mht_6_v, 317, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "CLProgram::CLProgram");

  program.program_ = nullptr;
}

CLProgram& CLProgram::operator=(CLProgram&& program) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_7(mht_7_v, 324, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "=");

  if (this != &program) {
    Release();
    std::swap(program_, program.program_);
    std::swap(device_id_, program.device_id_);
  }
  return *this;
}

CLProgram::~CLProgram() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_8(mht_8_v, 336, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "CLProgram::~CLProgram");
 Release(); }

void CLProgram::Release() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_9(mht_9_v, 341, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "CLProgram::Release");

  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }
}

absl::Status CLProgram::GetBinary(std::vector<uint8_t>* result) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_10(mht_10_v, 351, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "CLProgram::GetBinary");

  size_t binary_size;
  RETURN_IF_ERROR(GetBinarySize(program_, &binary_size));
  result->resize(result->size() + binary_size);
  uint8_t* binary_ptr = result->data() + result->size() - binary_size;
  cl_int error_code = clGetProgramInfo(program_, CL_PROGRAM_BINARIES,
                                       binary_size, &binary_ptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to get program binary - ",
                                           CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateCLProgram(const std::string& code,
                             const std::string& compiler_options,
                             const CLContext& context, const CLDevice& device,
                             CLProgram* result) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("code: \"" + code + "\"");
   mht_11_v.push_back("compiler_options: \"" + compiler_options + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_11(mht_11_v, 373, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "CreateCLProgram");

  int error_code;
  const char* source = code.c_str();

  cl_program program = clCreateProgramWithSource(context.context(), 1, &source,
                                                 nullptr, &error_code);
  if (!program || error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create compute program - ",
                     CLErrorCodeToString(error_code)));
  }

  *result = CLProgram(program, device.id());
  RETURN_IF_ERROR(BuildProgram(program, device, compiler_options));
  return absl::OkStatus();
}

absl::Status CreateCLProgramFromBinary(const CLContext& context,
                                       const CLDevice& device,
                                       absl::Span<const uint8_t> binary,
                                       CLProgram* result) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_programDTcc mht_12(mht_12_v, 396, "", "./tensorflow/lite/delegates/gpu/cl/cl_program.cc", "CreateCLProgramFromBinary");

  cl_int binary_status;
  cl_int error_code;
  cl_device_id devices_list[] = {device.id()};
  size_t binary_size = binary.size();
  const uint8_t* binary_pointer = binary.data();
  cl_program program = clCreateProgramWithBinary(
      context.context(), 1, devices_list, &binary_size, &binary_pointer,
      &binary_status, &error_code);
  if (binary_status != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat(
        "Something wrong with binary after clCreateProgramWithBinary - ",
        binary_status));
  }
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to create program - ",
                                           CLErrorCodeToString(error_code)));
  }
  *result = CLProgram(program, device.id());
  return BuildProgram(program, device, "");
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
