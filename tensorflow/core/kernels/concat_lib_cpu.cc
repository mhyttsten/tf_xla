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
class MHTracer_DTPStensorflowPScorePSkernelsPSconcat_lib_cpuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSconcat_lib_cpuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSconcat_lib_cpuDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/concat_lib_cpu.h"
#include <vector>
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/concat_lib.h"

namespace tensorflow {

namespace {
template <typename T>
struct MemCpyCopier {
  inline void Copy(T* dst, const T* src, int input_index, size_t n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconcat_lib_cpuDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/concat_lib_cpu.cc", "Copy");

    if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
      memcpy(dst, src, n * sizeof(T));
    } else {
      for (size_t k = 0; k < n; ++k) {
        *dst++ = *src++;
      }
    }
  }
};
template <>
struct MemCpyCopier<ResourceHandle> {
  inline void Copy(ResourceHandle* dst, const ResourceHandle* src,
                   int input_index, size_t n) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconcat_lib_cpuDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/kernels/concat_lib_cpu.cc", "Copy");

    for (size_t k = 0; k < n; ++k) {
      *dst++ = *src++;
    }
  }
};

template <typename T>
int64_t EstimateBytesPerElement(
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconcat_lib_cpuDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/kernels/concat_lib_cpu.cc", "EstimateBytesPerElement");

  return sizeof(T);
}

// EstimateBytesPerElement for strings estimates the total bytes involved in
// concatenating the strings in the "inputs" matrices (higher-level code
// reshapes all the inputs to matrices), by sampling the lengths of the actual
// strings in the various tensors.
template <>
int64_t EstimateBytesPerElement<tstring>(
    const std::vector<
        std::unique_ptr<typename TTypes<tstring, 2>::ConstMatrix>>& inputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconcat_lib_cpuDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/kernels/concat_lib_cpu.cc", "EstimateBytesPerElement<tstring>");

  // randomly sample a few input strings to get a sense of the average size
  // of each element
  int num_samples = 0;
  int64_t num_bytes_in_samples = 0;
  for (const auto& input : inputs) {
    const auto dim0 = input->dimension(0);
    const auto dim1 = input->dimension(1);
    const auto zero = dim0 - dim0;  // Make type match
    if (dim0 > 0 && dim1 > 0) {
      // Draw 9 samples of string sizes from the input, in this sort of pattern
      // ("*" is sample), to get an estimate of the lengths of each string
      // element in the tensors:
      //
      //    *...*...*
      //    .........
      //    *...*...*
      //    .........
      //    *...*...*
      for (auto i : {zero, dim0 / 2, dim0 - 1}) {
        for (auto j : {zero, dim1 / 2, dim1 - 1}) {
          num_bytes_in_samples += (*input)(i, j).size();
          num_samples++;
        }
      }
    }
  }
  // We don't use sizeof(std::string) as the overhead, since that would
  // overestimate the memory touched for copying a string.
  int64_t string_overhead = sizeof(char*) + sizeof(size_t);
  return string_overhead +
         ((num_samples > 0) ? (num_bytes_in_samples / num_samples) : 0);
}

}  // namespace

template <typename T>
void ConcatCPU(
    DeviceBase* d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconcat_lib_cpuDTcc mht_4(mht_4_v, 284, "", "./tensorflow/core/kernels/concat_lib_cpu.cc", "ConcatCPU");

  int64_t cost_per_unit = EstimateBytesPerElement<T>(inputs);
  ConcatCPUImpl<T>(d, inputs, cost_per_unit, MemCpyCopier<T>(), output);
}

#define REGISTER(T)                                                            \
  template void ConcatCPU<T>(                                                  \
      DeviceBase*,                                                             \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&, \
      typename TTypes<T, 2>::Matrix* output);
TF_CALL_ALL_TYPES(REGISTER)
REGISTER(quint8)
REGISTER(qint8)
REGISTER(quint16)
REGISTER(qint16)
REGISTER(qint32)

#if defined(IS_MOBILE_PLATFORM) && !defined(SUPPORT_SELECTIVE_REGISTRATION) && \
    !defined(__ANDROID_TYPES_FULL__)
// Primarily used for SavedModel support on mobile. Registering it here only
// if __ANDROID_TYPES_FULL__ is not defined (which already registers string)
// to avoid duplicate registration.
REGISTER(tstring);
#endif  // defined(IS_MOBILE_PLATFORM) &&
        // !defined(SUPPORT_SELECTIVE_REGISTRATION) &&
        // !defined(__ANDROID_TYPES_FULL__)

}  // namespace tensorflow
