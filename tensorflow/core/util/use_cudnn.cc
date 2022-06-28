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
class MHTracer_DTPStensorflowPScorePSutilPSuse_cudnnDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSuse_cudnnDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSuse_cudnnDTcc() {
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

#include "tensorflow/core/util/use_cudnn.h"

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

#define ADD_BOOL_CUDNN_FLAG(func_name, flag_name, default_value)           \
  bool func_name() {                                                       \
    bool value = default_value;                                            \
    Status status = ReadBoolFromEnvVar(#flag_name, default_value, &value); \
    if (!status.ok()) {                                                    \
      LOG(ERROR) << status;                                                \
    }                                                                      \
    return value;                                                          \
  }

bool CudnnUseFrontend() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSuse_cudnnDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/util/use_cudnn.cc", "CudnnUseFrontend");

  static bool result = [] {
    bool value = false;
#if GOOGLE_CUDA
    if (CUDNN_VERSION >= 8100) {
      // cuDNN 8.1.0 + the frontend has issues regarding fused convolution.
      Status status = ReadBoolFromEnvVar("TF_CUDNN_USE_FRONTEND",
                                         CUDNN_VERSION >= 8200, &value);
      if (!status.ok()) {
        LOG(ERROR) << status;
      }
    }
#endif  // GOOGLE_CUDA
    return value;
  }();
  return result;
}

ADD_BOOL_CUDNN_FLAG(CudnnUseAutotune, TF_CUDNN_USE_AUTOTUNE, true);
// Whether to auto-tuning Cudnn RNN forward and backward pass to pick
// statistically the best cudnnRNNAlgo_t and cudnnMathType_t.
// The flag is disabled when TF_DEBUG_CUDNN_RNN is turned on.
ADD_BOOL_CUDNN_FLAG(CudnnRnnUseAutotune, TF_CUDNN_RNN_USE_AUTOTUNE, true);
ADD_BOOL_CUDNN_FLAG(CudnnDisableConv1x1Optimization,
                    TF_CUDNN_DISABLE_CONV_1X1_OPTIMIZATION, false);

// Whether to run Cudnn RNN forward and backward in debug mode, where users can
// force a specified cudnnRNNAlgo_t and cudnnMathType_t, when used together with
// the following two env vars:
// TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS
// TF_DEBUG_CUDNN_RNN_ALGO
// By default it is disabled and only intended for testing and profiling.
ADD_BOOL_CUDNN_FLAG(DebugCudnnRnn, TF_DEBUG_CUDNN_RNN, false);
// If using TENSOR_OP_MATH in Cudnn RNN for both forward and backward pass. Only
// effective when TF_DEBUG_CUDNN_RNN is true.
// Note none of the persistent RNN algorithm support TENSOR_OP_MATH before
// Cudnn 7.1. See Nvidia Cudnn manual for more details.
ADD_BOOL_CUDNN_FLAG(DebugCudnnRnnUseTensorOps,
                    TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS, false);
#undef ADD_BOOL_CUDNN_FLAG

#define ADD_INT64_CUDNN_FLAG(func_name, flag_name, default_value)           \
  int64_t func_name() {                                                     \
    int64_t value = default_value;                                          \
    Status status = ReadInt64FromEnvVar(#flag_name, default_value, &value); \
    if (!status.ok()) {                                                     \
      LOG(ERROR) << status;                                                 \
    }                                                                       \
    return value;                                                           \
  }
// Cudnn RNN algorithm to use for both forward and backward pass. Only effective
// when TF_DEBUG_CUDNN_RNN is true. See Nvidia Cudnn manual for allowed
// cudnnRNNAlgo_t.
ADD_INT64_CUDNN_FLAG(DebugCudnnRnnAlgo, TF_DEBUG_CUDNN_RNN_ALGO, -1);
#undef ADD_INT64_CUDNN_FLAG

bool IsCudnnSupportedFilterSize(const int32_t filter_rows,
                                const int32_t filter_cols,
                                const int32_t in_depth,
                                const int32_t out_depth) {
  return in_depth == out_depth && filter_rows == filter_cols &&
         (filter_rows == 1 || filter_rows == 3 || filter_rows == 5 ||
          filter_rows == 7);
}

}  // namespace tensorflow
