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
class MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_gpu_0DTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_gpu_0DTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_gpu_0DTcuDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/segment_reduction_ops_gpu.cu.h"

namespace tensorflow {

bool UseDeterministicSegmentReductions() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_gpu_0DTcuDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/kernels/segment_reduction_ops_gpu_0.cu.cc", "UseDeterministicSegmentReductions");

  // See comment below regarding CI build error on Windows.
#if !defined(PLATFORM_WINDOWS)
  static bool cached_result = [] {
    bool result = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
        "TF_USE_DETERMINISTIC_SEGMENT_REDUCTIONS",
        /*default_val=*/false, &result));
    return result;
  }();
  return cached_result;
#else
  return false;
#endif
}

bool DisableSegmentReductionOpDeterminismExceptions() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_gpu_0DTcuDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/kernels/segment_reduction_ops_gpu_0.cu.cc", "DisableSegmentReductionOpDeterminismExceptions");

  static bool cached_disable = [] {
    bool disable = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
        "TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS",
        /*default_val=*/false, &disable));
    return disable;
  }();
  return cached_disable;
}

namespace functor {

#define DEFINE_SORTED_GPU_SPECS_INDEX(T, Index)               \
  template struct SegmentReductionFunctor<                    \
      T, Index, /*InitialValueF=*/functor::Zero<T>,           \
      /*EmptySegmentValueF=*/functor::Zero<T>, functor::Sum>; \
  template struct SegmentReductionFunctor<                    \
      T, Index, /*InitialValueF=*/functor::One<T>,            \
      /*EmptySegmentValueF=*/functor::One<T>, functor::Prod>; \
  template struct SegmentReductionFunctor<                    \
      T, Index, /*InitialValueF=*/functor::Highest<T>,        \
      /*EmptySegmentValueF=*/functor::Zero<T>, functor::Min>; \
  template struct SegmentReductionFunctor<                    \
      T, Index, /*InitialValueF=*/functor::Lowest<T>,         \
      /*EmptySegmentValueF=*/functor::Zero<T>, functor::Max>;

#define DEFINE_SORTED_GPU_SPECS(T) DEFINE_SORTED_GPU_SPECS_INDEX(T, int32);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_SORTED_GPU_SPECS);

#define DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, Index)                         \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index,                  \
                                         functor::Lowest<T>, functor::Max>;    \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index,                  \
                                         functor::Highest<T>, functor::Min>;   \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index, functor::One<T>, \
                                         functor::Prod>;

// Sum is the only op that supports all input types currently.
#define DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, Index)         \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index, \
                                         functor::Zero<T>, functor::Sum>;

#define DEFINE_REAL_GPU_SPECS(T) DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int32);

#define DEFINE_SUM_GPU_SPECS(T) DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int32);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_REAL_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DEFINE_SUM_GPU_SPECS);

#undef DEFINE_SORTED_GPU_SPECS_INDEX
#undef DEFINE_SORTED_GPU_SPECS
#undef DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_REAL_GPU_SPECS
#undef DEFINE_SUM_GPU_SPECS

// TODO(benbarsdell): These kernels are disabled on Windows as a workaround for
// a CI build error: "formal parameter with requested alignment of 128 won't be
// aligned". The root cause is suspected to be an aligned type (AlignedVector)
// being passed to a function by value, possibly inside the CUB library
// somewhere, but I have not yet been able to reproduce it in isolation outside
// of the GitHub CI.
#if !defined(PLATFORM_WINDOWS)

#define DEFINE_SPARSE_SEGMENT_REDUCTION_FUNCTOR(T)                \
  template struct SparseSegmentReductionFunctor<T, int32, int32>; \
  template struct SparseSegmentReductionFunctor<T, int32, int64_t>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_SPARSE_SEGMENT_REDUCTION_FUNCTOR);
#undef DEFINE_SPARSE_SEGMENT_REDUCTION_FUNCTOR

#define DEFINE_SPARSE_SEGMENT_GRAD_FUNCTOR(T)                           \
  template struct SparseSegmentGradFunctor<GPUDevice, T, int32, int32>; \
  template struct SparseSegmentGradFunctor<GPUDevice, T, int32, int64_t>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_SPARSE_SEGMENT_GRAD_FUNCTOR);
#undef DEFINE_SPARSE_SEGMENT_GRAD_FUNCTOR

#endif  // !defined(PLATFORM_WINDOWS)

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
