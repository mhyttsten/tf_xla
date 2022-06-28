/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_TPU_EMBEDDING_OPTIMIZATION_PARAMETERS_UTILS_H_
#define TENSORFLOW_CORE_TPU_TPU_EMBEDDING_OPTIMIZATION_PARAMETERS_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTh() {
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


#include <string>

#include "absl/base/casts.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/tpu/optimization_parameters.pb.h"

namespace tensorflow {
namespace tpu {

using OptimizationAlgorithm = OptimizationParameters::ParametersCase;

// Returns the name of the optimization algorithm.
string GetOptimizationAlgorithmName(OptimizationAlgorithm alg);

// Returns a user-friendly name for the optimization algorithm.
string GetOptimizationAlgorithmFriendlyName(OptimizationAlgorithm alg);

// Returns all supported optimization algorithms.
std::vector<OptimizationAlgorithm> GetOptimizationAlgorithms();

enum class GradientAccumulationSupport {
  // Accumulation cannot be used with this optimizer.
  kNotSupported,

  // Accumulation is allowed and changes optimizer behavior.
  kSupported,
};

// Returns the number of optimization parameter vectors used by the optimization
// algorithm, excluding the weights themselves and assuming no gradient
// accumulation.
Status GetBaseAuxiliaryParameterCount(const OptimizationParameters &params,
                                      int *count);

// Returns whether (and how) an optimization algorithm supports gradient
// accumulation.
Status GetGradientAccumulationSupport(const OptimizationParameters &params,
                                      GradientAccumulationSupport *support);

// Returns whether both the given set of optimization parameters has gradient
// accumulation turned on and that the algorithm used supports it or should
// ignore that setting. Returns an error if gradient accumulation is enabled and
// the algorithm does not support it.
Status UseGradientAccumulation(const OptimizationParameters &params,
                               bool *use_gradient_accumulation);

// Returns the parameter specifications for the optimization algorithm (the main
// parameters first, followed by any auxiliary parameters such as Adagrad
// accumulators).
Status GetOptimizationAlgorithmStateVariables(
    const OptimizationParameters &params,
    std::vector<StateVariableSpecification> *state_variables);

// Maximum value of auxiliary_parametery_count for any optimization algorithm.
// This count is used by TPU embedding load/retrieve and needs to be independent
// of any particular TPU version and hence, we take the maximum across all TPU
// versions.
static constexpr int kMaxAuxiliaryParameterCount = 7;

// Fill value for gradient accumulators. This is a denormal so that it will be
// flushed to zero on the current TPU platforms and needs to continue to have
// the following properties in the future:
//
// 1. Does not have the same bit pattern as a zero and can be distinguished from
// it using integer operations.
// 2. Treated as zero by floating-point arithmetic operations (at least addition
// and subtraction).
// 3. Cannot be produced by any floating-point arithmetic operation, including
// those involving itself.
//
// It does not need to compare equal or not equal to zero in floating point. We
// need to use a non-zero value here because some optimization algorithms are
// not no-ops on zero gradients, so we need to distinguish an accumulated
// gradient of zero from one that has been cleared after its gradients have
// already been applied to the parameters and accumulators.
inline float GradientAccumulatorInitialValue() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTh mht_0(mht_0_v, 264, "", "./tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h", "GradientAccumulatorInitialValue");

  return absl::bit_cast<float, uint32>(1);
}

// Generic shape function for per-optimization-algorithm load ops.
class LoadOpShapeFunction {
 public:
  // Computes resulting shape and does parameter checking.
  Status operator()(shape_inference::InferenceContext *c) const;
};

// Generic shape function for per-optimization-algorithm retrieve ops.
class RetrieveOpShapeFunction {
 public:
  // Computes resulting shape and does parameter checking.
  Status operator()(shape_inference::InferenceContext *c) const;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_EMBEDDING_OPTIMIZATION_PARAMETERS_UTILS_H_
