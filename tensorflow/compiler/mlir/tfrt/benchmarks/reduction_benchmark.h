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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_REDUCTION_BENCHMARK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_REDUCTION_BENCHMARK_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSreduction_benchmarkDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSreduction_benchmarkDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSreduction_benchmarkDTh() {
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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {

std::string GetSumF32IR(llvm::ArrayRef<int32_t> input_shape,
                        llvm::ArrayRef<bool> dynamic_dims,
                        llvm::ArrayRef<int32_t> dims_to_reduce);

std::string GetMeanF32IR(llvm::ArrayRef<int32_t> input_shape,
                         llvm::ArrayRef<bool> dynamic_dims,
                         llvm::ArrayRef<int32_t> dims_to_reduce);

template <size_t N>
TensorShape ReducedTensorShape(TensorShape input_shape,
                               std::array<int32_t, N> dims_to_reduce) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSreduction_benchmarkDTh mht_0(mht_0_v, 205, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/reduction_benchmark.h", "ReducedTensorShape");

  std::vector<int64_t> result_shape;
  int j = 0;
  for (int i = 0; i < input_shape.dims(); ++i) {
    if (j < dims_to_reduce.size() && i == dims_to_reduce[j]) {
      j++;
      continue;
    }
    result_shape.push_back(input_shape.dim_size(i));
  }
  return TensorShape(result_shape);
}

template <int32_t INPUT_RANK, size_t N_DIMS_TO_REDUCE>
auto GetEigenSumF32Function(
    std::array<int32_t, N_DIMS_TO_REDUCE> dims_to_reduce) {
  return [dims_to_reduce](llvm::ArrayRef<Tensor> inputs,
                          llvm::Optional<Eigen::ThreadPoolDevice> device) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSreduction_benchmarkDTh mht_1(mht_1_v, 225, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/reduction_benchmark.h", "lambda");

    Tensor output(DT_FLOAT,
                  ReducedTensorShape(inputs[0].shape(), dims_to_reduce));
    auto in = inputs[0].tensor<float, INPUT_RANK>();
    auto out = output.tensor<float, INPUT_RANK - N_DIMS_TO_REDUCE>();
    out.setZero();
    if (device.hasValue()) {
      out.device(*device) = in.sum(dims_to_reduce);
    } else {
      out = in.sum(dims_to_reduce);
    }
  };
}

template <int32_t INPUT_RANK, size_t N_DIMS_TO_REDUCE>
auto GetEigenMeanF32Function(
    std::array<int32_t, N_DIMS_TO_REDUCE> dims_to_reduce) {
  return [dims_to_reduce](llvm::ArrayRef<Tensor> inputs,
                          llvm::Optional<Eigen::ThreadPoolDevice> device) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSreduction_benchmarkDTh mht_2(mht_2_v, 246, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/reduction_benchmark.h", "lambda");

    Tensor output(DT_FLOAT,
                  ReducedTensorShape(inputs[0].shape(), dims_to_reduce));
    auto in = inputs[0].tensor<float, INPUT_RANK>();
    auto out = output.tensor<float, INPUT_RANK - N_DIMS_TO_REDUCE>();
    out.setZero();
    if (device.hasValue()) {
      out.device(*device) = in.mean(dims_to_reduce);
    } else {
      out = in.mean(dims_to_reduce);
    }
  };
}

llvm::SmallVector<InputTensorSpec> GetInputSpec(
    llvm::ArrayRef<ssize_t> input_shape);

}  // namespace tensorflow

#define INTS(...) __VA_ARGS__
#define BOOLS(...) __VA_ARGS__

#define BM(KIND, ...) BM_##KIND(__VA_ARGS__)->Arg(0);

#define BM_SUITE_SUM_F32(NAME, INPUT_RANK, INPUT_SHAPE, DYNAMIC_DIMS,          \
                         N_DIMS_TO_REDUCE, DIMS_TO_REDUCE)                     \
  BM(JitrtV, NAME,                                                             \
     GetSumF32IR({INPUT_SHAPE}, {DYNAMIC_DIMS}, {DIMS_TO_REDUCE}), "main",     \
     GetInputSpec({INPUT_SHAPE}));                                             \
  BM(Eigen, NAME,                                                              \
     (GetEigenSumF32Function<INPUT_RANK>(                                      \
         std::array<int32_t, N_DIMS_TO_REDUCE>{DIMS_TO_REDUCE})),              \
     GetInputSpec({INPUT_SHAPE}));                                             \
  BM(Tfrt, NAME, GetSumF32IR({INPUT_SHAPE}, {DYNAMIC_DIMS}, {DIMS_TO_REDUCE}), \
     "main", GetInputSpec({INPUT_SHAPE}))

#define BM_SUITE_MEAN_F32(NAME, INPUT_RANK, INPUT_SHAPE, DYNAMIC_DIMS,      \
                          N_DIMS_TO_REDUCE, DIMS_TO_REDUCE)                 \
  BM(JitrtV, NAME,                                                          \
     GetMeanF32IR({INPUT_SHAPE}, {DYNAMIC_DIMS}, {DIMS_TO_REDUCE}), "main", \
     GetInputSpec({INPUT_SHAPE}));                                          \
  BM(Eigen, NAME,                                                           \
     (GetEigenMeanF32Function<INPUT_RANK>(                                  \
         std::array<int32_t, N_DIMS_TO_REDUCE>{DIMS_TO_REDUCE})),           \
     GetInputSpec({INPUT_SHAPE}));                                          \
  BM(Tfrt, NAME,                                                            \
     GetMeanF32IR({INPUT_SHAPE}, {DYNAMIC_DIMS}, {DIMS_TO_REDUCE}), "main", \
     GetInputSpec({INPUT_SHAPE}))

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_REDUCTION_BENCHMARK_H_
