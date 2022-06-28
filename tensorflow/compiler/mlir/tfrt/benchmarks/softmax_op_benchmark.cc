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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSsoftmax_op_benchmarkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSsoftmax_op_benchmarkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSsoftmax_op_benchmarkDTcc() {
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

#include <string>

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {
namespace {

const char* kSoftmaxIR = R"(
  func.func @main(%input: {0}) -> {0} {
    %result = "tf.Softmax"(%input)
      {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
      : ({0}) -> {0}
    func.return %result : {0}
  }
)";

std::string Softmax(llvm::ArrayRef<bool> dynamic_dims,
                    llvm::ArrayRef<ssize_t> input_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSsoftmax_op_benchmarkDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/softmax_op_benchmark.cc", "Softmax");

  llvm::SmallVector<int64_t, 2> mlir_input_shape;
  for (int i = 0; i < input_shape.size(); ++i) {
    mlir_input_shape.push_back(dynamic_dims[i] ? kDynSize : input_shape[i]);
  }
  return llvm::formatv(kSoftmaxIR, PrintTensorType(mlir_input_shape, "f32"));
}

// Eigen code implementing SoftmaxFunctor::operator() carefully taken from
// tensorflow/core/kernels/softmax_op_functor.h
template <typename InT, typename OutT>
static void ComputeSoftmax(const Eigen::DefaultDevice& d, InT logits,
                           OutT softmax) {
  const int kBatchDim = 0;
  const int kClassDim = 1;

  const int batch_size = logits.dimension(kBatchDim);
  const int num_classes = logits.dimension(kClassDim);

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
  Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
  Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
  batch_by_one.set(0, batch_size);
  Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
  one_by_class.set(1, num_classes);
  // shifted_logits = logits - max(logits along classes);
  auto shifted_logits = (logits - logits.maximum(along_class)
                                      .eval()
                                      .reshape(batch_by_one)
                                      .broadcast(one_by_class));
  softmax.device(d) = shifted_logits.exp();
  softmax.device(d) = (softmax * softmax.sum(along_class)
                                     .inverse()
                                     .eval()
                                     .reshape(batch_by_one)
                                     .broadcast(one_by_class));
}

auto EigenSoftmax() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSsoftmax_op_benchmarkDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/softmax_op_benchmark.cc", "EigenSoftmax");

  return [](llvm::ArrayRef<Tensor> inputs,
            llvm::Optional<Eigen::ThreadPoolDevice>) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSsoftmax_op_benchmarkDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/softmax_op_benchmark.cc", "lambda");

    Tensor output(DT_FLOAT, {inputs[0].dim_size(0), inputs[0].dim_size(1)});

    auto in = inputs[0].tensor<float, 2>();
    auto out = output.tensor<float, 2>();
    out.setZero();

    Eigen::DefaultDevice default_device;
    ComputeSoftmax<decltype(in), decltype(out)>(default_device, in, out);
  };
}

llvm::SmallVector<InputTensorSpec> Inputs(ssize_t rows, ssize_t cols) {
  return {InputTensorSpec(DT_FLOAT, {rows, cols})};
}

#define BM(FN) BM_##FN->Arg(0);

#define BM_SUITE(NAME, DYNAMIC_ROW, DYNAMIC_COL, ROWS, COLS)                 \
  BM(JitrtV(NAME, Softmax({DYNAMIC_ROW, DYNAMIC_COL}, {ROWS, COLS}), "main", \
            Inputs(ROWS, COLS)));                                            \
  BM(Eigen(NAME, EigenSoftmax(), Inputs(ROWS, COLS)));                       \
  BM(Tfrt(NAME, Softmax({DYNAMIC_ROW, DYNAMIC_COL}, {ROWS, COLS}), "main",   \
          Inputs(ROWS, COLS)))

#define BM_DYNAMIC_ALL(ROWS, COLS)                                            \
  BM_SUITE(SoftmaxDynamicAll_##ROWS##_##COLS, kDynamicDim, kDynamicDim, ROWS, \
           COLS)
BM_DYNAMIC_ALL(2, 80);
BM_DYNAMIC_ALL(8, 6);
BM_DYNAMIC_ALL(80, 1);
BM_DYNAMIC_ALL(80, 60);
BM_DYNAMIC_ALL(81, 61);
BM_DYNAMIC_ALL(800, 600);
BM_DYNAMIC_ALL(802, 602);

#define BM_STATIC_ROW(ROWS, COLS) \
  BM_SUITE(SoftmaxStaticRow##ROWS##_##COLS, kStaticDim, kDynamicDim, ROWS, COLS)
BM_STATIC_ROW(2, 80);
BM_STATIC_ROW(8, 6);
BM_STATIC_ROW(80, 1);
BM_STATIC_ROW(80, 60);
BM_STATIC_ROW(81, 61);
BM_STATIC_ROW(800, 600);
BM_STATIC_ROW(802, 602);

#define BM_STATIC_COL(ROWS, COLS)                                           \
  BM_SUITE(SoftmaxStaticCol_##ROWS##_##COLS, kDynamicDim, kStaticDim, ROWS, \
           COLS)
BM_STATIC_COL(2, 80);
BM_STATIC_COL(8, 6);
BM_STATIC_COL(80, 1);
BM_STATIC_COL(80, 60);
BM_STATIC_COL(81, 61);
BM_STATIC_COL(800, 600);
BM_STATIC_COL(802, 602);

#define BM_STATIC_ALL(ROWS, COLS) \
  BM_SUITE(SoftmaxStaticAll_##ROWS##_##COLS, kStaticDim, kStaticDim, ROWS, COLS)
BM_STATIC_ALL(2, 80);
BM_STATIC_ALL(8, 6);
BM_STATIC_ALL(80, 1);
BM_STATIC_ALL(80, 60);
BM_STATIC_ALL(81, 61);
BM_STATIC_ALL(800, 600);
BM_STATIC_ALL(802, 602);

}  // namespace
}  // namespace tensorflow
