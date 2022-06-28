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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSfused_reduction_benchmarkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSfused_reduction_benchmarkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSfused_reduction_benchmarkDTcc() {
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

#include <array>
#include <string>

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {
namespace {

const char* kReductionIR = R"(
  func.func @main(%lhs: {0}, %rhs: {0}) -> tensor<f32> {
    %lhs_abs = "tf.Abs"(%lhs) {{
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : ({0}) -> {0}
    %rhs_exp = "tf.Exp"(%rhs) {{
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : ({0}) -> {0}

    %add = "tf.Add"(%lhs_abs, %rhs_exp) {{
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : ({0}, {0}) -> {0}

    %dim_to_reduce = "tf.Const"() {{
      value = dense<[0]> : tensor<1xi32>,
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : () -> tensor<1xi32>
    %result = "tf.Prod"(%add, %dim_to_reduce) {{
      keep_dims = false,
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : ({0}, tensor<1xi32>) -> tensor<f32>
    func.return %result : tensor<f32>
  }
)";

std::string FusedReduction1D(bool dynamic, int64_t size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSfused_reduction_benchmarkDTcc mht_0(mht_0_v, 220, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/fused_reduction_benchmark.cc", "FusedReduction1D");

  return llvm::formatv(kReductionIR,
                       PrintTensorType({dynamic ? kDynSize : size}, "f32"));
}

auto EigenFusedReduction1D() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSfused_reduction_benchmarkDTcc mht_1(mht_1_v, 228, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/fused_reduction_benchmark.cc", "EigenFusedReduction1D");

  return [](llvm::ArrayRef<Tensor> inputs,
            llvm::Optional<Eigen::ThreadPoolDevice> device) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSfused_reduction_benchmarkDTcc mht_2(mht_2_v, 233, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/fused_reduction_benchmark.cc", "lambda");

    std::array<int64_t, 1> dims_to_reduce{0};
    Tensor output(DT_FLOAT, {});

    auto lhs = inputs[0].tensor<float, 1>();
    auto rhs = inputs[1].tensor<float, 1>();
    auto out = output.tensor<float, 0>();
    out.setZero();

    if (device.hasValue()) {
      out.device(*device) = (lhs.abs() + rhs.exp()).sum(dims_to_reduce);
    } else {
      out = (lhs.abs() + rhs.exp()).prod(dims_to_reduce);
    }
  };
}

llvm::SmallVector<InputTensorSpec> Inputs(ssize_t dim) {
  return {InputTensorSpec(DT_FLOAT, {dim}), InputTensorSpec(DT_FLOAT, {dim})};
}

#define BM(FN) BM_##FN->Arg(0);

#define BM_SUITE(NAME, DYNAMIC, SIZE)                                      \
  BM(JitrtV(NAME, FusedReduction1D(DYNAMIC, SIZE), "main", Inputs(SIZE))); \
  BM(Eigen(NAME, EigenFusedReduction1D(), Inputs(SIZE)));                  \
  BM(Tfrt(NAME, FusedReduction1D(DYNAMIC, SIZE), "main", Inputs(SIZE)))

#define BM_DYNAMIC(SIZE) \
  BM_SUITE(FusedReductionDynamic_##SIZE, kDynamicDim, SIZE)
BM_DYNAMIC(3);
BM_DYNAMIC(8);
BM_DYNAMIC(80);
BM_DYNAMIC(800);
BM_DYNAMIC(8000);
BM_DYNAMIC(8131);
BM_DYNAMIC(1000000);
BM_DYNAMIC(1010131);

#define BM_STATIC(SIZE) BM_SUITE(FusedReductionStatic_##SIZE, kStaticDim, SIZE)
BM_STATIC(3);
BM_STATIC(8);
BM_STATIC(80);
BM_STATIC(800);
BM_STATIC(8000);
BM_STATIC(8131);
BM_STATIC(1000000);
BM_STATIC(1010131);

}  // namespace
}  // namespace tensorflow
