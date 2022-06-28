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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPStranspose_op_benchmarkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPStranspose_op_benchmarkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPStranspose_op_benchmarkDTcc() {
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
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {

static const char* mlir_2d_input = R"(
func.func @compute(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {{
    %0 = "tf.Const"()
         {{value = dense<[1, 0]> : tensor<2xi64>,
          device = "/job:localhost/replica:0/task:0/device:CPU:0"}
         : () -> tensor<2xi64>
    %1 = "tf.Transpose"(%arg0, %0)
         {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
         : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    func.return %1 : tensor<?x?xf32>
  }
)";

static const char* mlir_3d_input = R"(
func.func @compute(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {{
    %0 = "tf.Const"()
         {{value = dense<[{0}, {1}, {2}]> : tensor<3xi64>,
          device = "/job:localhost/replica:0/task:0/device:CPU:0"}
         : () -> tensor<3xi64>
    %1 = "tf.Transpose"(%arg0, %0)
         {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
         : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
    func.return %1 : tensor<?x?x?xf32>
  }
)";

static std::string Transpose2D() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPStranspose_op_benchmarkDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/transpose_op_benchmark.cc", "Transpose2D");
 return llvm::formatv(mlir_2d_input); }

static std::string Transpose3D(std::array<int32_t, 3> perm) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPStranspose_op_benchmarkDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/transpose_op_benchmark.cc", "Transpose3D");

  return llvm::formatv(mlir_3d_input, perm[0], perm[1], perm[2]);
}

template <int32_t size>
static auto Shuffle(std::array<int32_t, size> perm) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPStranspose_op_benchmarkDTcc mht_2(mht_2_v, 232, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/transpose_op_benchmark.cc", "Shuffle");

  return [perm](llvm::ArrayRef<Tensor> inputs,
                llvm::Optional<Eigen::ThreadPoolDevice> device) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPStranspose_op_benchmarkDTcc mht_3(mht_3_v, 237, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/transpose_op_benchmark.cc", "lambda");

    std::array<int64_t, size> shuffled;
    for (unsigned d = 0; d < size; d++)
      shuffled[d] = inputs[0].dim_size(perm[d]);

    Tensor output(DT_FLOAT, TensorShape(shuffled));

    auto in0 = inputs[0].tensor<float, size>();
    auto out0 = output.tensor<float, size>();

    if (device.hasValue()) {
      out0.device(*device) = in0.shuffle(perm);
    } else {
      out0 = in0.shuffle(perm);
    }
  };
}

static llvm::SmallVector<InputTensorSpec> Inputs(llvm::ArrayRef<ssize_t> dims) {
  return {InputTensorSpec(DT_FLOAT, dims)};
}

#define BM(FN) BM_##FN->Arg(0)->Arg(4)->Arg(8);

// Small 2D Transpose: [1, 0]
BM(Jitrt(Transpose_small_1x0, Transpose2D(), "compute", Inputs({128, 128})));
BM(JitrtV(Transpose_small_1x0, Transpose2D(), "compute", Inputs({128, 128})));
BM(Tfrt(Transpose_small_1x0, Transpose2D(), "compute", Inputs({128, 128})));
BM(Eigen(Transpose_small_1x0, Shuffle<2>({1, 0}), Inputs({128, 128})));

// Small 3D Transpose: [0, 2, 1]
BM(Jitrt(Transpose_small_0x2x1, Transpose3D({0, 2, 1}), "compute",
         Inputs({32, 32, 16})));
BM(JitrtV(Transpose_small_0x2x1, Transpose3D({0, 2, 1}), "compute",
          Inputs({32, 32, 16})));
BM(Tfrt(Transpose_small_0x2x1, Transpose3D({0, 2, 1}), "compute",
        Inputs({32, 32, 16})));
BM(Eigen(Transpose_small_0x2x1, Shuffle<3>({0, 2, 1}), Inputs({32, 32, 16})));

// Small 3D Transpose: [2, 0, 1]
BM(Jitrt(Transpose_small_2x0x1, Transpose3D({2, 0, 1}), "compute",
         Inputs({32, 32, 16})));
BM(JitrtV(Transpose_small_2x0x1, Transpose3D({2, 0, 1}), "compute",
          Inputs({32, 32, 16})));
BM(Tfrt(Transpose_small_2x0x1, Transpose3D({2, 0, 1}), "compute",
        Inputs({32, 32, 16})));
BM(Eigen(Transpose_small_2x0x1, Shuffle<3>({2, 0, 1}), Inputs({32, 32, 16})));

// Small 3D Transpose: [2, 1, 0]
BM(Jitrt(Transpose_small_2x1x0, Transpose3D({2, 1, 0}), "compute",
         Inputs({32, 32, 16})));
BM(JitrtV(Transpose_small_2x1x0, Transpose3D({2, 1, 0}), "compute",
          Inputs({32, 32, 16})));
BM(Tfrt(Transpose_small_2x1x0, Transpose3D({2, 1, 0}), "compute",
        Inputs({32, 32, 16})));
BM(Eigen(Transpose_small_2x1x0, Shuffle<3>({2, 1, 0}), Inputs({32, 32, 16})));

// Small 3D Transpose: [1, 2, 0]
BM(Jitrt(Transpose_small_1x2x0, Transpose3D({1, 2, 0}), "compute",
         Inputs({32, 32, 16})));
BM(JitrtV(Transpose_small_1x2x0, Transpose3D({1, 2, 0}), "compute",
          Inputs({32, 32, 16})));
BM(Tfrt(Transpose_small_1x2x0, Transpose3D({1, 2, 0}), "compute",
        Inputs({32, 32, 16})));
BM(Eigen(Transpose_small_1x2x0, Shuffle<3>({1, 2, 0}), Inputs({32, 32, 16})));

// Small 3D Transpose: [1, 0, 2]
BM(Jitrt(Transpose_small_1x0x2, Transpose3D({1, 0, 2}), "compute",
         Inputs({32, 32, 16})));
BM(JitrtV(Transpose_small_1x0x2, Transpose3D({1, 0, 2}), "compute",
          Inputs({32, 32, 16})));
BM(Tfrt(Transpose_small_1x0x2, Transpose3D({1, 0, 2}), "compute",
        Inputs({32, 32, 16})));
BM(Eigen(Transpose_small_1x0x2, Shuffle<3>({1, 0, 2}), Inputs({32, 32, 16})));

// Medium 2D Transpose: [1, 0]
BM(Jitrt(Transpose_medium_1x0, Transpose2D(), "compute", Inputs({4096, 4096})));
BM(JitrtV(Transpose_medium_1x0, Transpose2D(), "compute",
          Inputs({4096, 4096})));
BM(Tfrt(Transpose_medium_1x0, Transpose2D(), "compute", Inputs({4096, 4096})));
BM(Eigen(Transpose_medium_1x0, Shuffle<2>({1, 0}), Inputs({4096, 4096})));

// Medium 3D Transpose: [0, 2, 1]
BM(Jitrt(Transpose_medium_0x2x1, Transpose3D({0, 2, 1}), "compute",
         Inputs({256, 256, 256})));
BM(JitrtV(Transpose_medium_0x2x1, Transpose3D({0, 2, 1}), "compute",
          Inputs({256, 256, 256})));
BM(Tfrt(Transpose_medium_0x2x1, Transpose3D({0, 2, 1}), "compute",
        Inputs({256, 256, 256})));
BM(Eigen(Transpose_medium_0x2x1, Shuffle<3>({0, 2, 1}),
         Inputs({256, 256, 256})));

// Medium 3D Transpose: [2, 0, 1]
BM(Jitrt(Transpose_medium_2x0x1, Transpose3D({2, 0, 1}), "compute",
         Inputs({256, 256, 256})));
BM(JitrtV(Transpose_medium_2x0x1, Transpose3D({2, 0, 1}), "compute",
          Inputs({256, 256, 256})));
BM(Tfrt(Transpose_medium_2x0x1, Transpose3D({2, 0, 1}), "compute",
        Inputs({256, 256, 256})));
BM(Eigen(Transpose_medium_2x0x1, Shuffle<3>({2, 0, 1}),
         Inputs({256, 256, 256})));

// Medium 3D Transpose: [2, 1, 0]
BM(Jitrt(Transpose_medium_2x1x0, Transpose3D({2, 1, 0}), "compute",
         Inputs({256, 256, 256})));
BM(JitrtV(Transpose_medium_2x1x0, Transpose3D({2, 1, 0}), "compute",
          Inputs({256, 256, 256})));
BM(Tfrt(Transpose_medium_2x1x0, Transpose3D({2, 1, 0}), "compute",
        Inputs({256, 256, 256})));
BM(Eigen(Transpose_medium_2x1x0, Shuffle<3>({2, 1, 0}),
         Inputs({256, 256, 256})));

// Medium 3D Transpose: [1, 2, 0]
BM(Jitrt(Transpose_medium_1x2x0, Transpose3D({1, 2, 0}), "compute",
         Inputs({256, 256, 256})));
BM(JitrtV(Transpose_medium_1x2x0, Transpose3D({1, 2, 0}), "compute",
          Inputs({256, 256, 256})));
BM(Tfrt(Transpose_medium_1x2x0, Transpose3D({1, 2, 0}), "compute",
        Inputs({256, 256, 256})));
BM(Eigen(Transpose_medium_1x2x0, Shuffle<3>({1, 2, 0}),
         Inputs({256, 256, 256})));

// Medium 3D Transpose: [1, 0, 2]
BM(Jitrt(Transpose_medium_1x0x2, Transpose3D({1, 0, 2}), "compute",
         Inputs({256, 256, 256})));
BM(JitrtV(Transpose_medium_1x0x2, Transpose3D({1, 0, 2}), "compute",
          Inputs({256, 256, 256})));
BM(Tfrt(Transpose_medium_1x0x2, Transpose3D({1, 0, 2}), "compute",
        Inputs({256, 256, 256})));
BM(Eigen(Transpose_medium_1x0x2, Shuffle<3>({1, 0, 2}),
         Inputs({256, 256, 256})));

// Large 2D Transpose: [1, 0]
BM(Jitrt(Transpose_large_1x0, Transpose2D(), "compute", Inputs({8192, 8192})));
BM(JitrtV(Transpose_large_1x0, Transpose2D(), "compute", Inputs({8192, 8192})));
BM(Tfrt(Transpose_large_1x0, Transpose2D(), "compute", Inputs({8192, 8192})));
BM(Eigen(Transpose_large_1x0, Shuffle<2>({1, 0}), Inputs({8192, 8192})));

// Large 3D Transpose: [0, 2, 1]
BM(Jitrt(Transpose_large_0x2x1, Transpose3D({0, 2, 1}), "compute",
         Inputs({448, 448, 448})));
BM(JitrtV(Transpose_large_0x2x1, Transpose3D({0, 2, 1}), "compute",
          Inputs({448, 448, 448})));
BM(Tfrt(Transpose_large_0x2x1, Transpose3D({0, 2, 1}), "compute",
        Inputs({448, 448, 448})));
BM(Eigen(Transpose_large_0x2x1, Shuffle<3>({0, 2, 1}),
         Inputs({448, 448, 448})));

// Large 3D Transpose: [2, 0, 1]
BM(Jitrt(Transpose_large_2x0x1, Transpose3D({2, 0, 1}), "compute",
         Inputs({448, 448, 448})));
BM(JitrtV(Transpose_large_2x0x1, Transpose3D({2, 0, 1}), "compute",
          Inputs({448, 448, 448})));
BM(Tfrt(Transpose_large_2x0x1, Transpose3D({2, 0, 1}), "compute",
        Inputs({448, 448, 448})));
BM(Eigen(Transpose_large_2x0x1, Shuffle<3>({2, 0, 1}),
         Inputs({448, 448, 448})));

// Large 3D Transpose: [2, 1, 0]
BM(Jitrt(Transpose_large_2x1x0, Transpose3D({2, 1, 0}), "compute",
         Inputs({448, 448, 448})));
BM(JitrtV(Transpose_large_2x1x0, Transpose3D({2, 1, 0}), "compute",
          Inputs({448, 448, 448})));
BM(Tfrt(Transpose_large_2x1x0, Transpose3D({2, 1, 0}), "compute",
        Inputs({448, 448, 448})));
BM(Eigen(Transpose_large_2x1x0, Shuffle<3>({2, 1, 0}),
         Inputs({448, 448, 448})));

// Large 3D Transpose: [1, 2, 0]
BM(Jitrt(Transpose_large_1x2x0, Transpose3D({1, 2, 0}), "compute",
         Inputs({448, 448, 448})));
BM(JitrtV(Transpose_large_1x2x0, Transpose3D({1, 2, 0}), "compute",
          Inputs({448, 448, 448})));
BM(Tfrt(Transpose_large_1x2x0, Transpose3D({1, 2, 0}), "compute",
        Inputs({448, 448, 448})));
BM(Eigen(Transpose_large_1x2x0, Shuffle<3>({1, 2, 0}),
         Inputs({448, 448, 448})));

// Large 3D Transpose: [1, 0, 2]
BM(Jitrt(Transpose_large_1x0x2, Transpose3D({1, 0, 2}), "compute",
         Inputs({448, 448, 448})));
BM(JitrtV(Transpose_large_1x0x2, Transpose3D({1, 0, 2}), "compute",
          Inputs({448, 448, 448})));
BM(Tfrt(Transpose_large_1x0x2, Transpose3D({1, 0, 2}), "compute",
        Inputs({448, 448, 448})));
BM(Eigen(Transpose_large_1x0x2, Shuffle<3>({1, 0, 2}),
         Inputs({448, 448, 448})));

}  // namespace tensorflow
