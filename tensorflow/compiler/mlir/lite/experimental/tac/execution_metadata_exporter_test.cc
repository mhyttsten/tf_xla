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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPSexecution_metadata_exporter_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPSexecution_metadata_exporter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPSexecution_metadata_exporter_testDTcc() {
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

// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/mlir/lite/experimental/tac/execution_metadata_exporter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/runtime_metadata_generated.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace tflite {

std::string CreateRuntimeMetadata() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPSexecution_metadata_exporter_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/mlir/lite/experimental/tac/execution_metadata_exporter_test.cc", "CreateRuntimeMetadata");

  flatbuffers::FlatBufferBuilder fb_builder;

  std::vector<flatbuffers::Offset<flatbuffers::String>> device_names = {
      fb_builder.CreateString("GPU"), fb_builder.CreateString("CPU")};

  const auto hardwares =
      CreateHardwareMetadata(fb_builder, fb_builder.CreateVector(device_names));
  const auto ops = {
      CreateOpMetadata(fb_builder, 0, 0,
                       fb_builder.CreateVector(std::vector<float>({1.0, 5.0}))),
      CreateOpMetadata(fb_builder, 1, 0,
                       fb_builder.CreateVector(std::vector<float>({1.0, 5.0}))),
      CreateOpMetadata(fb_builder, 2, 0,
                       fb_builder.CreateVector(std::vector<float>({1.0, 5.0}))),
      CreateOpMetadata(
          fb_builder, 3, 1,
          fb_builder.CreateVector(std::vector<float>({-1.0, 2.0}))),
  };
  const auto subgraphs = {CreateSubgraphMetadata(
      fb_builder, fb_builder.CreateVector(ops.begin(), ops.size()))};

  const auto metadata = CreateRuntimeMetadata(
      fb_builder, hardwares,
      fb_builder.CreateVector(subgraphs.begin(), subgraphs.size()));
  fb_builder.Finish(metadata);

  return std::string(
      reinterpret_cast<const char*>(fb_builder.GetBufferPointer()),
      fb_builder.GetSize());
}

void Verify(const RuntimeMetadata* result, const RuntimeMetadata* expected) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPSexecution_metadata_exporter_testDTcc mht_1(mht_1_v, 235, "", "./tensorflow/compiler/mlir/lite/experimental/tac/execution_metadata_exporter_test.cc", "Verify");

  EXPECT_EQ(result->subgraph_metadata()->size(),
            expected->subgraph_metadata()->size());
  for (int i = 0; i < result->subgraph_metadata()->size(); ++i) {
    auto result_subgraph_metadata =
        result->subgraph_metadata()->GetAs<SubgraphMetadata>(i);
    auto expected_subgraph_metadata =
        expected->subgraph_metadata()->GetAs<SubgraphMetadata>(i);
    if (expected_subgraph_metadata->op_metadata() == nullptr &&
        result_subgraph_metadata->op_metadata() == nullptr) {
      return;
    }
    ASSERT_EQ(expected_subgraph_metadata->op_metadata()->size(),
              result_subgraph_metadata->op_metadata()->size());
    for (int j = 0; j < expected_subgraph_metadata->op_metadata()->size();
         ++j) {
      auto result_op_metadata =
          result_subgraph_metadata->op_metadata()->GetAs<OpMetadata>(j);
      auto expected_op_metadata =
          expected_subgraph_metadata->op_metadata()->GetAs<OpMetadata>(j);
      EXPECT_EQ(result_op_metadata->index(), expected_op_metadata->index());
      EXPECT_EQ(result_op_metadata->hardware(),
                expected_op_metadata->hardware());

      EXPECT_EQ(result_op_metadata->op_costs()->size(),
                expected_op_metadata->op_costs()->size());
      for (int i = 0; i < result_op_metadata->op_costs()->size(); ++i) {
        EXPECT_FLOAT_EQ(result_op_metadata->op_costs()->Get(i),
                        expected_op_metadata->op_costs()->Get(i));
      }
    }
  }
}

TEST(ExporterTest, Valid) {
  const std::string kMLIR = R"(
func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>) -> tensor<2x1xf32> {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU6",  per_device_costs = {CPU = 5.0 : f32, GPU = 1.0 : f32}, tac.device = "GPU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "tfl.mul"(%0, %arg2) {fused_activation_function = "RELU6", per_device_costs = {CPU = 5.0 : f32, GPU = 1.0 : f32}, tac.device = "GPU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.add"(%arg0, %arg3) {fused_activation_function = "RELU6", per_device_costs = {CPU = 5.0 : f32, GPU = 1.0 : f32}, tac.device = "GPU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.pack"(%1, %2) {axis = 0 : i32, per_device_costs = {CPU = 2.0 : f32, GPU = -1.0 : f32}, values_count = 2 : i32, tac.device = "CPU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  func.return %3 : tensor<2x1xf32>
})";
  const std::string kExpectedFB = CreateRuntimeMetadata();
  mlir::DialectRegistry registry;
  registry.insert<mlir::TFL::TensorFlowLiteDialect,
                  mlir::arith::ArithmeticDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::parseSourceString<mlir::ModuleOp>(kMLIR, &context));
  auto module_op = module.get();
  auto serialized_result_fb = ExportRuntimeMetadata(module_op);
  const auto* result =
      GetRuntimeMetadata(serialized_result_fb.getValue().c_str());
  const auto* expected = GetRuntimeMetadata(kExpectedFB.c_str());
  ASSERT_TRUE(result != nullptr);
  ASSERT_TRUE(result->subgraph_metadata() != nullptr);
  ASSERT_TRUE(expected->subgraph_metadata() != nullptr);
  Verify(result, expected);
}

}  // namespace tflite
