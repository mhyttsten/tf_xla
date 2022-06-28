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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_inst_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_inst_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_inst_testDTcc() {
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
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"

#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/metrics/types_util.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace mlir {
namespace TFL {
namespace {
using stream_executor::port::StatusOr;

// MockSuccessPass reports errors but doesn't fail.
class MockSuccessPass
    : public PassWrapper<MockSuccessPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_inst_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst_test.cc", "getDependentDialects");

    registry.insert<TF::TensorFlowDialect>();
  }

 public:
  explicit MockSuccessPass() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_inst_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst_test.cc", "MockSuccessPass");
}

 private:
  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_inst_testDTcc mht_2(mht_2_v, 231, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst_test.cc", "runOnOperation");

    getOperation().walk([](Operation* nestedOp) {
      nestedOp->emitError()
          << "Error at " << nestedOp->getName().getStringRef().str() << " op";
    });
  };
};

// MockFailurePass reports errors and fails.
class MockFailurePass
    : public PassWrapper<MockFailurePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_inst_testDTcc mht_3(mht_3_v, 245, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst_test.cc", "getDependentDialects");

    registry.insert<TF::TensorFlowDialect>();
  }

 public:
  explicit MockFailurePass() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_inst_testDTcc mht_4(mht_4_v, 253, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst_test.cc", "MockFailurePass");
}

 private:
  void runOnOperation() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPSerror_collector_inst_testDTcc mht_5(mht_5_v, 259, "", "./tensorflow/compiler/mlir/lite/metrics/error_collector_inst_test.cc", "runOnOperation");

    getOperation().walk([](Operation* nestedOp) {
      if (nestedOp->getName().getStringRef().str().rfind("tf.") != -1) {
        AttachErrorCode(
            nestedOp->emitError()
                << "Failed at " << nestedOp->getName().getStringRef().str()
                << " op",
            tflite::metrics::ConverterErrorData::ERROR_NEEDS_FLEX_OPS);
      }
    });
    signalPassFailure();
  };
};

StatusOr<OwningOpRef<mlir::ModuleOp>> LoadModule(MLIRContext* context,
                                                 const std::string& file_name) {
  std::string error_message;
  auto file = openInputFile(file_name, &error_message);
  if (!file) {
    return tensorflow::errors::InvalidArgument("fail to open input file");
  }

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return OwningOpRef<mlir::ModuleOp>(
      parseSourceFile<mlir::ModuleOp>(source_mgr, context));
}

TEST(ErrorCollectorTest, TessSuccessPass) {
  std::string input_file = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/lite/metrics/testdata/strided_slice.mlir");
  MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  context.enableMultithreading();

  auto module = LoadModule(&context, input_file);
  EXPECT_EQ(module.ok(), true);

  PassManager pm(&context, OpPassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<MockSuccessPass>());

  pm.addInstrumentation(
      std::make_unique<ErrorCollectorInstrumentation>(&context));
  EXPECT_EQ(succeeded(pm.run(module.ValueOrDie().get())), true);

  auto collected_errors =
      ErrorCollector::GetErrorCollector()->CollectedErrors();
  EXPECT_EQ(collected_errors.size(), 0);
}

TEST(ErrorCollectorTest, TessFailurePass) {
  using tflite::metrics::ConverterErrorData;
  MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  const std::string input_file =
      "tensorflow/compiler/mlir/lite/metrics/testdata/strided_slice.mlir";
  auto input_file_id = StringAttr::get(&context, input_file);

  context.allowUnregisteredDialects();
  context.enableMultithreading();

  auto module =
      LoadModule(&context, tensorflow::GetDataDependencyFilepath(input_file));
  EXPECT_EQ(module.ok(), true);

  PassManager pm(&context, OpPassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<MockSuccessPass>());
  pm.addPass(std::make_unique<MockFailurePass>());

  pm.addInstrumentation(
      std::make_unique<ErrorCollectorInstrumentation>(&context));
  EXPECT_EQ(succeeded(pm.run(module.ValueOrDie().get())), false);

  auto collected_errors =
      ErrorCollector::GetErrorCollector()->CollectedErrors();

  EXPECT_EQ(collected_errors.size(), 3);
  EXPECT_EQ(collected_errors.count(NewConverterErrorData(
                "MockFailurePass",
                "Failed at tf.Const op\nsee current operation: %0 = "
                "\"tf.Const\"() {value = dense<1> : tensor<4xi32>} : () -> "
                "tensor<4xi32>\nError code: ERROR_NEEDS_FLEX_OPS",
                ConverterErrorData::ERROR_NEEDS_FLEX_OPS, "tf.Const",
                mlir::FileLineColLoc::get(input_file_id, 2, 9))),
            1);
  EXPECT_EQ(collected_errors.count(NewConverterErrorData(
                "MockFailurePass",
                "Failed at tf.Const op\nsee current operation: %1 = "
                "\"tf.Const\"() {value = dense<0> : tensor<4xi32>} : () -> "
                "tensor<4xi32>\nError code: ERROR_NEEDS_FLEX_OPS",
                ConverterErrorData::ERROR_NEEDS_FLEX_OPS, "tf.Const",
                mlir::FileLineColLoc::get(input_file_id, 2, 9))),
            1);
  EXPECT_EQ(collected_errors.count(NewConverterErrorData(
                "MockFailurePass",
                "Failed at tf.StridedSlice op\nsee current operation: %2 = "
                "\"tf.StridedSlice\"(%arg0, %1, %1, %0) {begin_mask = 11 : "
                "i64, device = \"\", ellipsis_mask = 0 : i64, end_mask = 11 : "
                "i64, new_axis_mask = 4 : i64, shrink_axis_mask = 0 : i64} : "
                "(tensor<*xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) "
                "-> tensor<*xf32>\nError code: ERROR_NEEDS_FLEX_OPS",
                ConverterErrorData::ERROR_NEEDS_FLEX_OPS, "tf.StridedSlice",
                mlir::FileLineColLoc::get(input_file_id, 4, 10))),
            1);

  // Check the location information.
  std::vector<std::string> locations;
  for (const auto& error : collected_errors) {
    EXPECT_TRUE(error.has_location());
    locations.push_back(error.location().DebugString());
  }

  EXPECT_THAT(locations, Each(testing::HasSubstr("CALLSITELOC")));
  EXPECT_THAT(locations, Each(testing::HasSubstr(input_file)));
  EXPECT_THAT(locations, Contains(testing::HasSubstr("line: 2")));
  EXPECT_THAT(locations, Contains(testing::HasSubstr("column: 9")));
  EXPECT_THAT(locations, Contains(testing::HasSubstr("line: 4")));
  EXPECT_THAT(locations, Contains(testing::HasSubstr("column: 10")));
}
}  // namespace
}  // namespace TFL
}  // namespace mlir
