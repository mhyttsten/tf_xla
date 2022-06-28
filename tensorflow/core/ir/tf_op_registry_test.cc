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
class MHTracer_DTPStensorflowPScorePSirPStf_op_registry_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSirPStf_op_registry_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSirPStf_op_registry_testDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/tf_op_registry.h"

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tfg {
namespace {

// Register the TFG dialect and register the op registry interface.
void PrepareContext(MLIRContext *context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_registry_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/ir/tf_op_registry_test.cc", "PrepareContext");

  DialectRegistry registry;
  registry.insert<TFGraphDialect>();
  registry.addExtension(+[](mlir::MLIRContext *ctx, TFGraphDialect *dialect) {
    dialect->addInterfaces<TensorFlowOpRegistryInterface>();
  });
  context->appendDialectRegistry(registry);
}

TEST(TensorFlowOpRegistryInterface, TestStatelessFuncAndReturn) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  PrepareContext(&context);

  const char *const code = R"mlir(
    tfg.func @test(%arg: tensor<i32>) -> (tensor<i32>) {
      return(%arg) : tensor<i32>
    }
  )mlir";
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);

  auto func_op = cast<GraphFuncOp>(&module->front());
  auto ret_op = cast<ReturnOp>(func_op.body().front().getTerminator());
  EXPECT_FALSE(dyn_cast<TensorFlowRegistryInterface>(*func_op).isStateful());
  EXPECT_FALSE(dyn_cast<TensorFlowRegistryInterface>(*ret_op).isStateful());
}

TEST(TensorFlowOpRegistryInterface, TestStatelessTFOps) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  PrepareContext(&context);

  const char *const code = R"mlir(
    tfg.func @test(%lhs: tensor<i32>, %rhs: tensor<i32>) -> (tensor<i32>) {
      %Add, %ctl = Add(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      return(%Add) : tensor<i32>
    }
  )mlir";
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);

  Operation *add = &cast<GraphFuncOp>(&module->front()).body().front().front();
  auto iface = dyn_cast<TensorFlowRegistryInterface>(add);
  ASSERT_TRUE(iface);
  EXPECT_FALSE(iface.isStateful());
}

TEST(TensorFlowOpRegistryInterface, TestStatelessAndStatefulRegionOps) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  PrepareContext(&context);

  const char *const code_template = R"mlir(
    tfg.func @test(%idx: tensor<i32>, %arg: tensor<i32>) -> (tensor<i32>) {{
      %Case, %ctl = {0}CaseRegion %idx {{
        yield(%arg) : tensor<i32>
      } : (tensor<i32>) -> (tensor<i32>)
      return(%Case) : tensor<i32>
    }
  )mlir";
  SmallVector<StringRef, 2> prefixes = {"", "Stateless"};
  SmallVector<bool, 2> expected = {true, false};
  for (auto it : llvm::zip(prefixes, expected)) {
    std::string code = llvm::formatv(code_template, std::get<0>(it)).str();
    OwningOpRef<ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(code, &context);
    ASSERT_TRUE(module);

    Operation *case_op =
        &cast<GraphFuncOp>(&module->front()).body().front().front();
    auto iface = dyn_cast<TensorFlowRegistryInterface>(case_op);
    ASSERT_TRUE(iface);
    EXPECT_EQ(iface.isStateful(), std::get<1>(it));
  }
}
}  // namespace
}  // namespace tfg
}  // namespace mlir
