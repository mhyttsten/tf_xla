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
class MHTracer_DTPStensorflowPScorePSirPSops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSirPSops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSirPSops_testDTcc() {
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

#include "tensorflow/core/ir/ops.h"

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tfg {
namespace {

template <typename OpT>
OpT findOp(ModuleOp module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSirPSops_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/ir/ops_test.cc", "findOp");

  OpT result;
  module.walk([&](OpT op) {
    result = op;
    return WalkResult::interrupt();
  });
  assert(result);
  return result;
}

//===----------------------------------------------------------------------===//
// Unit tests for TFG region ops RegionBranchOpInterface API.
//===----------------------------------------------------------------------===//

TEST(TestTFGRegionOps, TestIfLikeRegionOpSuccessorRegions) {
  const char *const code = R"mlir(
    tfg.func @test(%arg0: tensor<i1>, %arg1: tensor<f32>) -> (tensor<f32>) {
      %IfRegion, %ctl = IfRegion %arg0 then {
        yield(%arg1) : tensor<f32>
      } else {
        yield(%arg1) : tensor<f32>
      } : (tensor<i1>) -> (tensor<f32>)
      return(%IfRegion) : tensor<f32>
    }
  )mlir";
  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  auto op = findOp<IfRegionOp>(*module);

  // Test region -> parent
  SmallVector<RegionSuccessor> regions;
  for (unsigned index = 0; index <= 1; ++index, regions.clear()) {
    op.getSuccessorRegions(index, /*operands=*/{Attribute()}, regions);
    ASSERT_EQ(regions.size(), 1u);
    EXPECT_TRUE(regions.front().isParent());
  }

  // Test parent -> regions
  op.getSuccessorRegions(/*index=*/llvm::None, /*operands=*/{Attribute()},
                         regions);
  EXPECT_EQ(regions.size(), 2u);
  regions.clear();

  // Test parent -> regions with known branch
  Builder b(&context);
  ShapedType tensor_type = RankedTensorType::get(/*shape=*/{}, b.getI1Type());
  Attribute cond = DenseElementsAttr::get(tensor_type, /*value=*/true);
  op.getSuccessorRegions(/*index=*/llvm::None, /*operands=*/{cond}, regions);
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.front().getSuccessor(), &op.then_region());
}

TEST(TestTFGRegionOps, TestCaseLikeRegionOpSuccessorRegions) {
  const char *const code = R"mlir(
    tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<f32>) {
      %CaseRegion, %ctl = CaseRegion %arg0 {
        yield(%arg1) : tensor<f32>
      }, {
        yield(%arg1) : tensor<f32>
      } : (tensor<i32>) -> (tensor<f32>)
      return(%CaseRegion) : tensor<f32>
    }
  )mlir";
  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  auto op = findOp<CaseRegionOp>(*module);

  // Test region -> parent
  SmallVector<RegionSuccessor> regions;
  for (unsigned index = 0; index < op.getNumRegions();
       ++index, regions.clear()) {
    op.getSuccessorRegions(index, /*operands=*/{Attribute()}, regions);
    ASSERT_EQ(regions.size(), 1u);
    EXPECT_TRUE(regions.front().isParent());
  }

  // Test parent -> region
  op.getSuccessorRegions(/*index=*/llvm::None, /*operands=*/{Attribute()},
                         regions);
  EXPECT_EQ(regions.size(), 2u);
  regions.clear();

  // Test parent -> region with known branch
  Builder b(&context);
  ShapedType tensor_type = RankedTensorType::get(/*shape=*/{}, b.getI32Type());
  Attribute branch = DenseElementsAttr::get(tensor_type, /*value=*/1);
  op.getSuccessorRegions(/*index=*/llvm::None, {branch}, regions);
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.front().getSuccessor(), &op.branches()[1]);
}

TEST(TestTFGRegionOps, TestWhileLikeRegionOpSuccessorRegions) {
  const char *const code = R"mlir(
    tfg.func @test(%arg0: tensor<f32>) -> (tensor<f32>) {
      %WhileRegion, %ctl = WhileRegion(%arg0) {
      ^bb0(%arg1: tensor<f32>, %arg2: !tf_type.control):
        %Cond, %ctl = Cond : () -> (tensor<i1>)
        condition %Cond : tensor<i1> (%arg1) : tensor<f32>
      } do {
      ^bb0(%arg1: tensor<f32>, %arg2: !tf_type.control):
        yield(%arg1) : tensor<f32>
      } {parallel_iterations = 10 : i64} : (tensor<f32>) -> (tensor<f32>)
      return(%WhileRegion) : tensor<f32>
    }
  )mlir";
  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  auto op = findOp<WhileRegionOp>(*module);

  // Test parent -> cond
  SmallVector<RegionSuccessor> regions;
  op.getSuccessorRegions(/*index=*/llvm::None, /*operands=*/{Attribute()},
                         regions);
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.front().getSuccessor(), &op.cond_region());
  regions.clear();

  // Test cond -> parent or body
  op.getSuccessorRegions(/*index=*/0, /*operands=*/{Attribute()}, regions);
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_TRUE(regions.front().isParent() ^ regions.back().isParent());
  regions.clear();

  // Test body -> cond
  op.getSuccessorRegions(/*index=*/1, /*operands=*/{Attribute()}, regions);
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.front().getSuccessor(), &op.cond_region());
  regions.clear();
}

TEST(TestTFGRegionOps, TestForLikeRegionOpSuccessorRegions) {
  const char *const code = R"mlir(
    tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<f32>) {
      %ForRegion, %ctl = ForRegion(%arg1) from %arg0 to %arg0 by %arg0 {
        ^bb0(%arg2: tensor<i32>, %arg3: tensor<f32>,
             %arg4: !tf_type.control, %arg5: !tf_type.control):
        yield(%arg3) : tensor<f32>
      } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>) -> (tensor<f32>)
      return(%ForRegion) : tensor<f32>
    }
  )mlir";
  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  auto op = findOp<ForRegionOp>(*module);

  // Test parent -> body
  SmallVector<RegionSuccessor> regions;
  op.getSuccessorRegions(/*index=*/llvm::None, /*operands=*/{Attribute()},
                         regions);
  EXPECT_EQ(regions.size(), 1u);
  regions.clear();

  // Test body -> body or parent
  op.getSuccessorRegions(/*index=*/0, /*operands=*/{Attribute()}, regions);
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_TRUE(regions.front().isParent() ^ regions.back().isParent());
}

}  // namespace
}  // namespace tfg
}  // namespace mlir
