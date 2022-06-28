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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_transformsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_transformsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_transformsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter_transforms.h"

#include "absl/algorithm/container.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace experimental {

using mlir::OpBuilder;

BoundAffineMap GetBoundAffineMapFrom(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_transformsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter_transforms.cc", "GetBoundAffineMapFrom");

  if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(op)) {
    return {load.getAffineMap(),
            std::vector<mlir::Value>(load.getMapOperands().begin(),
                                     load.getMapOperands().end())};
  } else if (auto store = mlir::dyn_cast<mlir::AffineStoreOp>(op)) {
    return {store.getAffineMap(),
            std::vector<mlir::Value>(store.getMapOperands().begin(),
                                     store.getMapOperands().end())};
  } else {
    CHECK(false);
  }
}

mlir::Operation* CloneWithNewAffineMap(mlir::Operation* op,
                                       BoundAffineMap new_affine,
                                       OpBuilder builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_transformsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter_transforms.cc", "CloneWithNewAffineMap");

  if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(op)) {
    return builder.create<mlir::AffineLoadOp>(
        builder.getUnknownLoc(), load.getMemRef(), new_affine.affine_map,
        new_affine.operands);
  } else if (auto store = mlir::dyn_cast<mlir::AffineStoreOp>(op)) {
    return builder.create<mlir::AffineStoreOp>(
        builder.getUnknownLoc(), store.getValueToStore(), store.getMemRef(),
        new_affine.affine_map, new_affine.operands);
  } else {
    CHECK(false);
  }
}

bool IsSimpleLoop(mlir::AffineForOp loop) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_transformsDTcc mht_2(mht_2_v, 234, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter_transforms.cc", "IsSimpleLoop");

  return loop.getLowerBoundMap().isSingleConstant() &&
         loop.getLowerBoundMap().getSingleConstantResult() == 0 &&
         loop.getStep() == 1 && loop.getUpperBoundMap().getNumResults() == 1 &&
         std::next(loop.region().begin()) == loop.region().end();
}

std::vector<mlir::AffineForOp> CreateNestedSimpleLoops(
    absl::Span<const int64_t> upper_bounds, OpBuilder builder) {
  std::vector<mlir::AffineForOp> loops;
  loops.reserve(upper_bounds.size());
  for (int64_t dim : upper_bounds) {
    auto loop =
        builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 0, dim);
    loops.push_back(loop);
    builder = OpBuilder::atBlockTerminator(loop.getBody());
  }
  return loops;
}

void SetBoundForSimpleLoop(mlir::AffineForOp loop, mlir::AffineExpr new_bound,
                           OpBuilder builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_transformsDTcc mht_3(mht_3_v, 258, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter_transforms.cc", "SetBoundForSimpleLoop");

  CHECK(IsSimpleLoop(loop));

  loop.setUpperBoundMap(mlir::AffineMap::get(
      loop.getUpperBoundMap().getNumDims(),
      loop.getUpperBoundMap().getNumSymbols(), {new_bound}));
}

mlir::AffineForOp TileLoop(mlir::AffineForOp loop, int64_t size,
                           mlir::AffineForOp target) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_transformsDTcc mht_4(mht_4_v, 270, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter_transforms.cc", "TileLoop");

  CHECK(IsSimpleLoop(loop));
  CHECK(IsSimpleLoop(target));
  {
    llvm::SmallVector<mlir::AffineForOp, 4> all_loops;
    getPerfectlyNestedLoops(all_loops, loop);
    CHECK(absl::c_linear_search(all_loops, target));
  }

  auto builder = OpBuilder::atBlockTerminator(target.getBody());

  auto inner_loop =
      builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 0, size);
  {
    auto& inner_operations = inner_loop.getBody()->getOperations();
    auto& target_operations = target.getBody()->getOperations();

    inner_operations.splice(inner_operations.begin(), target_operations,
                            target_operations.begin(),
                            std::prev(target_operations.end(), 2));

    mlir::AffineExpr length = loop.getUpperBoundMap().getResult(0);
    CHECK_EQ(0, length.cast<mlir::AffineConstantExpr>().getValue() % size);
    SetBoundForSimpleLoop(loop, length.ceilDiv(size), builder);
  }

  for (auto& use :
       llvm::make_early_inc_range(loop.getInductionVar().getUses())) {
    mlir::Operation* owner = use.getOwner();
    BoundAffineMap affine_map = GetBoundAffineMapFrom(owner);
    unsigned new_dim = affine_map.operands.size();
    affine_map.operands.push_back(inner_loop.getInductionVar());
    std::vector<mlir::AffineExpr> replacements;
    for (int i = 0; i < affine_map.affine_map.getNumDims(); i++) {
      if (affine_map.operands[i] == loop.getInductionVar()) {
        replacements.push_back(builder.getAffineDimExpr(i) * size +
                               builder.getAffineDimExpr(new_dim));
      } else {
        replacements.push_back(builder.getAffineDimExpr(i));
      }
    }
    affine_map.affine_map = affine_map.affine_map.replaceDimsAndSymbols(
        replacements, {}, affine_map.operands.size(), 0);
    auto new_op = CloneWithNewAffineMap(owner, affine_map, OpBuilder(owner));
    owner->replaceAllUsesWith(new_op);
    owner->erase();
  }
  return inner_loop;
}

void SinkPerfectlyNestedLoops(llvm::MutableArrayRef<mlir::AffineForOp> loops,
                              int rotate_amount) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_transformsDTcc mht_5(mht_5_v, 324, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter_transforms.cc", "SinkPerfectlyNestedLoops");

  CHECK_GE(rotate_amount, 0);
  std::vector<unsigned> permutation(loops.size());
  std::iota(permutation.begin(), permutation.end(), unsigned(0));
  std::rotate(permutation.begin(),
              permutation.begin() + loops.size() - rotate_amount,
              permutation.end());
  mlir::permuteLoops(loops, permutation);
}

}  // namespace experimental
}  // namespace xla
