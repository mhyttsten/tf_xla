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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSargmax_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSargmax_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSargmax_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/argmax_spmd_expander.h"

#include <cstdint>
#include <string>
#include <utility>

#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {
namespace {

StatusOr<Layout> ComputeResultLayout(mlir::Operation* op,
                                     const Layout& input_layout) {
  if (!mlir::isa<mlir::TF::ArgMaxOp>(op))
    return errors::Unimplemented("SPMD expansion for op type: ", OpName(op),
                                 " not yet implemented.");

  auto argmax_op = llvm::cast<mlir::TF::ArgMaxOp>(op);
  const auto input_rank = ValueRank(argmax_op.input());
  TF_ASSIGN_OR_RETURN(int64_t axis,
                      ExtractConstIntFromValue(argmax_op.dimension()));

  if (axis < 0) axis += input_rank;

  LayoutProto output_layout_proto;
  *output_layout_proto.mutable_mesh_config() = input_layout.mesh().ToProto();

  for (int i = 0; i < input_rank; ++i) {
    if (i != axis)
      output_layout_proto.add_sharding_specs()->set_sharding_spec(
          input_layout.sharding_spec(i));
  }
  return Layout::FromProto(output_layout_proto).ValueOrDie();
}
}  // namespace

StatusOr<mlir::Operation*> ArgMaxSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSargmax_spmd_expanderDTcc mht_0(mht_0_v, 234, "", "./tensorflow/dtensor/mlir/expansions/argmax_spmd_expander.cc", "ArgMaxSPMDExpander::ExpandOp");

  auto argmax_op = llvm::cast<mlir::TF::ArgMaxOp>(op);
  TF_ASSIGN_OR_RETURN(int64_t axis,
                      ExtractConstIntFromValue(argmax_op.dimension()));
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractLayoutFromOperand(argmax_op.input()));
  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(argmax_op));
  if (!input_layout || !output_layout)
    return errors::InvalidArgument(
        OpName(op), " is missing layouts during SPMD Expansion.");

  auto input = argmax_op.input();
  const auto input_rank = ValueRank(input);

  TF_ASSIGN_OR_RETURN(auto input_shape, GetShapeOfValue(input));

  if (input_rank == -1) return errors::Unimplemented("missing rank for input.");
  if (axis < 0) axis += input_rank;

  mlir::OpBuilder builder(op);
  {
    LayoutProto tgt_input_layout_proto;
    *tgt_input_layout_proto.mutable_mesh_config() =
        input_layout->mesh().ToProto();

    for (int i = 0; i < input_shape.size(); ++i) {
      // const auto dim_name
      if (i == axis) {
        // Set replicated for `axis` dim.
        tgt_input_layout_proto.add_sharding_specs()->set_sharding_spec(
            Layout::kUnshardedDim);
      } else {
        // Keep the rest dimension.
        tgt_input_layout_proto.add_sharding_specs()->set_sharding_spec(
            input_layout->sharding_spec(i));
      }
    }

    if (!Layout::IsUnshardedDimension(input_layout->sharding_spec(axis))) {
      TF_ASSIGN_OR_RETURN(
          input, EmitAllGather(
                     builder, input, *input_layout,
                     Layout::FromProto(tgt_input_layout_proto).ValueOrDie()));
    }
  }

  auto new_argmax = builder.create<mlir::TF::ArgMaxOp>(
      argmax_op.getLoc(), argmax_op.getResult().getType(), input,
      argmax_op.dimension());
  op->getResult(0).replaceAllUsesWith(new_argmax.output());
  op->erase();

  return InferSPMDExpandedLocalShape(new_argmax);
}

StatusOr<llvm::DenseMap<int, Layout>> ArgMaxSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If the input layout is missing, don't return an output layout.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  const Layout& input_layout = input_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(auto result_layout,
                      ComputeResultLayout(op, input_layout));
  if (result_layout.rank() != input_layout.rank() - 1)
    return errors::FailedPrecondition(
        OpName(op), " derived output layout rank is ", result_layout.rank(),
        " not ", input_layout.rank() - 1, " as expected.");

  return llvm::DenseMap<int, Layout>({{0, result_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> ArgMaxSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If no output layout, then do not infer any operand layouts.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto argmax_op = llvm::cast<mlir::TF::ArgMaxOp>(op);
  TF_ASSIGN_OR_RETURN(int64_t axis,
                      ExtractConstIntFromValue(argmax_op.dimension()));
  auto input = argmax_op.input();
  const auto input_rank = ValueRank(input);

  // Handle the case of negative axis.
  if (axis < 0) axis += input_rank;

  const Layout& output_layout = output_layouts.lookup(0);

  TF_ASSIGN_OR_RETURN(auto input_shape, GetShapeOfValue(input));

  std::vector<std::string> layout_sharding;

  int output_dim = 0;
  for (int i = 0; i < input_shape.size(); ++i) {
    if (i == axis) {
      layout_sharding.emplace_back(Layout::kUnshardedDim);
    } else {
      layout_sharding.emplace_back(output_layout.sharding_spec(output_dim));
      output_dim += 1;
    }
  }

  // Add Layout for first input attribute, while the second one is axis as a
  // scalar, we don't need to set its layout.
  TF_ASSIGN_OR_RETURN(const Layout result_layout,
                      Layout::GetLayout(layout_sharding, output_layout.mesh()));
  return llvm::DenseMap<int, Layout>({{0, result_layout}});
}

}  // namespace dtensor
}  // namespace tensorflow
