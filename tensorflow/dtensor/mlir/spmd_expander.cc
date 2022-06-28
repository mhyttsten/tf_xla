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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/spmd_expander.h"

#include <climits>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

// static
SPMDExpanderRegistry* SPMDExpanderRegistry::Global() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTcc mht_0(mht_0_v, 216, "", "./tensorflow/dtensor/mlir/spmd_expander.cc", "SPMDExpanderRegistry::Global");

  static SPMDExpanderRegistry* registry = new SPMDExpanderRegistry();
  return registry;
}

SPMDExpanderBase* SPMDExpanderRegistry::GetPropagateFnForOp(
    mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTcc mht_1(mht_1_v, 225, "", "./tensorflow/dtensor/mlir/spmd_expander.cc", "SPMDExpanderRegistry::GetPropagateFnForOp");

  auto key = OpName(op);
  auto fn = op_to_propagate_fn_map_.find(key);
  if (fn == op_to_propagate_fn_map_.end()) return nullptr;
  return fn->second.get();
}

InitOnStartupMarker SPMDExpanderRegistry::RegisterPropagateFn(
    std::string opName, std::unique_ptr<SPMDExpanderBase> prop) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("opName: \"" + opName + "\"");
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTcc mht_2(mht_2_v, 237, "", "./tensorflow/dtensor/mlir/spmd_expander.cc", "SPMDExpanderRegistry::RegisterPropagateFn");

  CHECK(op_to_propagate_fn_map_  // Crash ok
            .insert_or_assign(opName, std::move(prop))
            .second);
  return {};
}

Status SPMDExpanderBase::ExpandOpAndSetLayout(mlir::Operation* op,
                                              mlir::Operation** output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTcc mht_3(mht_3_v, 248, "", "./tensorflow/dtensor/mlir/spmd_expander.cc", "SPMDExpanderBase::ExpandOpAndSetLayout");

  TF_ASSIGN_OR_RETURN(std::vector<absl::optional<Layout>> computed_layout,
                      ExtractLayoutFromOp(op));

  if (computed_layout.empty() && op->getNumResults() != 0) {
    return errors::InvalidArgument(
        absl::StrCat("No attachced layout found for op : ", OpName(op),
                     " This might be due to an error in layout propagation.")
            .c_str());
  }

  // `op` may be removed/replaced from the graph during SPMD expansion, so
  // extract the global output shape before expansion.
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> global_output_shapes;
  global_output_shapes.reserve(op->getNumResults());
  for (auto output_value : op->getResults()) {
    auto maybe_ranked =
        output_value.getType().dyn_cast<mlir::RankedTensorType>();
    // Do not extract global shape if the shape isn't statically known.
    //
    // This is a bit subtle and relies on the check of static shape of output
    // value below when extracting local_shape. We probably should consider a
    // placeholder for unknown shapes to avoid surprises in the future.
    //
    // Given the nature of RestoreV2 op and its output ranks, we only special
    // case for RestoreV2 for now.
    if (llvm::isa<mlir::TF::RestoreV2Op, mlir::TF::DTensorRestoreV2Op>(op) &&
        (!maybe_ranked || !maybe_ranked.hasStaticShape()))
      continue;
    TF_ASSIGN_OR_RETURN(auto global_shape,
                        ExtractGlobalOutputShape(output_value));
    global_output_shapes.emplace_back(llvm::SmallVector<int64_t, 4>{
        global_shape.begin(), global_shape.end()});
  }

  TF_ASSIGN_OR_RETURN(*output, this->ExpandOp(op));

  // TODO(hthu): Use ToString() instead.
  SetLayoutOnOp(*output, absl::Span<absl::optional<Layout>>(
                             computed_layout.data(), computed_layout.size()));

  // Verify the local shape of the expanded operation matches the shape expected
  // from the layout. Note that this does **not** catch all errors. When tensor
  // dimension is sharded in a wrong mesh with the same device cardinality as
  // the correct/expected mesh, this check will still pass.
  for (const auto& output_layout_and_index :
       llvm::enumerate(llvm::zip((*output)->getResults(), computed_layout))) {
    const int index = output_layout_and_index.index();
    const auto& output_and_layout = output_layout_and_index.value();

    auto output_value = std::get<0>(output_and_layout);
    // Extract the static shape of `output_value` if possible, otherwise ignore
    // this output.
    auto local_expanded_shape_or_status = GetShapeOfValue(output_value);
    if (!local_expanded_shape_or_status.ok()) continue;

    const auto local_expanded_shape =
        local_expanded_shape_or_status.ValueOrDie();
    const auto& layout = std::get<1>(output_and_layout);
    const auto expected_global_shape =
        layout->GlobalShapeFromLocalShape(local_expanded_shape);

    for (const auto& expanded_and_true_global_shape :
         llvm::zip(global_output_shapes[index], expected_global_shape)) {
      const auto expanded_shape = std::get<0>(expanded_and_true_global_shape);
      const auto expected_shape = std::get<1>(expanded_and_true_global_shape);
      // If any of the shape has unknown dimension, do not check/validate the
      // shape.
      if (expanded_shape <= 0 || expected_shape <= 0) continue;

      if (expanded_shape != expected_shape) {
        return errors::Internal(
            "SPMD expansion resulted in op output inconsistent with the "
            "provided layout.");
      }
    }
  }

  return Status::OK();
}

StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  return errors::Unimplemented(
      "ComputeLayoutForward API must be implemented via the subclass.");
}

StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
    const llvm::DenseMap<int, Layout>& output_layouts) {
  return ComputeLayoutForward(op, input_layouts);
}

StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return errors::Unimplemented(
      "ComputeLayoutBackward API must be implemented via the subclass.");
}

StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
    const llvm::DenseMap<int, Layout>& output_layouts) {
  return ComputeLayoutBackward(op, output_layouts);
}

Status RunSPMDExpansion(mlir::Operation* op, mlir::Operation** output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTcc mht_4(mht_4_v, 356, "", "./tensorflow/dtensor/mlir/spmd_expander.cc", "RunSPMDExpansion");

  SPMDExpanderBase* expander =
      SPMDExpanderRegistry::Global()->GetPropagateFnForOp(op);
  if (expander != nullptr) {
    return expander->ExpandOpAndSetLayout(op, output);
  } else {
    VLOG(1) << "No expansion found for " << OpName(op) << "\n";
    *output = op;
  }
  return Status::OK();
}

}  // namespace dtensor
}  // namespace tensorflow
