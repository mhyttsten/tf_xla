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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsegmentation_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsegmentation_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsegmentation_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/segmentation_spmd_expander.h"

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

// We always forward replicated layout to operands/output of
// UnsortedSegmentedSum op as SPMD logic sharded UnsortedSegmentedSum op is not
// implemented yet.
// TODO(b/171079751): Implement layout propagation for non-trivial layouts
StatusOr<llvm::DenseMap<int, Layout>>
UnsortedSegmentSumSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto unsorted_segmented_sum = llvm::cast<mlir::TF::UnsortedSegmentSumOp>(op);
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(
               mesh, /*rank=*/ValueRank(unsorted_segmented_sum.output()))}});
}

StatusOr<llvm::DenseMap<int, Layout>>
UnsortedSegmentSumSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto unsorted_segmented_sum = llvm::cast<mlir::TF::UnsortedSegmentSumOp>(op);
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(
               mesh, /*rank=*/ValueRank(unsorted_segmented_sum.data()))},
       {1,
        Layout::ReplicatedOnMesh(mesh, /*rank=*/ValueRank(op->getOperand(1)))},
       {2, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)}});
}

StatusOr<mlir::Operation*> UnsortedSegmentSumSPMDExpander::ExpandOp(
    mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsegmentation_spmd_expanderDTcc mht_0(mht_0_v, 228, "", "./tensorflow/dtensor/mlir/expansions/segmentation_spmd_expander.cc", "UnsortedSegmentSumSPMDExpander::ExpandOp");

  // The algorithm is simple
  //
  // 1. Up to rank of the segment_ids, if the data or ids are sharded, perform
  //    all-concat, respectively.
  // 2. We do not care the sharding dims of data[rank(ids):] and just leave as
  //    is
  // 3. output.layout[0] is expected to be replicated due to the steps above.
  //    otherwise, perform a slicing.
  // 4. output.layout[1:] is expected to be same as data.layout[rank(ids):] as
  //    untouched. otherwise, perform all-concat or slicing.
  //
  // For item 3 and 4, we perform a single all-concat Op.
  //
  // Alternative to the steps 1 and 2 above could be
  //   a. all-concat data
  //   b. local unsorted seg sum followed by a all reduce with some masks.
  //
  // Alternative to the step 4 above is merging it with step 1 (upon the dim is
  // compatible).

  auto sum_op = mlir::cast<mlir::TF::UnsortedSegmentSumOp>(op);
  auto data = sum_op.data();
  auto segment_ids = sum_op.segment_ids();

  TF_ASSIGN_OR_RETURN(auto data_layout, ExtractLayoutFromOperand(data));
  TF_ASSIGN_OR_RETURN(auto segment_ids_layout,
                      ExtractLayoutFromOperand(segment_ids));

  const auto data_rank = ValueRank(data);
  const auto segment_ids_rank = ValueRank(segment_ids);

  // Prepares the resulting output layout. Fills the default unsharded dim for
  // the first axis (dim size is num_segments).
  LayoutProto result_output_layout;
  *result_output_layout.mutable_mesh_config() = data_layout->mesh().ToProto();
  result_output_layout.add_sharding_specs()->set_sharding_spec(
      Layout::kUnshardedDim);

  // Prepares the replicated target data output (up to segment_ids_rank).
  LayoutProto tgt_data_layout;
  *tgt_data_layout.mutable_mesh_config() = data_layout->mesh().ToProto();

  bool need_data_all_concat = false;
  for (int i = 0; i < data_rank; i++) {
    if (i < segment_ids_rank) {
      tgt_data_layout.add_sharding_specs()->set_sharding_spec(
          Layout::kUnshardedDim);
      if (data_layout->sharding_spec(i) != Layout::kUnshardedDim) {
        need_data_all_concat = true;
      }
    } else {
      tgt_data_layout.add_sharding_specs()->set_sharding_spec(
          data_layout->sharding_spec(i));
      result_output_layout.add_sharding_specs()->set_sharding_spec(
          data_layout->sharding_spec(i));
    }
  }

  mlir::OpBuilder builder(op);
  if (need_data_all_concat) {
    TF_ASSIGN_OR_RETURN(
        auto data_concat,
        EmitAllGather(builder, data, *data_layout,
                      Layout::FromProto(tgt_data_layout).ValueOrDie()));
    data = data_concat;
  }

  // Ensure segment IDs are fully replicated.
  if (!segment_ids_layout->IsFullyReplicated()) {
    TF_ASSIGN_OR_RETURN(
        auto segment_ids_concat,
        EmitAllGather(builder, segment_ids, *segment_ids_layout,
                      Layout::ReplicatedOnMesh(segment_ids_layout->mesh(),
                                               segment_ids_layout->rank())));
    segment_ids = segment_ids_concat;
  }

  auto new_sum_op = builder.create<mlir::TF::UnsortedSegmentSumOp>(
      op->getLoc(), sum_op.output().getType(), data, segment_ids,
      sum_op.num_segments());

  InferSPMDExpandedLocalShape(new_sum_op);

  // Transform the result to the expected output_layout, if necessary.
  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(
      auto final_output,
      EmitRelayout(new_sum_op.getResult(),
                   Layout::FromProto(result_output_layout).ValueOrDie(),
                   *output_layout));
  op->getResult(0).replaceAllUsesWith(final_output);
  op->erase();

  return final_output.getDefiningOp();
}

}  // namespace dtensor
}  // namespace tensorflow
