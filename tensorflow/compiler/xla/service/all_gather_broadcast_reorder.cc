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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_broadcast_reorderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_broadcast_reorderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_broadcast_reorderDTcc() {
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

#include "tensorflow/compiler/xla/service/all_gather_broadcast_reorder.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

StatusOr<bool> AllGatherBroadcastReorder::Run(HloModule *module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_broadcast_reorderDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/all_gather_broadcast_reorder.cc", "AllGatherBroadcastReorder::Run");

  if (hlo_query::ContainsLayoutConstrainedCollective(*module,
                                                     HloOpcode::kAllGather)) {
    VLOG(1) << "Skip AllGatherBroadcastReorder because the module contains "
               "all-gather with constrained layouts";
    return false;
  }

  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (auto computation : module->computations()) {
    for (HloInstruction *inst : computation->MakeInstructionPostOrder()) {
      // Check for all-gather with a broadcast operand.
      if (inst->opcode() != HloOpcode::kAllGather || !inst->shape().IsArray() ||
          inst->operand(0)->opcode() != HloOpcode::kBroadcast) {
        continue;
      }

      HloAllGatherInstruction *ag = Cast<HloAllGatherInstruction>(inst);
      HloBroadcastInstruction *bcast =
          Cast<HloBroadcastInstruction>(inst->mutable_operand(0));

      // We categorize each dimension of the all-gather result as either
      // uniform (same data along that dimension) or not. The all_gather
      // dimension is not uniform since we don't know uniformity of data across
      // the all-gather participants. In addition, the set of dimensions
      // for the broadcast instruction feeding into all-gather are also not
      // uniform. If there are any remaining uniform dims and their size > 1,
      // then doing the broadcast after the all-gather is beneficial as the
      // all-gather will be operating on smaller data.
      absl::flat_hash_set<int64_t> non_uniform_dims;
      non_uniform_dims.insert(bcast->dimensions().begin(),
                              bcast->dimensions().end());
      const bool all_gather_along_uniform_dim =
          non_uniform_dims.insert(ag->all_gather_dimension()).second;

      // Find the product of the size of uniform dims.
      int64_t uniform_dim_size = 1;
      for (int64_t i = 0; i < ag->shape().rank(); ++i) {
        if (non_uniform_dims.count(i) == 0) {
          uniform_dim_size *= ag->shape().dimensions(i);
        }
      }

      if (uniform_dim_size == 1) {
        continue;
      }

      HloInstruction *replacement;
      const int64_t ag_dim = ag->all_gather_dimension();
      // Transform the all-gather(broadcast(x)) to broadcast(all-gather(x)).
      // There are 2 cases here:
      if (!all_gather_along_uniform_dim) {
        // If the all-gather happens along one of the non-uniform dimensions of
        // the broadcast, then issue all-gather(x) and then a broadcast.
        // Example:
        //       x = f32[128, 5] ..
        //       bc = f32[5, 4, 8, 128] broadcast(x) dimensions={3, 0}
        //       ag = f32[5, 4, 8, 256] all-gather(bc) all_gather_dimension={3}
        // to:
        //       ag = f32[256, 5] all-gather(x) all_gather_dimension={0}
        //       bc = f32[5, 4, 8, 256] broadcast(ag) dimensions={3, 0}

        VLOG(2) << "All-gather along non uniform dimension";

        // Find the index of the all_gather dimension in the broadcast dims.
        auto ag_dim_index = PositionInContainer(bcast->dimensions(), ag_dim);

        // The new all-gather shape is just the shape of x, with the dimension
        // that was gathered multiplied by some factor.
        Shape new_ag_shape = bcast->operand(0)->shape();
        new_ag_shape.set_dimensions(ag_dim_index,
                                    ag->shape().dimensions(ag_dim));

        // Create a new gather, which is going to gather along `ag_dim_index`.
        auto *new_ag =
            Cast<HloAllGatherInstruction>(computation->AddInstruction(
                ag->CloneWithNewOperands(new_ag_shape, bcast->operands())));
        if (ag->channel_id()) {
          new_ag->set_channel_id(next_channel_id++);
        }
        new_ag->set_all_gather_dimension(ag_dim_index);

        // Finally broadcast after the gather. This new broadcast uses the same
        // broadcast dimensions as the original broadcast, as illustrated in the
        // example above.
        replacement = computation->AddInstruction(
            bcast->CloneWithNewOperands(ag->shape(), {new_ag}));
      } else {
        // If the all-gather happens along one of the uniform dimensions of the
        // broadcast, that dimension does not exists in x. Use the following
        // representative sequence for this case:
        //
        //       x = f32[128, 5] ..
        //       bc = f32[5, 4, 8, 128] broadcast(x) dimensions={3, 0}
        //       ag = f32[5, 12, 8, 128] all-gather(bc) all_gather_dimension={1}
        // to:
        //       rs0 = f32[1, 128, 5] reshape(x)
        //       ag = f32[3, 128, 5] all-gather(rs0) all_gather_dimension={0}
        //       bc = f32[5, 3, 4, 8, 128] broadcast(ag) dimensions={1, 4, 0}
        //       rs1 = f32[5, 12, 8, 128] reshape(bc)

        VLOG(2) << "All-gather along uniform dimension";
        HloInstruction *x = bcast->mutable_operand(0);

        // Reshape to add a leading '1' dimension.
        std::vector<int64_t> shape_dims{1};
        absl::Span<const int64_t> x_dims = x->shape().dimensions();
        shape_dims.insert(shape_dims.end(), x_dims.begin(), x_dims.end());
        Shape shape =
            ShapeUtil::MakeShape(x->shape().element_type(), shape_dims);

        HloInstruction *rs0 = computation->AddInstruction(
            HloInstruction::CreateReshape(shape, x));

        // Number of participants in the all-gather.
        const int64_t ag_factor = ag->shape().dimensions(ag_dim) /
                                  ag->operand(0)->shape().dimensions(ag_dim);

        shape.set_dimensions(0, ag_factor);

        auto *new_ag =
            Cast<HloAllGatherInstruction>(computation->AddInstruction(
                ag->CloneWithNewOperands(shape, {rs0})));
        if (ag->channel_id()) {
          new_ag->set_channel_id(next_channel_id++);
        }
        new_ag->set_all_gather_dimension(0);

        // Now issue a broadcast which matches the existing all-gather shape,
        // except the all-gather dim is split into [ag_factor,
        // ag_dim_size/ag_factor].
        std::vector<int64_t> bcast_shape_dims =
            SpanToVector(ag->shape().dimensions());
        bcast_shape_dims[ag_dim] = ag_factor;
        bcast_shape_dims.insert(bcast_shape_dims.begin() + ag_dim + 1,
                                ag->shape().dimensions(ag_dim) / ag_factor);
        Shape bcast_shape =
            ShapeUtil::MakeShape(x->shape().element_type(), bcast_shape_dims);

        // The broadcast dims have 1 extra dim as compared to the existing
        // broadcast (due to the ag_factor dimension). This corresponds to dim
        // 0 of the new broadcast inputs. Also, we need to adjust the dimensions
        // from old -> new broadcast as follows:
        //     if the dim value > ag_dim, add +1 to account for the extra dim.
        //     if the dim value < ag_dim, keep it unmodified.
        // As an example, in the running case, the broadcast input is
        //  [ag_factor=3, 128, 5].
        // The new broadcast will have 3 dimensions. The first one will be
        // ag_dim = 1. The existing dims are {3, 0}. Per the adjustment rules, 3
        // will be adjusted to 4 and 0 will stay unmodified, giving the final
        // dims = {1, 4, 0}
        std::vector<int64_t> bcast_dims;
        bcast_dims.push_back(ag_dim);
        for (int64_t d : bcast->dimensions()) {
          bcast_dims.push_back(d + (d > ag_dim));
        }
        HloInstruction *bcast = computation->AddInstruction(
            HloInstruction::CreateBroadcast(bcast_shape, new_ag, bcast_dims));

        // Finally, "flatten" the [ag_factor, ag_dim_size/ag_factor] to just
        // ag_dim_size by issusing a final reshape.
        replacement = computation->AddInstruction(
            HloInstruction::CreateReshape(ag->shape(), bcast));
      }

      TF_RETURN_IF_ERROR(ag->ReplaceAllUsesWith(replacement));
      TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(ag));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
