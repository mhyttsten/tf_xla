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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_reduce_scatter_creatorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_reduce_scatter_creatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_reduce_scatter_creatorDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/gpu_reduce_scatter_creator.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/reduce_scatter_utils.h"

namespace xla {
namespace gpu {

StatusOr<bool> ReduceScatterCreator::Run(HloModule *module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_reduce_scatter_creatorDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/xla/service/gpu/gpu_reduce_scatter_creator.cc", "ReduceScatterCreator::Run");

  const HloModuleConfig &config = module->config();
  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (HloComputation *computation : module->MakeNonfusionComputations()) {
    for (HloInstruction *instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kAllReduce) {
        continue;
      }
      auto *ar = Cast<HloAllReduceInstruction>(instruction);
      auto ar_spec = MatchReduceScatter(ar, config.num_partitions(),
                                        config.replica_count(),
                                        /*allow_multiple_split_dims=*/false,
                                        /*allow_intervening_reshape=*/true);
      if (!ar_spec) {
        VLOG(2) << "Cannot match reduce-scatter " << ar->ToString();
        continue;
      }

      HloInstruction *ds = ar_spec->dynamic_slice;

      // Convert to all-reduce scatter. The output shape of the all-reduce
      // scatter will the same as the input shape, except the split dim size is
      // that of the result of the dynamic slice.
      const int64_t split_dim = ar_spec->split_dim;
      Shape scatter_shape = ar->shape();
      const int64_t split_dim_size = scatter_shape.dimensions(split_dim);
      HloInstruction *rs_input = ar->mutable_operand(0);
      const int64_t scatter_dim_size = split_dim_size / ar_spec->group_size;
      TF_RET_CHECK(scatter_dim_size * ar_spec->group_size <= split_dim_size);
      if (split_dim_size % ar_spec->group_size != 0) {
        // The dynamic-slice does not evenly split the scatter dim. In that
        // case, create a reduce-scatter with the relevant slice of the
        // all-reduce input.
        scatter_shape.set_dimensions(split_dim,
                                     scatter_dim_size * ar_spec->group_size);
        rs_input = computation->AddInstruction(HloInstruction::CreateSlice(
            scatter_shape, rs_input,
            std::vector<int64_t>(scatter_shape.rank(), 0),
            scatter_shape.dimensions(),
            std::vector<int64_t>(scatter_shape.rank(), 1)));
      }
      scatter_shape.set_dimensions(split_dim, scatter_dim_size);

      absl::optional<int64_t> channel_id;
      if (ar->channel_id()) {
        // We cannot reuse the channel_id on all-reduce for reduce-scatter.
        channel_id = next_channel_id++;
      }

      HloInstruction *ars =
          computation->AddInstruction(HloInstruction::CreateReduceScatter(
              scatter_shape, {rs_input}, ar->to_apply(), ar->replica_groups(),
              ar->constrain_layout(), channel_id, ar->use_global_device_ids(),
              ar_spec->split_dim));

      // If there was an intervening reshape, reshape the non-split dimensions
      // to match that existing reshape. Basically we can just reshape the ars
      // result to the dynamic slice shape.
      HloInstruction *result = ars;
      HloInstruction *reshape = nullptr;
      if (ds->operand(0) != ar) {
        reshape = ds->mutable_operand(0);
        result = computation->AddInstruction(
            HloInstruction::CreateReshape(ds->shape(), result));
      }

      // Note that RemoveInstructionAndUnusedOperands may not always remove the
      // all-reduce operand of the dynamic-slice, so remove all the dead
      // instructions manually.
      TF_RETURN_IF_ERROR(ds->ReplaceAllUsesWith(result));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ds));
      if (reshape) {
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(reshape));
      }
      TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(ar));
      changed = true;
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
