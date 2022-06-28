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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_degenerate_dim_removerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_degenerate_dim_removerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_degenerate_dim_removerDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/reduction_degenerate_dim_remover.h"

#include <algorithm>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

class ReductionDegenerateDimRemoverVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleReduce(HloInstruction *hlo) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_degenerate_dim_removerDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/service/gpu/reduction_degenerate_dim_remover.cc", "HandleReduce");

    auto instr = Cast<HloReduceInstruction>(hlo);
    absl::InlinedVector<HloInstruction *, 2> input_reshapes;
    absl::InlinedVector<Shape, 2> canonical_reduce_shapes;

    int idx = -1;
    std::vector<int64_t> updated_reduced_dimensions;
    for (HloInstruction *reduced_op : instr->inputs()) {
      idx++;
      const Shape &input_shape = reduced_op->shape();
      const Shape &reduce_shape = instr->shape().IsTuple()
                                      ? instr->shape().tuple_shapes(idx)
                                      : instr->shape();

      if (!ShapeUtil::HasDegenerateDimensions(reduced_op->shape())) {
        return Status::OK();
      }
      Shape canonical_input_shape =
          ShapeUtil::DropDegenerateDimensions(input_shape);

      Shape canonical_reduce_shape =
          ShapeUtil::DropDegenerateDimensions(reduce_shape);

      auto reduced_dimensions = instr->dimensions();
      int64_t shift = 0;

      for (int dim = 0; dim < input_shape.rank(); dim++) {
        if (input_shape.dimensions(dim) == 1) {
          shift++;
        } else {
          if (absl::c_linear_search(reduced_dimensions, dim) && idx == 0) {
            // Only populate on first iteration.
            updated_reduced_dimensions.push_back(dim - shift);
          }
        }
      }

      if (updated_reduced_dimensions.empty()) {
        std::unique_ptr<HloInstruction> reshape =
            HloInstruction::CreateBitcast(reduce_shape, reduced_op);
        return ReplaceWithNewInstruction(instr, std::move(reshape));
      }

      input_reshapes.push_back(instr->parent()->AddInstruction(
          HloInstruction::CreateBitcast(canonical_input_shape, reduced_op)));
      canonical_reduce_shapes.push_back(canonical_reduce_shape);
    }

    Shape canonical_reduce_shape =
        ShapeUtil::MakeMaybeTupleShape(canonical_reduce_shapes);
    const Shape &orig_reduce_shape = instr->shape();
    std::unique_ptr<HloInstruction> new_reduce = HloInstruction::CreateReduce(
        canonical_reduce_shape, input_reshapes, instr->init_values(),
        updated_reduced_dimensions, instr->to_apply());

    if (canonical_reduce_shape != instr->shape()) {
      HloInstruction *wrapped_reduce =
          instr->parent()->AddInstruction(std::move(new_reduce));
      absl::InlinedVector<HloInstruction *, 2> out;
      if (!canonical_reduce_shape.IsTuple()) {
        new_reduce =
            HloInstruction::CreateBitcast(orig_reduce_shape, wrapped_reduce);
      } else {
        for (int oidx = 0; oidx < instr->input_count(); oidx++) {
          HloInstruction *gte = instr->parent()->AddInstruction(
              HloInstruction::CreateGetTupleElement(wrapped_reduce, oidx));
          out.push_back(
              instr->parent()->AddInstruction(HloInstruction::CreateBitcast(
                  orig_reduce_shape.tuple_shapes(oidx), gte)));
        }
        new_reduce = HloInstruction::CreateTuple(out);
      }
    }

    return ReplaceWithNewInstruction(instr, std::move(new_reduce));
  }
};

StatusOr<bool> ReductionDegenerateDimRemover::Run(HloModule *module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_degenerate_dim_removerDTcc mht_1(mht_1_v, 289, "", "./tensorflow/compiler/xla/service/gpu/reduction_degenerate_dim_remover.cc", "ReductionDegenerateDimRemover::Run");

  TF_ASSIGN_OR_RETURN(
      bool changed, ReductionDegenerateDimRemoverVisitor().RunOnModule(module));
  return changed;
}

}  // namespace gpu
}  // namespace xla
