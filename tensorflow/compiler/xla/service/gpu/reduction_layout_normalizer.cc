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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_layout_normalizerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_layout_normalizerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_layout_normalizerDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/reduction_layout_normalizer.h"

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

class EnforceMinorToMajorReduceOpVisitor : public DfsHloRewriteVisitor {
  Status HandleReduce(HloInstruction *hlo) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_layout_normalizerDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/service/gpu/reduction_layout_normalizer.cc", "HandleReduce");

    auto reduce = Cast<HloReduceInstruction>(hlo);
    VLOG(5) << "Input: " << reduce->ToString();

    int operand_idx = -1;

    absl::InlinedVector<HloInstruction *, 2> canonical_reduce_inputs;
    absl::InlinedVector<Shape, 2> new_reduce_shapes;

    absl::InlinedVector<int64_t, 6> out_reduce_dimensions;
    const Shape &first_instruction_shape = reduce->inputs()[0]->shape();

    for (HloInstruction *operand : reduce->inputs()) {
      operand_idx++;

      if (operand_idx != 0 &&
          operand->shape().layout() != first_instruction_shape.layout()) {
        HloInstruction *copy =
            reduce->parent()->AddInstruction(HloInstruction::CreateUnary(
                operand->shape(), HloOpcode::kCopy, operand));

        LayoutUtil::ClearLayout(copy->mutable_shape());
        TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
            first_instruction_shape, copy->mutable_shape()));

        copy->set_metadata(operand->metadata());
        operand = copy;
        VLOG(3) << "Copying to establish consistent inputs layout: "
                << copy->ToString();
      }

      const Shape &operand_shape = operand->shape();
      const Layout &operand_layout = operand_shape.layout();

      const Shape &reduce_shape =
          reduce->shape().IsTuple() ? reduce->shape().tuple_shapes(operand_idx)
                                    : reduce->shape();

      absl::InlinedVector<int64_t, 6> new_reduce_dimensions;
      absl::InlinedVector<int64_t, 6> new_operand_shape_data;
      absl::InlinedVector<int64_t, 6> new_reduce_shape_data;

      // The layout order of the reduction output can be different to the
      // ordering of kept dimensions in the input operand, thus we need to
      // calculate the new layout.
      absl::InlinedVector<int64_t, 6> new_reduce_shape_layout(
          reduce_shape.rank());
      std::vector<int64_t> reduce_shape_logical_to_physical =
          LayoutUtil::MakeLogicalToPhysical(reduce_shape.layout());

      auto to_reduce_logical_dim = [&](int64_t op_logical_dim) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_layout_normalizerDTcc mht_1(mht_1_v, 260, "", "./tensorflow/compiler/xla/service/gpu/reduction_layout_normalizer.cc", "lambda");

        return op_logical_dim -
               absl::c_count_if(reduce->dimensions(), [&](int64_t dim) {
                 CHECK(dim != op_logical_dim);
                 return dim < op_logical_dim;
               });
      };

      for (int i = 0; i < operand_shape.rank(); i++) {
        // Process the dimensions in the major-to-minor order in order to
        // enforce the default layout.
        int64_t major_to_minor_dim_idx = operand_shape.rank() - i - 1;
        int64_t logical_dim =
            operand_layout.minor_to_major(major_to_minor_dim_idx);
        int64_t dim_size = operand_shape.dimensions(logical_dim);
        VLOG(5) << "Processing logical dimension " << logical_dim << " of size "
                << dim_size;
        new_operand_shape_data.push_back(dim_size);

        if (absl::c_linear_search(reduce->dimensions(), logical_dim)) {
          new_reduce_dimensions.push_back(i);
        } else {
          new_reduce_shape_data.push_back(dim_size);
          int64_t logical_reduce_dim = to_reduce_logical_dim(logical_dim);
          int64_t physical_reduce_dim =
              reduce_shape_logical_to_physical[logical_reduce_dim];
          VLOG(5) << "logical_reduce_dim = " << logical_reduce_dim << ", "
                  << "physical_reduce_dim = " << physical_reduce_dim;
          new_reduce_shape_layout[reduce_shape.rank() - physical_reduce_dim -
                                  1] = new_reduce_shape_data.size() - 1;
        }
      }

      Shape new_operand_shape = ShapeUtil::MakeShape(
          operand_shape.element_type(), new_operand_shape_data);
      Shape new_reduce_shape = ShapeUtil::MakeShapeWithLayout(
          reduce_shape.element_type(), new_reduce_shape_data,
          new_reduce_shape_layout);

      if (new_operand_shape == operand_shape && reduce->inputs().size() == 1) {
        return Status::OK();
      }

      HloInstruction *canonical_reduce_input =
          new_operand_shape != operand_shape
              ? reduce->parent()->AddInstruction(
                    HloInstruction::CreateBitcast(new_operand_shape, operand))
              : operand;
      canonical_reduce_input->set_metadata(operand->metadata());
      VLOG(5) << "Reduction input: " << canonical_reduce_input->ToString();

      new_reduce_shapes.push_back(new_reduce_shape);
      canonical_reduce_inputs.push_back(canonical_reduce_input);

      if (out_reduce_dimensions.empty()) {
        out_reduce_dimensions = new_reduce_dimensions;
      } else {
        TF_RET_CHECK(out_reduce_dimensions == new_reduce_dimensions);
      }
    }

    Shape new_reduce_shape = ShapeUtil::MakeMaybeTupleShape(new_reduce_shapes);

    std::unique_ptr<HloInstruction> new_reduce = HloInstruction::CreateReduce(
        new_reduce_shape, canonical_reduce_inputs, reduce->init_values(),
        out_reduce_dimensions, reduce->to_apply());
    VLOG(5) << "Generated new reduction: " << new_reduce->ToString();
    const Shape &orig_reduce_shape = reduce->shape();

    if (new_reduce_shape != orig_reduce_shape) {
      HloInstruction *wrapped_reduce =
          reduce->parent()->AddInstruction(std::move(new_reduce));

      if (!new_reduce_shape.IsTuple()) {
        new_reduce =
            HloInstruction::CreateBitcast(reduce->shape(), wrapped_reduce);
      } else {
        // Bitcast each element of the tuple.
        absl::InlinedVector<HloInstruction *, 2> out;
        for (int oidx = 0; oidx < reduce->input_count(); oidx++) {
          HloInstruction *gte = reduce->parent()->AddInstruction(
              HloInstruction::CreateGetTupleElement(wrapped_reduce, oidx));
          out.push_back(
              reduce->parent()->AddInstruction(HloInstruction::CreateBitcast(
                  orig_reduce_shape.tuple_shapes(oidx), gte)));
        }
        new_reduce = HloInstruction::CreateTuple(out);
      }
    }

    VLOG(5) << "Generated output: " << new_reduce->ToString();
    return ReplaceWithNewInstruction(reduce, std::move(new_reduce));
  }
};

StatusOr<bool> ReductionLayoutNormalizer::Run(HloModule *module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_layout_normalizerDTcc mht_2(mht_2_v, 358, "", "./tensorflow/compiler/xla/service/gpu/reduction_layout_normalizer.cc", "ReductionLayoutNormalizer::Run");

  TF_ASSIGN_OR_RETURN(bool changed,
                      EnforceMinorToMajorReduceOpVisitor().RunOnModule(module));
  return changed;
}

}  // namespace gpu
}  // namespace xla
