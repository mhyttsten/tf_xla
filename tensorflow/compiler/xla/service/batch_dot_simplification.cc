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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatch_dot_simplificationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatch_dot_simplificationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatch_dot_simplificationDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/batch_dot_simplification.h"

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"

namespace xla {
StatusOr<bool>
BatchDotSimplification::ElideDegenerateBatchDimensionFromBatchDot(
    HloInstruction* batch_dot) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatch_dot_simplificationDTcc mht_0(mht_0_v, 194, "", "./tensorflow/compiler/xla/service/batch_dot_simplification.cc", "BatchDotSimplification::ElideDegenerateBatchDimensionFromBatchDot");

  // This pass assumes the lhs and rhs batch dimensions are equal and strictly
  // ascending.
  const auto& is_iota = [](absl::Span<const int64_t> dims) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatch_dot_simplificationDTcc mht_1(mht_1_v, 200, "", "./tensorflow/compiler/xla/service/batch_dot_simplification.cc", "lambda");

    for (int64_t i = 0; i < dims.size(); ++i) {
      if (dims[i] != i) {
        return false;
      }
    }
    return true;
  };
  if (!absl::c_equal(
          batch_dot->dot_dimension_numbers().lhs_batch_dimensions(),
          batch_dot->dot_dimension_numbers().rhs_batch_dimensions()) ||
      !is_iota(batch_dot->dot_dimension_numbers().lhs_batch_dimensions())) {
    return false;
  }

  const DotDimensionNumbers& dim_numbers = batch_dot->dot_dimension_numbers();
  HloInstruction *lhs = batch_dot->mutable_operand(0),
                 *rhs = batch_dot->mutable_operand(1);
  const Shape& lhs_shape = lhs->shape();

  // A dot with no contracting dims will be rewritten into a multiply by
  // AlgebraicSimplifier. Dots with multiple contracting dims are currently
  // unsupported.
  if (dim_numbers.lhs_contracting_dimensions_size() != 1) {
    return false;
  }

  std::vector<int64_t> degenerate_dims;
  for (int64_t batch_dim : dim_numbers.lhs_batch_dimensions()) {
    if (lhs_shape.dimensions(batch_dim) == 1) {
      degenerate_dims.push_back(batch_dim);
    }
  }

  if (degenerate_dims.empty()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * new_lhs,
                      ElideDegenerateDims(lhs, degenerate_dims));
  TF_ASSIGN_OR_RETURN(HloInstruction * new_rhs,
                      ElideDegenerateDims(rhs, degenerate_dims));

  DotDimensionNumbers new_dim_numbers = dim_numbers;
  new_dim_numbers.clear_lhs_batch_dimensions();
  new_dim_numbers.clear_rhs_batch_dimensions();

  for (int64_t i = 0, e = dim_numbers.lhs_batch_dimensions_size() -
                          degenerate_dims.size();
       i < e; i++) {
    new_dim_numbers.add_lhs_batch_dimensions(i);
    new_dim_numbers.add_rhs_batch_dimensions(i);
  }

  new_dim_numbers.set_lhs_contracting_dimensions(
      0,
      new_dim_numbers.lhs_contracting_dimensions(0) - degenerate_dims.size());
  new_dim_numbers.set_rhs_contracting_dimensions(
      0,
      new_dim_numbers.rhs_contracting_dimensions(0) - degenerate_dims.size());

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_dot,
      MakeDotHlo(new_lhs, new_rhs, new_dim_numbers,
                 batch_dot->precision_config(),
                 /*preferred_element_type=*/batch_dot->shape().element_type()));

  TF_ASSIGN_OR_RETURN(HloInstruction * new_dot_reshaped,
                      MakeReshapeHlo(batch_dot->shape(), new_dot));

  VLOG(2) << "Replaced " << batch_dot->ToString() << " with "
          << new_dot->ToString();

  TF_RETURN_IF_ERROR(
      batch_dot->parent()->ReplaceInstruction(batch_dot, new_dot_reshaped));

  return true;
}

absl::string_view BatchDotSimplification::name() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatch_dot_simplificationDTcc mht_2(mht_2_v, 282, "", "./tensorflow/compiler/xla/service/batch_dot_simplification.cc", "BatchDotSimplification::name");

  return "batch-dot-simplification";
}

StatusOr<bool> BatchDotSimplification::Run(HloModule* module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatch_dot_simplificationDTcc mht_3(mht_3_v, 289, "", "./tensorflow/compiler/xla/service/batch_dot_simplification.cc", "BatchDotSimplification::Run");

  bool changed = false;
  std::vector<HloInstruction*> dot_instrs;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    absl::c_copy_if(computation->instructions(), std::back_inserter(dot_instrs),
                    [](HloInstruction* instr) {
                      return instr->opcode() == HloOpcode::kDot;
                    });
  }
  for (HloInstruction* dot_instr : dot_instrs) {
    TF_ASSIGN_OR_RETURN(bool elided_batch_dim_from_one,
                        ElideDegenerateBatchDimensionFromBatchDot(dot_instr));
    changed |= elided_batch_dim_from_one;
  }
  return changed;
}
}  // namespace xla
