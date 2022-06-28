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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemmsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemmsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemmsDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/cublas_pad_for_gemms.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace gpu {

static StatusOr<bool> PadForGemm(HloDotInstruction* dot, PrimitiveType datatype,
                                 int pad_to_multiple_of) {
  auto* lhs = dot->mutable_operand(0);
  auto* rhs = dot->mutable_operand(1);

  Shape lshape = lhs->shape();
  Shape rshape = rhs->shape();
  Shape result_shape = dot->shape();

  if (lshape.element_type() != datatype || rshape.element_type() != datatype) {
    return false;
  }

  auto pad_dim = [&](Shape& s, int dim) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemmsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/gpu/cublas_pad_for_gemms.cc", "lambda");

    s.set_dimensions(dim,
                     RoundUpTo<int64_t>(s.dimensions(dim), pad_to_multiple_of));
  };

  auto pad_matrix_dims = [&pad_dim](Shape s) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemmsDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/service/gpu/cublas_pad_for_gemms.cc", "lambda");

    // Since the dot instruction is canonicalized, the last two dimensions for
    // each operand represent non-batch dimensions, and the others are the same
    // for both operands and correspond to batch dimensions.
    pad_dim(s, s.rank() - 2);
    pad_dim(s, s.rank() - 1);
    return s;
  };

  Shape new_lshape = pad_matrix_dims(lshape);
  Shape new_rshape = pad_matrix_dims(rshape);
  Shape new_result_shape = pad_matrix_dims(result_shape);

  if (new_lshape == lshape && new_rshape == rshape) {
    return false;
  }

  VLOG(3) << "old shape: " << lshape << " " << rshape << " " << result_shape;
  VLOG(3) << "new shape: " << new_lshape << " " << new_rshape << " "
          << new_result_shape;

  auto create_padding_config = [](Shape& shape, Shape& new_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemmsDTcc mht_2(mht_2_v, 243, "", "./tensorflow/compiler/xla/service/gpu/cublas_pad_for_gemms.cc", "lambda");

    PaddingConfig padding_config;
    for (int i = 0; i < shape.rank(); ++i) {
      auto dimension = padding_config.add_dimensions();
      dimension->set_edge_padding_high(new_shape.dimensions()[i] -
                                       shape.dimensions()[i]);
      dimension->set_edge_padding_low(0);
      dimension->set_interior_padding(0);
    }
    return padding_config;
  };

  auto l_padding_config = create_padding_config(lshape, new_lshape);
  auto r_padding_config = create_padding_config(rshape, new_rshape);

  HloComputation* parent = dot->parent();

  HloInstruction* zero_float = parent->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(datatype)));
  zero_float->set_metadata(dot->metadata());

  HloInstruction* lpad = parent->AddInstruction(
      HloInstruction::CreatePad(new_lshape, lhs, zero_float, l_padding_config));
  lpad->set_metadata(dot->metadata());

  HloInstruction* rpad = parent->AddInstruction(
      HloInstruction::CreatePad(new_rshape, rhs, zero_float, r_padding_config));
  rpad->set_metadata(dot->metadata());

  HloInstruction* new_dot = parent->AddInstruction(
      dot->CloneWithNewOperands(new_result_shape, {lpad, rpad}));

  std::vector<int64_t> start_indices(result_shape.rank(), 0);
  std::vector<int64_t> strides(result_shape.rank(), 1);
  HloInstruction* slice = parent->AddInstruction(
      HloInstruction::CreateSlice(result_shape, new_dot, start_indices,
                                  result_shape.dimensions(), strides));
  slice->set_metadata(dot->metadata());

  bool is_root = dot->user_count() == 0;

  TF_CHECK_OK(parent->ReplaceInstruction(dot, slice));

  if (is_root) {
    parent->set_root_instruction(slice);
  }

  return true;
}

namespace {

// We need this check because PadForGemm works in the assumption that
// the dot instruction is canonicalized.
bool CheckCanonical(HloDotInstruction* dot) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemmsDTcc mht_3(mht_3_v, 300, "", "./tensorflow/compiler/xla/service/gpu/cublas_pad_for_gemms.cc", "CheckCanonical");

  auto dimension_numbers = dot->dot_dimension_numbers();

  if (dimension_numbers.lhs_batch_dimensions_size() + 2 !=
          dot->operand(0)->shape().rank() ||
      dimension_numbers.rhs_batch_dimensions_size() + 2 !=
          dot->operand(1)->shape().rank()) {
    LOG(ERROR) << "Dot is not canonical: Expected all dimensions but 2 to be "
                  "batch_dimensions.";
    return false;
  }

  std::vector<int64_t> canonical_batch_dims(
      dimension_numbers.lhs_batch_dimensions_size());
  absl::c_iota(canonical_batch_dims, 0);
  if (!absl::c_equal(dimension_numbers.lhs_batch_dimensions(),
                     canonical_batch_dims) ||
      !absl::c_equal(dimension_numbers.rhs_batch_dimensions(),
                     canonical_batch_dims)) {
    LOG(ERROR) << "Dot is not canonical: Expected batch dimensions to be all "
                  "dimensions except for the last 2 ones.";
    return false;
  }

  return true;
}

}  // namespace

static std::vector<HloDotInstruction*> GetRelevantDots(HloComputation* comp,
                                                       PrimitiveType datatype) {
  std::vector<HloDotInstruction*> gemms;

  for (HloInstruction* instr : comp->instructions()) {
    if (IsMatrixMultiplication(*instr)) {
      HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
      if (instr->operand(0)->shape().element_type() == datatype &&
          CheckCanonical(dot)) {
        gemms.push_back(dot);
      }
    }
  }
  return gemms;
}

StatusOr<bool> CublasPadForGemms::Run(HloModule* module) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScublas_pad_for_gemmsDTcc mht_4(mht_4_v, 348, "", "./tensorflow/compiler/xla/service/gpu/cublas_pad_for_gemms.cc", "CublasPadForGemms::Run");

  bool changed = false;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloDotInstruction* dot : GetRelevantDots(comp, datatype_)) {
      TF_ASSIGN_OR_RETURN(bool result,
                          PadForGemm(dot, datatype_, pad_to_multiple_of_));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
