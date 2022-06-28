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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_rewriterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_rewriterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_rewriterDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/cusolver_rewriter.h"

#include <cstdlib>
#include <functional>
#include <numeric>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/blas.h"

namespace xla {
namespace gpu {

namespace {

void SetFortranLayout(Shape* shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_rewriterDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/xla/service/gpu/cusolver_rewriter.cc", "SetFortranLayout");

  LayoutUtil::SetToDefaultLayout(shape);
  int n = shape->mutable_layout()->minor_to_major_size();
  CHECK_GE(n, 2);
  std::swap(shape->mutable_layout()->mutable_minor_to_major()->at(0),
            shape->mutable_layout()->mutable_minor_to_major()->at(1));
}

StatusOr<HloInstruction*> CreateCholesky(GpuSolverContext* context,
                                         HloInstruction* operand,
                                         const CholeskyOptions& options,
                                         const OpMetadata& metadata) {
  HloComputation* computation = operand->parent();

  Shape a_shape = operand->shape();
  int ndim = a_shape.dimensions_size();
  CHECK_GE(ndim, 2);
  int64_t n = a_shape.dimensions(ndim - 1);

  std::vector<int64_t> batch_dims(a_shape.dimensions().begin(),
                                  a_shape.dimensions().end() - 2);
  std::vector<int64_t> batch_dim_ids(batch_dims.size());
  absl::c_iota(batch_dim_ids, 0);
  int64_t batch_size = absl::c_accumulate(batch_dims, 1, std::multiplies<>{});

  // Find the workspace size.
  se::blas::UpperLower uplo = options.lower() ? se::blas::UpperLower::kLower
                                              : se::blas::UpperLower::kUpper;
  int64_t workspace_size;  // Number of elements of size a_shape.element_type()
  TF_ASSIGN_OR_RETURN(
      workspace_size,
      context->PotrfBufferSize(a_shape.element_type(), uplo, n, n, batch_size));

  // TODO(phawkins): Ideally we would relax this constraint. What we actually
  // want is that:
  // a) the batch dimensions are major, in no particular order.
  // b) the two minor dimensions are in fortran (column-major) order,

  SetFortranLayout(&a_shape);

  // This call returns a tuple of (cholesky_result, workspace, info) where:
  // * cholesky_result is the result of the Cholesky decomposition,
  // * workspace is temporary scratch memory used by cuSolver.
  // * info contains the Potrf success/failure status.
  // Currently we have no meaningful way to report an error, so we simply
  // discard the success/failure information. Obviously this is suboptimal.
  Shape info_shape = ShapeUtil::MakeShape(S32, batch_dims);
  Shape call_shape = ShapeUtil::MakeTupleShape(
      {a_shape,
       ShapeUtil::MakeShape(operand->shape().element_type(), {workspace_size}),
       info_shape});

  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, {operand}, kCusolverCholeskyCallTarget, {a_shape}));
  custom_call->set_metadata(metadata);
  TF_RETURN_IF_ERROR(custom_call->set_backend_config(options));
  HloInstruction* out = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(a_shape, custom_call, 0));
  HloInstruction* info = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(info_shape, custom_call, 2));

  // If info was non-zero, indicating that the Cholesky decomposition failed,
  // returns an array full of NaNs for the corresponding batch element.
  HloInstruction* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  HloInstruction* zeros =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          info_shape, zero, /*broadcast_dimensions=*/{}));
  HloInstruction* ok = computation->AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, batch_dims),
                                    info, zeros, ComparisonDirection::kEq));
  ok = computation->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(PRED, a_shape.dimensions()), ok,
      /*broadcast_dimensions=*/batch_dim_ids));

  TF_ASSIGN_OR_RETURN(Literal nan_literal,
                      LiteralUtil::NanValue(a_shape.element_type()));
  HloInstruction* nan = computation->AddInstruction(
      HloInstruction::CreateConstant(std::move(nan_literal)));
  HloInstruction* nans =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          a_shape, nan, /*broadcast_dimensions=*/{}));

  HloInstruction* select =
      computation->AddInstruction(HloInstruction::CreateTernary(
          a_shape, HloOpcode::kSelect, ok, out, nans));
  return select;
}

// Tries to rewrite a single convolution into a call to cudnn.
StatusOr<bool> RunOnInstruction(GpuSolverContext* context,
                                HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCholesky) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * custom_call,
      CreateCholesky(context, instruction->mutable_operand(0),
                     instruction->cholesky_options(), instruction->metadata()));

  VLOG(1) << "Replacing " << instruction->ToString() << " with "
          << custom_call->ToString();

  TF_RETURN_IF_ERROR(
      instruction->parent()->ReplaceInstruction(instruction, custom_call));
  return true;
}

}  // namespace

// Rewrites the convolutions in the given computation into calls to cudnn.
// Returns true if it made any changes.
StatusOr<bool> GpusolverRewriter::RunOnComputation(
    HloComputation* computation) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_rewriterDTcc mht_1(mht_1_v, 330, "", "./tensorflow/compiler/xla/service/gpu/cusolver_rewriter.cc", "GpusolverRewriter::RunOnComputation");

  std::vector<HloInstruction*> cusolver_calls;
  for (auto* hlo : computation->instructions()) {
    if (hlo->opcode() == HloOpcode::kCholesky) {
      cusolver_calls.push_back(hlo);
    }
  }

  if (cusolver_calls.empty()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(GpuSolverContext context,
                      GpuSolverContext::Create(/*stream=*/nullptr));

  bool changed = false;
  for (HloInstruction* instruction : cusolver_calls) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(&context, instruction));
    changed |= result;
  }
  return changed;
}

GpusolverRewriter::GpusolverRewriter() = default;

StatusOr<bool> GpusolverRewriter::Run(HloModule* module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_rewriterDTcc mht_2(mht_2_v, 358, "", "./tensorflow/compiler/xla/service/gpu/cusolver_rewriter.cc", "GpusolverRewriter::Run");

  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
