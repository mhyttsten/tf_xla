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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgemm_rewriterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgemm_rewriterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgemm_rewriterDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

// Give this instruction a more useful name than "custom-call.42".
Status SetName(HloModule *module, HloInstruction *gemm) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgemm_rewriterDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/service/gpu/gemm_rewriter.cc", "SetName");

  GemmBackendConfig config;
  TF_ASSIGN_OR_RETURN(config, gemm->backend_config<GemmBackendConfig>());
  bool is_batch_dot = config.batch_size() > 1;

  module->SetAndUniquifyInstrName(
      gemm, is_batch_dot ? "cublas-batch-gemm" : "cublas-gemm");
  return Status::OK();
}

// The rewriting proceeds in a bottom-up way:
//
// (kDot A B) is rewritten into a (kCustomCall:gemm A B)
//
// (kMultiply (kCustomCall:gemm A B) C) is folding C (provided it's a constant)
// into an alpha parameter of the custom call.
//
// (kAdd (kCustomCall:gemm A B) C) is rewritten into (kCustomCall:gemm A B C),
// where the "beta" parameter is set to 1 (provided it was zero before,
// and provided C has no other users).
// We then guide the buffer assignment to alias the buffer of the custom call
// and C.
class GemmRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleDot(HloInstruction *instr) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgemm_rewriterDTcc mht_1(mht_1_v, 234, "", "./tensorflow/compiler/xla/service/gpu/gemm_rewriter.cc", "HandleDot");

    if (IsMatrixMultiplication(*instr)) {
      CHECK(!instr->IsRank2Transpose());
      HloInstruction *lhs = instr->mutable_operand(0);
      HloInstruction *rhs = instr->mutable_operand(1);
      CHECK(!lhs->IsRank2Transpose());
      CHECK(!rhs->IsRank2Transpose());
      const Shape &output_shape = instr->shape();
      int64_t batch_size = std::accumulate(output_shape.dimensions().begin(),
                                           output_shape.dimensions().end() - 2,
                                           1, std::multiplies<int64_t>());
      std::unique_ptr<HloInstruction> gemm_call =
          HloInstruction::CreateCustomCall(output_shape, {lhs, rhs},
                                           kGemmCallTarget);
      GemmBackendConfig gemm_config;
      gemm_config.set_alpha_real(1.0);
      gemm_config.set_alpha_imag(0.0);
      gemm_config.set_beta(0.0);
      *gemm_config.mutable_dot_dimension_numbers() =
          instr->dot_dimension_numbers();
      gemm_config.set_batch_size(batch_size);

      int64_t lhs_batch_dims_size =
          instr->dot_dimension_numbers().lhs_batch_dimensions_size();
      int64_t lhs_stride = lhs->shape().dimensions(lhs_batch_dims_size) *
                           lhs->shape().dimensions(lhs_batch_dims_size + 1);
      int64_t rhs_stride = rhs->shape().dimensions(lhs_batch_dims_size) *
                           rhs->shape().dimensions(lhs_batch_dims_size + 1);

      gemm_config.set_lhs_stride(lhs_stride);
      gemm_config.set_rhs_stride(rhs_stride);
      TF_RETURN_IF_ERROR(gemm_call->set_backend_config(gemm_config));
      TF_RETURN_IF_ERROR(SetName(instr->GetModule(), gemm_call.get()));
      TF_RETURN_IF_ERROR(
          ReplaceWithNewInstruction(instr, std::move(gemm_call)));
    }
    return Status::OK();
  }

  Status HandleMultiply(HloInstruction *instr) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgemm_rewriterDTcc mht_2(mht_2_v, 276, "", "./tensorflow/compiler/xla/service/gpu/gemm_rewriter.cc", "HandleMultiply");

    HloInstruction *alpha, *existing_gemm;
    if (Match(instr,
              m::MultiplyAnyOrder(
                  m::Op(&existing_gemm).WithCustomCallTarget(kGemmCallTarget),
                  m::Broadcast(m::ConstantScalar(&alpha))))) {
      TF_ASSIGN_OR_RETURN(auto config,
                          existing_gemm->backend_config<GemmBackendConfig>());

      // Do not fuse alpha into S32 GEMM, as they only support fixed values for
      // alpha/beta.
      if (existing_gemm->shape().element_type() == S32) {
        return Status::OK();
      }

      if (config.beta() == 0.0 && existing_gemm->user_count() == 1) {
        complex128 prev_alpha = {config.alpha_real(), config.alpha_imag()};
        complex128 new_alpha =
            *alpha->literal().GetAsComplex128({}) * prev_alpha;
        config.set_alpha_real(new_alpha.real());
        config.set_alpha_imag(new_alpha.imag());
        TF_RETURN_IF_ERROR(existing_gemm->set_backend_config(config));
        TF_RETURN_IF_ERROR(ReplaceInstruction(instr, existing_gemm));
      }
    }
    return Status::OK();
  }

  Status HandleAdd(HloInstruction *instr) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgemm_rewriterDTcc mht_3(mht_3_v, 307, "", "./tensorflow/compiler/xla/service/gpu/gemm_rewriter.cc", "HandleAdd");

    HloInstruction *bias, *existing_gemm;
    if (Match(instr,
              m::AddAnyOrder(
                  m::Op(&existing_gemm).WithCustomCallTarget(kGemmCallTarget),
                  m::Op(&bias)))) {
      // Do not fuse bias into S32 GEMM, as for this datatype cuBLAS only
      // supports fixed values for alpha/beta.
      if (existing_gemm->shape().element_type() == S32) {
        return Status::OK();
      }
      auto config =
          existing_gemm->backend_config<GemmBackendConfig>().ValueOrDie();
      if (config.beta() == 0 && bias->user_count() == 1 &&
          existing_gemm->user_count() == 1 &&
          bias->shape() == existing_gemm->shape()) {
        config.set_beta(1.0);
        CHECK_EQ(existing_gemm->operand_count(), 2);
        std::unique_ptr<HloInstruction> gemm_call =
            HloInstruction::CreateCustomCall(
                instr->shape(),
                {existing_gemm->mutable_operand(0),
                 existing_gemm->mutable_operand(1), bias},
                kGemmCallTarget);
        TF_RETURN_IF_ERROR(gemm_call->set_backend_config(config));
        TF_RETURN_IF_ERROR(SetName(instr->GetModule(), gemm_call.get()));
        TF_RETURN_IF_ERROR(
            ReplaceWithNewInstruction(instr, std::move(gemm_call)));
      }
    }
    return Status::OK();
  }
};

StatusOr<bool> RunOnComputation(HloComputation *computation) {
  GemmRewriterVisitor visitor;
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace

StatusOr<bool> GemmRewriter::Run(HloModule *module) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgemm_rewriterDTcc mht_4(mht_4_v, 352, "", "./tensorflow/compiler/xla/service/gpu/gemm_rewriter.cc", "GemmRewriter::Run");

  bool changed = false;
  for (HloComputation *computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
