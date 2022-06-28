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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_cost_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_cost_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_cost_analysisDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.h"

#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"

namespace xla {
namespace gpu {

Status GpuHloCostAnalysis::HandleCustomCall(const HloInstruction* custom_call) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_cost_analysisDTcc mht_0(mht_0_v, 193, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.cc", "GpuHloCostAnalysis::HandleCustomCall");

  if (custom_call->custom_call_target() == gpu::kGemmCallTarget) {
    // The naming conventions and meanings of gemm parameters are documented
    // here:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    TF_ASSIGN_OR_RETURN(auto gemm_config,
                        custom_call->backend_config<gpu::GemmBackendConfig>());

    // Technically, in addition to the dot product (A * B), cuBLAS gemm also
    // performs additional scaling (by factor 'alpha') and addition with a
    // scaled third matrix (beta * C), which will introduce additional
    // multiplications and additions. But total FLOPS will be dominated by the
    // dot product, so we don't include these extra multiplications and
    // additions in the FLOPS calculation.

    // Also, this calculation assumes that the strides for the gemm are
    // properly set such that none of the inputs in a batch overlap with any
    // other batches. If they do, this will undercount the FLOPS, because it
    // assumes that the strides are implicit in the sizes of the batch
    // dimensions.

    // Finally, this is technically incorrect if the element type of this
    // gemm is an integer type, because in that case no floating point
    // operations are involved at all! But we still calculate FLOPS because the
    // number is sometimes required for ad-hoc calculations.
    current_properties_[kFlopsKey] =
        GetDotFlops(custom_call->operand(0)->shape(), custom_call->shape(),
                    gemm_config.dot_dimension_numbers());
    return Status::OK();
  }

  if (IsCustomCallToDnnConvolution(*custom_call)) {
    // As with dots, this flops calculation has the following inaccuracies.
    //
    //  - We may have a fused conv which does additional ops (multiplying by a
    //    scalar `alpha`, adding a bias or side-input, doing a relu, etc).  But
    //    we can safely ignore this because the overall computation is dominated
    //    by the convolution itself.
    //
    //  - cudnn may use complex conv algorithms that do fewer (or more!) flops
    //    than we calculate.
    //
    //  - for int8_t convs, these aren't *fl*ops, but we fudge it.
    current_properties_[kFlopsKey] = GetConvolutionFlops(custom_call);

    // conv custom-calls return a tuple (real_output, temp_bytes).  Count just
    // the real_output in output bytes accessed.  The main purpose of
    // hlo_cost_analysis is to figure out if ops are running "as fast as
    // possible", and if we were to include temp memory in here, we'd
    // essentially be *rewarding* convs that use additional temp memory!
    if (custom_call->shape().IsTuple()) {
      SetOutputBytesAccessed(
          options_.shape_size(custom_call->shape().tuple_shapes(0)));
    }
    return Status::OK();
  }

  return HloCostAnalysis::HandleCustomCall(custom_call);
}

int64_t GpuHloCostAnalysis::GetConvolutionFlops(
    const HloInstruction* convolution) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_cost_analysisDTcc mht_1(mht_1_v, 257, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.cc", "GpuHloCostAnalysis::GetConvolutionFlops");

  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& result_shape = [&]() -> const Shape& {
    // convolution custom-calls return a tuple of (actual_result, temp_buffer).
    const Shape& shape = convolution->shape();
    if (IsCustomCallToDnnConvolution(*convolution) &&
        convolution->shape().IsTuple()) {
      return shape.tuple_shapes(0);
    }
    return shape;
  }();

  return HloCostAnalysis::GetConvolutionFlops(convolution, lhs_shape, rhs_shape,
                                              result_shape);
}

std::unique_ptr<HloCostAnalysis>
GpuHloCostAnalysis::CreateNestedCostAnalysis() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_cost_analysisDTcc mht_2(mht_2_v, 280, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.cc", "GpuHloCostAnalysis::CreateNestedCostAnalysis");

  return std::make_unique<GpuHloCostAnalysis>(options_);
}

}  // namespace gpu
}  // namespace xla
