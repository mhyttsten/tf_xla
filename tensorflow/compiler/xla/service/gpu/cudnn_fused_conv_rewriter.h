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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_FUSED_CONV_REWRITER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_FUSED_CONV_REWRITER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScudnn_fused_conv_rewriterDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScudnn_fused_conv_rewriterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScudnn_fused_conv_rewriterDTh() {
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


#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites custom-calls targeting cudnnConvolutionForward to
// cudnnConvolutionBiasActivationForward by fusing operations following forward
// convolution.  This transform must run after cudnn_conv_rewriter.
//
// Semantics of underlying cudnn ops:
//
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
// https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters
//
// ## Floating-point convs
//
// A "complete" fused floating-point conv has the form
//
//   max(0, alpha1 * conv(x, w) + alpha2 * side_input + broadcast(bias)),
//
// which we fuse to
//
//   cudnnConvolutionBiasActivationForward(x, w, bias, side_input).
//
// You can leave out side_input, bias, alpha1, alpha2, and max(x, 0) and still
// get a fused convolution.  alpha1/2 must be broadcasts of scalar constants.
//
// f16 convs accumulate in f32.  We represent this in HLO as an f32 convolution
// whose inputs can be converted to f16 without loss of precision and whose
// output is immediately converted to f16.  A fused f16 conv must follow one of
// the following idioms.
//
//   1. convert_f16(conv_f32(x_f32, w_f32)) +
//      side_input_f16 + broadcast(bias_f16)
//
//   2. convert_f16(conv_f32(x_f32, w_f32) +
//                  side_input_f32 + broadcast(bias_f32))
//
// (These are not strictly mathematically equivalent, but cudnn doesn't tell us
// which one it does, and we deem them "close enough".)
//
// The foo_f32 HLOs must all be losslessly-convertible to f16.  Some valid
// examples:
//
//   - foo_f32 = convert_f32(foo_f16)
//   - foo_f32 = an f32 constant whose values all fit within f16
//   - foo_f32 = broadcast/transpose/reshape(one of the above)
//
// If you have a relu, it can appear before or after the convert_f16.
//
// Note that here `bias` must be losslessly-convertible to f16; this is
// different than for s8 convolutions, where bias is f32.
//
// ## Integer convs
//
// In pure HLO, a "complete" integer conv is spelled as one of the following
// `result`s.
//
//   base = alpha1_f32 * convert_f32(conv_s32(input_s32, filter_s32)) +
//          alpha2_f32 * side_input +
//          bias_f32
//
//   result_f32        = max(result_f32, 0)
//   result_s8_option1 = max(convert_s8(clamp(-128, base, 127)), 0)
//   result_s8_option2 = convert_s8(clamp(-128, max(base, 0), 127))
//
// The foo_s32 HLOs must be losslessly-convertible to s8.  If the `result_s8`
// case, side_input should be an f32 HLO that's losslessly-convertible to s8;
// otherwise, it should be losslessly-convertible to f32.
//
// In the `result_s8` case where there's no bias, side-input, or alpha1, you can
// skip the convert_f32 on conv.
//
// If you have an integer convolution that doesn't fit one of these idioms, this
// pass returns an error -- cudnn will not be able to run it.
class CudnnFusedConvRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScudnn_fused_conv_rewriterDTh mht_0(mht_0_v, 266, "", "./tensorflow/compiler/xla/service/gpu/cudnn_fused_conv_rewriter.h", "name");

    return "cudnn-fused-convolution-rewriter";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_FUSED_CONV_REWRITER_H_
