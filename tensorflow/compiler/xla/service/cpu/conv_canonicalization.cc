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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSconv_canonicalizationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSconv_canonicalizationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSconv_canonicalizationDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/conv_canonicalization.h"

#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace cpu {

StatusOr<bool> ConvCanonicalization::Run(HloModule* module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSconv_canonicalizationDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/xla/service/cpu/conv_canonicalization.cc", "ConvCanonicalization::Run");

  bool changed = false;
  for (HloInstruction* hlo :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (hlo->opcode() == HloOpcode::kConvolution &&
        !PotentiallyImplementedAsEigenConvolution(*hlo,
                                                  target_machine_features_)) {
      const ConvolutionDimensionNumbers& dnums =
          hlo->convolution_dimension_numbers();
      auto input_batch_dim = dnums.input_batch_dimension();
      auto input_feature_dim = dnums.input_feature_dimension();
      auto kernel_input_feature_dim = dnums.kernel_input_feature_dimension();
      auto kernel_output_feature_dim = dnums.kernel_output_feature_dimension();

      const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
      const int64_t num_dims = num_spatial_dims + 2;

      // A canonical convolution's dimension numbers need to satisfy the
      // following conditions (see cs/PotentiallyImplementedAsEigenConvolution).
      //
      // - the input is in NHWC order.
      // - the kernel is in HWIO order.
      //
      // For simplicity, as a first step, we reshape the input and filter to
      // NHWC and HWIO order, respectively. This may lose precision but won't
      // break the soundness.
      HloInstruction* input = hlo->mutable_operand(0);

      std::vector<int64_t> new_input_dim_order(num_dims);
      std::vector<int64_t> new_input_dims(num_dims);
      new_input_dim_order[0] = input_batch_dim;
      new_input_dims[0] = input->shape().dimensions(input_batch_dim);
      for (int64_t i = 0; i < num_spatial_dims; ++i) {
        new_input_dim_order[i + 1] = dnums.input_spatial_dimensions(i);
        new_input_dims[i + 1] =
            input->shape().dimensions(dnums.input_spatial_dimensions(i));
      }
      new_input_dim_order[num_dims - 1] = input_feature_dim;
      new_input_dims[num_dims - 1] =
          input->shape().dimensions(input_feature_dim);

      Shape new_input_shape =
          ShapeUtil::MakeShape(input->shape().element_type(), new_input_dims);
      HloInstruction* new_input = module->entry_computation()->AddInstruction(
          HloInstruction::CreateTranspose(new_input_shape, input,
                                          new_input_dim_order));

      HloInstruction* kernel = hlo->mutable_operand(1);

      std::vector<int64_t> new_kernel_dim_order(num_dims);
      std::vector<int64_t> new_kernel_dims(num_dims);
      for (int64_t i = 0; i < num_spatial_dims; ++i) {
        new_kernel_dim_order[i] = dnums.kernel_spatial_dimensions(i);
        new_kernel_dims[i] =
            kernel->shape().dimensions(dnums.kernel_spatial_dimensions(i));
      }
      new_kernel_dim_order[num_dims - 2] = kernel_input_feature_dim;
      new_kernel_dims[num_dims - 2] =
          kernel->shape().dimensions(kernel_input_feature_dim);
      new_kernel_dim_order[num_dims - 1] = kernel_output_feature_dim;
      new_kernel_dims[num_dims - 1] =
          kernel->shape().dimensions(kernel_output_feature_dim);

      Shape new_kernel_shape =
          ShapeUtil::MakeShape(kernel->shape().element_type(), new_kernel_dims);
      HloInstruction* new_kernel = module->entry_computation()->AddInstruction(
          HloInstruction::CreateTranspose(new_kernel_shape, kernel,
                                          new_kernel_dim_order));

      std::vector<int64_t> new_output_dim_order(num_dims);
      std::vector<int64_t> new_conv_dims(num_dims);
      auto output_batch_dim = dnums.output_batch_dimension();
      auto output_feature_dim = dnums.output_feature_dimension();
      new_output_dim_order[0] = output_batch_dim;
      new_conv_dims[0] = hlo->shape().dimensions(output_batch_dim);
      for (int64_t i = 0; i < num_spatial_dims; ++i) {
        new_output_dim_order[i + 1] = dnums.output_spatial_dimensions(i);
        new_conv_dims[i + 1] =
            hlo->shape().dimensions(dnums.output_spatial_dimensions(i));
      }
      new_output_dim_order[num_dims - 1] = output_feature_dim;
      new_conv_dims[num_dims - 1] = hlo->shape().dimensions(output_feature_dim);
      Shape new_conv_shape =
          ShapeUtil::MakeShape(hlo->shape().element_type(), new_conv_dims);

      ConvolutionDimensionNumbers new_dnums;
      new_dnums.set_input_batch_dimension(0);
      new_dnums.set_output_batch_dimension(0);
      for (int64_t i = 0; i < num_spatial_dims; ++i) {
        new_dnums.add_input_spatial_dimensions(i + 1);
        new_dnums.add_kernel_spatial_dimensions(i);
        new_dnums.add_output_spatial_dimensions(i + 1);
      }
      new_dnums.set_input_feature_dimension(num_dims - 1);
      new_dnums.set_output_feature_dimension(num_dims - 1);
      new_dnums.set_kernel_input_feature_dimension(num_dims - 2);
      new_dnums.set_kernel_output_feature_dimension(num_dims - 1);

      // The window of the old convolution is reused, because reshapes only
      // change the dimension mapping but not the dimension sizes. For
      // example, input height and width are the same as before the reshapes.
      HloInstruction* new_conv = module->entry_computation()->AddInstruction(
          HloInstruction::CreateConvolve(
              new_conv_shape, new_input, new_kernel, hlo->feature_group_count(),
              hlo->batch_group_count(), hlo->window(), new_dnums,
              hlo->precision_config()));

      // Reshape the output back to the shape of the original convolution.
      TF_RETURN_IF_ERROR(module->entry_computation()->ReplaceWithNewInstruction(
          hlo, HloInstruction::CreateTranspose(
                   hlo->shape(), new_conv,
                   InversePermutation(new_output_dim_order))));
      changed = true;
    }
  }

  return changed;
}

}  // namespace cpu
}  // namespace xla
