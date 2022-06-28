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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_4d_expanderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_4d_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_4d_expanderDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/convolution_4d_expander.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

bool Convolution4DExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_4d_expanderDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/xla/service/convolution_4d_expander.cc", "Convolution4DExpander::InstructionMatchesPattern");

  if (instruction->opcode() != HloOpcode::kConvolution) {
    return false;
  }

  // Check whether it is a 4D convolution and whether there is at least one
  // trivial dimension.
  const ConvolutionDimensionNumbers& dim_nums =
      instruction->convolution_dimension_numbers();
  if (dim_nums.input_spatial_dimensions().size() != 4) {
    return false;
  }
  Shape input = instruction->operand(0)->shape();
  for (int64_t i = 0; i < dim_nums.input_spatial_dimensions().size(); ++i) {
    int64_t spatial_dim = dim_nums.input_spatial_dimensions(i);
    if (input.dimensions(spatial_dim) == 1 &&
        instruction->window().dimensions(i).padding_low() == 0 &&
        instruction->window().dimensions(i).padding_high() == 0) {
      return true;
    }
  }
  return false;
}

StatusOr<HloInstruction*> Convolution4DExpander::ExpandInstruction(
    HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_4d_expanderDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/xla/service/convolution_4d_expander.cc", "Convolution4DExpander::ExpandInstruction");

  HloComputation* computation = instruction->parent();
  ConvolutionDimensionNumbers dim_nums =
      instruction->convolution_dimension_numbers();
  ConvolutionDimensionNumbers new_dim_nums = dim_nums;

  std::vector<int64_t> removed_input_dimensions;
  std::vector<int64_t> removed_kernel_dimensions;
  std::vector<int64_t> removed_output_dimensions;
  new_dim_nums.clear_input_spatial_dimensions();
  new_dim_nums.clear_output_spatial_dimensions();
  new_dim_nums.clear_kernel_spatial_dimensions();
  Window new_window;
  HloInstruction* input = instruction->mutable_operand(0);

  // Collect all trivial input spatial dimensions, and the corresponding
  // dimensions of the kernel and the output. Those will be removed.
  for (int64_t i = 0; i < dim_nums.input_spatial_dimensions().size(); ++i) {
    int64_t input_spatial_dim = dim_nums.input_spatial_dimensions(i);
    int64_t output_spatial_dim = dim_nums.output_spatial_dimensions(i);
    int64_t kernel_spatial_dim = dim_nums.kernel_spatial_dimensions(i);
    if (input->shape().dimensions(input_spatial_dim) == 1 &&
        instruction->window().dimensions(i).padding_low() == 0 &&
        instruction->window().dimensions(i).padding_high() == 0) {
      removed_input_dimensions.push_back(input_spatial_dim);
      removed_output_dimensions.push_back(output_spatial_dim);
      removed_kernel_dimensions.push_back(kernel_spatial_dim);
    } else {
      *new_window.add_dimensions() = instruction->window().dimensions(i);
      new_dim_nums.add_input_spatial_dimensions(input_spatial_dim);
      new_dim_nums.add_output_spatial_dimensions(output_spatial_dim);
      new_dim_nums.add_kernel_spatial_dimensions(kernel_spatial_dim);
    }
  }
  // We sort the removed dimensions into descending order, because we need to
  // delete higher dimensions first, otherwise we would have to adjust dimension
  // indices.
  std::sort(removed_input_dimensions.begin(), removed_input_dimensions.end(),
            std::greater<>());
  std::sort(removed_output_dimensions.begin(), removed_output_dimensions.end(),
            std::greater<>());
  std::sort(removed_kernel_dimensions.begin(), removed_kernel_dimensions.end(),
            std::greater<>());

  // Compute the new shapes.
  Shape new_input_shape = input->shape();
  for (int64_t dim : removed_input_dimensions) {
    new_input_shape.DeleteDimension(dim);
  }
  HloInstruction* kernel = instruction->mutable_operand(1);
  Shape new_kernel_shape = kernel->shape();
  for (int64_t dim : removed_kernel_dimensions) {
    new_kernel_shape.DeleteDimension(dim);
  }
  Shape new_output_shape = instruction->shape();
  for (int64_t dim : removed_output_dimensions) {
    new_output_shape.DeleteDimension(dim);
  }

  // Relabel the dimension numbers to account for the deleted dimensions. For
  // each dimension number, we need to reduce its value by the number of removed
  // smaller dimensions.
  auto compute_new_dimension =
      [](const std::vector<int64_t>& removed_dimensions,
         int64_t old_dimension) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconvolution_4d_expanderDTcc mht_2(mht_2_v, 296, "", "./tensorflow/compiler/xla/service/convolution_4d_expander.cc", "lambda");

        int64_t num_smaller = absl::c_count_if(
            removed_dimensions, [old_dimension](int64_t removed_dimension) {
              return removed_dimension < old_dimension;
            });
        return old_dimension - num_smaller;
      };
  new_dim_nums.set_input_batch_dimension(compute_new_dimension(
      removed_input_dimensions, new_dim_nums.input_batch_dimension()));
  new_dim_nums.set_input_feature_dimension(compute_new_dimension(
      removed_input_dimensions, new_dim_nums.input_feature_dimension()));
  for (int64_t i = 0; i < new_dim_nums.input_spatial_dimensions().size(); ++i) {
    new_dim_nums.set_input_spatial_dimensions(
        i, compute_new_dimension(removed_input_dimensions,
                                 new_dim_nums.input_spatial_dimensions(i)));
  }
  new_dim_nums.set_output_batch_dimension(compute_new_dimension(
      removed_output_dimensions, new_dim_nums.output_batch_dimension()));
  new_dim_nums.set_output_feature_dimension(compute_new_dimension(
      removed_output_dimensions, new_dim_nums.output_feature_dimension()));
  for (int64_t i = 0; i < new_dim_nums.output_spatial_dimensions().size();
       ++i) {
    new_dim_nums.set_output_spatial_dimensions(
        i, compute_new_dimension(removed_output_dimensions,
                                 new_dim_nums.output_spatial_dimensions(i)));
  }
  new_dim_nums.set_kernel_input_feature_dimension(
      compute_new_dimension(removed_kernel_dimensions,
                            new_dim_nums.kernel_input_feature_dimension()));
  new_dim_nums.set_kernel_output_feature_dimension(
      compute_new_dimension(removed_kernel_dimensions,
                            new_dim_nums.kernel_output_feature_dimension()));
  for (int64_t i = 0; i < new_dim_nums.kernel_spatial_dimensions().size();
       ++i) {
    new_dim_nums.set_kernel_spatial_dimensions(
        i, compute_new_dimension(removed_kernel_dimensions,
                                 new_dim_nums.kernel_spatial_dimensions(i)));
  }

  // Reshape the input and the kernel.
  HloInstruction* reshaped_input = computation->AddInstruction(
      HloInstruction::CreateReshape(new_input_shape, input));
  HloInstruction* reshaped_kernel = computation->AddInstruction(
      HloInstruction::CreateReshape(new_kernel_shape, kernel));

  // We want to use CloneWithNewOperands, but that doesn't support substituting
  // the window and the ConvolutionDimensionNumbers. So we set this on the old
  // instruction (which is going to be removed anyway) before cloning it.
  instruction->set_convolution_dimension_numbers(new_dim_nums);
  instruction->set_window(new_window);
  HloInstruction* new_convolution =
      computation->AddInstruction(instruction->CloneWithNewOperands(
          new_output_shape, {reshaped_input, reshaped_kernel}));
  return computation->AddInstruction(
      HloInstruction::CreateReshape(instruction->shape(), new_convolution));
}

}  // namespace xla
