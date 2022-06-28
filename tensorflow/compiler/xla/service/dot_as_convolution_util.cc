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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_as_convolution_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_as_convolution_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_as_convolution_utilDTcc() {
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

#include "tensorflow/compiler/xla/service/dot_as_convolution_util.h"

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace dot_as_convolution_util {

SpatialBatchRepresentation SpatialIsBatch(int64_t lhs_spatial_size,
                                          const WindowDimension& spatial_wd) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_as_convolution_utilDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/xla/service/dot_as_convolution_util.cc", "SpatialIsBatch");

  if (lhs_spatial_size == spatial_wd.size() &&
      lhs_spatial_size == spatial_wd.base_dilation() &&
      ((std::max<int64_t>(1, lhs_spatial_size - 1) == spatial_wd.stride() &&
        spatial_wd.window_dilation() == 1) ||
       (std::max<int64_t>(1, lhs_spatial_size - 1) ==
            spatial_wd.window_dilation() &&
        spatial_wd.stride() == 1)) &&
      spatial_wd.padding_high() == 0 && spatial_wd.padding_low() == 0 &&
      !spatial_wd.window_reversal()) {
    return SpatialBatchRepresentation::kUnpaddedVersion;
  } else if (lhs_spatial_size == spatial_wd.size() &&
             spatial_wd.padding_high() == lhs_spatial_size - 1 &&
             spatial_wd.padding_low() == lhs_spatial_size - 1 &&
             spatial_wd.window_reversal() &&
             spatial_wd.window_dilation() == 1 &&
             spatial_wd.stride() == lhs_spatial_size &&
             spatial_wd.base_dilation() == lhs_spatial_size - 1) {
    return SpatialBatchRepresentation::kPaddedVersion;
  }
  return SpatialBatchRepresentation::kNone;
}

bool SpatialIsLhsNonContracting(int64_t rhs_spatial_size,
                                const WindowDimension& spatial_wd) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_as_convolution_utilDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/service/dot_as_convolution_util.cc", "SpatialIsLhsNonContracting");

  return spatial_wd.stride() == 1 && spatial_wd.window_dilation() == 1 &&
         spatial_wd.base_dilation() == 1 && rhs_spatial_size == 1 &&
         spatial_wd.size() == 1 && spatial_wd.padding_high() == 0 &&
         spatial_wd.padding_low() == 0 && !spatial_wd.window_reversal();
}

bool SpatialIsRhsNonContracting(int64_t lhs_spatial_size,
                                int64_t rhs_spatial_size,
                                const WindowDimension& spatial_wd) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_as_convolution_utilDTcc mht_2(mht_2_v, 236, "", "./tensorflow/compiler/xla/service/dot_as_convolution_util.cc", "SpatialIsRhsNonContracting");

  return spatial_wd.stride() == 1 && spatial_wd.window_dilation() == 1 &&
         spatial_wd.base_dilation() == 1 && lhs_spatial_size == 1 &&
         spatial_wd.size() == rhs_spatial_size &&
         spatial_wd.padding_high() == rhs_spatial_size - 1 &&
         spatial_wd.padding_low() == rhs_spatial_size - 1 &&
         spatial_wd.window_reversal();
}

bool SpatialIsContracting(int64_t lhs_spatial_size, int64_t rhs_spatial_size,
                          const WindowDimension& spatial_wd) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_as_convolution_utilDTcc mht_3(mht_3_v, 249, "", "./tensorflow/compiler/xla/service/dot_as_convolution_util.cc", "SpatialIsContracting");

  return lhs_spatial_size == spatial_wd.size() &&
         spatial_wd.base_dilation() == 1 && spatial_wd.window_dilation() == 1 &&
         spatial_wd.padding_high() == 0 && spatial_wd.padding_low() == 0 &&
         !spatial_wd.window_reversal();
}

/* static */ DotConvolutionDimsInfo ParseConvolutionDimsInfo(
    const HloInstruction* conv) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_as_convolution_utilDTcc mht_4(mht_4_v, 260, "", "./tensorflow/compiler/xla/service/dot_as_convolution_util.cc", "ParseConvolutionDimsInfo");

  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);
  const auto& conv_dims = conv->convolution_dimension_numbers();
  DotConvolutionDimsInfo dims;
  dims.lhs_non_contracting_dims.push_back(
      {conv_dims.input_batch_dimension(), -1,
       conv_dims.output_batch_dimension(), -1});
  dims.rhs_non_contracting_dims.push_back(
      {-1, conv_dims.kernel_output_feature_dimension(),
       conv_dims.output_feature_dimension(), -1});
  dims.contracting_dims.push_back({conv_dims.input_feature_dimension(),
                                   conv_dims.kernel_input_feature_dimension(),
                                   -1, -1});

  for (int64_t i = 0; i < conv_dims.input_spatial_dimensions_size(); ++i) {
    int64_t lhs = conv_dims.input_spatial_dimensions(i);
    int64_t lhs_size = conv->operand(0)->shape().dimensions(lhs);
    int64_t rhs = conv_dims.kernel_spatial_dimensions(i);
    int64_t rhs_size = conv->operand(1)->shape().dimensions(rhs);
    int64_t output = conv_dims.output_spatial_dimensions(i);
    const auto& wd = conv->window().dimensions(i);
    if (SpatialIsBatch(lhs_size, wd) != SpatialBatchRepresentation::kNone) {
      dims.batch_dims.push_back({lhs, rhs, output, i});
    } else if (lhs_size == wd.size() && wd.base_dilation() == 1 &&
               wd.window_dilation() == 1 && wd.padding_high() == 0 &&
               wd.padding_low() == 0 && !wd.window_reversal()) {
      // A contracting dimension be represented as a spatial dimension with
      // window size C (contracting dimension size). Stride can be any size
      // since there is only one window.
      dims.contracting_dims.push_back({lhs, rhs, output, i});
    } else if (wd.stride() == 1 && wd.window_dilation() == 1 &&
               wd.base_dilation() == 1) {
      if (rhs_size == 1 && wd.size() == 1 && wd.padding_high() == 0 &&
          wd.padding_low() == 0 && !wd.window_reversal()) {
        // A LHS non-contracting dimension can be represented as a spatial
        // dimension with window size 1.
        dims.lhs_non_contracting_dims.push_back({lhs, rhs, output, i});
      } else if (lhs_size == 1 && wd.size() == rhs_size &&
                 wd.padding_high() == rhs_size - 1 &&
                 wd.padding_low() == rhs_size - 1 && wd.window_reversal()) {
        // A RHS non-contracting dimension can be represented as a spatial
        // dimension with window size N (non-contracting dimension size), low
        // padding N - 1,  high padding N - 1 and window reversal.
        dims.rhs_non_contracting_dims.push_back({lhs, rhs, output, i});
      } else {
        dims.conv_spatial_dims.push_back({lhs, rhs, output, i});
      }
    } else {
      dims.conv_spatial_dims.push_back({lhs, rhs, output, i});
    }
  }

  return dims;
}

StatusOr<std::unique_ptr<HloInstruction>>
CreateShardedConvForDotGeneralConvolution(
    const HloInstruction& conv, const DotConvolutionDimsInfo& dot_dnums,
    HloInstruction* sharded_lhs_hlo, HloInstruction* sharded_rhs_hlo) {
  CHECK_EQ(conv.opcode(), HloOpcode::kConvolution);
  const auto& conv_dnums = conv.convolution_dimension_numbers();
  auto window = conv.window();
  for (const auto& dim : dot_dnums.batch_dims) {
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    wd->set_size(sharded_lhs_hlo->shape().dimensions(
        conv_dnums.input_spatial_dimensions(dim.spatial_dim)));
    wd->set_stride(std::max<int64_t>(1, wd->size() - 1));
    wd->set_base_dilation(wd->size());
  }
  for (const auto& dim : dot_dnums.contracting_dims) {
    if (dim.spatial_dim < 0) {
      continue;
    }
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    wd->set_size(sharded_lhs_hlo->shape().dimensions(
        conv_dnums.input_spatial_dimensions(dim.spatial_dim)));
  }
  for (const auto& dim : dot_dnums.rhs_non_contracting_dims) {
    if (dim.spatial_dim < 0) {
      continue;
    }
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    wd->set_size(sharded_rhs_hlo->shape().dimensions(
        conv_dnums.kernel_spatial_dimensions(dim.spatial_dim)));
    wd->set_padding_high(wd->size() - 1);
    wd->set_padding_low(wd->size() - 1);
  }
  TF_ASSIGN_OR_RETURN(
      Shape sharded_conv_shape,
      ShapeInference::InferConvolveShape(
          sharded_lhs_hlo->shape(), sharded_rhs_hlo->shape(),
          /*feature_group_count=*/conv.feature_group_count(),
          /*batch_group_count=*/conv.batch_group_count(), window, conv_dnums,
          /*preferred_element_type=*/conv.shape().element_type()));
  *sharded_conv_shape.mutable_layout() = conv.shape().layout();
  return HloInstruction::CreateConvolve(
      sharded_conv_shape, sharded_lhs_hlo, sharded_rhs_hlo,
      /*feature_group_count=*/conv.feature_group_count(),
      /*batch_group_count=*/conv.batch_group_count(), window, conv_dnums,
      conv.precision_config());
}

DotConvolutionDimsInfo ParseDotGeneralFromDot(const HloInstruction* dot) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_as_convolution_utilDTcc mht_5(mht_5_v, 365, "", "./tensorflow/compiler/xla/service/dot_as_convolution_util.cc", "ParseDotGeneralFromDot");

  const auto& dot_dim_numbs = dot->dot_dimension_numbers();
  dot_as_convolution_util::DotConvolutionDimsInfo dnums;
  for (int64_t i = 0; i < dot_dim_numbs.lhs_batch_dimensions().size(); ++i) {
    dnums.batch_dims.emplace_back();
    dnums.batch_dims.back().lhs = dot_dim_numbs.lhs_batch_dimensions(i);
    dnums.batch_dims.back().rhs = dot_dim_numbs.rhs_batch_dimensions(i);
    dnums.batch_dims.back().output = i;
    dnums.batch_dims.back().spatial_dim = -1;
  }
  for (int64_t i = 0; i < dot_dim_numbs.lhs_contracting_dimensions().size();
       ++i) {
    dnums.contracting_dims.emplace_back();
    dnums.contracting_dims.back().lhs =
        dot_dim_numbs.lhs_contracting_dimensions(i);
    dnums.contracting_dims.back().rhs =
        dot_dim_numbs.rhs_contracting_dimensions(i);
    dnums.contracting_dims.back().output = -1;
    dnums.contracting_dims.back().spatial_dim = -1;
  }
  for (int64_t i = 0; i < dot->operand(0)->shape().rank(); ++i) {
    if (!absl::c_linear_search(dot_dim_numbs.lhs_batch_dimensions(), i) &&
        !absl::c_linear_search(dot_dim_numbs.lhs_contracting_dimensions(), i)) {
      dnums.lhs_non_contracting_dims.emplace_back();
      dnums.lhs_non_contracting_dims.back().lhs = i;
      dnums.lhs_non_contracting_dims.back().rhs = -1;
      dnums.lhs_non_contracting_dims.back().output =
          dot_dim_numbs.lhs_batch_dimensions_size() +
          dnums.lhs_non_contracting_dims.size() - 1;
      dnums.lhs_non_contracting_dims.back().spatial_dim = -1;
    }
  }
  for (int64_t i = 0; i < dot->operand(1)->shape().rank(); ++i) {
    if (!absl::c_linear_search(dot_dim_numbs.rhs_batch_dimensions(), i) &&
        !absl::c_linear_search(dot_dim_numbs.rhs_contracting_dimensions(), i)) {
      dnums.rhs_non_contracting_dims.emplace_back();
      dnums.rhs_non_contracting_dims.back().lhs = -1;
      dnums.rhs_non_contracting_dims.back().rhs = i;
      dnums.rhs_non_contracting_dims.back().output =
          dot_dim_numbs.lhs_batch_dimensions_size() +
          dnums.lhs_non_contracting_dims.size() +
          dnums.rhs_non_contracting_dims.size() - 1;
      dnums.rhs_non_contracting_dims.back().spatial_dim = -1;
    }
  }
  return dnums;
}

}  // namespace dot_as_convolution_util
}  // namespace xla
