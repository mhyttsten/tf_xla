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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSdepthwise_conv_plus_1x1_convDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSdepthwise_conv_plus_1x1_convDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSdepthwise_conv_plus_1x1_convDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/special/depthwise_conv_plus_1x1_conv.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/relu.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {
void UploadWeights(const DepthwiseConvolution2DAttributes& dw_attr,
                   const Convolution2DAttributes& conv_attr,
                   const GpuInfo& gpu_info, CalculationsPrecision precision,
                   GPUOperation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSdepthwise_conv_plus_1x1_convDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/depthwise_conv_plus_1x1_conv.cc", "UploadWeights");

  int dw_dst_ch_aligned = AlignByN(dw_attr.weights.shape.i, 4);
  int dw_weights_count =
      dw_dst_ch_aligned * dw_attr.weights.shape.h * dw_attr.weights.shape.w;
  int conv_src_ch_aligned = AlignByN(conv_attr.weights.shape.i, 4);
  int conv_dst_ch_aligned = AlignByN(conv_attr.weights.shape.o, 4);
  int conv_weights_count = conv_src_ch_aligned * conv_dst_ch_aligned;
  std::vector<float> gpu_data;
  gpu_data.reserve(dw_dst_ch_aligned + dw_weights_count + conv_dst_ch_aligned +
                   conv_weights_count);
  // dw bias loading
  for (int i = 0; i < dw_dst_ch_aligned; ++i) {
    if (i < dw_attr.bias.shape.v) {
      gpu_data.push_back(dw_attr.bias.data[i]);
    } else {
      gpu_data.push_back(0.0f);
    }
  }
  // dw weights loading
  for (int d = 0; d < dw_dst_ch_aligned / 4; ++d) {
    for (int y = 0; y < dw_attr.weights.shape.h; ++y) {
      for (int x = 0; x < dw_attr.weights.shape.w; ++x) {
        for (int i = 0; i < 4; ++i) {
          const int d_ch = d * 4 + i;
          if (d_ch < dw_attr.weights.shape.i) {
            const int f_index =
                dw_attr.weights.shape.LinearIndex({0, y, x, d_ch});
            gpu_data.push_back(dw_attr.weights.data[f_index]);
          } else {
            gpu_data.push_back(0.0f);
          }
        }
      }
    }
  }
  // conv bias loading
  for (int i = 0; i < conv_dst_ch_aligned; ++i) {
    if (i < conv_attr.bias.shape.v) {
      gpu_data.push_back(conv_attr.bias.data[i]);
    } else {
      gpu_data.push_back(0.0f);
    }
  }
  // conv weights loading
  for (int d = 0; d < conv_dst_ch_aligned / 4; ++d) {
    for (int s = 0; s < conv_src_ch_aligned / 4; ++s) {
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const int s_ch = s * 4 + j;
          const int d_ch = d * 4 + i;
          if (s_ch < conv_attr.weights.shape.i &&
              d_ch < conv_attr.weights.shape.o) {
            const int f_index =
                conv_attr.weights.shape.LinearIndex({d_ch, 0, 0, s_ch});
            gpu_data.push_back(conv_attr.weights.data[f_index]);
          } else {
            gpu_data.push_back(0.0f);
          }
        }
      }
    }
  }

  const bool fp32_weights = precision == CalculationsPrecision::F32;
  const int float_size = fp32_weights ? 4 : 2;
  BufferDescriptor desc;
  desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  desc.memory_type =
      gpu_info.IsMali() ? MemoryType::GLOBAL : MemoryType::CONSTANT;
  desc.size = float_size * gpu_data.size();
  desc.data.resize(desc.size);

  if (fp32_weights) {
    memcpy(desc.data.data(), gpu_data.data(), desc.size);
  } else {
    half* gpu_data_half = reinterpret_cast<half*>(desc.data.data());
    for (int i = 0; i < gpu_data.size(); ++i) {
      gpu_data_half[i] = gpu_data[i];
    }
  }
  op->args_.AddObject("constants",
                      absl::make_unique<BufferDescriptor>(std::move(desc)));
}

std::string GenerateCode(const OperationDef& op_def, const GpuInfo& gpu_info,
                         const DepthwiseConvolution2DAttributes& dw_attr,
                         ReLUAttributes* relu_attr_ptr, int result_depth,
                         GPUOperation* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSdepthwise_conv_plus_1x1_convDTcc mht_1(mht_1_v, 294, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/depthwise_conv_plus_1x1_conv.cc", "GenerateCode");

  auto src_desc = op_def.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  result->AddSrcTensor("src_tensor", src_desc);
  result->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  result->args_.AddInt("stride_x", dw_attr.strides.w);
  result->args_.AddInt("padding_x", -dw_attr.padding.prepended.w);
  result->args_.AddInt("dilation_x", dw_attr.dilations.w);
  result->args_.AddInt("stride_y", dw_attr.strides.h);
  result->args_.AddInt("padding_y", -dw_attr.padding.prepended.h);
  result->args_.AddInt("dilation_y", dw_attr.dilations.h);

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) { "
       "\n";
  c += "    return; \n";
  c += "  } \n";
  int intermediate_depth = DivideRoundUp(dw_attr.weights.shape.i, 4);
  int weights_counter = 0;
  for (int d = 0; d < intermediate_depth; ++d) {
    c += "  FLT4 dw_res_" + std::to_string(d) + " = args.constants.Read(" +
         std::to_string(weights_counter++) + ");\n";
  }
  c += "  int x_offseted = X * args.stride_x + args.padding_x;\n";
  c += "  int y_offseted = Y * args.stride_y + args.padding_y;\n";
  c += "  int x_c, y_c;\n";

  auto generate_check = [&]() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSdepthwise_conv_plus_1x1_convDTcc mht_2(mht_2_v, 336, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/depthwise_conv_plus_1x1_conv.cc", "lambda");

    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"x_in", "y_in", "z_in"};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis) && !src_desc.SupportsZeroClamp(axis)) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i];
      }
    }
    return check;
  };
  const std::string check = generate_check();
  if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
    c += "  bool y_in;\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::WIDTH)) {
    c += "  bool x_in;\n";
  }

  const std::string postfixes[] = {".x", ".xy", ".xyz", ""};
  c += "  FLT4 src;\n";
  for (int d = 0; d < intermediate_depth; ++d) {
    const int src_ch_count = std::min(4, dw_attr.weights.shape.i - d * 4);
    const std::string s_postfix = postfixes[src_ch_count - 1];
    for (int ky = 0; ky < dw_attr.weights.shape.h; ++ky) {
      c += "  y_c = y_offseted + " + std::to_string(ky) +
           " * args.dilation_y;\n";
      if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
        c += "  y_in = y_c >= 0 && y_c < args.src_tensor.Height();\n";
        c += "  y_c = clamp(y_c, 0, args.src_tensor.Height() - 1);\n";
      }
      for (int kx = 0; kx < dw_attr.weights.shape.w; ++kx) {
        c += "  x_c = x_offseted + " + std::to_string(kx) +
             " * args.dilation_x;\n";
        if (!src_desc.SupportsZeroClamp(Axis::WIDTH)) {
          c += "  x_in = x_c >= 0 && x_c < args.src_tensor.Width();\n";
          c += "  x_c = clamp(x_c, 0, args.src_tensor.Width() - 1);\n";
        }
        std::string multiplier =
            check.empty() ? "" : " * INIT_FLT(" + check + ")";
        c += "  src" + s_postfix + " = args.src_tensor.Read(x_c, y_c, " +
             std::to_string(d) + ")" + s_postfix + multiplier + ";\n";
        c += "  dw_res_" + std::to_string(d) + s_postfix + " += src" +
             s_postfix + " * args.constants.Read(" +
             std::to_string(weights_counter++) + ")" + s_postfix + ";\n";
      }
    }
  }
  if (relu_attr_ptr) {
    std::string elementwise_code;
    CreateReLU(*relu_attr_ptr, op_def.precision, &result->args_,
               &elementwise_code);
    for (int d = 0; d < intermediate_depth; ++d) {
      const std::string var_name = "dw_res_" + std::to_string(d);
      const std::string elementwise_new_code =
          absl::StrReplaceAll(elementwise_code, {{"in_out_value", var_name}});
      c += "  {  " + elementwise_new_code + "  }\n";
    }
  }
  for (int d = 0; d < result_depth; ++d) {
    c += "  FLT4 conv_res_" + std::to_string(d) + " = args.constants.Read(" +
         std::to_string(weights_counter++) + ");\n";
  }
  for (int d = 0; d < result_depth; ++d) {
    for (int s = 0; s < intermediate_depth; ++s) {
      std::string src = "dw_res_" + std::to_string(s);
      std::string dst = "conv_res_" + std::to_string(d);
      c += "  " + dst + " += " + src + ".x * args.constants.Read(" +
           std::to_string(weights_counter++) + ");\n";
      c += "  " + dst + " += " + src + ".y * args.constants.Read(" +
           std::to_string(weights_counter++) + ");\n";
      c += "  " + dst + " += " + src + ".z * args.constants.Read(" +
           std::to_string(weights_counter++) + ");\n";
      c += "  " + dst + " += " + src + ".w * args.constants.Read(" +
           std::to_string(weights_counter++) + ");\n";
    }
    c += "  args.dst_tensor.Write(conv_res_" + std::to_string(d) + ", X, Y, " +
         std::to_string(d) + ");\n";
  }
  c += "}\n";

  return c;
}

}  // namespace

bool IsDepthwiseConvPlus1x1ConvSupported(
    const OperationDef& definition, const GpuInfo& gpu_info,
    const DepthwiseConvolution2DAttributes& dw_attr,
    const Convolution2DAttributes& conv_attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSdepthwise_conv_plus_1x1_convDTcc mht_3(mht_3_v, 432, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/depthwise_conv_plus_1x1_conv.cc", "IsDepthwiseConvPlus1x1ConvSupported");

  const auto dw_shape = dw_attr.weights.shape;
  const auto conv_shape = conv_attr.weights.shape;
  bool good_dw = dw_shape.o == 1;
  bool good_conv =
      conv_shape.w == 1 && conv_shape.h == 1 && conv_attr.dilations.w == 1 &&
      conv_attr.dilations.h == 1 && conv_attr.strides.w == 1 &&
      conv_attr.strides.h == 1 && conv_attr.padding.prepended.w == 0 &&
      conv_attr.padding.prepended.h == 0 && conv_attr.padding.appended.w == 0 &&
      conv_attr.padding.appended.h == 0;
  if (gpu_info.IsApple()) {
    if (definition.precision == CalculationsPrecision::F16) {
      bool recommended_dw = dw_shape.i <= 16 &&
                            dw_shape.i * dw_shape.h * dw_shape.w <= 3 * 3 * 16;
      bool recommended_conv =
          conv_shape.o <= 16 && conv_shape.i * conv_shape.o <= 16 * 16;
      return good_dw && good_conv && recommended_dw && recommended_conv;
    } else {
      bool recommended_dw = dw_shape.i <= 16 &&
                            dw_shape.i * dw_shape.h * dw_shape.w <= 3 * 3 * 16;
      bool recommended_conv =
          conv_shape.o <= 8 && conv_shape.i * conv_shape.o <= 8 * 16;
      return good_dw && good_conv && recommended_dw && recommended_conv;
    }
  } else if (gpu_info.IsMali()) {
    if (!gpu_info.mali_info.IsValhall()) {
      return false;
    }
    if (gpu_info.mali_info.IsValhallGen1()) {
      return false;
    }
    if (definition.precision == CalculationsPrecision::F16 &&
        definition.src_tensors[0].SupportsZeroClamp(Axis::WIDTH) &&
        definition.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT)) {
      bool recommended_dw = dw_shape.i <= 16 &&
                            dw_shape.i * dw_shape.h * dw_shape.w <= 3 * 3 * 16;
      bool recommended_conv =
          conv_shape.o <= 16 && conv_shape.i * conv_shape.o <= 16 * 16;
      return good_dw && good_conv && recommended_dw && recommended_conv;
    } else {
      return false;
    }
  } else {
    if (definition.precision == CalculationsPrecision::F16) {
      bool recommended_dw = dw_shape.i <= 32 &&
                            dw_shape.i * dw_shape.h * dw_shape.w <= 3 * 3 * 32;
      bool recommended_conv =
          conv_shape.o <= 32 && conv_shape.i * conv_shape.o <= 32 * 32;
      return good_dw && good_conv && recommended_dw && recommended_conv;
    } else {
      bool recommended_dw = dw_shape.i <= 16 &&
                            dw_shape.i * dw_shape.h * dw_shape.w <= 3 * 3 * 16;
      bool recommended_conv =
          conv_shape.o <= 32 && conv_shape.i * conv_shape.o <= 16 * 32;
      return good_dw && good_conv && recommended_dw && recommended_conv;
    }
  }
}

GPUOperation CreateDepthwiseConvPlus1x1Conv(
    const OperationDef& definition, const GpuInfo& gpu_info,
    const DepthwiseConvolution2DAttributes& dw_attr,
    const Convolution2DAttributes& conv_attr, ReLUAttributes* relu_attr_ptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSdepthwise_conv_plus_1x1_convDTcc mht_4(mht_4_v, 497, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/depthwise_conv_plus_1x1_conv.cc", "CreateDepthwiseConvPlus1x1Conv");

  GPUOperation result(definition);
  result.code_ =
      GenerateCode(definition, gpu_info, dw_attr, relu_attr_ptr,
                   DivideRoundUp(conv_attr.weights.shape.o, 4), &result);
  result.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
  UploadWeights(dw_attr, conv_attr, gpu_info, definition.precision, &result);
  return result;
}

}  // namespace gpu
}  // namespace tflite
