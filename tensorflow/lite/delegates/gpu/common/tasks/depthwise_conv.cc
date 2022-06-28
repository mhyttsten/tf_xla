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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {

bool IsSpecializedCase(int channel_multiplier) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.cc", "IsSpecializedCase");

  return channel_multiplier == 1 || channel_multiplier == 2 ||
         channel_multiplier == 4;
}

std::string GetSrcValue(int channel_multiplier, const std::string coords) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("coords: \"" + coords + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc mht_1(mht_1_v, 209, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.cc", "GetSrcValue");

  std::string c;
  if (channel_multiplier == 1) {
    c += "      FLT4 src_final = args.src_tensor.Read(" + coords + ", S);\n";
  } else if (channel_multiplier == 2) {
    c += "      int s_layer = S / 2;\n";
    c += "      FLT4 src = args.src_tensor.Read(" + coords + ", s_layer);\n";
    c += "      FLT2 t0 = S % 2 == 0 ? src.xy : src.zw;\n";
    c += "      FLT4 src_final = INIT_FLT4v4(t0.x, t0.x, t0.y, t0.y);\n";
  } else if (channel_multiplier == 4) {
    c += "      int s_layer = S / 4;\n";
    c += "      FLT4 src = args.src_tensor.Read(" + coords + ", s_layer);\n";
    c += "      FLT t0 = src.x;\n";
    c += "      int reminder = S % 4;\n";
    c += "      if (reminder == 1) t0 = src.y;\n";
    c += "      if (reminder == 2) t0 = src.z;\n";
    c += "      if (reminder == 3) t0 = src.w;\n";
    c += "      FLT4 src_final = INIT_FLT4v4(t0, t0, t0, t0);\n";
  } else {
    c += "      int s_layer = S / args.ch_multiplier;\n";
    c += "      FLT4 src = args.src_tensor.Read(" + coords + ", s_layer);\n";
    c += "      int s_offset = (S % args.ch_multiplier) * 4;\n";
    c += "      FLT4 src_final;\n";
    c += "      FLT temp_arr[4] = {src.x, src.y, src.z, src.w};\n";
    c += "      src_final.x = temp_arr[(s_offset + 0) / args.ch_multiplier];\n";
    c += "      src_final.y = temp_arr[(s_offset + 1) / args.ch_multiplier];\n";
    c += "      src_final.z = temp_arr[(s_offset + 2) / args.ch_multiplier];\n";
    c += "      src_final.w = temp_arr[(s_offset + 3) / args.ch_multiplier];\n";
  }

  return c;
}

std::string GenerateDepthwiseConvolutionCode(
    const OperationDef& op_def, bool stride_correction, int channel_multiplier,
    bool weights_are_buffer, bool dynamic_weights, GPUOperation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.cc", "GenerateDepthwiseConvolutionCode");

  auto src_desc = op_def.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddSrcTensor("src_tensor", src_desc);
  if (dynamic_weights) {
    op->AddSrcTensor("weights", op_def.src_tensors[1]);
  }

  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddDstTensor("dst_tensor", dst_desc);

  std::string c;

  c += "MAIN_FUNCTION(\n";
  c += "$0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  ACCUM_FLT4 r = INIT_ACCUM_FLT4(0.0f);\n";
  if (stride_correction) {
    c += "  int x_offseted = " +
         GetXStrideCorrectedV2("X", "args.src_tensor.Batch()", "args.stride_x",
                               "args.padding_x") +
         ";\n";
  } else {
    if (op_def.IsBatchSupported()) {
      c += "  int x_offseted = X * args.stride_x + args.padding_x * "
           "args.src_tensor.Batch();\n";
    } else {
      c += "  int x_offseted = X * args.stride_x + args.padding_x;\n";
    }
  }
  c += "  int y_offseted = Y * args.stride_y + args.padding_y;\n";
  if (!dynamic_weights) {
    std::string weights_offset = "args.kernel_size_x * args.kernel_size_y";
    if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
      c += "  int z_offseted = Z * args.stride_z + args.padding_z;\n";
      weights_offset += " * args.kernel_size_z";
    }
    if (weights_are_buffer) {
      c += "  int fx_c = S * " + weights_offset + ";\n";
    } else {
      c += "  int fx_c = 0;\n";
    }
  }
  std::string kernel_size_x =
      dynamic_weights ? "args.weights.Width()" : "args.kernel_size_x";
  std::string kernel_size_y =
      dynamic_weights ? "args.weights.Height()" : "args.kernel_size_y";
  std::string kernel_size_z =
      dynamic_weights ? "args.weights.Depth()" : "args.kernel_size_z";

  auto generate_check = [&]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc mht_3(mht_3_v, 318, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.cc", "lambda");

    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"outside_x", "outside_y", "outside_z"};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis) && !src_desc.SupportsZeroClamp(axis)) {
        if (!check.empty()) {
          check += " && ";
        }
        check += "!" + names[i];
      }
    }
    return check;
  };
  auto generate_coords = [&]() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc mht_4(mht_4_v, 336, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.cc", "lambda");

    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"x_c", "y_c", "z_c"};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis)) {
        if (!check.empty()) {
          check += ", ";
        }
        check += names[i];
      }
    }
    return check;
  };
  const std::string check = generate_check();
  const std::string coords = generate_coords();

  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  for (int kz = 0; kz < " + kernel_size_z + "; ++kz) {\n";
    c += "    int z_c = z_offseted + kz * args.dilation_z;\n";
    if (!src_desc.SupportsZeroClamp(Axis::DEPTH)) {
      c += "    bool outside_z = z_c < 0 || z_c >= args.src_tensor.Depth();\n";
    }
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::HEIGHT)) {
    c += "  for (int ky = 0; ky < " + kernel_size_y + "; ++ky) {\n";
    c += "    int y_c = y_offseted + ky * args.dilation_y;\n";
    if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
      c += "    bool outside_y = y_c < 0 || y_c >= args.src_tensor.Height();\n";
    }
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::WIDTH)) {
    c += "  for (int kx = 0; kx < " + kernel_size_x + "; ++kx) {\n";
    const std::string dilation_x =
        op_def.IsBatchSupported() ? "args.dilation_x * args.src_tensor.Batch()"
                                  : "args.dilation_x";
    c += "    int x_c = x_offseted + kx * " + dilation_x + ";\n";
    if (!src_desc.SupportsZeroClamp(Axis::WIDTH)) {
      c += "    bool outside_x = x_c < 0 || x_c >= args.src_tensor.Width();\n";
    }
  }
  if (!check.empty()) {
    c += "    if (" + check + ") {\n";
  }
  if (dynamic_weights) {
    c += "      FLT4 f = args.weights.Read(kx, ky, S);\n";
  } else {
    if (weights_are_buffer) {
      c += "      FLT4 f = args.weights.Read(fx_c);\n";
    } else {
      c += "      FLT4 f = args.weights.Read(fx_c, S);\n";
    }
  }
  c += GetSrcValue(channel_multiplier, coords);
  c += "      r += TO_ACCUM_TYPE(src_final * f);\n";
  if (!check.empty()) {
    c += "    }\n";
  }
  if (!dynamic_weights) {
    c += "    fx_c++;\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::WIDTH)) {
    c += "  }\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::HEIGHT)) {
    c += "  }\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  }\n";
  }
  c += "  FLT4 res0 = TO_FLT4(r) + args.biases.Read(S);\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  args.dst_tensor.Write(res0, X, Y, Z, S);\n";
  } else {
    c += "  args.dst_tensor.Write(res0, X, Y, S);\n";
  }
  c += "}\n";
  return c;
}

bool UseBuffersForWeights(const GpuInfo& gpu_info) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc mht_5(mht_5_v, 420, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.cc", "UseBuffersForWeights");

  if (gpu_info.IsApple()) {
    if (gpu_info.apple_info.IsA7GenerationGpu() ||
        gpu_info.apple_info.IsA8GenerationGpu()) {
      return false;
    }
  }
  return !gpu_info.SupportsImages() || gpu_info.IsMali() || gpu_info.IsApple();
}
}  // namespace

GPUOperation CreateDepthwiseConvolution2D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc mht_6(mht_6_v, 436, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.cc", "CreateDepthwiseConvolution2D");

  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.weights.shape.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("kernel_size_y", attr.weights.shape.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  if (!IsSpecializedCase(attr.weights.shape.o)) {
    op.args_.AddInt("ch_multiplier", attr.weights.shape.o);
  }
  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  op.code_ = GenerateDepthwiseConvolutionCode(definition, stride_correction,
                                              attr.weights.shape.o,
                                              weights_are_buffer, false, &op);
  UploadWeightsForDWConv2D(attr.weights, weights_are_buffer,
                           definition.precision, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  TensorLinearDescriptor desc;
  desc.storage_type = weights_are_buffer ? LinearStorageType::BUFFER
                                         : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

GPUOperation CreateDepthwiseConvolution2DDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc mht_7(mht_7_v, 474, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.cc", "CreateDepthwiseConvolution2DDynamicWeights");

  GPUOperation op(definition);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  op.code_ = GenerateDepthwiseConvolutionCode(definition, stride_correction, 1,
                                              false, true, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  TensorLinearDescriptor desc;
  desc.storage_type =
      !gpu_info.SupportsImages() || gpu_info.IsMali() || gpu_info.IsApple()
          ? LinearStorageType::BUFFER
          : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

GPUOperation CreateDepthwiseConvolution3D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution3DAttributes& attr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_convDTcc mht_8(mht_8_v, 505, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.cc", "CreateDepthwiseConvolution3D");

  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.weights.shape.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("kernel_size_y", attr.weights.shape.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  op.args_.AddInt("kernel_size_z", attr.weights.shape.d);
  op.args_.AddInt("stride_z", attr.strides.d);
  op.args_.AddInt("padding_z", -attr.padding.prepended.d);
  op.args_.AddInt("dilation_z", attr.dilations.d);
  if (!IsSpecializedCase(attr.weights.shape.o)) {
    op.args_.AddInt("ch_multiplier", attr.weights.shape.o);
  }
  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  op.code_ = GenerateDepthwiseConvolutionCode(definition, stride_correction,
                                              attr.weights.shape.o,
                                              weights_are_buffer, false, &op);
  UploadWeightsForDWConv3D(attr.weights, weights_are_buffer,
                           definition.precision, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  TensorLinearDescriptor desc;
  desc.storage_type = weights_are_buffer ? LinearStorageType::BUFFER
                                         : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

}  // namespace gpu
}  // namespace tflite
