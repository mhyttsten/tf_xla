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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

ConvolutionTransposed::ConvolutionTransposed(
    const OperationDef& definition, const ConvolutionTransposedAttributes& attr,
    const GpuInfo& gpu_info, bool weights_are_buffer)
    : GPUOperation(definition),
      stride_(attr.stride.w, attr.stride.h, 1, 1),
      block_size_(2, 2, 1, 2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "ConvolutionTransposed::ConvolutionTransposed");

  if (weights_are_buffer) {
    if (gpu_info.IsApple()) {
      weights_layout_ = WeightsLayout::kOSpatialIOGroupO4I4;
    } else {
      weights_layout_ = WeightsLayout::kOSpatialIOGroupI4O4;
    }
  } else {
    if (gpu_info.IsApple()) {
      weights_layout_ = WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4;
    } else {
      weights_layout_ = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    }
  }
  const bool is_f16 = definition.precision == CalculationsPrecision::F16;
  if (gpu_info.IsMali()) {
    if (gpu_info.mali_info.IsMidgard()) {
      block_size_ = is_f16 ? int4(2, 1, 1, 2) : int4(2, 1, 1, 1);
    } else {
      block_size_ = is_f16 ? int4(2, 2, 1, 2) : int4(2, 2, 1, 1);
    }
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  if (dst_depth == 1 || dst_depth == 3) {
    if (!gpu_info.IsMali()) {
      block_size_.y *= block_size_.w;
    }
    block_size_.w = 1;
  }

  args_.AddInt("stride_x", stride_.x);
  args_.AddInt("stride_y", stride_.y);
  args_.AddInt("padding_x", attr.padding.prepended.w);
  args_.AddInt("padding_y", attr.padding.prepended.h);
  args_.AddInt("kernel_size_x", attr.weights.shape.w);
  args_.AddInt("kernel_size_y", attr.weights.shape.h);
  code_ = GenerateConvolutionTransposedCode(definition_, gpu_info,
                                            weights_are_buffer, block_size_);
}

ConvolutionTransposed::ConvolutionTransposed(
    const OperationDef& definition,
    const ConvolutionTransposed3DAttributes& attr, const GpuInfo& gpu_info,
    bool weights_are_buffer)
    : GPUOperation(definition),
      stride_(attr.stride.w, attr.stride.h, attr.stride.d, 1),
      block_size_(2, 2, 1, 2) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_1(mht_1_v, 256, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "ConvolutionTransposed::ConvolutionTransposed");

  if (weights_are_buffer) {
    if (gpu_info.IsApple()) {
      weights_layout_ = WeightsLayout::kOSpatialIOGroupO4I4;
    } else {
      weights_layout_ = WeightsLayout::kOSpatialIOGroupI4O4;
    }
  } else {
    if (gpu_info.IsApple()) {
      weights_layout_ = WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4;
    } else {
      weights_layout_ = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    }
  }
  const bool is_f16 = definition.precision == CalculationsPrecision::F16;
  if (gpu_info.IsMali()) {
    if (gpu_info.mali_info.IsMidgard()) {
      block_size_ = is_f16 ? int4(2, 1, 1, 2) : int4(2, 1, 1, 1);
    } else {
      block_size_ = is_f16 ? int4(2, 2, 1, 2) : int4(2, 2, 1, 1);
    }
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  if (dst_depth == 1 || dst_depth == 3) {
    if (!gpu_info.IsMali()) {
      block_size_.y *= block_size_.w;
    }
    block_size_.w = 1;
  }

  args_.AddInt("stride_x", stride_.x);
  args_.AddInt("stride_y", stride_.y);
  args_.AddInt("stride_z", stride_.z);
  args_.AddInt("padding_x", attr.padding.prepended.w);
  args_.AddInt("padding_y", attr.padding.prepended.h);
  args_.AddInt("padding_z", attr.padding.prepended.d);
  args_.AddInt("kernel_size_x", attr.weights.shape.w);
  args_.AddInt("kernel_size_y", attr.weights.shape.h);
  args_.AddInt("kernel_size_z", attr.weights.shape.d);
  args_.AddInt("grid_size_y");
  code_ = GenerateConvolutionTransposedCode(definition_, gpu_info,
                                            weights_are_buffer, block_size_);
}

std::string ConvolutionTransposed::GenerateConvolutionTransposedCode(
    const OperationDef& op_def, const GpuInfo& gpu_info,
    bool weights_are_buffer, const int4& block_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_2(mht_2_v, 306, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "ConvolutionTransposed::GenerateConvolutionTransposedCode");

  auto src_desc = op_def.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  AddSrcTensor("src_tensor", src_desc);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  if (op_def.src_tensors.size() != 1) {
    // dynamic weights
    if (weights_layout_ == WeightsLayout::kOSpatialIOGroupI4O4 ||
        weights_layout_ == WeightsLayout::kOSpatialIOGroupO4I4) {
      BufferDescriptor desc;
      desc.element_type = op_def.src_tensors[1].data_type;
      desc.element_size = 16;
      desc.memory_type = MemoryType::GLOBAL;
      AddSrcBuffer("weights", desc);
    } else {
      for (int i = 0; i < 4; ++i) {
        Texture2DDescriptor desc;
        desc.element_type = op_def.src_tensors[1 + i].data_type;
        const std::string name = "weights" + std::to_string(i);
        AddSrcTexture2D("weights" + std::to_string(i), desc);
      }
    }
  }

  const auto& src_def = op_def.src_tensors[0];

  std::string c;

  for (int s = 0; s < block_size.w; ++s) {
    std::string f0, f1, f2, f3;
    if (weights_are_buffer) {
      if (gpu_info.SupportsPointersInKernels()) {
        f0 = "FLT16_0123(weights_cache[" + std::to_string(s) + "])";
        f1 = "FLT16_4567(weights_cache[" + std::to_string(s) + "])";
        f2 = "FLT16_89ab(weights_cache[" + std::to_string(s) + "])";
        f3 = "FLT16_cdef(weights_cache[" + std::to_string(s) + "])";
      } else {
        f0 = "FLT16_0123(flt16val)";
        f1 = "FLT16_4567(flt16val)";
        f2 = "FLT16_89ab(flt16val)";
        f3 = "FLT16_cdef(flt16val)";
      }
    } else {
      f0 = "f" + std::to_string(s * 4 + 0);
      f1 = "f" + std::to_string(s * 4 + 1);
      f2 = "f" + std::to_string(s * 4 + 2);
      f3 = "f" + std::to_string(s * 4 + 3);
    }
    if (GetWeightsDescription().IsI4O4()) {
      switch (op_def.precision) {
        case CalculationsPrecision::F32:
        case CalculationsPrecision::F16:
          c += "#define CONV" + std::to_string(s) + "(R, S)    \\\n";
          c += "R += S.x * " + f0 + "; \\\n";
          c += "R += S.y * " + f1 + "; \\\n";
          c += "R += S.z * " + f2 + "; \\\n";
          c += "R += S.w * " + f3 + ";   \n";
          break;
        case CalculationsPrecision::F32_F16:
          c += "#define CONV" + std::to_string(s) + "(R, S) \\\n";
          c += "R += TO_ACCUM_TYPE(S.x * " + f0 + " + S.y * " + f1 +
               " + S.z * " + f2 + " + S.w * " + f3 + ");\n";
          break;
      }
    } else {
      // O4I4
      c += "#define CONV" + std::to_string(s) + "(R, S)    \\\n";
      c += "R.x += dot(S, " + f0 + "); \\\n";
      c += "R.y += dot(S, " + f1 + "); \\\n";
      c += "R.z += dot(S, " + f2 + "); \\\n";
      c += "R.w += dot(S, " + f3 + ");   \n";
    }
  }

  auto generate_id = [&](const std::string& x, const std::string& y,
                         const std::string& z) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("x: \"" + x + "\"");
   mht_3_v.push_back("y: \"" + y + "\"");
   mht_3_v.push_back("z: \"" + z + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_3(mht_3_v, 388, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "lambda");

    std::string id;
    if (src_def.HasAxis(Axis::WIDTH)) {
      id += "_w" + x;
    }
    if (src_def.HasAxis(Axis::HEIGHT)) {
      id += "_h" + y;
    }
    if (src_def.HasAxis(Axis::DEPTH)) {
      id += "_d" + z;
    }
    return id;
  };

  auto generate_id_full = [&](const std::string& x, const std::string& y,
                              const std::string& z, const std::string& s) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("x: \"" + x + "\"");
   mht_4_v.push_back("y: \"" + y + "\"");
   mht_4_v.push_back("z: \"" + z + "\"");
   mht_4_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_4(mht_4_v, 410, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "lambda");

    return generate_id(x, y, z) + "_s" + s;
  };

  auto generate_check = [&](const std::string& x, const std::string& y,
                            const std::string& z) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("x: \"" + x + "\"");
   mht_5_v.push_back("y: \"" + y + "\"");
   mht_5_v.push_back("z: \"" + z + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_5(mht_5_v, 421, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "lambda");

    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"in_x", "in_y", "in_z"};
    const std::vector<std::string> coords{x, y, z};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_def.HasAxis(axis) && !src_def.SupportsZeroClamp(axis) &&
          block_size[i] != 1) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i] + coords[i];
      }
    }
    return check;
  };

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
      c += "#define FLT16 float16\n";
      break;
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      c += "#define FLT16 half16\n";
      break;
  }

  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int dst_x = (linear_id / args.dst_tensor.Batch());\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int dst_x = GLOBAL_ID_0;\n";
  }
  c += "  int rem_x = dst_x % args.stride_x;\n";
  c += "  int ceil_x = dst_x / args.stride_x;\n";
  c += "  dst_x = ceil_x * args.stride_x * " + std::to_string(block_size.x) +
       " + rem_x;\n";
  if (src_def.HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_y = GLOBAL_ID_1;\n";
    c += "  int dst_y = linear_id_y % args.grid_size_y;\n";
    c += "  int dst_z = linear_id_y / args.grid_size_y;\n";
    c += "  int rem_z = dst_z % args.stride_z;\n";
    c += "  int ceil_z = dst_z / args.stride_z;\n";
    c += "  dst_z = ceil_z * args.stride_z * " + std::to_string(block_size.z) +
         " + rem_z;\n";
    c += "  if (dst_z >= args.dst_tensor.Depth()) return;\n";
  } else {
    c += "  int dst_y = GLOBAL_ID_1;\n";
  }
  c += "  int rem_y = dst_y % args.stride_y;\n";
  c += "  int ceil_y = dst_y / args.stride_y;\n";
  c += "  dst_y = ceil_y * args.stride_y * " + std::to_string(block_size.y) +
       " + rem_y;\n";
  c += "  int dst_s = GLOBAL_ID_2 * " + std::to_string(block_size.w) + ";\n";
  c += "  if (dst_x >= args.dst_tensor.Width() || dst_y >= "
       "args.dst_tensor.Height() || dst_s >= "
       "args.dst_tensor.Slices()) return;\n";
  if (weights_are_buffer) {
    c += "  int f_base = dst_s * args.src_tensor.Slices() * args.kernel_size_x "
         "* args.kernel_size_y";
    if (src_def.HasAxis(Axis::DEPTH)) {
      c += " * args.kernel_size_z";
    }
    c += ";\n";
  }
  for (int s = 0; s < block_size.w; ++s) {
    const std::string sind = std::to_string(s);
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          c += "  ACCUM_FLT4 r" + generate_id_full(xind, yind, zind, sind) +
               " = INIT_ACCUM_FLT4(0.0f);\n";
        }
      }
    }
  }
  c += "  int kernel_first_dst_x = dst_x + args.padding_x;\n";
  c += "  int kernel_first_dst_y = dst_y + args.padding_y;\n";
  c += "  int kernel_last_dst_x = kernel_first_dst_x - args.kernel_size_x;\n";
  c += "  int kernel_last_dst_y = kernel_first_dst_y - args.kernel_size_y;\n";
  c += "  int offset_x = abs(args.padding_x);\n";
  c += "  int offset_x_strided = offset_x * args.stride_x;\n";
  c +=
      "  int src_x = (kernel_first_dst_x + offset_x_strided) / args.stride_x - "
      "offset_x;\n";
  c += "  int offset_y = abs(args.padding_y);\n";
  c += "  int offset_y_strided = offset_y * args.stride_y;\n";
  c +=
      "  int src_y = (kernel_first_dst_y + offset_y_strided) / args.stride_y - "
      "offset_y;\n";
  if (src_def.HasAxis(Axis::DEPTH)) {
    c += "  int kernel_first_dst_z = dst_z + args.padding_z;\n";
    c += "  int kernel_last_dst_z = kernel_first_dst_z - args.kernel_size_z;\n";
    c += "  int offset_z = abs(args.padding_z);\n";
    c += "  int offset_z_strided = offset_z * args.stride_z;\n";
    c += "  int src_z = (kernel_first_dst_z + offset_z_strided) / "
         "args.stride_z - offset_z;\n";
    c += "  int src_as_dst_z = src_z * args.stride_z;\n";
    c +=
        "  for (;src_as_dst_z > kernel_last_dst_z; src_z -= 1, src_as_dst_z -= "
        "args.stride_z) {\n";
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zindex = std::to_string(z);
      c += "    int sz" + zindex + " = src_z + " + zindex + ";\n";
      if (!src_def.SupportsZeroClamp(Axis::DEPTH)) {
        c += "    bool in_z" + zindex + " = sz" + zindex + " >= 0 && sz" +
             zindex + " < args.src_tensor.Depth();\n";
        if (!src_def.CanReadOutOfBorder(Axis::DEPTH)) {
          c += "    sz" + zindex + " = clamp(sz" + zindex +
               ", 0, args.src_tensor.Depth() - 1);\n";
        }
      }
    }
    if (block_size.z == 1 && !src_def.SupportsZeroClamp(Axis::DEPTH)) {
      c += "    if (!in_z0) continue;\n";
    }
    c += "    int kernel_z = kernel_first_dst_z - src_as_dst_z;\n";
    c += "    int src_as_dst_y = src_y * args.stride_y;\n";
    c += "    int src_y_copy = src_y;\n";
    c += "    for (;src_as_dst_y > kernel_last_dst_y; src_y_copy -= 1, "
         "src_as_dst_y -= args.stride_y) {\n";
  } else {
    c += "  int src_as_dst_y = src_y * args.stride_y;\n";
    c += "  for (;src_as_dst_y > kernel_last_dst_y; src_y -= 1, src_as_dst_y "
         "-= args.stride_y) {\n";
  }
  for (int y = 0; y < block_size.y; ++y) {
    const std::string yindex = std::to_string(y);
    const std::string src_y =
        src_def.HasAxis(Axis::DEPTH) ? "src_y_copy" : "src_y";
    c += "    int sy" + yindex + " = " + src_y + " + " + yindex + ";\n";
    if (!src_def.SupportsZeroClamp(Axis::HEIGHT)) {
      c += "    bool in_y" + yindex + " = sy" + yindex + " >= 0 && sy" +
           yindex + " < args.src_tensor.Height();\n";
      if (!src_def.CanReadOutOfBorder(Axis::HEIGHT)) {
        c += "    sy" + yindex + " = clamp(sy" + yindex +
             ", 0, args.src_tensor.Height() - 1);\n";
      }
    }
  }
  if (block_size.y == 1 && !src_def.SupportsZeroClamp(Axis::HEIGHT)) {
    c += "      if (!in_y0) continue;\n";
  }
  c += "    int kernel_y = kernel_first_dst_y - src_as_dst_y;\n";
  c += "    int src_as_dst_x = src_x * args.stride_x;\n";
  c += "    int src_x_copy = src_x;\n";
  c += "    for (;src_as_dst_x > kernel_last_dst_x; src_x_copy -= 1, "
       "src_as_dst_x "
       "-= args.stride_x) {\n";
  for (int x = 0; x < block_size.x; ++x) {
    const std::string xindex = std::to_string(x);
    c += "      int sx" + xindex + " = src_x_copy + " + xindex + ";\n";
    if (!src_def.SupportsZeroClamp(Axis::WIDTH)) {
      c += "      bool in_x" + xindex + " = sx" + xindex + " >= 0 && sx" +
           xindex + " < args.src_tensor.Width();\n";
      if (!src_def.CanReadOutOfBorder(Axis::WIDTH)) {
        c += "      sx" + xindex + " = clamp(sx" + xindex +
             ", 0, args.src_tensor.Width() - 1);\n";
      }
    }
  }
  if (block_size.x == 1 && !src_def.SupportsZeroClamp(Axis::WIDTH)) {
    c += "      if (!in_x0) continue;\n";
  }
  for (int z = 0; z < block_size.z; ++z) {
    const std::string zind = std::to_string(z);
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yind = std::to_string(y);
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xind = std::to_string(x);
        const std::string id = generate_id(xind, yind, zind);
        const std::string check = generate_check(xind, yind, zind);
        std::string coords = "sx" + xind + ", sy" + yind;
        if (src_def.HasAxis(Axis::DEPTH)) {
          coords += ", sz" + zind;
        }
        if (src_def.IsLinear()) {
          c += "      args.src_tensor.GetAddress(addr" + id + ", " + coords +
               ", 0);\n";
        }
        if (src_def.ReturnsZeroForNegOneRead()) {
          c += "      addr" + id + " = select(-1, addr" + id + ", (" + check +
               "));\n";
          c += "      int ds" + id +
               " = select(0, args.src_tensor.SliceStride(), (" + check +
               "));\n";
        }
      }
    }
  }
  if (src_def.storage_type == TensorStorageType::BUFFER) {
    c += "      int ds = args.src_tensor.SliceStride();\n";
  }
  c += "      int kernel_x = kernel_first_dst_x - src_as_dst_x;\n";
  if (src_def.HasAxis(Axis::DEPTH)) {
    c += "      int kernel_index = (kernel_z * args.kernel_size_y + kernel_y) "
         "*  args.kernel_size_x + kernel_x;\n";
  } else {
    c += "      int kernel_index = kernel_y * args.kernel_size_x + kernel_x;\n";
  }
  if (weights_are_buffer) {
    c += "      int f_offset = f_base + kernel_index * "
         "args.src_tensor.Slices() * " +
         std::to_string(block_size.w) + ";\n";
  } else {
    c += "      int x_c = kernel_index * args.src_tensor.Slices();\n";
  }
  c += "      for (int s = 0; s < args.src_tensor.Slices(); ++s) {\n";
  const bool conditional_read = gpu_info.IsMali();
  for (int z = 0; z < block_size.z; ++z) {
    const std::string zind = std::to_string(z);
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yind = std::to_string(y);
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xind = std::to_string(x);
        const std::string id = generate_id(xind, yind, zind);
        std::string address;
        if (src_def.IsLinear()) {
          address = "addr" + id;
        } else {
          address = "sx" + xind + ", sy" + yind;
          if (src_def.HasAxis(Axis::DEPTH)) {
            address += ", sz" + zind;
          }
          address += ", s";
        }
        if (src_def.ReturnsZeroForNegOneRead()) {
          c += "        FLT4 src" + id + " = args.src_tensor.Read(" + address +
               "); " + address + " += ds" + id + ";\n";
        } else {
          const std::string check = generate_check(xind, yind, zind);
          if (!check.empty()) {
            if (conditional_read) {
              c += "        FLT4 src" + id + " = " + check +
                   " ? args.src_tensor.Read(" + address +
                   ") : INIT_FLT4(0.0f);\n";
            } else {
              c += "        FLT4 src" + id + " = args.src_tensor.Read(" +
                   address + ") * INIT_FLT(" + check + ");\n";
            }
          } else {
            c += "        FLT4 src" + id + " = args.src_tensor.Read(" +
                 address + ");\n";
          }
          if (src_def.IsLinear()) {
            c += "        addr" + id + " += ds;\n";
          }
        }
      }
    }
  }
  if (weights_are_buffer) {
    if (gpu_info.SupportsPointersInKernels()) {
      c += "        __global FLT16* weights_cache = "
           "args.weights.GetPtr(f_offset);\n";
    }
  } else {
    for (int s = 0; s < block_size.w; ++s) {
      c += absl::Substitute(
          R"(        FLT4 f$1 = args.weights0.Read(dst_s + $0, x_c);
        FLT4 f$2 = args.weights1.Read(dst_s + $0, x_c);
        FLT4 f$3 = args.weights2.Read(dst_s + $0, x_c);
        FLT4 f$4 = args.weights3.Read(dst_s + $0, x_c);
)",
          s, s * 4 + 0, s * 4 + 1, s * 4 + 2, s * 4 + 3);
    }
    c += "        x_c++;\n";
  }
  if (weights_are_buffer && !gpu_info.SupportsPointersInKernels()) {
    c += "      FLT16 flt16val;\n";
  }
  for (int s = 0; s < block_size.w; ++s) {
    if (weights_are_buffer && !gpu_info.SupportsPointersInKernels()) {
      c += "        flt16val = args.weights.Read(f_offset + " +
           std::to_string(s) + ");\n";
    }
    const std::string sind = std::to_string(s);
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          const std::string id = generate_id(xind, yind, zind);
          const std::string full_id = generate_id_full(xind, yind, zind, sind);
          c += "        CONV" + sind + "(r" + full_id + ", src" + id + ");\n";
        }
      }
    }
  }
  if (weights_are_buffer) {
    c += "        f_offset += " + std::to_string(block_size.w) + ";\n";
  }
  c += "      }\n";
  c += "    }\n";
  c += "  }\n";
  if (src_def.HasAxis(Axis::DEPTH)) {
    c += "  }\n";
  }
  for (int s = 0; s < block_size.w; ++s) {
    const std::string sind = std::to_string(s);
    c += "  if (dst_s < args.dst_tensor.Slices()) {\n";
    c += "    FLT4 bias_val = args.biases.Read(dst_s);\n";
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          const std::string id = generate_id_full(xind, yind, zind, sind);
          std::string checks =
              "xc < args.dst_tensor.Width() && yc < args.dst_tensor.Height()";
          std::string coords = "xc, yc";
          c += "    {\n";
          c += "      int xc = dst_x + args.stride_x * " + xind + ";\n";
          c += "      int yc = dst_y + args.stride_y * " + yind + ";\n";
          if (src_def.HasAxis(Axis::DEPTH)) {
            c += "      int zc = dst_z + args.stride_z * " + zind + ";\n";
            checks += " && zc < args.dst_tensor.Depth()";
            coords += ", zc";
          }
          c += "      if (" + checks + ") {\n";
          c += "        FLT4 res = TO_FLT4(r" + id + ") + bias_val;\n";
          c += "        args.dst_tensor.Write(res, " + coords + ", dst_s);\n";
          c += "      }\n";
          c += "    }\n";
        }
      }
    }
    c += "  }\n";
    c += "  dst_s++;\n";
  }
  c += "}\n";
  return c;
}

absl::Status ConvolutionTransposed::BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_6(mht_6_v, 768, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "ConvolutionTransposed::BindArguments");

  if (definition_.src_tensors[0].HasAxis(Axis::DEPTH)) {
    const int aligned_h =
        AlignByN(dst_[0]->Height(), stride_.y * block_size_.y);
    RETURN_IF_ERROR(
        args->SetInt("grid_size_y", DivideRoundUp(aligned_h, block_size_.y)));
  }
  return absl::OkStatus();
}

int3 ConvolutionTransposed::GetGridSize() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_7(mht_7_v, 781, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "ConvolutionTransposed::GetGridSize");

  const int aligned_w = AlignByN(dst_[0]->Width(), stride_.x * block_size_.x);
  const int aligned_h = AlignByN(dst_[0]->Height(), stride_.y * block_size_.y);
  const int aligned_d = AlignByN(dst_[0]->Depth(), stride_.z * block_size_.z);
  const int grid_x = DivideRoundUp(aligned_w, block_size_.x) * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(aligned_h, block_size_.y) *
                     DivideRoundUp(aligned_d, block_size_.z);
  const int grid_z = DivideRoundUp(dst_[0]->Slices(), block_size_.w);
  return int3(grid_x, grid_y, grid_z);
}

void ConvolutionTransposed::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_8(mht_8_v, 797, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "ConvolutionTransposed::GetPossibleKernelWorkGroups");

  GetPossibleWorkGroupsConv(tuning_type, gpu_info, kernel_info, grid_size_,
                            work_groups);
}

ConvolutionTransposed CreateConvolutionTransposed(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_9(mht_9_v, 807, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "CreateConvolutionTransposed");

  const bool weights_are_buffer = gpu_info.IsMali() || gpu_info.IsApple();
  ConvolutionTransposed result(definition, attr, gpu_info, weights_are_buffer);
  result.UploadWeights(attr.weights, weights_are_buffer);

  TensorLinearDescriptor desc;
  desc.storage_type =
      DeduceLinearStorageType(definition.GetPrimaryStorageType());
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  result.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return result;
}

ConvolutionTransposed CreateConvolutionTransposed3D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposed3DAttributes& attr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_10(mht_10_v, 827, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "CreateConvolutionTransposed3D");

  const bool weights_are_buffer = gpu_info.IsMali() || gpu_info.IsApple();
  ConvolutionTransposed result(definition, attr, gpu_info, weights_are_buffer);
  result.UploadWeights(attr.weights, weights_are_buffer);

  TensorLinearDescriptor desc;
  desc.storage_type =
      DeduceLinearStorageType(definition.GetPrimaryStorageType());
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  result.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return result;
}

ConvolutionTransposed CreateConvolutionTransposedDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTcc mht_11(mht_11_v, 847, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.cc", "CreateConvolutionTransposedDynamicWeights");

  const bool weights_are_buffer = gpu_info.IsMali();
  OperationDef new_def = definition;
  new_def.src_tensors = {
      definition.src_tensors[0]};  // leaving only src_tensor def, weights defs
                                   // will be added later
  const DataType weights_type = definition.GetDataType();
  if (weights_are_buffer) {
    // add 1 src_tensor(buffer) for weights
    new_def.src_tensors.push_back(
        {weights_type, TensorStorageType::BUFFER, Layout::HWC});
  } else {
    // add 4 src_tensors(4X textures 2d) for weights
    new_def.src_tensors.push_back(
        {weights_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
    new_def.src_tensors.push_back(
        {weights_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
    new_def.src_tensors.push_back(
        {weights_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
    new_def.src_tensors.push_back(
        {weights_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  }
  ConvolutionTransposed result(new_def, attr, gpu_info, weights_are_buffer);

  TensorLinearDescriptor desc;
  desc.storage_type = DeduceLinearStorageType(new_def.GetPrimaryStorageType());
  desc.element_type = new_def.GetDataType();
  desc.UploadLinearData(attr.bias);
  result.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return result;
}

}  // namespace gpu
}  // namespace tflite
