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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/conv_constants.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
// Adreno can provide up to ~3-4KB of constant memory, but in some cases even
// 3KB can have very bad performance.
int GetAdrenoOptimalMaxConstantSize(const AdrenoInfo& adreno_info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_constants.cc", "GetAdrenoOptimalMaxConstantSize");

  if (adreno_info.IsAdreno3xx() || adreno_info.IsAdreno4xx() ||
      adreno_info.IsAdreno5xx()) {
    return 256 * 10;  // 2.5KB
  } else {
    return 256 * 14;  // 3.5KB
  }
}

int GetOptimalMaxConstantSize(const GpuInfo& info) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_constants.cc", "GetOptimalMaxConstantSize");

  if (!info.IsAdreno()) {
    // In general we do not expect that this kernel will be used with non Adreno
    // so as it tuned for __constant memory that have big profit on Adreno
    return 1024;  // 1KB
  } else {
    return GetAdrenoOptimalMaxConstantSize(info.adreno_info);
  }
}

// src_size and dst_size must be <= 4;
std::string GenerateConv(int src_size, int dst_size, bool use_dot_conv,
                         int const_mem_offset, CalculationsPrecision precision,
                         const std::string& dst, const std::string& src) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("dst: \"" + dst + "\"");
   mht_2_v.push_back("src: \"" + src + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc mht_2(mht_2_v, 232, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_constants.cc", "GenerateConv");

  std::string result;
  const std::string postfixes[] = {".x", ".y", ".z", ".w"};
  if (use_dot_conv) {
    const std::string src_postfixes[] = {".x", ".xy", ".xyz", ""};
    const std::string src_postfix = src_postfixes[src_size - 1];
    for (int i = 0; i < dst_size; ++i) {
      result += "    " + dst + postfixes[i] + " += dot(" + src +
                ", constants[" + std::to_string(const_mem_offset + i) + "]" +
                src_postfix + ");\n";
    }
  } else {
    const std::string dst_postfixes[] = {".x", ".xy", ".xyz", ""};
    const std::string dst_postfix = dst_postfixes[dst_size - 1];
    if (precision == CalculationsPrecision::F32_F16) {
      for (int i = 0; i < src_size; ++i) {
        if (i != 0) {
          result += " + ";
        }
        std::string src_name = src;
        if (src_size != 1) {
          src_name += postfixes[i];
        }
        result += src_name + " * constants[" +
                  std::to_string(const_mem_offset + i) + "]" + dst_postfix;
      }
      std::string size = dst_size == 1 ? "" : std::to_string(dst_size);
      result = "    " + dst + dst_postfix + " += TO_ACCUM_FLT" + size + "(" +
               result + ");\n";
    } else {
      for (int i = 0; i < src_size; ++i) {
        std::string src_name = src;
        if (src_size != 1) {
          src_name += postfixes[i];
        }
        result += "    " + dst + dst_postfix + " += " + src_name +
                  " * constants[" + std::to_string(const_mem_offset + i) + "]" +
                  dst_postfix + ";\n";
      }
    }
  }
  return result;
}

std::string GenerateConvolutionConstantCode(const OperationDef& op_def,
                                            const OHWI& weights_shape,
                                            bool stride_correction,
                                            bool use_dot_conv,
                                            GPUOperation* op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc mht_3(mht_3_v, 283, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_constants.cc", "GenerateConvolutionConstantCode");

  auto src_desc = op_def.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddSrcTensor("src_tensor", src_desc);

  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddDstTensor("dst_tensor", dst_desc);

  const int out_z = DivideRoundUp(weights_shape.o, 4);
  const std::string kOutZ = std::to_string(out_z);
  const int src_depth = DivideRoundUp(weights_shape.i, 4);

  const std::string postfixes[] = {".x", ".xy", ".xyz", ""};

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return;\n";
  if (stride_correction) {
    c += "  int start_x = " +
         GetXStrideCorrectedV2("X", "args.src_tensor.Batch()", "args.stride_x",
                               "args.padding_x") +
         ";\n";
  } else {
    if (op_def.IsBatchSupported()) {
      c += "  int start_x = X * args.stride_x + args.padding_x * "
           "args.src_tensor.Batch();\n";
    } else {
      c += "  int start_x = X * args.stride_x + args.padding_x;\n";
    }
  }
  c += "  int start_y = Y * args.stride_y + args.padding_y;\n";
  c += "  __constant FLT4* constants = args.weights.GetPtr();\n";
  for (int i = 0; i < out_z; ++i) {
    c += "  ACCUM_FLT4 r" + std::to_string(i) + " = INIT_ACCUM_FLT4(0.0f);\n";
  }
  auto generate_check = [&]() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc mht_4(mht_4_v, 330, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_constants.cc", "lambda");

    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"x_out", "y_out", "z_out"};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis) && !src_desc.SupportsZeroClamp(axis)) {
        if (!check.empty()) {
          check += " || ";
        }
        check += names[i];
      }
    }
    return check;
  };
  const std::string check = generate_check();
  int filters_counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    const int src_ch_count = std::min(4, weights_shape.i - s * 4);
    const std::string s_count =
        src_ch_count == 1 ? "" : std::to_string(src_ch_count);
    const std::string s_type = absl::StrCat("FLT", s_count);
    const std::string s_postfix = postfixes[src_ch_count - 1];
    const std::string dilation_x =
        op_def.IsBatchSupported() ? "args.dilation_x * args.src_tensor.Batch()"
                                  : "args.dilation_x";
    for (int ky = 0; ky < weights_shape.h; ++ky) {
      std::string s_y = absl::StrCat("(start_y + ", ky, " * args.dilation_y)");
      if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
        c += "  {\n";
        c += "  bool y_out = " + s_y + " < 0 || " + s_y +
             " >= args.src_tensor.Height();\n";
      }
      for (int kx = 0; kx < weights_shape.w; ++kx) {
        c += "  {\n";
        std::string s_x =
            absl::StrCat("(start_x + ", kx, " * " + dilation_x + ")");
        if (!src_desc.SupportsZeroClamp(Axis::WIDTH)) {
          c += "    bool x_out = " + s_x + " < 0 || " + s_x +
               ">= args.src_tensor.Width();\n";
        }
        if (check.empty()) {
          c += "    " + s_type + " src = args.src_tensor.Read(" + s_x + ", " +
               s_y + ", " + std::to_string(s) + ")" + s_postfix + ";\n";
        } else {
          c += "    FLT4 zero_vec = INIT_FLT4(0.0);\n";
          c += "    " + s_type + " src = x_out || y_out ? ";
          c += "zero_vec" + s_postfix + " : args.src_tensor.Read(" + s_x +
               ", " + s_y + ", " + std::to_string(s) + ")" + s_postfix + ";\n";
        }
        for (int d = 0; d < out_z; ++d) {
          const int dst_ch_count = std::min(4, weights_shape.o - d * 4);
          c += GenerateConv(src_ch_count, dst_ch_count, use_dot_conv,
                            filters_counter, op_def.precision,
                            "r" + std::to_string(d), "src");
          filters_counter += use_dot_conv ? dst_ch_count : src_ch_count;
        }
        c += "  }\n";
      }
      if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
        c += "  }\n";
      }
    }
  }
  for (int i = 0; i < out_z; ++i) {
    std::string s_i = std::to_string(i);
    c += "  {\n";
    c += "    FLT4 res = TO_FLT4(r" + s_i + ") + args.biases.Read(" + s_i +
         ");\n";
    c += "    args.dst_tensor.Write(res, X, Y, " + s_i + ");\n";
    c += "  }\n";
  }
  c += "}\n";
  return c;
}

bool IsDotConvBetter(int src_channels, int dst_channels) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc mht_5(mht_5_v, 409, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_constants.cc", "IsDotConvBetter");

  if (dst_channels % 4 == 0) {
    return false;
  }

  // dst_channels % 4 != 0
  if (src_channels % 4 == 0) {
    return true;
  }

  // dst_channels % 4 != 0 && src_channels % 4 != 0
  const int src_depth = DivideRoundUp(src_channels, 4);
  const int dst_depth = DivideRoundUp(dst_channels, 4);
  return dst_channels * src_depth < src_channels * dst_depth;
}

}  // namespace

bool IsConvConstantsSupported(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc mht_6(mht_6_v, 432, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_constants.cc", "IsConvConstantsSupported");

  if (gpu_info.IsAMD() && definition.precision != CalculationsPrecision::F32 &&
      definition.src_tensors[0].storage_type != TensorStorageType::BUFFER) {
    // BUG, some AMD GPUs crash without it
    return false;
  }

  if (gpu_info.IsApiOpenCl() && gpu_info.IsAdreno()) {
    const std::string kBadDriver =
        "OpenCL 2.0 QUALCOMM build: commit #7ff4f54 changeid #I4460aa6217 "
        "Date: 12/30/18";
    if (absl::StrContains(gpu_info.opencl_info.platform_version, kBadDriver)) {
      return false;
    }
  }

  if (attr.groups != 1) {
    return false;
  }

  const bool use_dot_conv =
      IsDotConvBetter(attr.weights.shape.i, attr.weights.shape.o);
  const auto& w_shape = attr.weights.shape;
  const int src_depth = DivideRoundUp(w_shape.i, 4);
  const int dst_depth = DivideRoundUp(w_shape.o, 4);
  const int aligned_ch_count =
      use_dot_conv ? w_shape.o * src_depth * 4 : w_shape.i * dst_depth * 4;
  const int filters_count = aligned_ch_count * w_shape.h * w_shape.w;
  const int float_size = definition.precision == CalculationsPrecision::F32
                             ? sizeof(float)
                             : sizeof(half);
  const int filters_buffer_size = filters_count * float_size;
  const int kConstantMaxSize = GetOptimalMaxConstantSize(gpu_info);
  const int flt4_registers = DivideRoundUp(w_shape.o, 4);
  return filters_buffer_size <= kConstantMaxSize && flt4_registers <= 8;
}

GPUOperation CreateConvConstants(const GpuInfo& gpu_info,
                                 const OperationDef& definition,
                                 const Convolution2DAttributes& attr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTcc mht_7(mht_7_v, 474, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_constants.cc", "CreateConvConstants");

  const bool use_dot_conv =
      IsDotConvBetter(attr.weights.shape.i, attr.weights.shape.o);
  GPUOperation op(definition);
  UploadWeightsForConvConstants(attr.weights, definition.precision,
                                use_dot_conv, &op);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;

  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;

  op.code_ = GenerateConvolutionConstantCode(
      definition, attr.weights.shape, stride_correction, use_dot_conv, &op);
  if (definition.precision == CalculationsPrecision::F16 &&
      gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx()) {
    op.compiler_options_.push_back(CompilerOptions::kAdrenoFullSimd);
  }
  if (definition.precision != CalculationsPrecision::F32 &&
      gpu_info.IsPowerVR()) {
    // BUG, some PowerVRs (GE8320) produce incorrect result without it
    op.compiler_options_.push_back(CompilerOptions::kClDisableOptimizations);
  }

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::BUFFER;
  desc.element_type = definition.GetDataType();
  desc.memory_type = MemoryType::CONSTANT;
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

}  // namespace gpu
}  // namespace tflite
