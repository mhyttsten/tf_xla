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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/winograd.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

namespace tflite {
namespace gpu {
namespace {
void VectorToKernelBufferDesc(const std::vector<float>& data,
                              DataType data_type,
                              BufferDescriptor* buffer_desc) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "VectorToKernelBufferDesc");

  buffer_desc->element_type = data_type;
  buffer_desc->element_size = 1;
  buffer_desc->memory_type = MemoryType::CONSTANT;
  buffer_desc->attributes.push_back("kernel_global_space");
  buffer_desc->size = SizeOf(data_type) * data.size();
  buffer_desc->data.resize(buffer_desc->size);
  if (data_type == DataType::FLOAT32) {
    memcpy(buffer_desc->data.data(), data.data(), buffer_desc->size);
  } else {
    half* hf_ptr = reinterpret_cast<half*>(buffer_desc->data.data());
    for (int i = 0; i < data.size(); ++i) {
      hf_ptr[i] = data[i];
    }
  }
}
std::string GetKernelWinograd4x4To36(const OperationDef& op_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_1(mht_1_v, 223, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "GetKernelWinograd4x4To36");

  std::string c;
  const auto src_desc = op_def.src_tensors[0];
  c += R"(
MAIN_FUNCTION($0) {
  int X = GLOBAL_ID_0 * 4;
  int Y = GLOBAL_ID_1 * 4;
  int S = GLOBAL_ID_2;

  if (GLOBAL_ID_0 >= args.tiles_x || GLOBAL_ID_1 >= args.tiles_y) return;

  FLT4 I[6][6];
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      I[y][x] = INIT_FLT4(0.0f);
    }
  }
)";
  if (src_desc.IsLinear()) {
    c += "  args.src_tensor.GetAddress(src_base, 0, 0, S);\n";
  }
  for (int y = 0; y < 6; ++y) {
    const std::string s_y = std::to_string(y);
    c += "  {\n";
    c += "    int coord_y = Y + " + s_y + " + args.padding_y;\n";
    if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
      c += "    bool in_y = coord_y >= 0 && coord_y < "
           "args.src_tensor.Height();\n";
      c += "    coord_y = clamp(coord_y, 0, args.src_tensor.Height() - 1);\n";
    }
    if (src_desc.IsLinear()) {
      c += "    int src_adress_y = src_base + coord_y * "
           "args.src_tensor.Width();\n";
    }
    for (int x = 0; x < 6; ++x) {
      const std::string s_x = std::to_string(x);
      c += "    {\n";
      c += "      int coord_x = X + " + s_x + " + args.padding_x;\n";
      if (!src_desc.SupportsZeroClamp(Axis::WIDTH)) {
        c += "      bool in_x = coord_x >= 0 && coord_x < "
             "args.src_tensor.Width();\n";
        c += "      coord_x = clamp(coord_x, 0, args.src_tensor.Width()-1);\n";
      }
      std::string multiplier;
      if (!src_desc.SupportsZeroClamp(Axis::WIDTH) &&
          !src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
        multiplier = " * INIT_FLT(in_y && in_x)";
      } else if (!src_desc.SupportsZeroClamp(Axis::WIDTH)) {
        multiplier = " * INIT_FLT(in_x)";
      } else if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
        multiplier = " * INIT_FLT(in_y)";
      }
      if (src_desc.IsLinear()) {
        c += "      FLT4 src = args.src_tensor.Read(src_adress_y + coord_x)" +
             multiplier + ";\n";
      } else {
        c += "      FLT4 src = args.src_tensor.Read(coord_x, coord_y, S)" +
             multiplier + ";\n";
      }
      c += "      I[0][" + s_x + "] += args.Bt.Read(" + std::to_string(y) +
           ") * src;\n";
      c += "      I[1][" + s_x + "] += args.Bt.Read(" + std::to_string(y + 6) +
           ") * src;\n";
      c += "      I[2][" + s_x + "] += args.Bt.Read(" + std::to_string(y + 12) +
           ") * src;\n";
      c += "      I[3][" + s_x + "] += args.Bt.Read(" + std::to_string(y + 18) +
           ") * src;\n";
      c += "      I[4][" + s_x + "] += args.Bt.Read(" + std::to_string(y + 24) +
           ") * src;\n";
      c += "      I[5][" + s_x + "] += args.Bt.Read(" + std::to_string(y + 30) +
           ") * src;\n";
      c += "    }\n";
    }
    c += "  }\n";
  }

  const auto dst_desc = op_def.dst_tensors[0];

  if (dst_desc.IsLinear()) {
    c += R"(
  int dst_x = GLOBAL_ID_1 * args.tiles_x + GLOBAL_ID_0;
  args.dst_tensor.GetAddress(dst_adress, dst_x, 0, S);
  for (int y = 0; y < 6; ++y) {
    FLT4 value = I[y][0] + args.Bt.Read(2) * I[y][2] + args.Bt.Read(4) * I[y][4];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
    value = args.Bt.Read(7) * I[y][1] + args.Bt.Read(8) * I[y][2] + args.Bt.Read(9) * I[y][3] + args.Bt.Read(10) * I[y][4];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
    value = args.Bt.Read(13) * I[y][1] + args.Bt.Read(14) * I[y][2] + args.Bt.Read(15) * I[y][3] + args.Bt.Read(16) * I[y][4];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
    value = args.Bt.Read(19) * I[y][1] + args.Bt.Read(20) * I[y][2] + args.Bt.Read(21) * I[y][3] + args.Bt.Read(22) * I[y][4];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
    value = args.Bt.Read(25) * I[y][1] + args.Bt.Read(26) * I[y][2] + args.Bt.Read(27) * I[y][3] + args.Bt.Read(28) * I[y][4];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
    value = args.Bt.Read(31) * I[y][1] + args.Bt.Read(33) * I[y][3] + I[y][5];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
  }
}
)";
  } else {
    c += R"(
  int dst_x = GLOBAL_ID_1 * args.tiles_x + GLOBAL_ID_0;
  for (int y = 0; y < 6; ++y) {
    FLT4 value = I[y][0] + args.Bt.Read(2) * I[y][2] + args.Bt.Read(4) * I[y][4];
    args.dst_tensor.Write(value, dst_x, y * 6 + 0, S);
    value = args.Bt.Read(7) * I[y][1] + args.Bt.Read(8) * I[y][2] + args.Bt.Read(9) * I[y][3] + args.Bt.Read(10) * I[y][4];
    args.dst_tensor.Write(value, dst_x, y * 6 + 1, S);
    value = args.Bt.Read(13) * I[y][1] + args.Bt.Read(14) * I[y][2] + args.Bt.Read(15) * I[y][3] + args.Bt.Read(16) * I[y][4];
    args.dst_tensor.Write(value, dst_x, y * 6 + 2, S);
    value = args.Bt.Read(19) * I[y][1] + args.Bt.Read(20) * I[y][2] + args.Bt.Read(21) * I[y][3] + args.Bt.Read(22) * I[y][4];
    args.dst_tensor.Write(value, dst_x, y * 6 + 3, S);
    value = args.Bt.Read(25) * I[y][1] + args.Bt.Read(26) * I[y][2] + args.Bt.Read(27) * I[y][3] + args.Bt.Read(28) * I[y][4];
    args.dst_tensor.Write(value, dst_x, y * 6 + 4, S);
    value = args.Bt.Read(31) * I[y][1] + args.Bt.Read(33) * I[y][3] + I[y][5];
    args.dst_tensor.Write(value, dst_x, y * 6 + 5, S);
  }
}
)";
  }
  return c;
}

std::string GetKernelWinograd36To4x4(const OperationDef& op_def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_2(mht_2_v, 353, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "GetKernelWinograd36To4x4");

  std::string c;
  const auto src_desc = op_def.src_tensors[0];

  c += R"(
MAIN_FUNCTION($0) {
  int tile_id = GLOBAL_ID_0;
  int Z = GLOBAL_ID_2;
  int tiles_count_x = (args.dst_tensor.Width() + 3) / 4;
  int tile_x = (tile_id % tiles_count_x) * 4;
  int tile_y = (tile_id / tiles_count_x) * 4;
  if (tile_x >= args.dst_tensor.Width() || tile_y >= args.dst_tensor.Height()) return;

  FLT4 I[4][6];
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 6; ++x) {
      I[y][x] = INIT_FLT4(0.0f);
    }
  }
)";
  if (src_desc.IsLinear()) {
    c += R"(
  args.src_tensor.GetAddress(src_adress, tile_id, 0, Z);
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x, src_adress += args.src_tensor.Width()) {
      FLT4 src = args.src_tensor.Read(src_adress);
      I[0][x] += src * args.At.Read(y);
      I[1][x] += src * args.At.Read(y + 6);
      I[2][x] += src * args.At.Read(y + 12);
      I[3][x] += src * args.At.Read(y + 18);
    }
  }
)";
  } else {
    c += R"(
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      FLT4 src = args.src_tensor.Read(tile_id, y * 6 + x, Z);
      I[0][x] += src * args.At.Read(y);
      I[1][x] += src * args.At.Read(y + 6);
      I[2][x] += src * args.At.Read(y + 12);
      I[3][x] += src * args.At.Read(y + 18);
    }
  }
)";
  }
  c += R"(

  FLT4 bias_val = args.biases.Read(Z);
  for (int y = 0; y < 4; ++y) {
    FLT4 t0 = I[y][1] + I[y][2];
    FLT4 t1 = I[y][3] + I[y][4];
    if (tile_x < args.dst_tensor.Width() && tile_y + y < args.dst_tensor.Height()) {
      FLT4 value = I[y][0] + t0 + t1 + bias_val;
      args.dst_tensor.Write(value, tile_x, tile_y + y, Z);
    }
    FLT4 t2 = I[y][1] - I[y][2];
    FLT4 t3 = I[y][3] - I[y][4];
    if (tile_x + 1 < args.dst_tensor.Width() && tile_y + y < args.dst_tensor.Height()) {
      FLT4 value = t2 * args.At.Read(7) + t3 * args.At.Read(9) + bias_val;
      args.dst_tensor.Write(value, tile_x + 1, tile_y + y, Z);
    }
    if (tile_x + 2 < args.dst_tensor.Width() && tile_y + y < args.dst_tensor.Height()) {
      FLT4 value = t0 * args.At.Read(13) + t1 * args.At.Read(15) + bias_val;
      args.dst_tensor.Write(value, tile_x + 2, tile_y + y, Z);
    }
    if (tile_x + 3 < args.dst_tensor.Width() && tile_y + y < args.dst_tensor.Height()) {
      FLT4 value = t2 * args.At.Read(19) + t3 * args.At.Read(21) + I[y][5] + bias_val;
      args.dst_tensor.Write(value, tile_x + 3, tile_y + y, Z);
    }
  }
}
)";
  return c;
}
}  // namespace

int3 Winograd4x4To36::GetGridSize() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_3(mht_3_v, 433, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd4x4To36::GetGridSize");

  int new_width =
      src_[0]->Width() + padding_.prepended.w + padding_.appended.w - 2;
  int new_height =
      src_[0]->Height() + padding_.prepended.h + padding_.appended.h - 2;
  int tiles_x = DivideRoundUp(new_width, 4);
  int tiles_y = DivideRoundUp(new_height, 4);
  return int3(tiles_x, tiles_y, src_[0]->Slices());
}

absl::Status Winograd4x4To36::BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_4(mht_4_v, 446, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd4x4To36::BindArguments");

  int new_width =
      src_[0]->Width() + padding_.prepended.w + padding_.appended.w - 2;
  int new_height =
      src_[0]->Height() + padding_.prepended.h + padding_.appended.h - 2;
  int tiles_x = DivideRoundUp(new_width, 4);
  int tiles_y = DivideRoundUp(new_height, 4);
  RETURN_IF_ERROR(args->SetInt("tiles_x", tiles_x));
  RETURN_IF_ERROR(args->SetInt("tiles_y", tiles_y));
  return absl::OkStatus();
}

Winograd4x4To36 CreateWinograd4x4To36(const OperationDef& definition,
                                      const Padding2D& padding) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_5(mht_5_v, 462, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "CreateWinograd4x4To36");

  Winograd4x4To36 desc(definition, padding);
  desc.code_ = GetKernelWinograd4x4To36(definition);

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args_.AddInt("padding_x", -padding.prepended.w);
  desc.args_.AddInt("padding_y", -padding.prepended.h);
  desc.args_.AddInt("tiles_x");
  desc.args_.AddInt("tiles_y");

  BufferDescriptor buffer_desc;
  VectorToKernelBufferDesc(BtMatrixForWinograd4x4To6x6(),
                           definition.GetDataType(), &buffer_desc);
  desc.args_.AddObject(
      "Bt", absl::make_unique<BufferDescriptor>(std::move(buffer_desc)));

  desc.work_group_size_ = int3(8, 4, 1);
  return desc;
}

Winograd4x4To36TileX6::Winograd4x4To36TileX6(const OperationDef& definition,
                                             const Padding2D& padding,
                                             const GpuInfo& gpu_info)
    : GPUOperation(definition), padding_(padding) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_6(mht_6_v, 490, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd4x4To36TileX6::Winograd4x4To36TileX6");

  work_group_size_ = int3(32, 1, 1);
  code_ = GetWinograd4x4To36TileX6Code(definition_, gpu_info);
  if (gpu_info.IsAdreno()) {
    compiler_options_.push_back(CompilerOptions::kAdrenoMoreWaves);
  }
  if (definition_.precision == CalculationsPrecision::F16 &&
      gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
}

std::string Winograd4x4To36TileX6::GetWinograd4x4To36TileX6Code(
    const OperationDef& op_def, const GpuInfo& gpu_info) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_7(mht_7_v, 506, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd4x4To36TileX6::GetWinograd4x4To36TileX6Code");

  std::string c;
  const auto& src_desc = op_def.src_tensors[0];
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddInt("padding_x");
  args_.AddInt("padding_y");
  args_.AddInt("tiles_total");
  args_.AddInt("tiles_x");

  c += "MAIN_FUNCTION($0) {\n";
  c += "  int DST_X = GLOBAL_ID_0;\n";
  c += "  int DST_Y = GLOBAL_ID_1;\n";
  c += "  int DST_Z = GLOBAL_ID_2;\n";
  c += "  if (DST_X >= args.tiles_total || DST_Y >= 6 || DST_Z >= "
       "args.dst_tensor.Slices()) {\n";
  c += "    return; \n";
  c += "  }\n";
  c += "  int tile_x = (DST_X % args.tiles_x) * 4;\n";
  c += "  int tile_y = (DST_X / args.tiles_x) * 4;\n";
  c += "  FLT4 I0, I1, I2, I3, I4, I5;\n";
  c += "  FLT bt_ar[6];\n";
  c += "  FLT4 t0 = args.bt_non_uniform.Read(DST_Y * 2 + 0);\n";
  c += "  FLT4 t1 = args.bt_non_uniform.Read(DST_Y * 2 + 1);\n";
  c += "  DST_Y *= 6;\n";
  c += "  bt_ar[0] = t0.x;\n";
  c += "  bt_ar[1] = t0.y;\n";
  c += "  bt_ar[2] = t0.z;\n";
  c += "  bt_ar[3] = t0.w;\n";
  c += "  bt_ar[4] = t1.x;\n";
  c += "  bt_ar[5] = t1.y;\n";
  auto read_src = [&](const std::string& src, const std::string& xs) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("src: \"" + src + "\"");
   mht_8_v.push_back("xs: \"" + xs + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_8(mht_8_v, 542, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "lambda");

    std::string read_statement;
    if (src_desc.IsLinear()) {
      read_statement = "args.src_tensor.Read(src_a_" + xs + " + offset)";
    } else {
      read_statement = "args.src_tensor.Read(xc" + xs + ", yc, DST_Z)";
    }
    std::string multiplier;
    if (!src_desc.SupportsZeroClamp(Axis::WIDTH)) {
      if (!(src_desc.IsLinear() && src_desc.ReturnsZeroForNegOneRead())) {
        multiplier = " * m" + xs + "_x";
      }
    }
    c += "    FLT4 " + src + " = " + read_statement + multiplier + ";\n";
  };
  for (int x = 0; x < 6; ++x) {
    const std::string xs = std::to_string(x);
    c += "  int xc" + xs + " = tile_x + args.padding_x + " + xs + ";\n";
    if (!src_desc.SupportsZeroClamp(Axis::WIDTH)) {
      c += "  bool inx" + xs + " = (xc" + xs + " >= 0 && xc" + xs +
           " < args.src_tensor.Width());\n";
      c += "  FLT m" + xs + "_x = INIT_FLT(inx" + xs + ");\n";
      c += "  xc" + xs + " = clamp(xc" + xs +
           ", 0, args.src_tensor.Width() - 1);\n";
    }
    if (src_desc.IsLinear()) {
      c += "  args.src_tensor.GetAddress(src_a_" + xs + ", xc" + xs +
           ", 0, DST_Z);\n";
      if (src_desc.ReturnsZeroForNegOneRead()) {
        c += "  src_a_" + xs +
             " = select(-args.src_tensor.Width() * args.src_tensor.Height(), "
             "src_a_" +
             xs + ", inx" + xs + ");\n";
      }
    }
  }
  const bool manual_unroll =
      !(op_def.precision == CalculationsPrecision::F32 && gpu_info.IsMali());
  if (manual_unroll) {
    c += "  {\n";
    c += "    int yc = tile_y + args.padding_y;\n";
    if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
      c += "    bool iny = (yc >= 0 && yc < args.src_tensor.Height());\n";
      c += "    yc = clamp(yc, 0, args.src_tensor.Height() - 1);\n";
      c += "    int offset = select(0, yc * args.src_tensor.Width(), iny);\n";
      c += "    FLT bt = bt_ar[0] * INIT_FLT(iny);\n";
    } else {
      c += "    FLT bt = bt_ar[0];\n";
    }
    for (int x = 0; x < 6; ++x) {
      const std::string xs = std::to_string(x);
      const std::string src = "src" + xs;
      read_src(src, xs);
      c += "    I" + xs + " = bt * " + src + ";\n";
    }
    c += "  }\n";
    for (int y = 1; y < 6; ++y) {
      const std::string ys = std::to_string(y);
      c += "  {\n";
      c += "    int yc = tile_y + args.padding_y + (" + ys + ");\n";
      if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
        c += "    bool iny = (yc >= 0 && yc < args.src_tensor.Height());\n";
        c += "    yc = clamp(yc, 0, args.src_tensor.Height() - 1);\n";
        c += "    int offset = select(0, yc * args.src_tensor.Width(), iny);\n";
        c += "    FLT bt = bt_ar[" + ys + "] * INIT_FLT(iny);\n";
      } else {
        c += "    FLT bt = bt_ar[" + ys + "];\n";
      }
      for (int x = 0; x < 6; ++x) {
        const std::string xs = std::to_string(x);
        const std::string src = "src" + xs;
        read_src(src, xs);
        c += "    I" + xs + " += bt * " + src + ";\n";
      }
      c += "  }\n";
    }
  } else {
    c += "  I0 = INIT_FLT4(0.0f);\n";
    c += "  I1 = INIT_FLT4(0.0f);\n";
    c += "  I2 = INIT_FLT4(0.0f);\n";
    c += "  I3 = INIT_FLT4(0.0f);\n";
    c += "  I4 = INIT_FLT4(0.0f);\n";
    c += "  I5 = INIT_FLT4(0.0f);\n";
    c += "  for (int y = 0; y < 6; ++y) {\n";
    c += "    int yc = tile_y + args.padding_y + y;\n";
    if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
      c += "    bool iny = (yc >= 0 && yc < args.src_tensor.Height());\n";
      c += "    yc = clamp(yc, 0, args.src_tensor.Height() - 1);\n";
      c += "    int offset = select(0, yc * args.src_tensor.Width(), iny);\n";
      c += "    FLT bt = bt_ar[y] * INIT_FLT(iny);\n";
    } else {
      c += "    FLT bt = bt_ar[y];\n";
    }
    for (int x = 0; x < 6; ++x) {
      const std::string xs = std::to_string(x);
      const std::string src = "src" + xs;
      read_src(src, xs);
      c += "    I" + xs + " += bt * " + src + ";\n";
    }
    c += "  }\n";
  }
  c += "  {\n";
  c += "    FLT4 r0 = I0 + args.Bt.Read(2) * I2 + args.Bt.Read(4) * I4;\n";
  c += "    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);\n";
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "  {\n";
  c += "    FLT4 r0 = args.Bt.Read(7) * I1 + args.Bt.Read(8) * I2 + "
       "args.Bt.Read(9) * I3 + args.Bt.Read(10) * I4;\n";
  c += "    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);\n";
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "  {\n";
  c += "    FLT4 r0 = args.Bt.Read(13) * I1 + args.Bt.Read(14) * I2 + "
       "args.Bt.Read(15) * I3 + args.Bt.Read(16) * I4;\n";
  c += "    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);\n";
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "  {\n";
  c += "    FLT4 r0 = args.Bt.Read(19) * I1 + args.Bt.Read(20) * I2 + "
       "args.Bt.Read(21) * I3 + args.Bt.Read(22) * I4;\n";
  c += "    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);\n";
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "  {\n";
  c += "    FLT4 r0 = args.Bt.Read(25) * I1 + args.Bt.Read(26) * I2 + "
       "args.Bt.Read(27) * I3 + args.Bt.Read(28) * I4;\n";
  c += "    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);\n";
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "  {\n";
  c += "    FLT4 r0 = args.Bt.Read(31) * I1 + args.Bt.Read(33) * I3 + I5;\n";
  c += "    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);\n";
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

void Winograd4x4To36TileX6::UploadBt() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_9(mht_9_v, 684, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd4x4To36TileX6::UploadBt");

  tflite::gpu::Tensor<Linear, DataType::FLOAT32> bt_aligned;
  bt_aligned.shape = Linear(6 * 8);
  bt_aligned.data.resize(6 * 8);
  auto bt_mat = BtMatrixForWinograd4x4To6x6();
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      bt_aligned.data[y * 8 + x] = bt_mat[y * 6 + x];
    }
    bt_aligned.data[y * 8 + 6] = 0.0f;
    bt_aligned.data[y * 8 + 7] = 0.0f;
  }

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::TEXTURE_2D;
  desc.element_type = definition_.GetDataType();
  desc.UploadLinearData(bt_aligned);
  args_.AddObject("bt_non_uniform",
                  absl::make_unique<TensorLinearDescriptor>(std::move(desc)));

  BufferDescriptor buffer_desc;
  VectorToKernelBufferDesc(bt_mat, definition_.GetDataType(), &buffer_desc);
  args_.AddObject("Bt",
                  absl::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

int3 Winograd4x4To36TileX6::SelectBestWorkGroup(
    const KernelInfo& kernel_info) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_10(mht_10_v, 714, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd4x4To36TileX6::SelectBestWorkGroup");

  const std::vector<int3> wgs = {{8, 6, 4}, {8, 6, 2}, {4, 6, 2},
                                 {4, 6, 2}, {2, 6, 2}, {2, 6, 1},
                                 {1, 6, 1}, {1, 3, 1}, {1, 1, 1}};
  return GetFirstSuitableWorkGroup(wgs, kernel_info.max_work_group_size);
}

absl::Status Winograd4x4To36TileX6::BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_11(mht_11_v, 724, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd4x4To36TileX6::BindArguments");

  const int tiles_x = DivideRoundUp(
      src_[0]->Width() + padding_.prepended.w + padding_.appended.w - 2, 4);
  const int tiles_y = DivideRoundUp(
      src_[0]->Height() + padding_.prepended.h + padding_.appended.h - 2, 4);
  const int tiles_total = tiles_x * tiles_y;
  RETURN_IF_ERROR(args->SetInt("padding_x", -padding_.prepended.w));
  RETURN_IF_ERROR(args->SetInt("padding_y", -padding_.prepended.h));
  RETURN_IF_ERROR(args->SetInt("tiles_total", tiles_total));
  RETURN_IF_ERROR(args->SetInt("tiles_x", tiles_x));
  return absl::OkStatus();
}

int3 Winograd4x4To36TileX6::GetGridSize() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_12(mht_12_v, 740, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd4x4To36TileX6::GetGridSize");

  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = 6;
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

void Winograd4x4To36TileX6::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_13(mht_13_v, 752, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd4x4To36TileX6::GetPossibleKernelWorkGroups");

  if (gpu_info.IsIntel()) {
    work_groups->push_back(int3(4, 6, 1));
    return;
  }
  switch (tuning_type) {
    case TuningType::kExhaustive:
      GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_,
                            work_groups);
      return;
    case TuningType::kFast:
    default:
      work_groups->push_back(SelectBestWorkGroup(kernel_info));
      return;
  }
}

Winograd4x4To36TileX6 CreateWinograd4x4To36TileX6(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Padding2D& padding) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_14(mht_14_v, 774, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "CreateWinograd4x4To36TileX6");

  Winograd4x4To36TileX6 result(definition, padding, gpu_info);
  result.UploadBt();
  return result;
}

int3 Winograd36To4x4::GetGridSize() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_15(mht_15_v, 783, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd36To4x4::GetGridSize");

  return int3(src_[0]->Width(), 1, src_[0]->Slices());
}

Winograd36To4x4 CreateWinograd36To4x4(
    const OperationDef& definition,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_16(mht_16_v, 792, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "CreateWinograd36To4x4");

  Winograd36To4x4 desc(definition);
  desc.code_ = GetKernelWinograd36To4x4(definition);

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  TensorLinearDescriptor bias_desc;
  bias_desc.storage_type = LinearStorageType::BUFFER;
  bias_desc.element_type = definition.GetDataType();
  bias_desc.UploadLinearData(biases);
  desc.args_.AddObject("biases", absl::make_unique<TensorLinearDescriptor>(
                                     std::move(bias_desc)));

  BufferDescriptor buffer_desc;
  VectorToKernelBufferDesc(AtMatrixForWinograd4x4To6x6(),
                           definition.GetDataType(), &buffer_desc);
  desc.args_.AddObject(
      "At", absl::make_unique<BufferDescriptor>(std::move(buffer_desc)));

  desc.work_group_size_ = int3(32, 1, 1);
  return desc;
}

Winograd36To4x4Tile4x1::Winograd36To4x4Tile4x1(const OperationDef& definition,
                                               const GpuInfo& gpu_info)
    : GPUOperation(definition) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_17(mht_17_v, 821, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd36To4x4Tile4x1::Winograd36To4x4Tile4x1");

  work_group_size_ = int3(32, 1, 1);
  if (definition_.precision == CalculationsPrecision::F16 &&
      gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  code_ = GetWinograd36To4x4Tile4x1Code(definition_, gpu_info);
}

std::string Winograd36To4x4Tile4x1::GetWinograd36To4x4Tile4x1Code(
    const OperationDef& op_def, const GpuInfo& gpu_info) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_18(mht_18_v, 834, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd36To4x4Tile4x1::GetWinograd36To4x4Tile4x1Code");

  std::string c;

  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddInt("tiles_x");

  c += "MAIN_FUNCTION($0) {\n";
  c += "  int tile_id = GLOBAL_ID_0;\n";
  c += "  int DST_Y = GLOBAL_ID_1;\n";
  c += "  int DST_Z = GLOBAL_ID_2;\n";
  c += "  int tile_x = (tile_id % args.tiles_x) * 4;\n";
  c += "  int tile_y = (tile_id / args.tiles_x) * 4 + DST_Y;\n";

  c += "  if (tile_x >= args.dst_tensor.Width() || tile_y >= "
       "args.dst_tensor.Height() || DST_Z >= args.dst_tensor.Slices()) {\n";
  c += "    return; \n";
  c += "  }\n";
  c += "  FLT4 I0, I1, I2, I3, I4, I5;\n";
  c += "  FLT at_ar[6];\n";
  c += "  FLT4 t00 = args.at_non_uniform.Read(DST_Y * 2 + 0);\n";
  c += "  FLT4 t01 = args.at_non_uniform.Read(DST_Y * 2 + 1);\n";
  c += "  at_ar[0] = t00.x;\n";
  c += "  at_ar[1] = t00.y;\n";
  c += "  at_ar[2] = t00.z;\n";
  c += "  at_ar[3] = t00.w;\n";
  c += "  at_ar[4] = t01.x;\n";
  c += "  at_ar[5] = t01.y;\n";
  const bool manual_unroll =
      !(op_def.precision == CalculationsPrecision::F32 && gpu_info.IsMali());
  if (manual_unroll) {
    c += "  {\n";
    c += "    FLT at = at_ar[0];\n";
    for (int x = 0; x < 6; ++x) {
      const std::string yc = std::to_string(x);
      const std::string src = "src" + std::to_string(x);
      c += "    FLT4 " + src + " = args.src_tensor.Read(tile_id, " + yc +
           ", DST_Z);\n";
      c += "    I" + std::to_string(x) + " = at * " + src + ";\n";
    }
    c += "  }\n";
    for (int y = 1; y < 6; ++y) {
      c += "  {\n";
      c += "    FLT at = at_ar[" + std::to_string(y) + "];\n";
      for (int x = 0; x < 6; ++x) {
        const std::string yc = std::to_string(y * 6 + x);
        const std::string src = "src" + std::to_string(x);
        c += "    FLT4 " + src + " = args.src_tensor.Read(tile_id, " + yc +
             ", DST_Z);\n";
        c += "    I" + std::to_string(x) + " += at * " + src + ";\n";
      }
      c += "  }\n";
    }
  } else {
    c += "  I0 = INIT_FLT4(0.0f);\n";
    c += "  I1 = INIT_FLT4(0.0f);\n";
    c += "  I2 = INIT_FLT4(0.0f);\n";
    c += "  I3 = INIT_FLT4(0.0f);\n";
    c += "  I4 = INIT_FLT4(0.0f);\n";
    c += "  I5 = INIT_FLT4(0.0f);\n";
    c += "  for (int y = 0; y < 6; ++y) {\n";
    c += "    FLT at = at_ar[y];\n";
    for (int x = 0; x < 6; ++x) {
      const std::string src = "src" + std::to_string(x);
      c += "    FLT4 " + src + " = args.src_tensor.Read(tile_id, y * 6 + " +
           std::to_string(x) + ", DST_Z);\n";
      c += "    I" + std::to_string(x) + " += at * " + src + ";\n";
    }
    c += "  }\n";
  }
  c += "  FLT4 t0 = I1 + I2;\n";
  c += "  FLT4 t1 = I3 + I4;\n";
  c += "  FLT4 bias_val = args.biases.Read(DST_Z);\n";
  c += "  {\n";
  c += "    FLT4 r0 = I0 + t0 + t1 + bias_val;\n";
  c += "    args.dst_tensor.Write(r0, tile_x, tile_y, DST_Z);\n";
  c += "    tile_x++;\n";
  c += "  }\n";
  c += "  FLT4 t2 = I1 - I2;\n";
  c += "  FLT4 t3 = I3 - I4;\n";
  c += "  if (tile_x < args.dst_tensor.Width()) {\n";
  c +=
      "    FLT4 r0 = t2 * args.At.Read(7) + t3 * args.At.Read(9) + bias_val;\n";
  c += "    args.dst_tensor.Write(r0, tile_x, tile_y, DST_Z);\n";
  c += "    tile_x++;\n";
  c += "  }\n";
  c += "  if (tile_x < args.dst_tensor.Width()) {\n";
  c += "    FLT4 r0 = t0 * args.At.Read(13) + t1 * args.At.Read(15) + "
       "bias_val;\n";
  c += "    args.dst_tensor.Write(r0, tile_x, tile_y, DST_Z);\n";
  c += "    tile_x++;\n";
  c += "  }\n";
  c += "  if (tile_x < args.dst_tensor.Width()) {\n";
  c += "    FLT4 r0 = t2 * args.At.Read(19) + t3 * args.At.Read(21) + I5 + "
       "bias_val;\n";
  c += "    args.dst_tensor.Write(r0, tile_x, tile_y, DST_Z);\n";
  c += "    tile_x++;\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

void Winograd36To4x4Tile4x1::UploadAt() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_19(mht_19_v, 939, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd36To4x4Tile4x1::UploadAt");

  tflite::gpu::Tensor<Linear, DataType::FLOAT32> at_aligned;
  at_aligned.shape = Linear(4 * 8);
  at_aligned.data.resize(4 * 8);
  auto at_mat = AtMatrixForWinograd4x4To6x6();
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 6; ++x) {
      at_aligned.data[y * 8 + x] = at_mat[y * 6 + x];
    }
    at_aligned.data[y * 8 + 6] = 0.0f;
    at_aligned.data[y * 8 + 7] = 0.0f;
  }

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::TEXTURE_2D;
  desc.element_type = definition_.GetDataType();
  desc.UploadLinearData(at_aligned);
  args_.AddObject("at_non_uniform",
                  absl::make_unique<TensorLinearDescriptor>(std::move(desc)));

  BufferDescriptor buffer_desc;
  VectorToKernelBufferDesc(at_mat, definition_.GetDataType(), &buffer_desc);
  args_.AddObject("At",
                  absl::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

int3 Winograd36To4x4Tile4x1::SelectBestWorkGroup(
    const KernelInfo& kernel_info) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_20(mht_20_v, 969, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd36To4x4Tile4x1::SelectBestWorkGroup");

  const std::vector<int3> wgs = {{32, 4, 2}, {16, 4, 2}, {16, 4, 1},
                                 {8, 4, 1},  {4, 4, 1},  {2, 4, 1},
                                 {1, 4, 1},  {1, 2, 1},  {1, 1, 1}};
  return GetFirstSuitableWorkGroup(wgs, kernel_info.max_work_group_size);
}

absl::Status Winograd36To4x4Tile4x1::BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_21(mht_21_v, 979, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd36To4x4Tile4x1::BindArguments");

  const int tiles_x = DivideRoundUp(dst_[0]->Width(), 4);
  RETURN_IF_ERROR(args->SetInt("tiles_x", tiles_x));
  return absl::OkStatus();
}

int3 Winograd36To4x4Tile4x1::GetGridSize() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_22(mht_22_v, 988, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd36To4x4Tile4x1::GetGridSize");

  const int tiles_x = DivideRoundUp(dst_[0]->Width(), 4);
  const int tiles_y = DivideRoundUp(dst_[0]->Height(), 4);
  const int grid_x = tiles_x * tiles_y * dst_[0]->Batch();
  const int grid_y = 4;
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

void Winograd36To4x4Tile4x1::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_23(mht_23_v, 1002, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "Winograd36To4x4Tile4x1::GetPossibleKernelWorkGroups");

  if (gpu_info.IsIntel()) {
    work_groups->push_back(int3(8, 4, 1));
    return;
  }
  switch (tuning_type) {
    case TuningType::kExhaustive:
      GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_,
                            work_groups);
      return;
    case TuningType::kFast:
    default:
      work_groups->push_back(SelectBestWorkGroup(kernel_info));
      return;
  }
}

Winograd36To4x4Tile4x1 CreateWinograd36To4x4Tile4x1(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTcc mht_24(mht_24_v, 1024, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.cc", "CreateWinograd36To4x4Tile4x1");

  Winograd36To4x4Tile4x1 result(definition, gpu_info);
  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(biases);
  result.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  result.UploadAt();
  return result;
}

}  // namespace gpu
}  // namespace tflite
