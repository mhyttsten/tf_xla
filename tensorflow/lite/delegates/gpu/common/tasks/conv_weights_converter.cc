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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converterDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converterDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.h"

#include <cstring>
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

ConverterToConvWeights::ConverterToConvWeights(
    const OperationDef& definition, const WeightsDescription& weights_desc)
    : GPUOperation(definition), weights_desc_(weights_desc) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converterDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.cc", "ConverterToConvWeights::ConverterToConvWeights");

  code_ = GetConverterToConvWeightsCode(definition_, weights_desc_);
}

ConverterToConvWeights::ConverterToConvWeights(
    ConverterToConvWeights&& operation)
    : GPUOperation(std::move(operation)),
      weights_desc_(std::move(operation.weights_desc_)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converterDTcc mht_1(mht_1_v, 209, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.cc", "ConverterToConvWeights::ConverterToConvWeights");
}

ConverterToConvWeights& ConverterToConvWeights::operator=(
    ConverterToConvWeights&& operation) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converterDTcc mht_2(mht_2_v, 215, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.cc", "=");

  if (this != &operation) {
    weights_desc_ = std::move(operation.weights_desc_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string ConverterToConvWeights::GetConverterToConvWeightsCode(
    const OperationDef& op_def, const WeightsDescription& conv_weights_desc) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converterDTcc mht_3(mht_3_v, 227, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.cc", "ConverterToConvWeights::GetConverterToConvWeightsCode");

  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  args_.AddFloat("mask_x");
  args_.AddFloat("mask_y");
  args_.AddFloat("mask_z");
  args_.AddFloat("mask_w");
  args_.AddInt("grid_x_size");

  if (conv_weights_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
      conv_weights_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    std::vector<int32_t> remap(conv_weights_desc.spatial_remap.size());
    for (int i = 0; i < remap.size(); ++i) {
      remap[i] = conv_weights_desc.spatial_remap[i];
    }
    BufferDescriptor desc;
    desc.element_type = DataType::INT32;
    desc.element_size = 1;
    desc.memory_type = MemoryType::GLOBAL;
    desc.size = remap.size() * sizeof(int32_t);
    desc.data.resize(desc.size);
    std::memcpy(desc.data.data(), remap.data(), desc.size);
    args_.AddObject("spatial_remap",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int O = GLOBAL_ID_0;\n";
  c += "  int I = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  int W = Z % args.src_tensor.Width();\n";
  c += "  int H = Z / args.src_tensor.Width();\n";
  c += "  if (O >= args.grid_x_size || I >= args.src_tensor.Slices() || "
       "H >= args.src_tensor.Height()) return;\n";
  c += "  O *= 4;\n";
  std::string x_kern = "W";
  std::string y_kern = "H";
  if (conv_weights_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
      conv_weights_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    c += "  int spatial_linear = H * args.src_tensor.Width() + W;\n";
    c += "  int linear_remap = args.spatial_remap.Read(spatial_linear);\n";
    c += "  int w_remap = linear_remap % args.src_tensor.Width();\n";
    c += "  int h_remap = linear_remap / args.src_tensor.Width();\n";
    x_kern = "w_remap";
    y_kern = "h_remap";
  }
  const std::string coords = x_kern + ", " + y_kern;
  c += "  FLT4 v0 = INIT_FLT4(0.0f);\n";
  c += "  FLT4 v1 = INIT_FLT4(0.0f);\n";
  c += "  FLT4 v2 = INIT_FLT4(0.0f);\n";
  c += "  FLT4 v3 = INIT_FLT4(0.0f);\n";
  c += "  if (O < args.src_tensor.Batch()) {\n";
  c += "    v0 = args.src_tensor.Read(" + coords + ", I, O);\n";
  c += "  }\n";
  c += "  if (O + 1 < args.src_tensor.Batch()) {\n";
  c += "    v1 = args.src_tensor.Read(" + coords + ", I, O + 1);\n";
  c += "  }\n";
  c += "  if (O + 2 < args.src_tensor.Batch()) {\n";
  c += "    v2 = args.src_tensor.Read(" + coords + ", I, O + 2);\n";
  c += "  }\n";
  c += "  if (O + 3 < args.src_tensor.Batch()) {\n";
  c += "    v3 = args.src_tensor.Read(" + coords + ", I, O + 3);\n";
  c += "  }\n";
  c += "  if (I == args.src_tensor.Slices() - 1) {\n";
  c += "    FLT4 mask = INIT_FLT4v4(args.mask_x, args.mask_y, args.mask_z, "
       "args.mask_w);\n";
  c += "    v0 *= mask;\n";
  c += "    v1 *= mask;\n";
  c += "    v2 *= mask;\n";
  c += "    v3 *= mask;\n";
  c += "  }\n";
  if (conv_weights_desc.IsI4O4()) {
    c += "  FLT4 r0 = INIT_FLT4v4(v0.x, v1.x, v2.x, v3.x);\n";
    c += "  FLT4 r1 = INIT_FLT4v4(v0.y, v1.y, v2.y, v3.y);\n";
    c += "  FLT4 r2 = INIT_FLT4v4(v0.z, v1.z, v2.z, v3.z);\n";
    c += "  FLT4 r3 = INIT_FLT4v4(v0.w, v1.w, v2.w, v3.w);\n";
  } else if (conv_weights_desc.IsO4I4()) {
    c += "  FLT4 r0 = v0;\n";
    c += "  FLT4 r1 = v1;\n";
    c += "  FLT4 r2 = v2;\n";
    c += "  FLT4 r3 = v3;\n";
  }
  if (conv_weights_desc.layout ==
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      conv_weights_desc.layout ==
          WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    // Writing to 4X Textures 2D
    AddDstTensor("dst_tensor0", op_def.dst_tensors[0]);
    AddDstTensor("dst_tensor1", op_def.dst_tensors[1]);
    AddDstTensor("dst_tensor2", op_def.dst_tensors[2]);
    AddDstTensor("dst_tensor3", op_def.dst_tensors[3]);
    c += "  int yc = (H * args.src_tensor.Width() + W) * "
         "args.src_tensor.Slices() + I;\n";
    c += "  args.dst_tensor0.Write2D(r0, O / 4, yc);\n";
    c += "  args.dst_tensor1.Write2D(r1, O / 4, yc);\n";
    c += "  args.dst_tensor2.Write2D(r2, O / 4, yc);\n";
    c += "  args.dst_tensor3.Write2D(r3, O / 4, yc);\n";
    c += "}\n";
  } else {
    // Writing to linear buffer
    AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
    c += "  int GROUP_SIZE = " +
         std::to_string(conv_weights_desc.GetOutputGroupSize()) + ";\n";
    c += "  int d_index = O / (GROUP_SIZE * 4);\n";
    c += "  int k_index = (O % (GROUP_SIZE * 4)) / 4;\n";
    std::string index;
    if (conv_weights_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
        conv_weights_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
      index =
          "((d_index * args.src_tensor.Slices() + I) * "
          "args.src_tensor.Height() "
          "+ H) * args.src_tensor.Width() + W";
    } else if (conv_weights_desc.layout ==
                   WeightsLayout::kOSpatialIOGroupI4O4 ||
               conv_weights_desc.layout ==
                   WeightsLayout::kOSpatialIOGroupO4I4) {
      index =
          "((d_index * args.src_tensor.Height() + H) * args.src_tensor.Width() "
          "+ "
          "W) * args.src_tensor.Slices() + I";
    }
    c += "  int dst_offset = (" + index + ") * GROUP_SIZE + k_index;\n";
    c += "  args.dst_tensor.WriteLinear(r0, dst_offset * 4 + 0);\n";
    c += "  args.dst_tensor.WriteLinear(r1, dst_offset * 4 + 1);\n";
    c += "  args.dst_tensor.WriteLinear(r2, dst_offset * 4 + 2);\n";
    c += "  args.dst_tensor.WriteLinear(r3, dst_offset * 4 + 3);\n";
    c += "}\n";
  }
  return c;
}

absl::Status ConverterToConvWeights::BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converterDTcc mht_4(mht_4_v, 361, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.cc", "ConverterToConvWeights::BindArguments");

  const int out_group_size = weights_desc_.GetOutputGroupSize();
  const int grid_x =
      DivideRoundUp(AlignByN(src_[0]->Batch(), 4 * out_group_size), 4);
  RETURN_IF_ERROR(args->SetInt("grid_x_size", grid_x));
  float4 mask = GetMaskForLastPlane(src_[0]->Channels());
  RETURN_IF_ERROR(args->SetFloat("mask_x", mask.x));
  RETURN_IF_ERROR(args->SetFloat("mask_y", mask.y));
  RETURN_IF_ERROR(args->SetFloat("mask_z", mask.z));
  return args->SetFloat("mask_w", mask.w);
}

int3 ConverterToConvWeights::GetGridSize() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converterDTcc mht_5(mht_5_v, 376, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.cc", "ConverterToConvWeights::GetGridSize");

  const int out_group_size = weights_desc_.GetOutputGroupSize();
  const int grid_x =
      DivideRoundUp(AlignByN(src_[0]->Batch(), 4 * out_group_size), 4);
  const int grid_y = src_[0]->Slices();
  const int grid_z = src_[0]->Width() * src_[0]->Height();
  return int3(grid_x, grid_y, grid_z);
}

ConverterToConvWeights CreateConverterToConvWeights(
    const OperationDef& definition, const WeightsDescription& weights_desc) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converterDTcc mht_6(mht_6_v, 389, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.cc", "CreateConverterToConvWeights");

  return ConverterToConvWeights(definition, weights_desc);
}

}  // namespace gpu
}  // namespace tflite
