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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpoolingDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpoolingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpoolingDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/pooling.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetAveragePoolingKernelCode(const OperationDef& op_def,
                                        bool stride_correction,
                                        GPUOperation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpoolingDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/gpu/common/tasks/pooling.cc", "GetAveragePoolingKernelCode");

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

  std::map<Axis, std::string> axis_to_src_coord = {
      {Axis::WIDTH, "x_c"},  {Axis::HEIGHT, "y_c"}, {Axis::DEPTH, "d_c"},
      {Axis::CHANNELS, "Z"}, {Axis::BATCH, "B"},
  };

  std::map<Axis, std::string> axis_to_dst_coord = {
      {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"}, {Axis::DEPTH, "D"},
      {Axis::CHANNELS, "Z"}, {Axis::BATCH, "B"},
  };

  std::vector<std::string> src_coords;
  std::vector<std::string> dst_coords;
  for (auto axis : {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH, Axis::CHANNELS}) {
    if (op_def.dst_tensors[0].HasAxis(axis)) {
      dst_coords.push_back(axis_to_dst_coord[axis]);
    }
    if (op_def.src_tensors[0].HasAxis(axis)) {
      src_coords.push_back(axis_to_src_coord[axis]);
    }
  }
  std::string src_coord = src_coords[0];
  for (int i = 1; i < src_coords.size(); ++i) {
    src_coord += ", " + src_coords[i];
  }
  std::string dst_coord = dst_coords[0];
  for (int i = 1; i < dst_coords.size(); ++i) {
    dst_coord += ", " + dst_coords[i];
  }

  const bool manual_clamp =
      op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER ||
      op_def.src_tensors[0].storage_type == TensorStorageType::IMAGE_BUFFER;

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  float4 r = INIT_FLOAT4(0.0f);\n";
  c += "  float window_size = 0.0;\n";
  if (stride_correction) {
    c += "  int xs = " +
         GetXStrideCorrectedV2("X", "args.src_tensor.Batch()", "args.stride_x",
                               "args.padding_x") +
         ";\n";
  } else {
    if (op_def.IsBatchSupported()) {
      c += "  int xs = X * args.stride_x + args.padding_x * "
           "args.src_tensor.Batch();\n";
    } else {
      c += "  int xs = X * args.stride_x + args.padding_x;\n";
    }
  }
  c += "  int ys = Y * args.stride_y + args.padding_y;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int ds = D * args.stride_z + args.padding_z;\n";
    c += "  for (int kz = 0; kz < args.kernel_size_z; ++kz) {\n";
    c += "    int d_c = ds + kz;\n";
    c += "    if (d_c < 0 || d_c >= args.src_tensor.Depth()) continue;\n";
  }
  c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
  c += "    int y_c = ys + ky;\n";
  c += "    bool outside_y = y_c < 0 || y_c >= args.src_tensor.Height();\n";
  c += "    for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
  if (op_def.IsBatchSupported()) {
    c += "      int x_c = xs + kx * args.src_tensor.Batch();\n";
  } else {
    c += "      int x_c = xs + kx;\n";
  }
  c += "      bool outside = outside_y || x_c < 0 || x_c >= "
       "args.src_tensor.Width();\n";
  if (manual_clamp) {
    c += "     r += !outside ? args.src_tensor.Read<float>(" + src_coord +
         ") : "
         "INIT_FLOAT4(0.0f);\n";
  } else {
    c += "      r += args.src_tensor.Read<float>(" + src_coord + ");\n";
  }
  c += "        window_size += !outside ? 1.0 : 0.0;\n";
  c += "    }\n";
  c += "  }\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  }  // Depth\n";
  }
  // If window_size==0, window covered nothing. This situation is a sign of
  // incorrectly constructed operation. NaNs are expected as output.
  c += "  FLT4 result = TO_FLT4(r / window_size);\n";
  c += "  args.dst_tensor.Write(result, " + dst_coord + ");\n";
  c += "}\n";

  return c;
}

std::string GetMaxPoolingKernelCode(const OperationDef& op_def,
                                    bool stride_correction, bool output_indices,
                                    GPUOperation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpoolingDTcc mht_1(mht_1_v, 319, "", "./tensorflow/lite/delegates/gpu/common/tasks/pooling.cc", "GetMaxPoolingKernelCode");

  auto src_desc = op_def.src_tensors[0];
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddSrcTensor("src_tensor", src_desc);
  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddDstTensor("dst_tensor", dst_desc);
  if (output_indices) {
    auto dst_ind_desc = op_def.dst_tensors[1];
    if (op_def.IsBatchSupported()) {
      dst_ind_desc.SetStateVar("BatchedWidth", "true");
    }
    op->AddDstTensor("dst_indices", dst_ind_desc);
  }

  std::map<Axis, std::string> axis_to_src_coord = {
      {Axis::WIDTH, "x_c"},  {Axis::HEIGHT, "y_c"}, {Axis::DEPTH, "d_c"},
      {Axis::CHANNELS, "Z"}, {Axis::BATCH, "B"},
  };

  std::map<Axis, std::string> axis_to_dst_coord = {
      {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"}, {Axis::DEPTH, "D"},
      {Axis::CHANNELS, "Z"}, {Axis::BATCH, "B"},
  };

  std::vector<std::string> src_coords;
  std::vector<std::string> dst_coords;
  for (auto axis : {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH, Axis::CHANNELS}) {
    if (op_def.dst_tensors[0].HasAxis(axis)) {
      dst_coords.push_back(axis_to_dst_coord[axis]);
    }
    if (op_def.src_tensors[0].HasAxis(axis)) {
      src_coords.push_back(axis_to_src_coord[axis]);
    }
  }
  std::string src_coord = src_coords[0];
  for (int i = 1; i < src_coords.size(); ++i) {
    src_coord += ", " + src_coords[i];
  }
  std::string dst_coord = dst_coords[0];
  for (int i = 1; i < dst_coords.size(); ++i) {
    dst_coord += ", " + dst_coords[i];
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT4 maximum = INIT_FLT4(-10000.0f);\n";
  if (output_indices) {
    c += "  int4 indexes = INIT_INT4v4(0, 0, 0, 0);\n";
  }
  if (stride_correction) {
    c += "  int xs = " +
         GetXStrideCorrectedV2("X", "args.src_tensor.Batch()", "args.stride_x",
                               "args.padding_x") +
         ";\n";
  } else {
    if (op_def.IsBatchSupported()) {
      c += "  int xs = X * args.stride_x + args.padding_x * "
           "args.src_tensor.Batch();\n";
    } else {
      c += "  int xs = X * args.stride_x + args.padding_x;\n";
    }
  }
  c += "  int ys = Y * args.stride_y + args.padding_y;\n";
  c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
  c += "    int y_c = ys + ky;\n";
  c += "    if (y_c < 0 || y_c >= args.src_tensor.Height()) continue;\n";
  c += "    for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
  if (op_def.IsBatchSupported()) {
    c += "      int x_c = xs + kx * args.src_tensor.Batch();\n";
  } else {
    c += "      int x_c = xs + kx;\n";
  }
  c += "      if (x_c < 0 || x_c >= args.src_tensor.Width()) continue;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "    int ds = D * args.stride_z + args.padding_z;\n";
    c += "    for (int kz = 0; kz < args.kernel_size_z; ++kz) {\n";
    c += "    int d_c = ds + kz;\n";
    c += "      if (d_c < 0 || d_c >= args.src_tensor.Depth()) continue;\n";
  }
  c += "      FLT4 src = args.src_tensor.Read(" + src_coord + ");\n";
  if (output_indices) {
    if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
      c += "      int index_counter = (ky * args.kernel_size_x + kx) * "
           "args.kernel_size_z + kz;\n";
    } else {
      c += "      int index_counter = ky * args.kernel_size_x + kx;\n";
    }
    c += "      if (src.x > maximum.x) {\n";
    c += "        indexes.x = index_counter;\n";
    c += "        maximum.x = src.x;\n";
    c += "      }\n";
    c += "      if (src.y > maximum.y) {\n";
    c += "        indexes.y = index_counter;\n";
    c += "        maximum.y = src.y;\n";
    c += "      }\n";
    c += "      if (src.z > maximum.z) {\n";
    c += "        indexes.z = index_counter;\n";
    c += "        maximum.z = src.z;\n";
    c += "      }\n";
    c += "      if (src.w > maximum.w) {\n";
    c += "        indexes.w = index_counter;\n";
    c += "        maximum.w = src.w;\n";
    c += "      }\n";
  } else {
    c += "      maximum = max(src, maximum);\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "    }  // Depth\n";
  }
  c += "    }\n";
  c += "  }\n";
  c += "  args.dst_tensor.Write(maximum, " + dst_coord + ");\n";
  if (output_indices) {
    c += "  args.dst_indices.Write<int>(indexes, " + dst_coord + ");\n";
  }
  c += "}\n";

  return c;
}
}  // namespace

GPUOperation CreatePooling(const OperationDef& definition,
                           const Pooling2DAttributes& attr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpoolingDTcc mht_2(mht_2_v, 462, "", "./tensorflow/lite/delegates/gpu/common/tasks/pooling.cc", "CreatePooling");

  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.kernel.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("kernel_size_y", attr.kernel.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  if (attr.type == PoolingType::AVERAGE) {
    op.code_ = GetAveragePoolingKernelCode(definition, stride_correction, &op);
  } else if (attr.type == PoolingType::MAX) {
    op.code_ = GetMaxPoolingKernelCode(definition, stride_correction,
                                       attr.output_indices, &op);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreatePooling(const OperationDef& definition,
                           const Pooling3DAttributes& attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpoolingDTcc mht_3(mht_3_v, 486, "", "./tensorflow/lite/delegates/gpu/common/tasks/pooling.cc", "CreatePooling");

  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.kernel.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("kernel_size_y", attr.kernel.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("kernel_size_z", attr.kernel.d);
  op.args_.AddInt("padding_z", -attr.padding.prepended.d);
  op.args_.AddInt("stride_z", attr.strides.d);
  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  if (attr.type == PoolingType::AVERAGE) {
    op.code_ = GetAveragePoolingKernelCode(definition, stride_correction, &op);
  } else if (attr.type == PoolingType::MAX) {
    op.code_ = GetMaxPoolingKernelCode(definition, stride_correction,
                                       attr.output_indices, &op);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
