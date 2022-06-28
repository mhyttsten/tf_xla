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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmax_unpoolingDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmax_unpoolingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmax_unpoolingDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/max_unpooling.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetMaxUnpoolingKernelCode(const OperationDef& op_def,
                                      GPUOperation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmax_unpoolingDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/delegates/gpu/common/tasks/max_unpooling.cc", "GetMaxUnpoolingKernelCode");

  auto src_desc = op_def.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddSrcTensor("src_tensor", src_desc);
  auto src_ind_desc = op_def.src_tensors[1];
  src_ind_desc.SetAddressMode(AddressMode::kZero);
  if (op_def.IsBatchSupported()) {
    src_ind_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddSrcTensor("src_indices", src_ind_desc);
  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddDstTensor("dst_tensor", dst_desc);

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_id_1 % args.dst_tensor.Depth();\n";
    c += "  int src_z = (Z + args.padding_z) / args.stride_z;\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id_0 = GLOBAL_ID_0;\n";
    c += "  int X0 = linear_id_0 / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id_0 % args.dst_tensor.Batch();\n";
    c += "  int src_x0 = (X0 + args.padding_x * args.dst_tensor.Batch()) / "
         "args.stride_x;\n";
    c += "  int src_x = src_x0 * args.dst_tensor.Batch() + B;\n";
  } else {
    c += "  int src_x = (X + args.padding_x) / args.stride_x;\n";
  }
  c += "  int src_y = (Y + args.padding_y) / args.stride_y;\n";
  std::string src_args = op_def.dst_tensors[0].HasAxis(Axis::DEPTH)
                             ? "src_x, src_y, src_z, S"
                             : "src_x, src_y, S";
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER) {
    if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
      c += "  bool outside = src_x < 0 || src_y < 0 || src_z < 0 || src_x >= "
           "args.src_tensor.Width() || src_y >= args.src_tensor.Height() || "
           "src_z >= args.src_tensor.Depth();\n";
    } else {
      c += "  bool outside = src_x < 0 || src_y < 0 || src_x >= "
           "args.src_tensor.Width() || src_y >= args.src_tensor.Height();\n";
    }
    c += "  FLT4 src = INIT_FLT4(0.0f);\n";
    c += "  int4 ind = INIT_INT4v4(0, 0, 0, 0);\n";
    c += "  if (!outside) {\n";
    c += "    src = args.src_tensor.Read(" + src_args + ");\n";
    c += "    ind = args.src_indices.Read<int>(" + src_args + ");\n";
    c += "  }\n";
  } else {
    c += "  FLT4 src = args.src_tensor.Read(" + src_args + ");\n";
    c += "  int4 ind = args.src_indices.Read<int>(" + src_args + ");\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int t_x = X0 - (src_x0 * args.stride_x - args.padding_x * "
         "args.dst_tensor.Batch());\n";
  } else {
    c += "  int t_x = X - (src_x * args.stride_x - args.padding_x);\n";
  }
  c += "  int t_y = Y - (src_y * args.stride_y - args.padding_y);\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int t_z = Z - (src_z * args.stride_z - args.padding_z);\n";
    c += "  int t_index = (t_y * args.kernel_size_x + t_x) * "
         "args.kernel_size_z + t_z;\n";
  } else {
    c += "  int t_index = t_y * args.kernel_size_x + t_x;\n";
  }
  c += "  FLT4 result;\n";
  const std::string channels[] = {".x", ".y", ".z", ".w"};
  for (int i = 0; i < 4; ++i) {
    const auto& s = channels[i];
    c += "  result" + s + "= t_index == ind" + s + "? src" + s +
         ": INIT_FLT(0.0f);\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  args.dst_tensor.Write(result, X, Y, Z, S);\n";
  } else {
    c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  }
  c += "}\n";

  return c;
}
}  // namespace

GPUOperation CreateMaxUnpooling(const OperationDef& definition,
                                const MaxUnpooling2DAttributes& attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmax_unpoolingDTcc mht_1(mht_1_v, 300, "", "./tensorflow/lite/delegates/gpu/common/tasks/max_unpooling.cc", "CreateMaxUnpooling");

  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.kernel.w);
  op.args_.AddInt("padding_x", attr.padding.appended.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("kernel_size_y", attr.kernel.h);
  op.args_.AddInt("padding_y", attr.padding.appended.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.code_ = GetMaxUnpoolingKernelCode(definition, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreateMaxUnpooling(const OperationDef& definition,
                                const MaxUnpooling3DAttributes& attr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmax_unpoolingDTcc mht_2(mht_2_v, 317, "", "./tensorflow/lite/delegates/gpu/common/tasks/max_unpooling.cc", "CreateMaxUnpooling");

  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.kernel.w);
  op.args_.AddInt("padding_x", attr.padding.appended.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("kernel_size_y", attr.kernel.h);
  op.args_.AddInt("padding_y", attr.padding.appended.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("kernel_size_z", attr.kernel.d);
  op.args_.AddInt("padding_z", attr.padding.appended.d);
  op.args_.AddInt("stride_z", attr.strides.d);
  op.code_ = GetMaxUnpoolingKernelCode(definition, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
