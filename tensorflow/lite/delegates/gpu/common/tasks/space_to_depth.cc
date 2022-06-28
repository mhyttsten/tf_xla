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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspace_to_depthDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspace_to_depthDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspace_to_depthDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/space_to_depth.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetSpaceToDepthCode(const OperationDef& op_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspace_to_depthDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/gpu/common/tasks/space_to_depth.cc", "GetSpaceToDepthCode");

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor::scalar_type tmp[4];\n";
  c += "  tmp[0] = args.src_tensor::scalar_zero_value;\n";
  c += "  tmp[1] = args.src_tensor::scalar_zero_value;\n";
  c += "  tmp[2] = args.src_tensor::scalar_zero_value;\n";
  c += "  tmp[3] = args.src_tensor::scalar_zero_value;\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_c = 4 * S + i;\n";
  c += "    int block_id = dst_c / args.src_tensor.Channels();\n";
  c += "    int src_x = X * args.block_size + block_id % args.block_size;\n";
  c += "    int src_y = Y * args.block_size + block_id / args.block_size;\n";
  c += "    int src_c = dst_c % args.src_tensor.Channels();\n";
  c += "    args.src_tensor.ReadPerChannel(tmp[i], src_x, src_y, src_c);\n";
  c += "  }\n";
  c += "  args.src_tensor::type result;\n";
  c += "  result.x = tmp[0];\n";
  c += "  result.y = tmp[1];\n";
  c += "  result.z = tmp[2];\n";
  c += "  result.w = tmp[3];\n";
  c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  c += "}\n";
  return c;
}

std::string GetDepthToSpaceCode(const OperationDef& op_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspace_to_depthDTcc mht_1(mht_1_v, 241, "", "./tensorflow/lite/delegates/gpu/common/tasks/space_to_depth.cc", "GetDepthToSpaceCode");

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT tmp[4];\n";
  c += "  tmp[0] = INIT_FLT(0.0f);\n";
  c += "  tmp[1] = INIT_FLT(0.0f);\n";
  c += "  tmp[2] = INIT_FLT(0.0f);\n";
  c += "  tmp[3] = INIT_FLT(0.0f);\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_c = 4 * S + i;\n";
  c += "    int block_x = X % args.block_size;\n";
  c += "    int src_x = X / args.block_size;\n";
  c += "    int block_y = Y % args.block_size;\n";
  c += "    int src_y = Y / args.block_size;\n";
  c += "    int block_id = block_y * args.block_size + block_x;\n";
  c += "    int src_c = block_id * args.dst_tensor.Channels() + dst_c;\n";
  c += "    args.src_tensor.ReadPerChannel(tmp[i], src_x, src_y, src_c);\n";
  c += "  }\n";
  c += "  FLT4 result;\n";
  c += "  result.x = tmp[0];\n";
  c += "  result.y = tmp[1];\n";
  c += "  result.z = tmp[2];\n";
  c += "  result.w = tmp[3];\n";
  c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateSpaceToDepth(const OperationDef& op_def,
                                const SpaceToDepthAttributes& attr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspace_to_depthDTcc mht_2(mht_2_v, 289, "", "./tensorflow/lite/delegates/gpu/common/tasks/space_to_depth.cc", "CreateSpaceToDepth");

  GPUOperation op(op_def);
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.args_.AddInt("block_size", attr.block_size);
  op.code_ = GetSpaceToDepthCode(op_def);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreateDepthToSpace(const OperationDef& op_def,
                                const SpaceToDepthAttributes& attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspace_to_depthDTcc mht_3(mht_3_v, 303, "", "./tensorflow/lite/delegates/gpu/common/tasks/space_to_depth.cc", "CreateDepthToSpace");

  GPUOperation op(op_def);
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.args_.AddInt("block_size", attr.block_size);
  op.code_ = GetDepthToSpaceCode(op_def);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
