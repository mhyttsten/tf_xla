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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpaddingDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpaddingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpaddingDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/padding.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetPaddingCode(const OperationDef& op_def,
                           const PadAttributes& attr, GPUOperation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpaddingDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/gpu/common/tasks/padding.cc", "GetPaddingCode");

  op->AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op->args_.AddInt("prepended_x", attr.prepended.w);
  op->args_.AddInt("prepended_y", attr.prepended.h);
  op->args_.AddInt("prepended_z", attr.prepended.c);
  op->args_.AddInt("prepended_w", attr.prepended.b);

  const std::string dst_batch =
      op_def.dst_tensors[0].HasAxis(Axis::BATCH) ? "B" : "0";
  std::string c;
  const std::string channels[] = {".x", ".y", ".z", ".w"};

  if (attr.type == PaddingContentType::REFLECT) {
    c += "int reflect_coord(int x, int size) {\n";
    c += "  int t = abs(x) - size + 1;\n";
    c += "  return size - 1 - abs(t);\n";
    c += "}\n\n";
  }

  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor::type result = args.src_tensor::zero_value;\n";
  c += "  int s_x = X - args.prepended_x;\n";
  c += "  int s_y = Y - args.prepended_y;\n";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int s_b = " + dst_batch + " - args.prepended_w;\n";
    c += "  args.src_tensor.SetBatchRef(s_b);\n";
  }
  if (attr.type == PaddingContentType::REFLECT) {
    c += "  s_x = reflect_coord(s_x, args.src_tensor.Width());\n";
    c += "  s_y = reflect_coord(s_y, args.src_tensor.Height());\n";
    if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
      c += "  int s_b = reflect_coord(s_b, args.src_tensor.Batch());\n";
    }
    if (attr.prepended.c == 0 && attr.appended.c == 0) {
      // optimized case
      c += "  result = args.src_tensor.Read(s_x, s_y, Z);\n";
    } else {
      c += "  int start_channel = Z * 4;\n";
      for (int i = 0; i < 4; ++i) {
        const auto& s = channels[i];
        c += "  {\n";
        c += "    int channel = start_channel + " + std::to_string(i) + ";\n";
        c += "    int s_z = channel - args.prepended_z;\n";
        // We need additional clamp for z, so that we use alignment for channels
        // and can proceed extra channels that can lead to reading out of
        // resource.
        c += "    s_z = clamp(reflect_coord(s_z, args.src_tensor.Channels()), "
             "0, "
             "args.src_tensor.Channels() - "
             "1);\n";
        c += "    args.src_tensor.ReadPerChannel(result" + s +
             ", s_x, s_y, s_z);\n";
        c += "  }\n";
      }
    }
  } else {
    c += "  bool inside_x = s_x >= 0 && s_x < args.src_tensor.Width();\n";
    c += "  bool inside_y = s_y >= 0 && s_y < args.src_tensor.Height();\n";
    if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
      c += "  inside_y = inside_y && (s_b >= 0 && s_b < "
           "args.src_tensor.Batch());\n";
    }
    c += "  if (inside_x && inside_y) {\n";
    if (attr.prepended.c == 0 && attr.appended.c == 0) {
      // optimized case
      c += "    result = args.src_tensor.Read(s_x, s_y, Z);\n";
    } else if (attr.prepended.c % 4 == 0) {
      c += "    int s_z = Z - args.prepended_z / 4;\n";
      c += "    if (s_z >= 0 && s_z < args.src_tensor.Slices()) {\n";
      c += "      result = args.src_tensor.Read(s_x, s_y, s_z);\n";
      c += "    }\n";
    } else {
      c += "    int start_channel = Z * 4;\n";
      for (int i = 0; i < 4; ++i) {
        const auto& s = channels[i];
        c += "    {\n";
        c += "    int channel = start_channel + " + std::to_string(i) + ";\n";
        c += "    int s_z = channel - args.prepended_z;\n";
        c += "    if (s_z >= 0 && s_z < args.src_tensor.Channels()) {\n";
        c += "      args.src_tensor.ReadPerChannel(result" + s +
             ", s_x, s_y, s_z);\n";
        c += "    }\n";
        c += "    }\n";
      }
    }
    c += "  }\n";
  }
  c += "  args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "}\n";

  return c;
}

}  // namespace

GPUOperation CreatePadding(const OperationDef& definition,
                           const PadAttributes& attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpaddingDTcc mht_1(mht_1_v, 311, "", "./tensorflow/lite/delegates/gpu/common/tasks/padding.cc", "CreatePadding");

  GPUOperation op(definition);
  op.code_ = GetPaddingCode(definition, attr, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
