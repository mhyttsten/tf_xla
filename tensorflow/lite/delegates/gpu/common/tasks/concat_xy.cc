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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_xyDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_xyDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_xyDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/concat_xy.h"

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {
std::string GetConcatKernelCode(const OperationDef& op_def,
                                const ConcatAttributes& attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_xyDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/delegates/gpu/common/tasks/concat_xy.cc", "GetConcatKernelCode");

  std::vector<std::string> tensor_names(op_def.src_tensors.size());
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    tensor_names[i] = "src_tensor_" + std::to_string(i);
  }

  std::map<Axis, std::string> axis_to_selector = {
      {Axis::WIDTH, "Width"}, {Axis::HEIGHT, "Height"},
      {Axis::DEPTH, "Depth"}, {Axis::CHANNELS, "Channels"},
      {Axis::BATCH, "Batch"},
  };
  std::map<Axis, std::string> axis_to_coord = {
      {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"}, {Axis::DEPTH, "D"},
      {Axis::CHANNELS, "S"}, {Axis::BATCH, "B"},
  };

  std::vector<std::string> src_coords;
  std::vector<std::string> dst_coords;
  for (auto axis :
       {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH, Axis::CHANNELS, Axis::BATCH}) {
    if (op_def.src_tensors[0].HasAxis(axis) && axis != Axis::BATCH) {
      if (axis == attr.axis) {
        src_coords.push_back("coord");
      } else {
        src_coords.push_back(axis_to_coord[axis]);
      }
    }
    if (op_def.dst_tensors[0].HasAxis(axis)) {
      dst_coords.push_back(axis_to_coord[axis]);
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
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id_0 = GLOBAL_ID_0;\n";
    c += "  int X = linear_id_0 / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id_0 % args.dst_tensor.Batch();\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor_0::type result = args.src_tensor_0::zero_value;\n";
  c += "  int coord = " + axis_to_coord[attr.axis] + ";\n";
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    const std::string field =
        "args." + tensor_names[i] + "." + axis_to_selector[attr.axis] + "()";
    c += "  if (coord >= 0 && coord < " + field + ") { \n";
    if (op_def.src_tensors[i].HasAxis(Axis::BATCH)) {
      if (attr.axis == Axis::BATCH) {
        c += "  args." + tensor_names[i] + ".SetBatchRef(coord);\n";
      } else {
        c += "  args." + tensor_names[i] + ".SetBatchRef(B);\n";
      }
    }
    c += "    result = args." + tensor_names[i] + ".Read(" + src_coord + ");\n";
    c += "  } \n";
    c += "  coord -= " + field + ";\n";
  }
  c += "  args.dst_tensor.Write(result, " + dst_coord + ");\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateConcatXY(const OperationDef& definition,
                            const ConcatAttributes& attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_xyDTcc mht_1(mht_1_v, 287, "", "./tensorflow/lite/delegates/gpu/common/tasks/concat_xy.cc", "CreateConcatXY");

  GPUOperation op(definition);
  for (int i = 0; i < definition.src_tensors.size(); ++i) {
    const std::string name = "src_tensor_" + std::to_string(i);
    op.AddSrcTensor(name, definition.src_tensors[i]);
  }
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetConcatKernelCode(definition, attr);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
