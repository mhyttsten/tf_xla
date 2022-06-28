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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsplitDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsplitDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsplitDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/split.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

Split::Split(const OperationDef& definition, const SplitAttributes& attr)
    : GPUOperation(definition), attr_(attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsplitDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/gpu/common/tasks/split.cc", "Split::Split");

  work_group_size_ = int3(8, 4, 1);
  code_ = attr.axis == Axis::CHANNELS ? GetSplitChannelsCode() : GetSplitCode();
}

std::string Split::GetSplitCode() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsplitDTcc mht_1(mht_1_v, 203, "", "./tensorflow/lite/delegates/gpu/common/tasks/split.cc", "Split::GetSplitCode");

  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  for (int i = 0; i < definition_.dst_tensors.size(); ++i) {
    AddDstTensor("dst_tensor_" + std::to_string(i), definition_.dst_tensors[i]);
  }
  const std::string task_width =
      attr_.axis == Axis::WIDTH ? "1" : "args.src_tensor.Width()";
  const std::string task_height =
      attr_.axis == Axis::HEIGHT ? "1" : "args.src_tensor.Height()";
  const std::string task_depth =
      attr_.axis == Axis::DEPTH ? "1" : "args.src_tensor.Depth()";
  const std::string task_batch =
      attr_.axis == Axis::BATCH ? "1" : "args.src_tensor.Batch()";
  const std::string task_slices =
      attr_.axis == Axis::CHANNELS ? "1" : "args.src_tensor.Slices()";

  std::map<Axis, std::string> axis_to_selector = {
      {Axis::WIDTH, "Width"}, {Axis::HEIGHT, "Height"},
      {Axis::DEPTH, "Depth"}, {Axis::CHANNELS, "Slices"},
      {Axis::BATCH, "Batch"},
  };
  std::map<Axis, std::string> axis_to_coord = {
      {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"}, {Axis::DEPTH, "D"},
      {Axis::CHANNELS, "S"}, {Axis::BATCH, "B"},
  };

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (definition_.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / " + task_batch + ";\n";
    c += "  int B = linear_id % " + task_batch + ";\n";
    c += "  if (X >= " + task_width + ") return;\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
    c += "  if (X >= " + task_width + ") return;\n";
  }
  if (definition_.src_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id % " + task_height + ";\n";
    c += "  int D = linear_id / " + task_height + ";\n";
    c += "  if (D >= " + task_depth + ") return;\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
    c += "  if (Y >= " + task_height + ") return;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (S >= " + task_slices + ") return;\n";
  c += "  int src_counter = 0;\n";
  std::vector<std::string> src_coords;
  for (auto axis :
       {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH, Axis::CHANNELS, Axis::BATCH}) {
    if (definition_.src_tensors[0].HasAxis(axis)) {
      const std::string coord_name =
          attr_.axis == axis ? "src_counter" : axis_to_coord[axis];
      src_coords.push_back(coord_name);
    }
  }
  std::string src_coords_str = src_coords[0];
  for (int i = 1; i < src_coords.size(); ++i) {
    src_coords_str += ", " + src_coords[i];
  }
  for (int i = 0; i < definition_.dst_tensors.size(); ++i) {
    std::vector<std::string> dst_coords;
    for (auto axis : {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH, Axis::CHANNELS,
                      Axis::BATCH}) {
      if (definition_.dst_tensors[i].HasAxis(axis)) {
        const std::string coord_name =
            attr_.axis == axis ? "i" : axis_to_coord[axis];
        dst_coords.push_back(coord_name);
      }
    }
    std::string dst_coords_str = dst_coords[0];
    for (int j = 1; j < dst_coords.size(); ++j) {
      dst_coords_str += ", " + dst_coords[j];
    }
    const std::string dst_name = "args.dst_tensor_" + std::to_string(i);
    c += "  for (int i = 0; i < " + dst_name + "." +
         axis_to_selector[attr_.axis] + "(); ++i, src_counter++) {\n";
    c += "    args.src_tensor::type result = args.src_tensor.Read(" +
         src_coords_str + ");\n";
    c += "    " + dst_name + ".Write(result, " + dst_coords_str + ");\n";
    c += "  }\n";
  }
  c += "}\n";
  return c;
}

std::string Split::GetSplitChannelsCode() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsplitDTcc mht_2(mht_2_v, 294, "", "./tensorflow/lite/delegates/gpu/common/tasks/split.cc", "Split::GetSplitChannelsCode");

  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  for (int i = 0; i < definition_.dst_tensors.size(); ++i) {
    AddDstTensor("dst_tensor_" + std::to_string(i), definition_.dst_tensors[i]);
  }

  const std::string batch_coord =
      definition_.src_tensors[0].HasAxis(Axis::BATCH) ? ", B" : "";
  std::string coords = "X, Y";
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (definition_.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.src_tensor.Batch();\n";
    c += "  int B = linear_id % args.src_tensor.Batch();\n";
    c += "  if (X >= args.src_tensor.Width()) return;\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
    c += "  if (X >= args.src_tensor.Width()) return;\n";
  }
  if (definition_.src_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id % args.src_tensor.Height();\n";
    c += "  int Z = linear_id / args.src_tensor.Height();\n";
    c += "  if (Z >= args.src_tensor.Depth()) return;\n";
    coords += ", Z";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
    c += "  if (Y >= args.src_tensor.Height()) return;\n";
  }
  c += "  int src_channel = 0;\n";
  const std::string postfixes[] = {"x", "y", "z", "w"};
  for (int i = 0; i < definition_.dst_tensors.size(); ++i) {
    const std::string dst_name = "args.dst_tensor_" + std::to_string(i);
    c += "  for (int i = 0; i < " + dst_name + ".Slices(); ++i) {\n";
    c += "    args.src_tensor::type result = args.src_tensor::zero_value;\n";
    for (int j = 0; j < 4; ++j) {
      c += "    if (i * 4 + " + std::to_string(j) + " < " + dst_name +
           ".Channels()) {\n";
      c += "      args.src_tensor.ReadPerChannel(result." + postfixes[j] +
           ", " + coords + ", src_channel" + batch_coord + ");\n";
      c += "      src_channel++;\n";
      c += "    }\n";
    }
    c += "    " + dst_name + ".Write(result, " + coords + ", i" + batch_coord +
         ");\n";
    c += "  }\n";
  }
  c += "}\n";
  return c;
}

int3 Split::GetGridSize() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsplitDTcc mht_3(mht_3_v, 349, "", "./tensorflow/lite/delegates/gpu/common/tasks/split.cc", "Split::GetGridSize");

  const int width = attr_.axis == Axis::WIDTH ? 1 : src_[0]->Width();
  const int height = attr_.axis == Axis::HEIGHT ? 1 : src_[0]->Height();
  const int depth = attr_.axis == Axis::DEPTH ? 1 : src_[0]->Depth();
  const int batch = attr_.axis == Axis::BATCH ? 1 : src_[0]->Batch();
  const int slices = attr_.axis == Axis::CHANNELS ? 1 : src_[0]->Slices();
  const int grid_x = width * batch;
  const int grid_y = height * depth;
  const int grid_z = slices;
  return int3(grid_x, grid_y, grid_z);
}

Split CreateSplit(const OperationDef& definition, const SplitAttributes& attr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsplitDTcc mht_4(mht_4_v, 364, "", "./tensorflow/lite/delegates/gpu/common/tasks/split.cc", "CreateSplit");

  return Split(definition, attr);
}

}  // namespace gpu
}  // namespace tflite
