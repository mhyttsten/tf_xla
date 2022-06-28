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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_zDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_zDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_zDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/concat_z.h"

#include <algorithm>
#include <string>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {

bool IsAllChannelsX4(const std::vector<int>& channels) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_zDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/gpu/common/tasks/concat_z.cc", "IsAllChannelsX4");

  for (int channel : channels) {
    if (channel % 4 != 0) {
      return false;
    }
  }
  return true;
}

std::string GetConcatKernelCode(const OperationDef& op_def,
                                const std::vector<int>& channels) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_zDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/delegates/gpu/common/tasks/concat_z.cc", "GetConcatKernelCode");

  std::vector<std::string> tensor_names(op_def.src_tensors.size());
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    tensor_names[i] = "src_tensor_" + std::to_string(i);
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  c += "  int Y = GLOBAL_ID_1;\n";
  std::string coords = "X, Y";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int Z = GLOBAL_ID_2;\n";
    c += "  if (Z >= args.dst_tensor.Depth()) return;\n";
    coords = "X, Y, Z";
  }
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return; \n";

  if (IsAllChannelsX4(channels)) {
    // When all channels % 4 == 0 we can read/assign/write VEC4 elements easily.
    // Also it is easy to write a loop in this case, to prevent long kernel
    // generation.
    c += "  int S = 0;\n";
    for (int i = 0; i < channels.size(); ++i) {
      std::string t_name = "args." + tensor_names[i];
      const int src_depth = DivideRoundUp(channels[i], 4);
      if (src_depth % 2 == 0) {
        // We can read more at once inside of loop in case src_depth % 2 == 0
        // it should be better for reading latency hiding
        c += "  for (int i = 0; i < " + t_name + ".Slices(); i += 2) {\n";
        c += "    " + t_name + "::type result0 = " + t_name + ".Read(" +
             coords + ", i);\n";
        c += "    " + t_name + "::type result1 = " + t_name + ".Read(" +
             coords + ", i + 1);\n";
        c += "    args.dst_tensor.Write(result0, " + coords + ", S);\n";
        c += "    args.dst_tensor.Write(result1, " + coords + ", S + 1);\n";
        c += "    S += 2;\n";
        c += "  }\n";
      } else {
        c += "  for (int i = 0; i < " + t_name + ".Slices(); ++i) {\n";
        c += "    " + t_name + "::type result = " + t_name + ".Read(" + coords +
             ", i);\n";
        c += "    args.dst_tensor.Write(result, " + coords + ", S);\n";
        c += "    S++;\n";
        c += "  }\n";
      }
    }
  } else {
    c += "  args.src_tensor_0::type result = args.src_tensor_0::zero_value;\n";
    int out_channel = 0;
    int read_index = 0;
    int z = 0;
    const std::string postfix[] = {".x", ".y", ".z", ".w"};
    for (int i = 0; i < channels.size(); ++i) {
      std::string tensor_name = "args." + tensor_names[i];
      const int depth = DivideRoundUp(channels[i], 4);
      for (int d = 0; d < depth; ++d) {
        const int channels_in_group = std::min(4, channels[i] - d * 4);
        const std::string temp_name = "t" + std::to_string(read_index);
        c += "  " + tensor_name + "::type " + temp_name + " = " + tensor_name +
             ".Read(" + coords + ", " + std::to_string(d) + ");\n";
        for (int ch = 0; ch < channels_in_group; ++ch) {
          c += "  result" + postfix[out_channel] + " = ";
          c += temp_name + postfix[ch] + ";\n";
          out_channel++;
          if (out_channel == 4) {
            out_channel = 0;
            c += "  args.dst_tensor.Write(result, " + coords + ", " +
                 std::to_string(z) + ");\n";
            z++;
          }
        }
        read_index++;
      }
    }
    if (out_channel != 0) {
      c += "  args.dst_tensor.Write(result, " + coords + ", " +
           std::to_string(z) + ");\n";
    }
  }
  c += "}\n";
  return c;
}

}  // namespace

GPUOperation CreateConcatZ(const OperationDef& definition,
                           const std::vector<int>& channels,
                           const GpuInfo& gpu_info) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_zDTcc mht_2(mht_2_v, 303, "", "./tensorflow/lite/delegates/gpu/common/tasks/concat_z.cc", "CreateConcatZ");

  GPUOperation op(definition);
  for (int i = 0; i < definition.src_tensors.size(); ++i) {
    const std::string name = "src_tensor_" + std::to_string(i);
    auto src_desc = definition.src_tensors[i];
    if (definition.IsBatchSupported()) {
      src_desc.SetStateVar("BatchedWidth", "true");
    }
    op.AddSrcTensor(name, src_desc);
  }
  auto dst_desc = definition.dst_tensors[0];
  if (definition.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op.AddDstTensor("dst_tensor", dst_desc);
  op.code_ = GetConcatKernelCode(definition, channels);
  if (gpu_info.IsPowerVR() &&
      definition.precision == CalculationsPrecision::F32 &&
      !IsAllChannelsX4(channels)) {
    // BUG, some PowerVRs (GE8320) produce incorrect result without it
    op.compiler_options_.push_back(CompilerOptions::kClDisableOptimizations);
  }
  if (gpu_info.IsAMD() && definition.precision != CalculationsPrecision::F32 &&
      definition.src_tensors[0].storage_type != TensorStorageType::BUFFER &&
      !IsAllChannelsX4(channels)) {
    // BUG, some AMD gpus crash without it
    op.compiler_options_.push_back(CompilerOptions::kClDisableOptimizations);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HToY_DToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
