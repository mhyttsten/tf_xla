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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSlstmDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSlstmDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSlstmDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/lstm.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetLSTMCode(const OperationDef& op_def, const GpuInfo& gpu_info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSlstmDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/gpu/common/tasks/lstm.cc", "GetLSTMCode");

  std::string c;
  c += "MAIN_FUNCTION(\n";
  c += "$0) {\n";
  c += "  int B = GLOBAL_ID_0;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (Z >= args.activation.Slices() || B >= args.activation.Batch()) "
       "return;\n";
  c += "  FLT4 prev_st = args.prev_state.Read(0, 0, Z, B);\n";
  c += "  FLT4 r0 = args.intermediate.Read(0, 0, Z, B);\n";
  c += "  int state_stride = args.activation.Slices();\n";
  c += "  FLT4 r1 = args.intermediate.Read(0, 0, Z + state_stride, B);\n";
  c += "  FLT4 r2 = args.intermediate.Read(0, 0, Z + state_stride * 2, B);\n";
  c += "  FLT4 r3 = args.intermediate.Read(0, 0, Z + state_stride * 3, B);\n";
  if (gpu_info.IsApiOpenCl() &&
      op_def.precision != CalculationsPrecision::F32 && gpu_info.IsAdreno()) {
    c += "  FLT4 input_gate;\n";
    c += "  FLT4 new_input;\n";
    c += "  FLT4 forget_gate;\n";
    c += "  FLT4 output_gate;\n";
    c += "  input_gate.x = native_recip(1.0h + native_exp(-r0.x));\n";
    c += "  input_gate.y = native_recip(1.0h + native_exp(-r0.y));\n";
    c += "  input_gate.z = native_recip(1.0h + native_exp(-r0.z));\n";
    c += "  input_gate.w = native_recip(1.0h + native_exp(-r0.w));\n";
    c += "  new_input.x = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.x));\n";
    c += "  new_input.y = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.y));\n";
    c += "  new_input.z = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.z));\n";
    c += "  new_input.w = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.w));\n";
    c += "  forget_gate.x = native_recip(1.0h + native_exp(-r2.x));\n";
    c += "  forget_gate.y = native_recip(1.0h + native_exp(-r2.y));\n";
    c += "  forget_gate.z = native_recip(1.0h + native_exp(-r2.z));\n";
    c += "  forget_gate.w = native_recip(1.0h + native_exp(-r2.w));\n";
    c += "  output_gate.x = native_recip(1.0h + native_exp(-r3.x));\n";
    c += "  output_gate.y = native_recip(1.0h + native_exp(-r3.y));\n";
    c += "  output_gate.z = native_recip(1.0h + native_exp(-r3.z));\n";
    c += "  output_gate.w = native_recip(1.0h + native_exp(-r3.w));\n";
  } else {
    c += "  FLT4 input_gate  = INIT_FLT4(1.0f) / (INIT_FLT4(1.0f) + "
         "exp(INIT_FLT4(-1.0f) "
         "* r0));\n";
    c += "  FLT4 new_input   = tanh(r1);\n";
    c += "  FLT4 forget_gate = INIT_FLT4(1.0f) / (INIT_FLT4(1.0f) + "
         "exp(INIT_FLT4(-1.0f) "
         "* r2));\n";
    c += "  FLT4 output_gate = INIT_FLT4(1.0f) / (INIT_FLT4(1.0f) + "
         "exp(INIT_FLT4(-1.0f) "
         "* r3));\n";
  }
  c += "  FLT4 new_st = input_gate * new_input + forget_gate * prev_st;\n";
  c += "  FLT4 act_value = output_gate * tanh(new_st);\n";
  c += "  args.activation.Write(act_value, 0, 0, Z, B);\n";
  c += "  args.new_state.Write(new_st, 0, 0, Z, B);\n";
  c += "}\n";
  return c;
}

}  // namespace

GPUOperation CreateLSTM(const OperationDef& definition,
                        const GpuInfo& gpu_info) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSlstmDTcc mht_1(mht_1_v, 261, "", "./tensorflow/lite/delegates/gpu/common/tasks/lstm.cc", "CreateLSTM");

  GPUOperation op(definition);
  op.AddSrcTensor("intermediate", definition.src_tensors[0]);
  op.AddSrcTensor("prev_state", definition.src_tensors[1]);
  op.AddDstTensor("new_state", definition.dst_tensors[0]);
  op.AddDstTensor("activation", definition.dst_tensors[1]);
  op.code_ = GetLSTMCode(definition, gpu_info);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
