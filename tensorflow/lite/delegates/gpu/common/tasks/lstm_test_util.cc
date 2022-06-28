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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSlstm_test_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSlstm_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSlstm_test_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/lstm_test_util.h"

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/lstm.h"

namespace tflite {
namespace gpu {

absl::Status LstmTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSlstm_test_utilDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/gpu/common/tasks/lstm_test_util.cc", "LstmTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 16);
  src_tensor.data = {
      -std::log(2.0f), -std::log(2.0f), -std::log(2.0f), -std::log(2.0f),
      std::log(3.0f),  std::log(3.0f),  std::log(3.0f),  std::log(3.0f),
      -std::log(4.0f), -std::log(4.0f), -std::log(4.0f), -std::log(4.0f),
      -std::log(5.0f), -std::log(5.0f), -std::log(5.0f), -std::log(5.0f)};
  // input_gate = 1.0 / (1.0 + exp(log(2.0f))) = 1.0 / 3.0;
  // new_input = tanh(log(3.0f)) = (exp(2 * log(3.0f)) - 1) / exp(2 * log(3.0f))
  // + 1 = (9 - 1) / (9 + 1) = 0.8;
  // forget_gate = 1.0 / (1.0 + exp(log(4.0f)))
  //  = 1.0 / 5.0;
  // output_gate = 1.0 / (1.0 + exp(log(5.0f))) = 1.0 / 6.0;
  // new_st = input_gate * new_input + forget_gate * prev_st
  //   = 1.0 / 3.0 * 0.8 + 1.0 / 5.0 * prev_st
  //   = 4.0 / 15.0 + 3.0 / 15.0 = 7.0 / 15.0
  // activation = output_gate * tanh(new_st)
  TensorFloat32 prev_state;
  prev_state.shape = BHWC(1, 1, 1, 4);
  prev_state.data = {1.0f, 2.0f, 3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorFloat32 new_state;
      TensorFloat32 new_activ;
      GPUOperation operation = CreateLSTM(op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor, prev_state},
          absl::make_unique<GPUOperation>(std::move(operation)),
          {BHWC(1, 1, 1, 4), BHWC(1, 1, 1, 4)}, {&new_state, &new_activ}));
      RETURN_IF_ERROR(
          PointWiseNear({7.0 / 15.0, 10.0 / 15.0, 13.0 / 15.0, 16.0 / 15.0},
                        new_state.data, eps))
          << ToString(storage) << ", " << ToString(precision);
      RETURN_IF_ERROR(PointWiseNear(
          {static_cast<float>((1.0 / 6.0) * std::tanh(7.0 / 15.0)),
           static_cast<float>((1.0 / 6.0) * std::tanh(10.0 / 15.0)),
           static_cast<float>((1.0 / 6.0) * std::tanh(13.0 / 15.0)),
           static_cast<float>((1.0 / 6.0) * std::tanh(16.0 / 15.0))},
          new_activ.data, eps))
          << ToString(storage) << ", " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
