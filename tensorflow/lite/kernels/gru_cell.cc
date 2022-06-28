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
class MHTracer_DTPStensorflowPSlitePSkernelsPSgru_cellDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSgru_cellDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSgru_cellDTcc() {
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

#include "tensorflow/lite/kernels/gru_cell.h"

#include <vector>

#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace tflite {
namespace ops {
namespace custom {
namespace gru_cell {

using optimized_ops::ArrayMap;
using optimized_ops::FullyConnected;
using optimized_ops::MapAsArrayWithLastDimAsRows;
using reference_ops::Concatenation;

void GruCell(const RuntimeShape& input_shape, const float* input,
             const RuntimeShape& state_shape, const float* input_state,
             const RuntimeShape& gate_weight_shape, const float* gate_weight,
             const RuntimeShape& gate_bias_shape, const float* gate_bias,
             const RuntimeShape& candidate_weight_shape,
             const float* candidate_weight,
             const RuntimeShape& candidate_bias_shape,
             const float* candidate_bias, const RuntimeShape& output_shape,
             float* output, float* output_state,
             const RuntimeShape& activation_shape, float* activation,
             const RuntimeShape& concat_shape, float* concat,
             const tflite::FullyConnectedParams& fc_params,
             tflite::CpuBackendContext* cpu_backend_context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgru_cellDTcc mht_0(mht_0_v, 213, "", "./tensorflow/lite/kernels/gru_cell.cc", "GruCell");

  const int n_batch = input_shape.Dims(0);
  const int n_input = input_shape.Dims(1);
  const int n_output = state_shape.Dims(1);

  // [x h] = concat(input, state)
  std::vector<float const*> concat_arrays_data;
  std::vector<RuntimeShape const*> concat_arrays_shapes;
  concat_arrays_data.push_back(input);
  concat_arrays_data.push_back(input_state);
  concat_arrays_shapes.push_back(&input_shape);
  concat_arrays_shapes.push_back(&state_shape);
  tflite::ConcatenationParams concat_params;
  concat_params.axis = 1;
  concat_params.inputs_count = concat_arrays_data.size();
  Concatenation(concat_params, &(concat_arrays_shapes[0]),
                &(concat_arrays_data[0]), concat_shape, concat);

  // [r u] = [x h] * gate_weight + gate_bias
  FullyConnected(fc_params, concat_shape, concat, gate_weight_shape,
                 gate_weight, gate_bias_shape, gate_bias, activation_shape,
                 activation, cpu_backend_context);

  // [r u] = sigmoid([r u])
  auto ru = MapAsArrayWithLastDimAsRows(activation, activation_shape);
  ru = ru.unaryExpr(Eigen::internal::scalar_logistic_op<float>());
  auto r = ru.block(0 * n_output, 0, n_output, n_batch);
  auto u = ru.block(1 * n_output, 0, n_output, n_batch);

  // hr = h .* r
  auto h = MapAsArrayWithLastDimAsRows(input_state, state_shape);
  auto xh = MapAsArrayWithLastDimAsRows(concat, concat_shape);
  auto hr = xh.block(n_input, 0, n_output, n_batch);
  hr = h * r;

  // c = [x hr] * candidate_weight + candidate_bias
  FullyConnected(fc_params, concat_shape, concat, candidate_weight_shape,
                 candidate_weight, candidate_bias_shape, candidate_bias,
                 output_shape, output, cpu_backend_context);

  auto c = MapAsArrayWithLastDimAsRows(output, output_shape);
  // output = (1 - u) .* tanh(c) + u .* h
  c = (1.0 - u) * c.tanh() + u * h;

  memcpy(output_state, output, n_batch * n_output * sizeof(float));
}

}  // namespace gru_cell
}  // namespace custom
}  // namespace ops
}  // namespace tflite
