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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_fake_quantDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_fake_quantDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_fake_quantDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

template <ArrayDataType A>
void GetBoundsForQuantizedDataType(float* min, float* max) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_fake_quantDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_fake_quant.cc", "GetBoundsForQuantizedDataType");

  using limits = std::numeric_limits<DataType<A>>;
  *min = limits::min();
  *max = limits::max();
}

void GetBoundsForQuantizedDataType(ArrayDataType quantized_data_type,
                                   float* min, float* max) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_fake_quantDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_fake_quant.cc", "GetBoundsForQuantizedDataType");

  // It is important for matching accuracy between TF training and TFLite
  // inference, that the min and max values are float to match TF's
  // FakeQuantWithMinMaxVarsFunctor.
  switch (quantized_data_type) {
    case ArrayDataType::kUint8:
      return GetBoundsForQuantizedDataType<ArrayDataType::kUint8>(min, max);
    case ArrayDataType::kInt8:
      return GetBoundsForQuantizedDataType<ArrayDataType::kInt8>(min, max);
    case ArrayDataType::kUint16:
      return GetBoundsForQuantizedDataType<ArrayDataType::kUint16>(min, max);
    case ArrayDataType::kInt16:
      return GetBoundsForQuantizedDataType<ArrayDataType::kInt16>(min, max);
    case ArrayDataType::kUint32:
      return GetBoundsForQuantizedDataType<ArrayDataType::kUint32>(min, max);
    case ArrayDataType::kInt32:
      return GetBoundsForQuantizedDataType<ArrayDataType::kInt32>(min, max);
    case ArrayDataType::kUint64:
      return GetBoundsForQuantizedDataType<ArrayDataType::kUint64>(min, max);
    case ArrayDataType::kInt64:
      return GetBoundsForQuantizedDataType<ArrayDataType::kInt64>(min, max);
    default:
      LOG(FATAL) << "unhandled quantized data type";
  }
}

::tensorflow::Status ResolveConstantFakeQuant::Run(Model* model,
                                                   std::size_t op_index,
                                                   bool* modified) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_fake_quantDTcc mht_2(mht_2_v, 239, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_fake_quant.cc", "ResolveConstantFakeQuant::Run");

  *modified = false;
  const auto fakequant_it = model->operators.begin() + op_index;
  const auto* fakequant_base_op = fakequant_it->get();
  if (fakequant_base_op->type != OperatorType::kFakeQuant) {
    return ::tensorflow::Status::OK();
  }

  const auto* fakequant_op =
      static_cast<const FakeQuantOperator*>(fakequant_base_op);

  // Yield until the fakequant MinMax has been resolved.
  if (!fakequant_op->minmax) {
    return ::tensorflow::Status::OK();
  }

  // This transformation only applies when the input array is constant.
  if (!IsConstantParameterArray(*model, fakequant_op->inputs[0])) {
    return ::tensorflow::Status::OK();
  }

  const auto& input_array = model->GetArray(fakequant_op->inputs[0]);
  CHECK(input_array.data_type == ArrayDataType::kFloat);

  // Determine the final data type in the same way as PropagateFakeQuantNumBits.
  ArrayDataType quantized_data_type = input_array.final_data_type;
  if (!InferQuantizedDataTypeFromFakeQuant(*fakequant_op,
                                           &quantized_data_type)) {
    AddMessageF("Unsupported FakeQuant num_bits=%d", fakequant_op->num_bits);
    return ::tensorflow::Status::OK();
  }

  AddMessageF("Resolving constant %s", LogName(*fakequant_op));

  auto& output_array = model->GetArray(fakequant_op->outputs[0]);
  CHECK(input_array.data_type == ArrayDataType::kFloat);
  output_array.data_type = ArrayDataType::kFloat;

  // We'll set the final data type to what the fake quant indicates we should
  // have (and would have been set if this stayed around until
  // PropagateFakeQuantNumBits).
  if (propagate_fake_quant_num_bits()) {
    output_array.final_data_type = quantized_data_type;
  }

  CHECK(!output_array.buffer);
  const auto& input_buffer = input_array.GetBuffer<ArrayDataType::kFloat>();
  output_array.GetOrCreateMinMax() = *fakequant_op->minmax;
  auto& output_buffer = output_array.GetMutableBuffer<ArrayDataType::kFloat>();
  const int size = input_buffer.data.size();
  output_buffer.data.resize(size);
  QuantizationParams qparams;
  ChooseQuantizationParamsForArrayAndQuantizedDataType(
      output_array, quantized_data_type, &qparams);
  float quantized_min, quantized_max;
  GetBoundsForQuantizedDataType(quantized_data_type, &quantized_min,
                                &quantized_max);
  if (fakequant_op->narrow_range) {
    quantized_min++;
    output_array.narrow_range = true;
  }

  // It is important for matching accuracy between TF training and TFLite
  // inference, that the following variables are float to match TF's
  // FakeQuantWithMinMaxVarsFunctor.
  const float scale = qparams.scale;
  const float nudged_min = (quantized_min - qparams.zero_point) * scale;
  const float nudged_max = (quantized_max - qparams.zero_point) * scale;
  tflite::FakeQuantizeArray(scale, nudged_min, nudged_max,
                            input_buffer.data.data(), output_buffer.data.data(),
                            size);
  DeleteOpAndArrays(model, fakequant_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
