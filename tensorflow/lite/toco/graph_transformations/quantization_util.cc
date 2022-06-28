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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool InferQuantizedDataTypeFromFakeQuant(
    const FakeQuantOperator& op, ArrayDataType* out_quantized_data_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/toco/graph_transformations/quantization_util.cc", "InferQuantizedDataTypeFromFakeQuant");

  if (op.num_bits <= 8) {
    *out_quantized_data_type = ArrayDataType::kUint8;
    return true;
  } else if (op.num_bits <= 16) {
    *out_quantized_data_type = ArrayDataType::kInt16;
    return true;
  } else {
    *out_quantized_data_type = ArrayDataType::kNone;
    return false;
  }
}

bool GetQuantizedDataTypeNumericalRange(ArrayDataType data_type,
                                        double* out_min_value,
                                        double* out_max_value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc mht_1(mht_1_v, 213, "", "./tensorflow/lite/toco/graph_transformations/quantization_util.cc", "GetQuantizedDataTypeNumericalRange");

  switch (data_type) {
    case ArrayDataType::kUint8:
      *out_min_value = 0;
      *out_max_value = 255;
      return true;
    case ArrayDataType::kInt16:
      *out_min_value = -32768;
      *out_max_value = 32767;
      return true;
    default:
      return false;
  }
}

ArrayDataType GetQuantizedDataType(const Array& array,
                                   ArrayDataType default_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc mht_2(mht_2_v, 232, "", "./tensorflow/lite/toco/graph_transformations/quantization_util.cc", "GetQuantizedDataType");

  switch (array.final_data_type) {
    case ArrayDataType::kInt8:
    case ArrayDataType::kUint8:
    case ArrayDataType::kInt16:
    case ArrayDataType::kUint16:
    case ArrayDataType::kInt32:
    case ArrayDataType::kUint32:
    case ArrayDataType::kInt64:
    case ArrayDataType::kUint64:
      return array.final_data_type;
    case ArrayDataType::kFloat:
    case ArrayDataType::kNone:
      return default_type;
    default:
      LOG(FATAL) << "Unhandled final quantization type "
                 << static_cast<int>(array.final_data_type);
  }
}

template <ArrayDataType A>
void ChooseQuantizationParamsForArrayAndQuantizedDataType(
    const Array& array, QuantizationParams* quantization_params) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc mht_3(mht_3_v, 257, "", "./tensorflow/lite/toco/graph_transformations/quantization_util.cc", "ChooseQuantizationParamsForArrayAndQuantizedDataType");

  *quantization_params = ::tflite::ChooseQuantizationParams<DataType<A>>(
      array.minmax->min, array.minmax->max, array.narrow_range);
}

void ChooseQuantizationParamsForArrayAndQuantizedDataType(
    const Array& array, ArrayDataType quantized_data_type,
    QuantizationParams* quantization_params) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc mht_4(mht_4_v, 267, "", "./tensorflow/lite/toco/graph_transformations/quantization_util.cc", "ChooseQuantizationParamsForArrayAndQuantizedDataType");

  switch (quantized_data_type) {
    case ArrayDataType::kInt8:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kInt8>(array, quantization_params);
      break;
    case ArrayDataType::kUint8:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kUint8>(array, quantization_params);
      break;
    case ArrayDataType::kInt16:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kInt16>(array, quantization_params);
      break;
    case ArrayDataType::kUint16:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kUint16>(array, quantization_params);
      break;
    case ArrayDataType::kInt32:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kInt32>(array, quantization_params);
      break;
    case ArrayDataType::kUint32:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kUint32>(array, quantization_params);
      break;
    case ArrayDataType::kInt64:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kInt64>(array, quantization_params);
      break;
    case ArrayDataType::kUint64:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kUint64>(array, quantization_params);
      break;
    case ArrayDataType::kFloat:
    case ArrayDataType::kComplex64:
    case ArrayDataType::kNone:
    default:
      LOG(FATAL) << "Unhandled final quantization type "
                 << static_cast<int>(quantized_data_type);
  }
}

namespace {

template <ArrayDataType A>
std::unique_ptr<GenericBuffer> QuantizeBuffer(
    const Array& array, const QuantizationParams& quantization_params) {
  const GenericBuffer& buffer = *array.buffer;
  const auto inverse_scale = 1. / quantization_params.scale;
  CHECK(buffer.type == ArrayDataType::kFloat);
  const auto& float_buffer =
      static_cast<const Buffer<ArrayDataType::kFloat>&>(buffer);
  auto* quantized_buffer = new Buffer<A>;
  quantized_buffer->data.resize(float_buffer.data.size());
  for (std::size_t i = 0; i < float_buffer.data.size(); i++) {
    const float src_val = float_buffer.data[i];
    double scaled_val;  // Astonishingly, using 'float' degrades accuracy just
                        // enough to make a few tests fail!
    if (quantization_params.scale == 0) {
      CHECK_EQ(src_val, 0) << "The quantization scale for this array is 0, "
                           << "so all its values should be 0.";
      scaled_val = quantization_params.zero_point;
    } else {
      scaled_val = quantization_params.zero_point + inverse_scale * src_val;
    }
    auto integer_val = tflite::SafeCast<DataType<A>>(std::round(scaled_val));
    // In addition to its effect on the choice of quantization params upstream
    // of here, narrow_range also means nudge the min quantized value by +1,
    // so e.g. uint8 values get constrained to [1, 255].
    if (integer_val == std::numeric_limits<DataType<A>>::min() &&
        array.narrow_range) {
      integer_val++;
    }
    quantized_buffer->data[i] = integer_val;
  }
  return std::unique_ptr<GenericBuffer>(quantized_buffer);
}

template <ArrayDataType A>
void QuantizeArray(GraphTransformation* transformation, Model* model,
                   const std::string& name,
                   const QuantizationParams& quantization_params) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc mht_5(mht_5_v, 353, "", "./tensorflow/lite/toco/graph_transformations/quantization_util.cc", "QuantizeArray");

  auto& array = model->GetArray(name);
  CHECK(array.data_type == ArrayDataType::kFloat);
  CHECK(!array.quantization_params);
  array.GetOrCreateQuantizationParams() = quantization_params;
  if (array.buffer) {
    array.buffer = QuantizeBuffer<A>(array, quantization_params);
  }
  array.data_type = A;
  array.final_data_type = A;
  transformation->AddMessageF(
      "Quantized array %s to %s zero_point=%g, scale=%g", name,
      ArrayDataTypeName(array.data_type), quantization_params.zero_point,
      quantization_params.scale);
}

}  // namespace

void QuantizeArray(GraphTransformation* transformation, Model* model,
                   const std::string& name, ArrayDataType quantized_data_type,
                   const QuantizationParams& quantization_params) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc mht_6(mht_6_v, 377, "", "./tensorflow/lite/toco/graph_transformations/quantization_util.cc", "QuantizeArray");

  ArrayDataType adjusted_data_type = quantized_data_type;
  auto& array = model->GetArray(name);
  if (array.final_data_type == ArrayDataType::kInt16) {
    adjusted_data_type = array.final_data_type;
  }

  switch (adjusted_data_type) {
    case ArrayDataType::kUint8:
      return QuantizeArray<ArrayDataType::kUint8>(transformation, model, name,
                                                  quantization_params);
    case ArrayDataType::kInt16:
      return QuantizeArray<ArrayDataType::kInt16>(transformation, model, name,
                                                  quantization_params);
    case ArrayDataType::kInt32:
      return QuantizeArray<ArrayDataType::kInt32>(transformation, model, name,
                                                  quantization_params);
    default:
      LOG(FATAL) << "Unhandled case.";
  }
}

bool IsArrayQuantizedRangeSubset(GraphTransformation* transformation,
                                 const Array& array, double clamp_min,
                                 double clamp_max) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSquantization_utilDTcc mht_7(mht_7_v, 404, "", "./tensorflow/lite/toco/graph_transformations/quantization_util.cc", "IsArrayQuantizedRangeSubset");

  ArrayDataType quantized_data_type =
      GetQuantizedDataType(array, array.data_type);
  if (quantized_data_type == ArrayDataType::kNone ||
      quantized_data_type == ArrayDataType::kFloat) {
    // The array is not (or never will be) quantized.
    return false;
  }

  QuantizationParams quantization_params;
  if (!array.quantization_params) {
    if (!array.minmax) {
      transformation->AddMessageF("No quantization params and no minmax");
      return false;
    } else {
      // Work around cases where we are asking for this prior to the Quantize
      // transformation having added the quantization_params.
      ChooseQuantizationParamsForArrayAndQuantizedDataType(
          array, quantized_data_type, &quantization_params);
      transformation->AddMessageF(
          "No quantization params - inferring from data type %s with minmax "
          "%g,%g as zero_point=%g, scale=%g",
          ArrayDataTypeName(quantized_data_type), array.minmax->min,
          array.minmax->max, quantization_params.zero_point,
          quantization_params.scale);
    }
  } else {
    quantization_params = array.GetQuantizationParams();
  }

  double quantized_min, quantized_max;
  CHECK(GetQuantizedDataTypeNumericalRange(quantized_data_type, &quantized_min,
                                           &quantized_max))
      << "Type is not quantized";

  bool has_nontrivial_min_bound = false;
  bool has_nontrivial_max_bound = false;

  double lowest_representable_output =
      (quantized_min - quantization_params.zero_point) *
      quantization_params.scale;
  if (lowest_representable_output < clamp_min) {
    has_nontrivial_min_bound = true;
    transformation->AddMessageF(
        "Quantized activation function is not trivial: "
        "the lowest representable output value %g"
        " less than the clamp min bound %g.",
        lowest_representable_output, clamp_min);
  }

  double highest_representable_output =
      (quantized_max - quantization_params.zero_point) *
      quantization_params.scale;
  if (highest_representable_output > clamp_max) {
    has_nontrivial_max_bound = true;
    transformation->AddMessageF(
        "Quantized activation function is not trivial: "
        "the highest representable output value %g"
        " is greater than the clamp max bound %g.",
        highest_representable_output, clamp_max);
  }

  return !has_nontrivial_min_bound && !has_nontrivial_max_bound;
}

}  // namespace toco
