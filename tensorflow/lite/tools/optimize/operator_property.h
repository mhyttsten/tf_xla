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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_OPERATOR_PROPERTY_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_OPERATOR_PROPERTY_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSoperator_propertyDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSoperator_propertyDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSoperator_propertyDTh() {
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


#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace operator_property {

// The scales of a certain tensor can be derived from the multiplications of all
// the scales. For example, for bias in conv, derived_scale = {{0, 1}, {}, {}}
// and for lstm gate bias, the derived scale is {{}, {0}, {2^-10}}
struct DerivedScale {
  // MSVC2015 version 14.0 and below doesn't support struct initialization with
  // initializer lists so emulate the behavior using a float initializer list.
#if _MSC_VER <= 1900
  DerivedScale() {}
  // Construct this object with a list of initializer lists. All list elements
  // are cast to float values to avoid ambiguous construction of a union-style
  // object that could take either std::initializer_list<float> or
  // std::initializer_list<int>.
  DerivedScale(std::initializer_list<std::initializer_list<float>> values) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSoperator_propertyDTh mht_0(mht_0_v, 206, "", "./tensorflow/lite/tools/optimize/operator_property.h", "DerivedScale");

    assert(values.size() == 3);
    std::vector<std::initializer_list<float>> items(values);
    for (auto& it : items[0]) {
      input_tensors.push_back(static_cast<int>(it));
    }
    for (auto& it : items[1]) {
      intermediate_tensors.push_back(static_cast<int>(it));
    }
    factors.assign(items[2]);
  }
#endif  // _MSC_VER <= 1900

  std::vector<int> input_tensors = {};
  std::vector<int> intermediate_tensors = {};
  // This is a list of extra factors that are not associated with any other
  // tensor.
  std::vector<float> factors = {};
};

struct TensorProperty {
  // per_axis also implies symmetric currently.
  bool per_axis = false;
  // TODO(jianlijianli): remove dimension index and read it from tensor instead.
  int per_axis_index = 0;
  bool symmetric = false;

  // Constraints.
  bool restriction = false;
  // scale/zero_point hardcoded.
  std::pair<float, int> restricted_value_int8 = {0.0f, 0};
  std::pair<float, int> restricted_value_int16 = {0.0f, 0};

  // Use derived scale.
  bool use_derived_scale = false;
  // The derived scale.
  DerivedScale derived_scale;

  // The number of bits for this tensor. It could be 8, 16, 32 or even not power
  // of two.
  int number_of_bits = 8;

  // Extend the range to power of two.
  bool extend_to_power_of_two = false;

  // State tensor.
  bool state_tensor = false;
};

struct OperatorProperty {
  // Is a quantized operations currently supported.
  bool quantizable = true;
  // Is a quantized operations currently supported for 16x8
  bool quantizable_int16 = true;
  // Op has arbitrary number of inputs, such as concat.
  bool arbitrary_inputs = false;
  // Op has arbitrary number of outputs, such as slice.
  bool arbitrary_outputs = false;
  // Input indexes -> input tensor property.
  // Must be topologically sorted since there are derived scales.
  std::vector<std::pair<int, TensorProperty>> inputs = {};
  // Output indexes -> output tensor property.
  std::vector<std::pair<int, TensorProperty>> outputs = {};
  // Bias indexes.
  // TODO(jianlijianli): remove this by putting biases into inputs as well since
  // we now can model "derived scale".
  std::vector<int> biases = {};

  // Intermediate indexes -> intermediate tensor property.
  std::vector<std::pair<int, TensorProperty>> intermediates = {};

  // Force output to reuse the same scale and zero point of input.
  bool restrict_same_input_output_scale = false;

  // Use same min of min and max of max for each group.
  // Incompatible with restrict_same_input_output_scale and restricted_value.
  // Currently it only supports scale pair of {input_index, output_index}.
  std::vector<std::vector<int>> restrict_scale = {};

  // Op version.
  int version = 1;

  // When we quantize activations into 16 bit and weights into 8 bit,
  // we want to quantize all inputs, including constant tensors,
  // for the operators like Add, Mul into 16-bit as well. The constant
  // inputs are quantized as weights and this variable indicates
  // that we want to do quantizations of these tensors as activations.
  bool quantize_input_as_activations = false;
};

// The op as well as it variants.
struct OpVariant {
  BuiltinOperator op_code;
  bool use_layer_norm = false;
  bool use_projection = false;
  bool use_peephole = false;
  // An attribute to indicate if quantization is supported for this Op.
  // This attribute is equivalent to the "quantizable" attribute in
  // "OperatorProperty". It added here since OpVariants peeks inside the Op and
  // determines its quantization related properties.
  bool is_quantizable = true;
};

OperatorProperty GetOperatorProperty(const ModelT* model, int subgraph_index,
                                     int op_index);
OperatorProperty GetOperatorProperty(OpVariant op_variant);

}  // namespace operator_property
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_OPERATOR_PROPERTY_H_
