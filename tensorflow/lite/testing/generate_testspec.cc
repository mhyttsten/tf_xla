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
class MHTracer_DTPStensorflowPSlitePStestingPSgenerate_testspecDTcc {
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
   MHTracer_DTPStensorflowPSlitePStestingPSgenerate_testspecDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPSgenerate_testspecDTcc() {
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

#include "tensorflow/lite/testing/generate_testspec.h"

#include <iostream>
#include <random>
#include <string>
#include <utility>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/lite/testing/join.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/testing/tf_driver.h"
#include "tensorflow/lite/testing/tflite_driver.h"

namespace tflite {
namespace testing {
namespace {

// Generates input name / value pairs according to given shape and distribution.
// Fills `out` with a pair of string, which the first element is input name and
// the second element is comma separated values in string.
template <typename T, typename RandomEngine, typename RandomDistribution>
void GenerateCsv(const string& name, const std::vector<int>& shape,
                 RandomEngine* engine, RandomDistribution distribution,
                 std::pair<string, string>* out) {
  std::vector<T> data =
      GenerateRandomTensor<T>(shape, [&]() { return distribution(*engine); });
  *out = std::make_pair(name, Join(data.data(), data.size(), ","));
}

// Generates random values for `input_layer` according to given value types and
// shapes.
// Fills `out` with a vector of string pairs, which the first element in the
// pair is the input name from `input_layer` and the second element is comma
// separated values in string.
template <typename RandomEngine>
std::vector<std::pair<string, string>> GenerateInputValues(
    RandomEngine* engine, const std::vector<string>& input_layer,
    const std::vector<string>& input_layer_type,
    const std::vector<string>& input_layer_shape) {
  std::vector<std::pair<string, string>> input_values;
  input_values.resize(input_layer.size());
  for (int i = 0; i < input_layer.size(); i++) {
    tensorflow::DataType type;
    CHECK(DataTypeFromString(input_layer_type[i], &type));
    auto shape = Split<int>(input_layer_shape[i], ",");
    const auto& name = input_layer[i];

    switch (type) {
      case tensorflow::DT_FLOAT:
        GenerateCsv<float>(name, shape, engine,
                           std::uniform_real_distribution<float>(-0.5, 0.5),
                           &input_values[i]);
        break;
      case tensorflow::DT_UINT8:
        GenerateCsv<uint8_t>(name, shape, engine,
                             std::uniform_int_distribution<uint32_t>(0, 255),
                             &input_values[i]);
        break;
      case tensorflow::DT_INT32:
        GenerateCsv<int32_t>(name, shape, engine,
                             std::uniform_int_distribution<int32_t>(-100, 100),
                             &input_values[i]);
        break;
      case tensorflow::DT_INT64:
        GenerateCsv<int64_t>(name, shape, engine,
                             std::uniform_int_distribution<int64_t>(-100, 100),
                             &input_values[i]);
        break;
      case tensorflow::DT_BOOL:
        GenerateCsv<int>(name, shape, engine,
                         std::uniform_int_distribution<int>(0, 1),
                         &input_values[i]);
        break;
      default:
        fprintf(stderr, "Unsupported type %d (%s) when generating testspec.\n",
                type, input_layer_type[i].c_str());
        input_values.clear();
        return input_values;
    }
  }
  return input_values;
}

bool GenerateTestSpecFromRunner(std::iostream& stream, int num_invocations,
                                const std::vector<string>& input_layer,
                                const std::vector<string>& input_layer_type,
                                const std::vector<string>& input_layer_shape,
                                const std::vector<string>& output_layer,
                                TestRunner* runner) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStestingPSgenerate_testspecDTcc mht_0(mht_0_v, 273, "", "./tensorflow/lite/testing/generate_testspec.cc", "GenerateTestSpecFromRunner");

  auto input_size = input_layer.size();
  if (input_layer_shape.size() != input_size ||
      input_layer_type.size() != input_size) {
    fprintf(stderr,
            "Input size not match. Expected %lu, got %lu input types, %lu "
            "input shapes.\n",
            input_size, input_layer_type.size(), input_layer_shape.size());
    return false;
  }

  stream << "reshape {\n";
  for (int i = 0; i < input_size; i++) {
    const auto& name = input_layer[i];
    const auto& shape = input_layer_shape[i];
    stream << "  input { key: \"" << name << "\" value: \"" << shape
           << "\" }\n";
  }
  stream << "}\n";

  // Generate inputs.
  std::mt19937 random_engine;
  for (int i = 0; i < num_invocations; ++i) {
    // Note that the input values are random, so each invocation will have a
    // different set.
    auto input_values = GenerateInputValues(
        &random_engine, input_layer, input_layer_type, input_layer_shape);
    if (input_values.empty()) {
      std::cerr << "Unable to generate input values for the TensorFlow model. "
                   "Make sure the correct values are defined for "
                   "input_layer, input_layer_type, and input_layer_shape."
                << std::endl;
      return false;
    }

    // Run TensorFlow.
    runner->Invoke(input_values);
    if (!runner->IsValid()) {
      std::cerr << runner->GetErrorMessage() << std::endl;
      return false;
    }

    // Write second part of test spec, with inputs and outputs.
    stream << "invoke {\n";
    for (const auto& entry : input_values) {
      stream << "  input { key: \"" << entry.first << "\" value: \""
             << entry.second << "\" }\n";
    }
    for (const auto& name : output_layer) {
      stream << "  output { key: \"" << name << "\" value: \""
             << runner->ReadOutput(name) << "\" }\n";
      if (!runner->IsValid()) {
        std::cerr << runner->GetErrorMessage() << std::endl;
        return false;
      }
    }
    stream << "}\n";
  }

  return true;
}

}  // namespace

bool GenerateTestSpecFromTensorflowModel(
    std::iostream& stream, const string& tensorflow_model_path,
    const string& tflite_model_path, int num_invocations,
    const std::vector<string>& input_layer,
    const std::vector<string>& input_layer_type,
    const std::vector<string>& input_layer_shape,
    const std::vector<string>& output_layer) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tensorflow_model_path: \"" + tensorflow_model_path + "\"");
   mht_1_v.push_back("tflite_model_path: \"" + tflite_model_path + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSgenerate_testspecDTcc mht_1(mht_1_v, 348, "", "./tensorflow/lite/testing/generate_testspec.cc", "GenerateTestSpecFromTensorflowModel");

  CHECK_EQ(input_layer.size(), input_layer_type.size());
  CHECK_EQ(input_layer.size(), input_layer_shape.size());

  // Invoke tensorflow model.
  TfDriver runner(input_layer, input_layer_type, input_layer_shape,
                  output_layer);
  if (!runner.IsValid()) {
    std::cerr << runner.GetErrorMessage() << std::endl;
    return false;
  }

  runner.LoadModel(tensorflow_model_path);
  if (!runner.IsValid()) {
    std::cerr << runner.GetErrorMessage() << std::endl;
    return false;
  }
  // Write first part of test spec, defining model and input shapes.
  stream << "load_model: " << tflite_model_path << "\n";
  return GenerateTestSpecFromRunner(stream, num_invocations, input_layer,
                                    input_layer_type, input_layer_shape,
                                    output_layer, &runner);
}

bool GenerateTestSpecFromTFLiteModel(
    std::iostream& stream, const string& tflite_model_path, int num_invocations,
    const std::vector<string>& input_layer,
    const std::vector<string>& input_layer_type,
    const std::vector<string>& input_layer_shape,
    const std::vector<string>& output_layer) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tflite_model_path: \"" + tflite_model_path + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSgenerate_testspecDTcc mht_2(mht_2_v, 381, "", "./tensorflow/lite/testing/generate_testspec.cc", "GenerateTestSpecFromTFLiteModel");

  TfLiteDriver runner;
  runner.LoadModel(tflite_model_path);
  if (!runner.IsValid()) {
    std::cerr << runner.GetErrorMessage() << std::endl;
    return false;
  }
  runner.AllocateTensors();
  return GenerateTestSpecFromRunner(stream, num_invocations, input_layer,
                                    input_layer_type, input_layer_shape,
                                    output_layer, &runner);
}

}  // namespace testing
}  // namespace tflite
