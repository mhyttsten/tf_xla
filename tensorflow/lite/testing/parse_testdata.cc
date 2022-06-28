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
class MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc {
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
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc() {
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
// Parses tflite example input data.
// Format is ASCII
// TODO(aselle): Switch to protobuf, but the android team requested a simple
// ASCII file.
#include "tensorflow/lite/testing/parse_testdata.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <utility>
#include <vector>

#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/testing/message.h"
#include "tensorflow/lite/testing/split.h"

namespace tflite {
namespace testing {
namespace {

const char kDefaultSignatureKey[] = "serving_default";

// Fatal error if parse error occurs
#define PARSE_CHECK_EQ(filename, current_line, x, y)                         \
  if ((x) != (y)) {                                                          \
    fprintf(stderr, "Parse Error @ %s:%d\n  File %s\n  Line %d, %s != %s\n", \
            __FILE__, __LINE__, filename, current_line + 1, #x, #y);         \
    return kTfLiteError;                                                     \
  }

// Breakup a "," delimited line into a std::vector<std::string>.
// This is extremely inefficient, and just used for testing code.
// TODO(aselle): replace with absl when we use it.
std::vector<std::string> ParseLine(const std::string& line) {
  size_t pos = 0;
  std::vector<std::string> elements;
  while (true) {
    size_t end = line.find(',', pos);
    if (end == std::string::npos) {
      elements.push_back(line.substr(pos));
      break;
    } else {
      elements.push_back(line.substr(pos, end - pos));
    }
    pos = end + 1;
  }
  return elements;
}

}  // namespace

// Given a `filename`, produce a vector of Examples corresponding
// to test cases that can be applied to a tflite model.
TfLiteStatus ParseExamples(const char* filename,
                           std::vector<Example>* examples) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + (filename == nullptr ? std::string("nullptr") : std::string((char*)filename)) + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_0(mht_0_v, 243, "", "./tensorflow/lite/testing/parse_testdata.cc", "ParseExamples");

  std::ifstream fp(filename);
  if (!fp.good()) {
    fprintf(stderr, "Could not read '%s'\n", filename);
    return kTfLiteError;
  }
  std::string str((std::istreambuf_iterator<char>(fp)),
                  std::istreambuf_iterator<char>());
  size_t pos = 0;

  // \n and , delimit parse a file.
  std::vector<std::vector<std::string>> csv;
  while (true) {
    size_t end = str.find('\n', pos);

    if (end == std::string::npos) {
      csv.emplace_back(ParseLine(str.substr(pos)));
      break;
    }
    csv.emplace_back(ParseLine(str.substr(pos, end - pos)));
    pos = end + 1;
  }

  int current_line = 0;
  PARSE_CHECK_EQ(filename, current_line, csv[0][0], "test_cases");
  int example_count = std::stoi(csv[0][1]);
  current_line++;

  auto parse_tensor = [&filename, &current_line,
                       &csv](FloatTensor* tensor_ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_1(mht_1_v, 275, "", "./tensorflow/lite/testing/parse_testdata.cc", "lambda");

    PARSE_CHECK_EQ(filename, current_line, csv[current_line][0], "dtype");
    current_line++;
    // parse shape
    PARSE_CHECK_EQ(filename, current_line, csv[current_line][0], "shape");
    size_t elements = 1;
    FloatTensor& tensor = *tensor_ptr;

    for (size_t i = 1; i < csv[current_line].size(); i++) {
      const auto& shape_part_to_parse = csv[current_line][i];
      if (shape_part_to_parse.empty()) {
        // Case of a 0-dimensional shape
        break;
      }
      int shape_part = std::stoi(shape_part_to_parse);
      elements *= shape_part;
      tensor.shape.push_back(shape_part);
    }
    current_line++;
    // parse data
    PARSE_CHECK_EQ(filename, current_line, csv[current_line].size() - 1,
                   elements);
    for (size_t i = 1; i < csv[current_line].size(); i++) {
      tensor.flat_data.push_back(std::stof(csv[current_line][i]));
    }
    current_line++;

    return kTfLiteOk;
  };

  for (int example_idx = 0; example_idx < example_count; example_idx++) {
    Example example;
    PARSE_CHECK_EQ(filename, current_line, csv[current_line][0], "inputs");
    int inputs = std::stoi(csv[current_line][1]);
    current_line++;
    // parse dtype
    for (int input_index = 0; input_index < inputs; input_index++) {
      example.inputs.push_back(FloatTensor());
      TF_LITE_ENSURE_STATUS(parse_tensor(&example.inputs.back()));
    }

    PARSE_CHECK_EQ(filename, current_line, csv[current_line][0], "outputs");
    int outputs = std::stoi(csv[current_line][1]);
    current_line++;
    for (int input_index = 0; input_index < outputs; input_index++) {
      example.outputs.push_back(FloatTensor());
      TF_LITE_ENSURE_STATUS(parse_tensor(&example.outputs.back()));
    }
    examples->emplace_back(example);
  }
  return kTfLiteOk;
}

TfLiteStatus FeedExample(tflite::Interpreter* interpreter,
                         const Example& example) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_2(mht_2_v, 332, "", "./tensorflow/lite/testing/parse_testdata.cc", "FeedExample");

  // Resize inputs to match example & allocate.
  for (size_t i = 0; i < interpreter->inputs().size(); i++) {
    int input_index = interpreter->inputs()[i];

    TF_LITE_ENSURE_STATUS(
        interpreter->ResizeInputTensor(input_index, example.inputs[i].shape));
  }
  TF_LITE_ENSURE_STATUS(interpreter->AllocateTensors());
  // Copy data into tensors.
  for (size_t i = 0; i < interpreter->inputs().size(); i++) {
    int input_index = interpreter->inputs()[i];
    if (float* data = interpreter->typed_tensor<float>(input_index)) {
      for (size_t idx = 0; idx < example.inputs[i].flat_data.size(); idx++) {
        data[idx] = example.inputs[i].flat_data[idx];
      }
    } else if (int32_t* data =
                   interpreter->typed_tensor<int32_t>(input_index)) {
      for (size_t idx = 0; idx < example.inputs[i].flat_data.size(); idx++) {
        data[idx] = example.inputs[i].flat_data[idx];
      }
    } else if (int64_t* data =
                   interpreter->typed_tensor<int64_t>(input_index)) {
      for (size_t idx = 0; idx < example.inputs[i].flat_data.size(); idx++) {
        data[idx] = example.inputs[i].flat_data[idx];
      }
    } else {
      fprintf(stderr, "input[%zu] was not float or int data\n", i);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus CheckOutputs(tflite::Interpreter* interpreter,
                          const Example& example) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_3(mht_3_v, 370, "", "./tensorflow/lite/testing/parse_testdata.cc", "CheckOutputs");

  constexpr double kRelativeThreshold = 1e-2f;
  constexpr double kAbsoluteThreshold = 1e-4f;

  ErrorReporter* context = DefaultErrorReporter();
  int model_outputs = interpreter->outputs().size();
  TF_LITE_ENSURE_EQ(context, model_outputs, example.outputs.size());
  for (size_t i = 0; i < interpreter->outputs().size(); i++) {
    bool tensors_differ = false;
    int output_index = interpreter->outputs()[i];
    if (const float* data = interpreter->typed_tensor<float>(output_index)) {
      for (size_t idx = 0; idx < example.outputs[i].flat_data.size(); idx++) {
        float computed = data[idx];
        float reference = example.outputs[0].flat_data[idx];
        float diff = std::abs(computed - reference);
        // For very small numbers, try absolute error, otherwise go with
        // relative.
        bool local_tensors_differ =
            std::abs(reference) < kRelativeThreshold
                ? diff > kAbsoluteThreshold
                : diff > kRelativeThreshold * std::abs(reference);
        if (local_tensors_differ) {
          fprintf(stdout, "output[%zu][%zu] did not match %f vs reference %f\n",
                  i, idx, data[idx], reference);
          tensors_differ = local_tensors_differ;
        }
      }
    } else if (const int32_t* data =
                   interpreter->typed_tensor<int32_t>(output_index)) {
      for (size_t idx = 0; idx < example.outputs[i].flat_data.size(); idx++) {
        int32_t computed = data[idx];
        int32_t reference = example.outputs[0].flat_data[idx];
        if (std::abs(computed - reference) > 0) {
          fprintf(stderr, "output[%zu][%zu] did not match %d vs reference %d\n",
                  i, idx, computed, reference);
          tensors_differ = true;
        }
      }
    } else if (const int64_t* data =
                   interpreter->typed_tensor<int64_t>(output_index)) {
      for (size_t idx = 0; idx < example.outputs[i].flat_data.size(); idx++) {
        int64_t computed = data[idx];
        int64_t reference = example.outputs[0].flat_data[idx];
        if (std::abs(computed - reference) > 0) {
          fprintf(stderr,
                  "output[%zu][%zu] did not match %" PRId64
                  " vs reference %" PRId64 "\n",
                  i, idx, computed, reference);
          tensors_differ = true;
        }
      }
    } else {
      fprintf(stderr, "output[%zu] was not float or int data\n", i);
      return kTfLiteError;
    }
    fprintf(stderr, "\n");
    if (tensors_differ) return kTfLiteError;
  }
  return kTfLiteOk;
}

// Processes Protobuf map<string, string> like message.
// Supports format of
// field_name {key: "KEY1" value: "VAL1"}
// field_name {key: "KEY2" value: "VAL2"}
// field_name {key: "KEY3" value: "VAL3"}
//
// for field `map<string, string> field_name = TAG;`
//
// Note: The parent of this field should track the ownership of the repeated
// field. By calling KvMap::Finish() means a new entry is added to the map
// instead of finish parsing of the whole map.
class KvMap : public Message, public std::vector<std::pair<string, string>> {
 public:
  void SetField(const std::string& name, const std::string& value) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   mht_4_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_4(mht_4_v, 449, "", "./tensorflow/lite/testing/parse_testdata.cc", "SetField");

    if (name == "key") {
      key_ = value;
    } else if (name == "value") {
      value_ = value;
    }
  }
  void Finish() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_5(mht_5_v, 459, "", "./tensorflow/lite/testing/parse_testdata.cc", "Finish");

    push_back(std::make_pair(key_, value_));
    key_.clear();
    value_.clear();
  }

 private:
  string key_;
  string value_;
};

// Processes an 'invoke' message, triggering execution of the test runner, as
// well as verification of outputs. An 'invoke' message looks like:
//   invoke {
//     id: "xyz"
//     input { key: "a" value: "1,2,1,1,1,2,3,4"}
//     input { key: "b" value: "1,2,1,1,1,2,3,4"}
//     output { key: "x" value: "4,5,6"}
//     output { key: "y" value: "14,15,16"}
//     output_shape { key: "x" value: "3"}
//     output_shape { key: "y" value: "1,3"}
//   }
class Invoke : public Message {
 public:
  explicit Invoke(TestRunner* test_runner) : test_runner_(test_runner) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_6(mht_6_v, 486, "", "./tensorflow/lite/testing/parse_testdata.cc", "Invoke");
}

  void SetField(const std::string& name, const std::string& value) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   mht_7_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_7(mht_7_v, 493, "", "./tensorflow/lite/testing/parse_testdata.cc", "SetField");

    if (name == "id") {
      test_runner_->SetInvocationId(value);
    }
  }

  Message* AddChild(const std::string& s) override {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_8(mht_8_v, 503, "", "./tensorflow/lite/testing/parse_testdata.cc", "AddChild");

    if (s == "input") {
      return MaybeInitializeChild(&inputs_);
    } else if (s == "output") {
      return MaybeInitializeChild(&expected_outputs_);
    } else if (s == "output_shape") {
      return MaybeInitializeChild(&expected_output_shapes_);
    }
    return nullptr;
  }

  // Invokes the test runner and checks expectations.
  void Finish() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_9(mht_9_v, 518, "", "./tensorflow/lite/testing/parse_testdata.cc", "Finish");

    using VectorT = std::vector<std::pair<string, string>>;
    test_runner_->Invoke(inputs_ ? *inputs_ : VectorT());
    test_runner_->CheckResults(
        expected_outputs_ ? *expected_outputs_ : VectorT(),
        expected_output_shapes_ ? *expected_output_shapes_ : VectorT());
  }

 private:
  // Checks whether `*child` is initialized and return the message pointer.
  // Initializes and owns it if it's not initialized.
  Message* MaybeInitializeChild(KvMap** child) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_10(mht_10_v, 532, "", "./tensorflow/lite/testing/parse_testdata.cc", "MaybeInitializeChild");

    if (*child == nullptr) {
      *child = new KvMap;
      Store(*child);
    }
    return *child;
  }

  TestRunner* test_runner_;

  KvMap* inputs_ = nullptr;
  KvMap* expected_outputs_ = nullptr;
  KvMap* expected_output_shapes_ = nullptr;
};

// Process an 'reshape' message, triggering resizing of the input tensors via
// the test runner. A 'reshape' message looks like:
//   reshape {
//     input { key: "a" value: "1,2,1,1,1,2,3,4"}
//     input { key: "b" value: "1,2,1,1,1,2,3,4"}
//   }
class Reshape : public Message {
 public:
  explicit Reshape(TestRunner* test_runner) : test_runner_(test_runner) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_11(mht_11_v, 558, "", "./tensorflow/lite/testing/parse_testdata.cc", "Reshape");
}

  Message* AddChild(const std::string& s) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_12(mht_12_v, 564, "", "./tensorflow/lite/testing/parse_testdata.cc", "AddChild");

    if (s != "input") return nullptr;
    if (input_shapes_ == nullptr) {
      input_shapes_ = new KvMap;
      Store(input_shapes_);
    }
    return input_shapes_;
  }

  // Reshapes tensors.
  void Finish() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_13(mht_13_v, 577, "", "./tensorflow/lite/testing/parse_testdata.cc", "Finish");

    if (!input_shapes_) return;
    for (const auto& item : *input_shapes_) {
      test_runner_->ReshapeTensor(item.first, item.second);
    }
  }

 private:
  TestRunner* test_runner_;

  KvMap* input_shapes_ = nullptr;
};

// This is the top-level message in a test file.
class TestData : public Message {
 public:
  explicit TestData(TestRunner* test_runner)
      : test_runner_(test_runner), num_invocations_(0), max_invocations_(-1) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_14(mht_14_v, 597, "", "./tensorflow/lite/testing/parse_testdata.cc", "TestData");
}
  void SetMaxInvocations(int max) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_15(mht_15_v, 601, "", "./tensorflow/lite/testing/parse_testdata.cc", "SetMaxInvocations");
 max_invocations_ = max; }
  void SetField(const std::string& name, const std::string& value) override {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   mht_16_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_16(mht_16_v, 607, "", "./tensorflow/lite/testing/parse_testdata.cc", "SetField");

    if (name == "load_model") {
      test_runner_->LoadModel(value, kDefaultSignatureKey);
    } else if (name == "init_state") {
      test_runner_->AllocateTensors();
      for (const auto& name : Split<string>(value, ",")) {
        test_runner_->ResetTensor(name);
      }
    }
  }
  Message* AddChild(const std::string& s) override {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_17(mht_17_v, 621, "", "./tensorflow/lite/testing/parse_testdata.cc", "AddChild");

    if (s == "invoke") {
      test_runner_->AllocateTensors();
      if (max_invocations_ == -1 || num_invocations_ < max_invocations_) {
        ++num_invocations_;
        return Store(new Invoke(test_runner_));
      } else {
        return nullptr;
      }
    } else if (s == "reshape") {
      return Store(new Reshape(test_runner_));
    }
    return nullptr;
  }

 private:
  TestRunner* test_runner_;
  int num_invocations_;
  int max_invocations_;
};

bool ParseAndRunTests(std::istream* input, TestRunner* test_runner,
                      int max_invocations) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStestingPSparse_testdataDTcc mht_18(mht_18_v, 646, "", "./tensorflow/lite/testing/parse_testdata.cc", "ParseAndRunTests");

  TestData test_data(test_runner);
  test_data.SetMaxInvocations(max_invocations);
  Message::Read(input, &test_data);
  return test_runner->IsValid() && test_runner->GetOverallSuccess();
}

}  // namespace testing
}  // namespace tflite
