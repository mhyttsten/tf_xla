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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc() {
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
#include <memory>
#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/logging.h"

ABSL_FLAG(std::string, model_file_path, "", "Path to the test model file.");
ABSL_FLAG(std::string, model_input_shapes, "",
          "List of different input shapes for testing, the input will "
          "resized for each one in order and tested. They Should be "
          "separated by : and each shape has dimensions separated by ,");
ABSL_FLAG(int, max_batch_size, -1,
          "Maximum batch size for a single run by hexagon.");
ABSL_FLAG(double, error_epsilon, 0.2,
          "Maximum error allowed while diffing the output.");

namespace tflite {
namespace {
// Returns a randomly generated data of size 'num_elements'.
std::vector<uint8_t> GetData(int num_elements) {
  std::vector<uint8_t> result(num_elements);
  std::random_device random_engine;
  std::uniform_int_distribution<uint32_t> distribution(0, 254);
  std::generate_n(result.data(), num_elements, [&]() {
    return static_cast<uint8_t>(distribution(random_engine));
  });
  return result;
}

// Returns the total number of elements.
int NumElements(const std::vector<int>& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc mht_0(mht_0_v, 227, "", "./tensorflow/lite/delegates/hexagon/builders/tests/batch_seq_config_test.cc", "NumElements");

  int num_elements = 1;
  for (int dim : shape) num_elements *= dim;
  return num_elements;
}

// Returns true if 'control' and 'exp' values match up to 'epsilon'
bool DiffOutput(const std::vector<float>& control,
                const std::vector<float>& exp, double epsilon) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/lite/delegates/hexagon/builders/tests/batch_seq_config_test.cc", "DiffOutput");

  if (control.size() != exp.size()) {
    TFLITE_LOG(ERROR) << "Mismatch size Expected" << control.size() << " got "
                      << exp.size();
    return false;
  }
  bool has_diff = false;
  for (int i = 0; i < control.size(); ++i) {
    if (abs(control[i] - exp[i]) > epsilon) {
      TFLITE_LOG(ERROR) << control[i] << " " << exp[i];
      has_diff = true;
    }
  }
  return !has_diff;
}

bool DiffOutput(const std::vector<float>& control,
                const std::vector<float>& exp) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc mht_2(mht_2_v, 258, "", "./tensorflow/lite/delegates/hexagon/builders/tests/batch_seq_config_test.cc", "DiffOutput");

  return DiffOutput(control, exp, absl::GetFlag(FLAGS_error_epsilon));
}
}  // namespace

class TestModel {
 public:
  TestModel() : delegate_(nullptr, [](TfLiteDelegate* delegate) {}) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc mht_3(mht_3_v, 268, "", "./tensorflow/lite/delegates/hexagon/builders/tests/batch_seq_config_test.cc", "TestModel");
}

  // Initialize the model by reading the model from file and build
  // interpreter.
  void Init() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc mht_4(mht_4_v, 275, "", "./tensorflow/lite/delegates/hexagon/builders/tests/batch_seq_config_test.cc", "Init");

    model_ = tflite::FlatBufferModel::BuildFromFile(
        absl::GetFlag(FLAGS_model_file_path).c_str());
    ASSERT_TRUE(model_ != nullptr);

    resolver_.reset(new ops::builtin::BuiltinOpResolver());
    InterpreterBuilder(*model_, *resolver_)(&interpreter_);
    ASSERT_TRUE(interpreter_ != nullptr);
  }

  // Add Hexagon delegate to the graph.
  void ApplyDelegate(int max_batch_size,
                     const std::vector<int>& input_batch_dimensions,
                     const std::vector<int>& output_batch_dimensions) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc mht_5(mht_5_v, 291, "", "./tensorflow/lite/delegates/hexagon/builders/tests/batch_seq_config_test.cc", "ApplyDelegate");

    TfLiteIntArray* input_batch_dim =
        TfLiteIntArrayCreate(input_batch_dimensions.size());
    TfLiteIntArray* output_batch_dim =
        TfLiteIntArrayCreate(output_batch_dimensions.size());
    for (int i = 0; i < input_batch_dimensions.size(); ++i)
      input_batch_dim->data[i] = input_batch_dimensions[i];
    for (int i = 0; i < output_batch_dimensions.size(); ++i)
      output_batch_dim->data[i] = output_batch_dimensions[i];
    ::TfLiteHexagonDelegateOptions options = {0};
    options.enable_dynamic_batch_size = true;
    options.max_batch_size = max_batch_size;
    options.input_batch_dimensions = input_batch_dim;
    options.output_batch_dimensions = output_batch_dim;
    TfLiteDelegate* delegate = TfLiteHexagonDelegateCreate(&options);
    ASSERT_TRUE(delegate != nullptr);
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        delegate, [](TfLiteDelegate* delegate) {
          TfLiteHexagonDelegateDelete(delegate);
        });
    ASSERT_TRUE(interpreter_->ModifyGraphWithDelegate(delegate_.get()) ==
                kTfLiteOk);
  }

  void Run(const std::vector<int>& input_shape,
           const std::vector<uint8_t>& input_data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc mht_6(mht_6_v, 319, "", "./tensorflow/lite/delegates/hexagon/builders/tests/batch_seq_config_test.cc", "Run");

    // Resize Inputs.
    auto interpreter_inputs = interpreter_->inputs();
    interpreter_->ResizeInputTensor(interpreter_inputs[0], input_shape);
    ASSERT_EQ(kTfLiteOk, interpreter_->AllocateTensors());

    TfLiteTensor* input_tensor =
        interpreter_->tensor(interpreter_->inputs()[0]);
    memcpy(input_tensor->data.raw, input_data.data(),
           input_data.size() * sizeof(uint8_t));

    ASSERT_EQ(kTfLiteOk, interpreter_->Invoke());
  }

  std::vector<float> GetOutput(int output_index) {
    auto* tensor = interpreter_->output_tensor(output_index);
    uint8_t* data = interpreter_->typed_output_tensor<uint8_t>(output_index);
    std::vector<float> result;
    result.resize(NumElements(tensor));
    const auto scale =
        reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params)
            ->scale->data[0];
    const auto zero_point =
        reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params)
            ->zero_point->data[0];
    for (int i = 0; i < result.size(); ++i) {
      result[i] = scale * (data[i] - zero_point);
    }
    return result;
  }

 private:
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<tflite::OpResolver> resolver_;
  std::unique_ptr<Interpreter> interpreter_;
};

std::vector<std::vector<int>> ParseInputShapes() {
  std::vector<string> str_input_shapes;
  benchmark::util::SplitAndParse(absl::GetFlag(FLAGS_model_input_shapes), ':',
                                 &str_input_shapes);
  std::vector<std::vector<int>> input_shapes(str_input_shapes.size());
  for (int i = 0; i < str_input_shapes.size(); ++i) {
    benchmark::util::SplitAndParse(str_input_shapes[i], ',', &input_shapes[i]);
  }
  return input_shapes;
}

TEST(HexagonDynamicBatch, MultipleResizes) {
  int num_failed_tests = 0;
  int num_test = 0;
  auto test_input_shapes = ParseInputShapes();
  auto default_model = std::make_unique<TestModel>();
  auto delegated_model = std::make_unique<TestModel>();
  default_model->Init();
  delegated_model->Init();
  delegated_model->ApplyDelegate(absl::GetFlag(FLAGS_max_batch_size), {0}, {0});
  for (const auto& input_shape : test_input_shapes) {
    const auto input = GetData(NumElements(input_shape));
    default_model->Run(input_shape, input);
    delegated_model->Run(input_shape, input);
    const auto default_output = default_model->GetOutput(0);
    const auto delegated_output = delegated_model->GetOutput(0);
    if (!DiffOutput(default_output, delegated_output)) {
      TFLITE_LOG(ERROR) << "Failed for input " << num_test;
      num_failed_tests++;
    }
    num_test++;
  }
  if (num_failed_tests == 0) {
    TFLITE_LOG(INFO) << "All Tests PASSED";
  } else {
    TFLITE_LOG(INFO) << "Failed " << num_failed_tests << " out of " << num_test;
  }
}
}  // namespace tflite

int main(int argc, char** argv) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSbatch_seq_config_testDTcc mht_7(mht_7_v, 400, "", "./tensorflow/lite/delegates/hexagon/builders/tests/batch_seq_config_test.cc", "main");

  ::tflite::LogToStderr();
  absl::ParseCommandLine(argc, argv);
  testing::InitGoogleTest();

  TfLiteHexagonInit();
  int return_val = RUN_ALL_TESTS();
  TfLiteHexagonTearDown();
  return return_val;
}
