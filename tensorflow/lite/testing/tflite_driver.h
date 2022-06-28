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
#ifndef TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_
#define TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_
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
class MHTracer_DTPStensorflowPSlitePStestingPStflite_driverDTh {
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
   MHTracer_DTPStensorflowPSlitePStestingPStflite_driverDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPStflite_driverDTh() {
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


#include <map>
#include <memory>

#include "tensorflow/lite/c/common.h"
#if !defined(__APPLE__)
#include "tensorflow/lite/delegates/flex/delegate.h"
#endif
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/signature_runner.h"
#include "tensorflow/lite/testing/test_runner.h"

namespace tflite {
namespace testing {

// A test runner that feeds inputs into TF Lite and verifies its outputs.
class TfLiteDriver : public TestRunner {
 public:
  enum class DelegateType {
    kNone,
    kNnapi,
    kGpu,
    kFlex,
  };

  // Initialize the global test delegate providers from commandline arguments
  // and returns true if successful.
  static bool InitTestDelegateProviders(int* argc, const char** argv);

  /**
   * Creates a new TfLiteDriver
   * @param  delegate         The (optional) delegate to use.
   * @param  reference_kernel Whether to use the builtin reference kernel
   * ops.
   */
  explicit TfLiteDriver(DelegateType delegate_type = DelegateType::kNone,
                        bool reference_kernel = false);
  ~TfLiteDriver() override;

  void LoadModel(const string& bin_file_path) override;
  void LoadModel(const string& bin_file_path, const string& signature) override;

  void ReshapeTensor(const string& name, const string& csv_values) override;
  void ResetTensor(const std::string& name) override;
  string ReadOutput(const string& name) override;
  void Invoke(const std::vector<std::pair<string, string>>& inputs) override;
  bool CheckResults(
      const std::vector<std::pair<string, string>>& expected_outputs,
      const std::vector<std::pair<string, string>>& expected_output_shapes)
      override;
  std::vector<string> GetOutputNames() override;

  void AllocateTensors() override;
  void SetThreshold(double relative_threshold, double absolute_threshold);
  void SetQuantizationErrorMultiplier(int quantization_error_multiplier);

 protected:
  Interpreter::TfLiteDelegatePtr delegate_;

 private:
  void SetInput(const string& name, const string& csv_values);
  void SetExpectation(const string& name, const string& csv_values);
  void SetShapeExpectation(const string& name, const string& csv_values);
  void DeallocateStringTensor(TfLiteTensor* t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStestingPStflite_driverDTh mht_0(mht_0_v, 252, "", "./tensorflow/lite/testing/tflite_driver.h", "DeallocateStringTensor");

    if (t) {
      free(t->data.raw);
      t->data.raw = nullptr;
    }
  }
  void AllocateStringTensor(int id, size_t num_bytes, TfLiteTensor* t) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStestingPStflite_driverDTh mht_1(mht_1_v, 261, "", "./tensorflow/lite/testing/tflite_driver.h", "AllocateStringTensor");

    t->data.raw = reinterpret_cast<char*>(malloc(num_bytes));
    t->bytes = num_bytes;
    tensors_to_deallocate_[id] = t;
  }

  void ResetLSTMStateTensors();
  // Formats tensor value to string in csv format.
  string TensorValueToCsvString(const TfLiteTensor* tensor);

  class DataExpectation;
  class ShapeExpectation;

  std::map<string, uint32_t> signature_inputs_;
  std::map<string, uint32_t> signature_outputs_;
  std::unique_ptr<OpResolver> resolver_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
  std::map<int, std::unique_ptr<DataExpectation>> expected_output_;
  std::map<int, std::unique_ptr<ShapeExpectation>> expected_output_shape_;
  SignatureRunner* signature_runner_ = nullptr;
  bool must_allocate_tensors_ = true;
  std::map<int, TfLiteTensor*> tensors_to_deallocate_;
  double relative_threshold_;
  double absolute_threshold_;
  int quantization_error_multiplier_;
};

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_
