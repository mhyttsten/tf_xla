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
#ifndef TENSORFLOW_LITE_TESTING_TEST_RUNNER_H_
#define TENSORFLOW_LITE_TESTING_TEST_RUNNER_H_
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
class MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh {
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
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh() {
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


#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace testing {

// This is the base class for processing test data. Each one of the virtual
// methods must be implemented to forward the data to the appropriate executor
// (e.g. TF Lite's interpreter, or the NNAPI).
class TestRunner {
 public:
  TestRunner() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_0(mht_0_v, 202, "", "./tensorflow/lite/testing/test_runner.h", "TestRunner");
}
  virtual ~TestRunner() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_1(mht_1_v, 206, "", "./tensorflow/lite/testing/test_runner.h", "~TestRunner");
}

  // Loads the given model, as a path relative to SetModelBaseDir().
  // DEPRECATED: use LoadModel with signature instead.
  virtual void LoadModel(const string& bin_file_path) = 0;
  // Loads the given model with signature specification.
  // Model path is relative to SetModelBaseDir().
  virtual void LoadModel(const string& bin_file_path,
                         const string& signature) = 0;

  // The following methods are supported by models with SignatureDef.
  //
  // Reshapes the tensors.
  // Keys are the input tensor names, values are csv string of the shape.
  virtual void ReshapeTensor(const string& name, const string& csv_values) = 0;

  // Sets the given tensor to some initial state, usually zero.
  virtual void ResetTensor(const std::string& name) = 0;

  // Reads the value of the output tensor and format it into a csv string.
  virtual string ReadOutput(const string& name) = 0;

  // Runs the model with signature.
  // Keys are the input tensor names, values are corresponding csv string.
  virtual void Invoke(const std::vector<std::pair<string, string>>& inputs) = 0;

  // Verifies that the contents of all outputs conform to the existing
  // expectations. Return true if there are no expectations or they are all
  // satisfied.
  // Keys are the input tensor names, values are corresponding csv string.
  virtual bool CheckResults(
      const std::vector<std::pair<string, string>>& expected_outputs,
      const std::vector<std::pair<string, string>>& expected_output_shapes) = 0;

  // Returns the list of output names in the loaded model for given signature.
  virtual std::vector<string> GetOutputNames() = 0;

  // Reserves memory for all tensors.
  virtual void AllocateTensors() = 0;

  // Sets the base path for loading models.
  void SetModelBaseDir(const string& path) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_2(mht_2_v, 251, "", "./tensorflow/lite/testing/test_runner.h", "SetModelBaseDir");

    model_base_dir_ = path;
    if (path[path.length() - 1] != '/') {
      model_base_dir_ += "/";
    }
  }

  // Returns the full path of a model.
  string GetFullPath(const string& path) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_3(mht_3_v, 263, "", "./tensorflow/lite/testing/test_runner.h", "GetFullPath");
 return model_base_dir_ + path; }

  // Gives an id to the next invocation to make error reporting more meaningful.
  void SetInvocationId(const string& id) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_4(mht_4_v, 270, "", "./tensorflow/lite/testing/test_runner.h", "SetInvocationId");
 invocation_id_ = id; }
  const string& GetInvocationId() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_5(mht_5_v, 274, "", "./tensorflow/lite/testing/test_runner.h", "GetInvocationId");
 return invocation_id_; }

  // Invalidates the test runner, preventing it from executing any further.
  void Invalidate(const string& error_message) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("error_message: \"" + error_message + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_6(mht_6_v, 281, "", "./tensorflow/lite/testing/test_runner.h", "Invalidate");

    std::cerr << error_message << std::endl;
    error_message_ = error_message;
  }
  bool IsValid() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_7(mht_7_v, 288, "", "./tensorflow/lite/testing/test_runner.h", "IsValid");
 return error_message_.empty(); }
  const string& GetErrorMessage() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_8(mht_8_v, 292, "", "./tensorflow/lite/testing/test_runner.h", "GetErrorMessage");
 return error_message_; }

  // Handles the overall success of this test runner. This will be true if all
  // invocations were successful.
  void SetOverallSuccess(bool value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_9(mht_9_v, 299, "", "./tensorflow/lite/testing/test_runner.h", "SetOverallSuccess");
 overall_success_ = value; }
  bool GetOverallSuccess() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_10(mht_10_v, 303, "", "./tensorflow/lite/testing/test_runner.h", "GetOverallSuccess");
 return overall_success_; }

 protected:
  // A helper to check of the given number of values is consistent with the
  // number of bytes in a tensor of type T. When incompatibles sizes are found,
  // the test runner is invalidated and false is returned.
  template <typename T>
  bool CheckSizes(size_t tensor_bytes, size_t num_values) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runnerDTh mht_11(mht_11_v, 313, "", "./tensorflow/lite/testing/test_runner.h", "CheckSizes");

    size_t num_tensor_elements = tensor_bytes / sizeof(T);
    if (num_tensor_elements != num_values) {
      Invalidate("Expected '" + std::to_string(num_tensor_elements) +
                 "' elements for a tensor, but only got '" +
                 std::to_string(num_values) + "'");
      return false;
    }
    return true;
  }

 private:
  string model_base_dir_;
  string invocation_id_;
  bool overall_success_ = true;

  string error_message_;
};

}  // namespace testing
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TESTING_TEST_RUNNER_H_
