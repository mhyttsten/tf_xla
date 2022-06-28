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
#ifndef TENSORFLOW_LITE_TESTING_TFLITE_DIFF_FLAGS_H_
#define TENSORFLOW_LITE_TESTING_TFLITE_DIFF_FLAGS_H_
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
class MHTracer_DTPStensorflowPSlitePStestingPStflite_diff_flagsDTh {
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
   MHTracer_DTPStensorflowPSlitePStestingPStflite_diff_flagsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPStflite_diff_flagsDTh() {
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


#include <cstring>

#include "absl/strings/match.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/testing/tflite_diff_util.h"
#include "tensorflow/lite/testing/tflite_driver.h"

namespace tflite {
namespace testing {

DiffOptions ParseTfliteDiffFlags(int* argc, char** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStestingPStflite_diff_flagsDTh mht_0(mht_0_v, 198, "", "./tensorflow/lite/testing/tflite_diff_flags.h", "ParseTfliteDiffFlags");

  struct {
    string tensorflow_model;
    string tflite_model;
    string input_layer;
    string input_layer_type;
    string input_layer_shape;
    string output_layer;
    int32_t num_runs_per_pass = 100;
    string delegate_name;
    string reference_tflite_model;
  } values;

  std::string delegate_name;
  std::vector<tensorflow::Flag> flags = {
      tensorflow::Flag("tensorflow_model", &values.tensorflow_model,
                       "Path of tensorflow model."),
      tensorflow::Flag("tflite_model", &values.tflite_model,
                       "Path of tensorflow lite model."),
      tensorflow::Flag("input_layer", &values.input_layer,
                       "Names of input tensors, separated by comma. Example: "
                       "input_1,input_2."),
      tensorflow::Flag("input_layer_type", &values.input_layer_type,
                       "Data types of input tensors, separated by comma. "
                       "Example: float,int."),
      tensorflow::Flag(
          "input_layer_shape", &values.input_layer_shape,
          "Shapes of input tensors, separated by colon. Example: 1,3,4,1:2."),
      tensorflow::Flag("output_layer", &values.output_layer,
                       "Names of output tensors, separated by comma. Example: "
                       "output_1,output_2."),
      tensorflow::Flag("num_runs_per_pass", &values.num_runs_per_pass,
                       "[optional] Number of full runs in each pass."),
      tensorflow::Flag("delegate", &values.delegate_name,
                       "[optional] Delegate to use for executing ops. Must be "
                       "`{\"\", NNAPI, GPU, FLEX}`"),
      tensorflow::Flag("reference_tflite_model", &values.reference_tflite_model,
                       "[optional] Path of the TensorFlow Lite model to "
                       "compare inference results against the model given in "
                       "`tflite_model`."),
  };

  bool no_inputs = *argc == 1;
  bool success = tensorflow::Flags::Parse(argc, argv, flags);
  if (!success || no_inputs || (*argc == 2 && !strcmp(argv[1], "--helpfull"))) {
    fprintf(stderr, "%s", tensorflow::Flags::Usage(argv[0], flags).c_str());
    return {};
  } else if (values.tensorflow_model.empty() || values.tflite_model.empty() ||
             values.input_layer.empty() || values.input_layer_type.empty() ||
             values.input_layer_shape.empty() || values.output_layer.empty()) {
    fprintf(stderr, "%s", tensorflow::Flags::Usage(argv[0], flags).c_str());
    return {};
  }

  TfLiteDriver::DelegateType delegate = TfLiteDriver::DelegateType::kNone;
  if (!values.delegate_name.empty()) {
    if (absl::EqualsIgnoreCase(values.delegate_name, "nnapi")) {
      delegate = TfLiteDriver::DelegateType::kNnapi;
    } else if (absl::EqualsIgnoreCase(values.delegate_name, "gpu")) {
      delegate = TfLiteDriver::DelegateType::kGpu;
    } else if (absl::EqualsIgnoreCase(values.delegate_name, "flex")) {
      delegate = TfLiteDriver::DelegateType::kFlex;
    } else {
      fprintf(stderr, "%s", tensorflow::Flags::Usage(argv[0], flags).c_str());
      return {};
    }
  }

  return {values.tensorflow_model,
          values.tflite_model,
          Split<string>(values.input_layer, ","),
          Split<string>(values.input_layer_type, ","),
          Split<string>(values.input_layer_shape, ":"),
          Split<string>(values.output_layer, ","),
          values.num_runs_per_pass,
          delegate,
          values.reference_tflite_model};
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_TFLITE_DIFF_FLAGS_H_
