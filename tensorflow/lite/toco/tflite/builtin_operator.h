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
#ifndef TENSORFLOW_LITE_TOCO_TFLITE_BUILTIN_OPERATOR_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_BUILTIN_OPERATOR_H_
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
class MHTracer_DTPStensorflowPSlitePStocoPStflitePSbuiltin_operatorDTh {
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
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSbuiltin_operatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStflitePSbuiltin_operatorDTh() {
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


#include "absl/memory/memory.h"
#include "tensorflow/lite/toco/tflite/operator.h"

namespace toco {

namespace tflite {

// Builtin operators have special TF Lite objects describing their options.
// This class has the boilerplate code for creating those.
//
// Template arguments:
//   - T1 must derive from ::toco::Operator.
//   - T2 must be one of TF Lite's objects defining Builtin Options, such as
//     ::tflite::Conv2DOptions.
template <typename T1, typename T2, ::tflite::BuiltinOptions TfLiteEnum>
class BuiltinOperator : public BaseOperator {
 public:
  using TocoOperator = T1;
  using TfLiteOptions = T2;

  BuiltinOperator(::tflite::BuiltinOperator op, OperatorType type)
      : BaseOperator(::tflite::EnumNameBuiltinOperator(op), type),
        builtin_op_(op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSbuiltin_operatorDTh mht_0(mht_0_v, 209, "", "./tensorflow/lite/toco/tflite/builtin_operator.h", "BuiltinOperator");
}

  // Build the configuration object in the given flatbuffer builder. Return
  // its offset.
  virtual flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const = 0;

  // Read options from the TF Lite object and set the corresponding values in
  // the tf.mini operator.
  virtual void ReadOptions(const TfLiteOptions& opt,
                           TocoOperator* op) const = 0;

  Options Serialize(const Operator& op,
                    flatbuffers::FlatBufferBuilder* builder) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSbuiltin_operatorDTh mht_1(mht_1_v, 226, "", "./tensorflow/lite/toco/tflite/builtin_operator.h", "Serialize");

    auto options = WriteOptions(static_cast<const TocoOperator&>(op), builder);
    return Options::Builtin(TfLiteEnum, options.Union());
  }

  std::unique_ptr<Operator> Deserialize(
      const BuiltinOptions* builtin_options,
      const CustomOptions* custom_options) const override {
    auto op = absl::make_unique<TocoOperator>();
    auto* options = static_cast<const TfLiteOptions*>(builtin_options);
    if (options) {
      ReadOptions(*options, op.get());
    }
    return std::unique_ptr<Operator>(op.release());
  }

  int GetVersion(const OperatorSignature& op_signature) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSbuiltin_operatorDTh mht_2(mht_2_v, 245, "", "./tensorflow/lite/toco/tflite/builtin_operator.h", "GetVersion");

    return ::tflite::GetBuiltinOperatorVersion(
        GetVersioningOpSig(builtin_op_, op_signature));
  }

  ::tflite::BuiltinOperator builtin_op() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSbuiltin_operatorDTh mht_3(mht_3_v, 253, "", "./tensorflow/lite/toco/tflite/builtin_operator.h", "builtin_op");
 return builtin_op_; }

 private:
  const ::tflite::BuiltinOperator builtin_op_;
};

}  // namespace tflite

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_BUILTIN_OPERATOR_H_
