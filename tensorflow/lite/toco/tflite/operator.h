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
#ifndef TENSORFLOW_LITE_TOCO_TFLITE_OPERATOR_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_OPERATOR_H_
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
class MHTracer_DTPStensorflowPSlitePStocoPStflitePSoperatorDTh {
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
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSoperatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStflitePSoperatorDTh() {
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


#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/tools/versioning/op_version.h"

namespace toco {

namespace tflite {

class BaseOperator;

// Return a map contained all know TF Lite Operators, keyed by their names.
// TODO(ycling): The pattern to propagate parameters (e.g. enable_select_tf_ops)
// is ugly here. Consider refactoring.
std::map<std::string, std::unique_ptr<BaseOperator>> BuildOperatorByNameMap(
    bool enable_select_tf_ops = false);

// Return a map contained all know TF Lite Operators, keyed by the type of
// their tf.mini counterparts.
std::map<OperatorType, std::unique_ptr<BaseOperator>> BuildOperatorByTypeMap(
    bool enable_select_tf_ops = false);

// Write the custom option FlexBuffer with a serialized TensorFlow NodeDef
// for a Flex op.
std::unique_ptr<flexbuffers::Builder> WriteFlexOpOptions(
    const std::string& tensorflow_node_def);

// These are the flatbuffer types for custom and builtin options.
using CustomOptions = flatbuffers::Vector<uint8_t>;
using BuiltinOptions = void;

// A simple wrapper around the flatbuffer objects used to describe options that
// configure operators.
struct Options {
  // Build custom options.
  static Options Custom(flatbuffers::Offset<CustomOptions> offset) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSoperatorDTh mht_0(mht_0_v, 223, "", "./tensorflow/lite/toco/tflite/operator.h", "Custom");

    return {::tflite::BuiltinOptions_NONE, 0, offset};
  }

  // Build builtin options of the given type.
  static Options Builtin(::tflite::BuiltinOptions type,
                         flatbuffers::Offset<BuiltinOptions> offset) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSoperatorDTh mht_1(mht_1_v, 232, "", "./tensorflow/lite/toco/tflite/operator.h", "Builtin");

    return {type, offset, 0};
  }

  ::tflite::BuiltinOptions type;
  flatbuffers::Offset<BuiltinOptions> builtin;
  flatbuffers::Offset<CustomOptions> custom;
};

// A BaseOperator encapsulates the relationship between operators in tf.mini
// and TF lite, and provides methods for converting between those two formats.
class BaseOperator {
 public:
  // Build an operator with the given TF Lite name and tf.mini type.
  BaseOperator(const std::string& name, OperatorType type)
      : name_(name), type_(type) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSoperatorDTh mht_2(mht_2_v, 251, "", "./tensorflow/lite/toco/tflite/operator.h", "BaseOperator");
}
  virtual ~BaseOperator() = default;

  std::string name() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSoperatorDTh mht_3(mht_3_v, 257, "", "./tensorflow/lite/toco/tflite/operator.h", "name");
 return name_; }
  OperatorType type() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSoperatorDTh mht_4(mht_4_v, 261, "", "./tensorflow/lite/toco/tflite/operator.h", "type");
 return type_; }

  // Given a tf.mini operator, create the corresponding flatbuffer options and
  // return their offsets.
  virtual Options Serialize(const Operator& op,
                            flatbuffers::FlatBufferBuilder* builder) const = 0;

  // Read TF Lite options and create the appropriate tf.mini operator.
  virtual std::unique_ptr<Operator> Deserialize(
      const BuiltinOptions* builtin_options,
      const CustomOptions* custom_options) const = 0;

  // Get the op version using the OperatorSignature.
  // The function needs to be overridden to return the op version based on the
  // parameters. Note:
  // * The first version for each op should be 1 (to be consistent with the
  //   default value in Flatbuffer. `return 1;` is okay for newly implemented
  //   ops.
  // * When multiple versions are defined for an op, this function could be
  //   overridden. (See example in `operator_test.cc` and
  //   'tools/versioning/op_version.cc`)
  virtual int GetVersion(const OperatorSignature& op_signature) const = 0;

  // Given a Toco `Operator`, return a list of booleans indicating the op
  // mutates which input variables.
  // * If the op mutates any input variables, it should return a list of bool
  //   with the same length as inputs.
  // * Otherwise, it will return an empty list.
  virtual std::vector<bool> GetMutatingInputVariables(
      const Operator& op) const {
    // Most ops don't have variable tensors. This function can be overridden.
    return std::vector<bool>();
  }

 private:
  std::string name_;
  OperatorType type_;
};

// Helper function to create ::tflite::OpSignature from the given
// ::tflite::BuiltinOperator and OperatorSignature.
::tflite::OpSignature GetVersioningOpSig(const ::tflite::BuiltinOperator op,
                                         const OperatorSignature& op_signature);

// Helper function to determine if a unsupported TensorFlow op should be
// exported as an Flex op or a regular custom op.
bool ShouldExportAsFlexOp(bool enable_select_tf_ops,
                          const std::string& tensorflow_op_name);

}  // namespace tflite

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_OPERATOR_H_
