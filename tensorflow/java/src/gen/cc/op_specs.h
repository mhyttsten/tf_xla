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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_OP_SPECS_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_OP_SPECS_H_
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
class MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh {
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
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh() {
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


#include <string>
#include <vector>

#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

namespace tensorflow {
namespace java {

constexpr const char kDefaultEndpointPackage[] = "core";

class EndpointSpec {
 public:
  // A specification for an operation endpoint
  //
  // package: package of this endpoint (from which also derives its package)
  // name: name of this endpoint class
  // javadoc: the endpoint class documentation
  // TODO(annarev): hardcode deprecated to false until deprecated is possible
  EndpointSpec(const string& package, const string& name,
               const Javadoc& javadoc)
      : package_(package), name_(name), javadoc_(javadoc), deprecated_(false) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("package: \"" + package + "\"");
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_0(mht_0_v, 213, "", "./tensorflow/java/src/gen/cc/op_specs.h", "EndpointSpec");
}

  const string& package() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_1(mht_1_v, 218, "", "./tensorflow/java/src/gen/cc/op_specs.h", "package");
 return package_; }
  const string& name() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_2(mht_2_v, 222, "", "./tensorflow/java/src/gen/cc/op_specs.h", "name");
 return name_; }
  const Javadoc& javadoc() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_3(mht_3_v, 226, "", "./tensorflow/java/src/gen/cc/op_specs.h", "javadoc");
 return javadoc_; }
  bool deprecated() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_4(mht_4_v, 230, "", "./tensorflow/java/src/gen/cc/op_specs.h", "deprecated");
 return deprecated_; }

 private:
  const string package_;
  const string name_;
  const Javadoc javadoc_;
  const bool deprecated_;
};

class ArgumentSpec {
 public:
  // A specification for an operation argument
  //
  // op_def_name: argument name, as known by TensorFlow core
  // var: a variable to represent this argument in Java
  // type: the tensor type of this argument
  // description: a description of this argument, in javadoc
  // iterable: true if this argument is a list
  ArgumentSpec(const string& op_def_name, const Variable& var, const Type& type,
               const string& description, bool iterable)
      : op_def_name_(op_def_name),
        var_(var),
        type_(type),
        description_(description),
        iterable_(iterable) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op_def_name: \"" + op_def_name + "\"");
   mht_5_v.push_back("description: \"" + description + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_5(mht_5_v, 259, "", "./tensorflow/java/src/gen/cc/op_specs.h", "ArgumentSpec");
}

  const string& op_def_name() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_6(mht_6_v, 264, "", "./tensorflow/java/src/gen/cc/op_specs.h", "op_def_name");
 return op_def_name_; }
  const Variable& var() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_7(mht_7_v, 268, "", "./tensorflow/java/src/gen/cc/op_specs.h", "var");
 return var_; }
  const Type& type() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_8(mht_8_v, 272, "", "./tensorflow/java/src/gen/cc/op_specs.h", "type");
 return type_; }
  const string& description() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_9(mht_9_v, 276, "", "./tensorflow/java/src/gen/cc/op_specs.h", "description");
 return description_; }
  bool iterable() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_10(mht_10_v, 280, "", "./tensorflow/java/src/gen/cc/op_specs.h", "iterable");
 return iterable_; }

 private:
  const string op_def_name_;
  const Variable var_;
  const Type type_;
  const string description_;
  const bool iterable_;
};

class AttributeSpec {
 public:
  // A specification for an operation attribute
  //
  // op_def_name: attribute name, as known by TensorFlow core
  // var: a variable to represent this attribute in Java
  // type: the type of this attribute
  // jni_type: the type of this attribute in JNI layer (see OperationBuilder)
  // description: a description of this attribute, in javadoc
  // iterable: true if this attribute is a list
  // default_value: default value for this attribute or nullptr if none. Any
  //                value referenced by this pointer must outlive the lifetime
  //                of the AttributeSpec. This is guaranteed if the value is
  //                issued by an OpDef of the global OpRegistry.
  AttributeSpec(const string& op_def_name, const Variable& var,
                const Type& type, const Type& jni_type,
                const string& description, bool iterable,
                const AttrValue* default_value)
      : op_def_name_(op_def_name),
        var_(var),
        type_(type),
        description_(description),
        iterable_(iterable),
        jni_type_(jni_type),
        default_value_(default_value) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("op_def_name: \"" + op_def_name + "\"");
   mht_11_v.push_back("description: \"" + description + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_11(mht_11_v, 319, "", "./tensorflow/java/src/gen/cc/op_specs.h", "AttributeSpec");
}

  const string& op_def_name() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_12(mht_12_v, 324, "", "./tensorflow/java/src/gen/cc/op_specs.h", "op_def_name");
 return op_def_name_; }
  const Variable& var() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_13(mht_13_v, 328, "", "./tensorflow/java/src/gen/cc/op_specs.h", "var");
 return var_; }
  const Type& type() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_14(mht_14_v, 332, "", "./tensorflow/java/src/gen/cc/op_specs.h", "type");
 return type_; }
  const string& description() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_15(mht_15_v, 336, "", "./tensorflow/java/src/gen/cc/op_specs.h", "description");
 return description_; }
  bool iterable() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_16(mht_16_v, 340, "", "./tensorflow/java/src/gen/cc/op_specs.h", "iterable");
 return iterable_; }
  const Type& jni_type() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_17(mht_17_v, 344, "", "./tensorflow/java/src/gen/cc/op_specs.h", "jni_type");
 return jni_type_; }
  bool has_default_value() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_18(mht_18_v, 348, "", "./tensorflow/java/src/gen/cc/op_specs.h", "has_default_value");
 return default_value_ != nullptr; }
  const AttrValue* default_value() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_19(mht_19_v, 352, "", "./tensorflow/java/src/gen/cc/op_specs.h", "default_value");
 return default_value_; }

 private:
  const string op_def_name_;
  const Variable var_;
  const Type type_;
  const string description_;
  const bool iterable_;
  const Type jni_type_;
  const AttrValue* default_value_;
};

class OpSpec {
 public:
  // Parses an op definition and its API to produce a specification used for
  // rendering its Java wrapper
  //
  // op_def: Op definition
  // api_def: Op API definition
  static OpSpec Create(const OpDef& op_def, const ApiDef& api_def);

  const string& graph_op_name() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_20(mht_20_v, 376, "", "./tensorflow/java/src/gen/cc/op_specs.h", "graph_op_name");
 return graph_op_name_; }
  bool hidden() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_21(mht_21_v, 380, "", "./tensorflow/java/src/gen/cc/op_specs.h", "hidden");
 return hidden_; }
  const string& deprecation_explanation() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_22(mht_22_v, 384, "", "./tensorflow/java/src/gen/cc/op_specs.h", "deprecation_explanation");

    return deprecation_explanation_;
  }
  const std::vector<EndpointSpec> endpoints() const { return endpoints_; }
  const std::vector<ArgumentSpec>& inputs() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_23(mht_23_v, 391, "", "./tensorflow/java/src/gen/cc/op_specs.h", "inputs");
 return inputs_; }
  const std::vector<ArgumentSpec>& outputs() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_24(mht_24_v, 395, "", "./tensorflow/java/src/gen/cc/op_specs.h", "outputs");
 return outputs_; }
  const std::vector<AttributeSpec>& attributes() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_25(mht_25_v, 399, "", "./tensorflow/java/src/gen/cc/op_specs.h", "attributes");
 return attributes_; }
  const std::vector<AttributeSpec>& optional_attributes() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_26(mht_26_v, 403, "", "./tensorflow/java/src/gen/cc/op_specs.h", "optional_attributes");

    return optional_attributes_;
  }

 private:
  // A specification for an operation
  //
  // graph_op_name: name of this op, as known by TensorFlow core engine
  // hidden: true if this op should not be visible through the Graph Ops API
  // deprecation_explanation: message to show if all endpoints are deprecated
  explicit OpSpec(const string& graph_op_name, bool hidden,
                  const string& deprecation_explanation)
      : graph_op_name_(graph_op_name),
        hidden_(hidden),
        deprecation_explanation_(deprecation_explanation) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("graph_op_name: \"" + graph_op_name + "\"");
   mht_27_v.push_back("deprecation_explanation: \"" + deprecation_explanation + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSop_specsDTh mht_27(mht_27_v, 422, "", "./tensorflow/java/src/gen/cc/op_specs.h", "OpSpec");
}

  const string graph_op_name_;
  const bool hidden_;
  const string deprecation_explanation_;
  std::vector<EndpointSpec> endpoints_;
  std::vector<ArgumentSpec> inputs_;
  std::vector<ArgumentSpec> outputs_;
  std::vector<AttributeSpec> attributes_;
  std::vector<AttributeSpec> optional_attributes_;
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_SPECS_H_
