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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_FUNCTIONS_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_FUNCTIONS_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTh() {
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


#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace tensorflow {
namespace grappler {

// Function input argument instantiated into an '_Arg' node in the function body
// graph, with an 'index' attribute corresponding to the input position.
struct InputArgInstantiation {
  InputArgInstantiation(string node_name, DataType data_type)
      : node_name(std::move(node_name)), data_type(data_type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTh mht_0(mht_0_v, 210, "", "./tensorflow/core/grappler/utils/functions.h", "InputArgInstantiation");
}
  string node_name;
  DataType data_type;
};

// Function output instantiated into a '_Retval' node in the function body
// graph, with an 'index' attribute corresponding to the output position.
struct OutputArgInstantiation {
  OutputArgInstantiation(string node_name, DataType data_type)
      : node_name(std::move(node_name)), data_type(data_type) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTh mht_1(mht_1_v, 223, "", "./tensorflow/core/grappler/utils/functions.h", "OutputArgInstantiation");
}
  string node_name;
  DataType data_type;
};

// A mapping from control output name to node name in function body graph.
struct ControlOutput {
  string output_name;
  string node_name;
  bool operator<(const ControlOutput& a) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTh mht_2(mht_2_v, 235, "", "./tensorflow/core/grappler/utils/functions.h", "operator<");

    return output_name < a.output_name;
  }
};

// A special case of GrapplerItem, constructed from a TensorFlow Function.
class GrapplerFunctionItem : public GrapplerItem {
 public:
  GrapplerFunctionItem() = default;

  const string& description() const;

  const std::vector<InputArgInstantiation>& inputs() const;
  const InputArgInstantiation& input(int i) const;
  const std::size_t input_size() const;

  const std::vector<OutputArgInstantiation>& outputs() const;
  const OutputArgInstantiation& output(int i) const;
  const std::size_t output_size() const;

  const std::vector<ControlOutput>& control_outputs() const;
  const std::size_t control_output_size() const;

  const AttrSlice& func_attr() const;
  const std::vector<const FunctionDef::ArgAttrs*>& arg_attr() const;
  const GraphDef& function_body() const;
  GraphDef& mutable_function_body();

  bool is_stateful() const;

  GrapplerFunctionItem& SwapFunctionBody(GraphDef&& other);

 private:
  friend Status MakeGrapplerFunctionItem(const FunctionDef&, const AttrSlice&,
                                         const FunctionLibraryDefinition&, int,
                                         GrapplerFunctionItem*);
  friend Status ReplaceInputWithConst(const NodeDef&, int,
                                      GrapplerFunctionItem*);
  friend Status RemoveFunctionOutputs(const absl::flat_hash_set<int>&,
                                      GrapplerFunctionItem*,
                                      std::vector<std::pair<int, int>>*);

  GrapplerFunctionItem(string func_name, string description,
                       AttrSlice func_attr,
                       std::vector<const FunctionDef::ArgAttrs*> arg_attr,
                       std::vector<InputArgInstantiation> input_args,
                       std::vector<OutputArgInstantiation> output_args,
                       std::vector<ControlOutput> control_outputs,
                       int graph_def_version, bool is_stateful,
                       GraphDef&& function_body);

  string description_;
  AttrSlice func_attr_;  // Attributes specific to function definition that
                         // produced this item (FuncDef.attr field).

  // Attributes of function arguments
  std::vector<const FunctionDef::ArgAttrs*> arg_attr_;

  std::vector<InputArgInstantiation> input_args_;
  std::vector<OutputArgInstantiation> output_args_;
  std::vector<ControlOutput> control_outputs_;

  bool is_stateful_ = false;
};

// Check if function input/output types are fully defined only at instantiation
// time (parametrized by its instantiation node).
bool HasParametrizedType(const FunctionDef& func);

// Check if a function body is parametrized by its instantiation node. Function
// body is parametrized, if it has at least one node with a 'placeholder'
// attribute.
bool HasParametrizedBody(const FunctionDef& func);

// Check if function has parametrized type or body.
bool IsParametrized(const FunctionDef& func);

// Resolve function instantiation type parameters from the attributes of the
// caller node. Return error if type can't be resolved.
Status InstantiationTypeParameters(
    const FunctionDef& func, const AttrSlice& func_instantiation_attr,
    absl::flat_hash_map<string, DataType>* type_parameters);

// Resolve function instantiation body parameters (values for the function body
// attr placeholders) from the attributes of the caller node. Return error if
// type can't be resolved.
Status InstantiationBodyParameters(
    const FunctionDef& func, const AttrSlice& func_instantiation_attr,
    absl::flat_hash_map<string, AttrValue>* body_parameters);

// Replace one of the function inputs with a constant.
Status ReplaceInputWithConst(const NodeDef& input_const, int input_index,
                             GrapplerFunctionItem* item);

// Removes outputs from instantiated grappler function item. For all active
// function outputs that changed its output index, this function adds an output
// mapping (std::pair<old index, new index>).
Status RemoveFunctionOutputs(const absl::flat_hash_set<int>& remove_outputs,
                             GrapplerFunctionItem* item,
                             std::vector<std::pair<int, int>>* output_mapping);

// TODO(ezhulenev, b/120103818): Add RemoveFunctionInputs.

// Make a GrapplerFunctionItem from the function definition and function
// instantiation attributes (caller node attributes). Returns error if the given
// function def cannot be converted (e.g. not all attributes are defined).
Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const AttrSlice& func_instantiation_attr,
                                const FunctionLibraryDefinition& flib,
                                int graph_def_version,
                                GrapplerFunctionItem* item);

// Make a GrapplerFunction item from the function definition. Function must be
// fully defined (no type or body parametrization).
// TODO(ezhulenev): Support parametrized functions without fully defined
// instantiation attributes? Do we ever want to optimize parametrized function
// without specializing it to its instantiation attributes (at least types)?
Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const FunctionLibraryDefinition& flib,
                                int graph_def_version,
                                GrapplerFunctionItem* item);

// Make a FunctionDef from the GrapplerFunctionItem. Use function library
// definition to lookup function body nodes output names and ranges.
Status MakeFunctionDef(const GrapplerFunctionItem& item,
                       const FunctionLibraryDefinition& flib,
                       FunctionDef* func);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_FUNCTIONS_H_
