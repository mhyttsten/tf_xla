/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_NODE_DEF_BUILDER_H_
#define TENSORFLOW_CORE_FRAMEWORK_NODE_DEF_BUILDER_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTh() {
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


#include <functional>
#include <vector>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

class NodeDefBuilder;
typedef std::function<Status(const OpDef&, int, const NodeDef&,
                             NodeDefBuilder*)>
    FakeInputFunctor;

// This is a helper for creating a NodeDef.  Automatically sets attrs
// that can be inferred from the inputs, and uses default values
// (where they exist) for unspecified attrs.  Example usage:
//
//  NodeDef node_def;
//  Status status = NodeDefBuilder(node_name, op_name)
//                           .Input(...)
//                           .Attr(...)
//                           .Finalize(&node_def);
//  if (!status.ok()) return status;
//  // Use node_def here.
class NodeDefBuilder {
 public:
  // To specify an output to be consumed by one of the Input() methods below.
  struct NodeOut {
    NodeOut(StringPiece n, int i, DataType dt);
    NodeOut();  // uninitialized, call Reset() before use.
    void Reset(StringPiece n, int i, DataType dt);
    string node;
    int index;
    DataType data_type;
  };

  // Specify the name and the Op (either via an OpDef or the name of
  // the Op plus a registry) for the NodeDef.  Other fields are
  // specified by calling the methods below.
  // REQUIRES: The OpDef must satisfy ValidateOpDef().
  NodeDefBuilder(StringPiece name, StringPiece op_name,
                 const OpRegistryInterface* op_registry = OpRegistry::Global(),
                 const NodeDebugInfo* debug = nullptr);
  NodeDefBuilder(StringPiece name, StringPiece op_name,
                 const NodeDebugInfo& debug);
  // REQUIRES: in addition, *op_def must outlive *this.
  NodeDefBuilder(StringPiece name, const OpDef* op_def);

  // You must call one Input() function per input_arg in the Op,
  // *and in the same order as the input_args appear in the OpDef.*

  // For inputs that take a single tensor.
  NodeDefBuilder& Input(StringPiece src_node, int src_index, DataType dt);
  NodeDefBuilder& Input(const NodeOut& src);

  // For inputs that take a list of tensors.
  NodeDefBuilder& Input(gtl::ArraySlice<NodeOut> src_list);

  // To create inputs in tests, see fake_input.h.
  NodeDefBuilder& Input(FakeInputFunctor fake_input);

  // Specify that this node must only run after src_node.
  NodeDefBuilder& ControlInput(StringPiece src_node);

  // Constrains what devices this node may be scheduled on.
  NodeDefBuilder& Device(StringPiece device_spec);

  // Sets the attr, if not already set.  If already set with a different
  // value, an error will be returned from Finalize().
  NodeDefBuilder& Attr(StringPiece name, const AttrValue& value);
  NodeDefBuilder& Attr(StringPiece name, AttrValue&& value);
  NodeDefBuilder& Attr(StringPiece name, StringPiece value);
  NodeDefBuilder& Attr(StringPiece name, const char* value);
  NodeDefBuilder& Attr(StringPiece name, int32_t value);
  NodeDefBuilder& Attr(StringPiece name, int64_t value);
  NodeDefBuilder& Attr(StringPiece name, float value);
  NodeDefBuilder& Attr(StringPiece name, double value);
  NodeDefBuilder& Attr(StringPiece name, bool value);
  NodeDefBuilder& Attr(StringPiece name, DataType value);
  NodeDefBuilder& Attr(StringPiece name, const PartialTensorShape& value);
  NodeDefBuilder& Attr(StringPiece name, const Tensor& value);
  NodeDefBuilder& Attr(StringPiece name, const TensorProto& value);
  NodeDefBuilder& Attr(StringPiece name, const NameAttrList& value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<StringPiece> value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<const char*> value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<string> value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<tstring> value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<int32> value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<int64_t> value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<float> value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<bool> value);
  NodeDefBuilder& Attr(StringPiece name, const std::vector<bool>& value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<DataType> value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<TensorShape> value);
  NodeDefBuilder& Attr(StringPiece name,
                       gtl::ArraySlice<PartialTensorShape> value);
  NodeDefBuilder& Attr(StringPiece name,
                       gtl::ArraySlice<TensorShapeProto> value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<Tensor> value);
  NodeDefBuilder& Attr(StringPiece name, gtl::ArraySlice<NameAttrList> value);

  template <class T>
  NodeDefBuilder& Attr(StringPiece name, std::initializer_list<T> value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTh mht_0(mht_0_v, 299, "", "./tensorflow/core/framework/node_def_builder.h", "Attr");

    return Attr(name, gtl::ArraySlice<T>(value));
  }

  // Finish building the NodeDef, returning any errors or setting
  // *node_def if none.
  // If `consume` is true, the builder state will be moved into `node_def`,
  // and the builder will be left in an undefined state.
  // WARNING: Not all problems are detected!  The resulting NodeDef may
  // not be valid!  Call ValidateNodeDef() from node_def_utils to be sure.
  Status Finalize(NodeDef* node_def, bool consume = false);

  // Accessors for the values set in the constructor.
  const string& node_name() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTh mht_1(mht_1_v, 315, "", "./tensorflow/core/framework/node_def_builder.h", "node_name");
 return node_def_.name(); }
  const OpDef& op_def() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTh mht_2(mht_2_v, 319, "", "./tensorflow/core/framework/node_def_builder.h", "op_def");
 return *op_def_; }

 private:
  // Called in the constructors.
  void Initialize();

  // Get the current ArgDef and advance to the next one. Returns nullptr
  // if no more inputs are available.
  const OpDef::ArgDef* NextArgDef();

  // Returns true if there is still an input_arg available in *op_def_,
  // otherwise adds to error_ and returns false.
  bool NextArgAvailable();

  // These do the main work of the Input() methods.
  void SingleInput(const OpDef::ArgDef* input_arg, StringPiece src_node,
                   int src_index, DataType dt);
  void ListInput(const OpDef::ArgDef* input_arg,
                 gtl::ArraySlice<NodeOut> src_list);

  // Add "src_node:src_index" to the list of inputs in the node_def_.
  void AddInput(StringPiece src_node, int src_index);

  // Generate an error if you can't pass dt when expected is expected.
  void VerifyInputType(const OpDef::ArgDef* input_arg, DataType expected,
                       DataType dt);

  // If input_arg->is_ref() is true, generate an error if dt is not a ref.
  void VerifyInputRef(const OpDef::ArgDef* input_arg, DataType dt);

  // Makes dt a ref type if that is what the input_arg specifies.
  DataType MaybeAddRef(const OpDef::ArgDef* input_arg, DataType dt) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTh mht_3(mht_3_v, 353, "", "./tensorflow/core/framework/node_def_builder.h", "MaybeAddRef");

    return input_arg->is_ref() ? MakeRefType(dt) : dt;
  }

  // Returns true if an attr named `name` is already present in the node_def_.
  // If such an attr is already present and `value` is not equal to the present
  // value, an error is generated.
  bool AttrValueAlreadyPresent(StringPiece name, const AttrValue& value);

  const OpDef* op_def_;
  NodeDef node_def_;
  int inputs_specified_;
  std::vector<string> control_inputs_;
  std::vector<string> errors_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_NODE_DEF_BUILDER_H_
