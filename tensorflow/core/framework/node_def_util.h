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

#ifndef TENSORFLOW_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_utilDTh() {
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
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

class AttrSlice;
// We forward declare protos so that kernels don't need to depend on them
class OpDef;
class AttrValue;
class NameAttrList;
class TensorProto;
class TensorShapeProto;

// Name of the attribute used to encode node colocation constraints.
//
// Nodes can be co-located on the same device. Desire for explicit co-location
// is described by list(string) attribute containing the name of colocation
// groups.
extern const char* const kColocationAttrName;

// String prefix applied to the operation name for colocation constraints.
extern const char* const kColocationGroupPrefix;

// Constants for host CPU staging op for TPUExecute.
extern const char* const kTpuExecuteStagingOp;
extern const char* const kTpuExecuteStagingNodeName;

// Produce a human-readable version of a Node or NodeDef that is more concise
// than a text-format proto.
//
// The parameter `max_inputs_in_summary` specifies how many inputs at most to
// serialize in the output (in order not to get a string which is overly large).
// The value `-1` specifies that all inputs will be shown.
std::string SummarizeNodeDef(const NodeDef& node_def,
                             int max_inputs_in_summary = -1);
std::string SummarizeAttrs(const NodeDef& node_def);
std::string SummarizeAttrsHelper(AttrSlice attrs, StringPiece device);

// Produces a formatted string pattern from the node which can uniquely identify
// this node upstream to produce an informative error message. The pattern
// followed is: {{node <node_name>}}
std::string FormatNodeDefForError(const NodeDef& node_def);
std::string FormatNodeDefForError(
    StringPiece node_name, bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info);

typedef protobuf::Map<string, AttrValue> AttrValueMap;

// Adds an attr with name <name> and value <value> to *node_def.
// The type of the attr is based on the type of value.
void AddNodeAttr(StringPiece name, const AttrValue& value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, AttrValue&& value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, StringPiece value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, const char* value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, int32_t value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, int64_t value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, float value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, double value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, bool value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, DataType value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, const PartialTensorShape& value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, const Tensor& value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, const TensorProto& value, NodeDef* node_def);
void AddNodeAttr(StringPiece name, const NameAttrList& value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<StringPiece> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<const char*> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<string> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<int32> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<int64_t> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<float> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<bool> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, const std::vector<bool>& value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<DataType> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<TensorShape> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<PartialTensorShape> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<TensorShapeProto> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<Tensor> value,
                 NodeDef* node_def);
void AddNodeAttr(StringPiece name, gtl::ArraySlice<NameAttrList> value,
                 NodeDef* node_def);

// Version to workaround C++'s "perfect" forwarding not being able to
// forward {...} initialization.
template <class T>
void AddNodeAttr(StringPiece name, std::initializer_list<T> value,
                 NodeDef* node_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_utilDTh mht_0(mht_0_v, 305, "", "./tensorflow/core/framework/node_def_util.h", "AddNodeAttr");

  AddNodeAttr(name, gtl::ArraySlice<T>(value), node_def);
}

// Adds an attr to an attr value map.
void AddAttr(StringPiece name, const AttrValue& value, AttrValueMap* map);
void AddAttr(StringPiece name, bool value, AttrValueMap* map);

class AttrSlice {
 public:
  AttrSlice(const NodeDef& node_def);  // NOLINT(runtime/explicit)

  AttrSlice();  // Empty
  explicit AttrSlice(const AttrValueMap* a);

  int size() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_utilDTh mht_1(mht_1_v, 323, "", "./tensorflow/core/framework/node_def_util.h", "size");
 return attrs()->size(); }

  // Returns the attr with attr_name if found.  Otherwise, returns
  // nullptr.
  const AttrValue* Find(StringPiece attr_name) const;
  const AttrValue* FindByString(const std::string& attr_name) const;

  // Returns the attr_value for attr_name if found. Otherwise, returns a
  // NotFound status.
  Status Find(StringPiece attr_name, const AttrValue** attr_value) const;

  // Helper class to avoid allocations in EqualAttrs.
  // TODO(irving): Will go away once NodeInfo is used.
  struct Scratch {
    std::string a;
    std::string b;
  };

  // Check if all attrs and attr values match.  Does not take defaults into
  // account.
  //
  // TODO(irving): There is a bug in this routine inherited from its
  // OptimizerCSE::EqualAttrs predecessor.  The same tensor attr can be
  // represented in more than one way as an AttrValue, since TensorProto is
  // not 1-1.  This bug will go away once I replace everything with NodeInfo,
  // which stores a Tensor object directly.  The Scratch object will also go
  // away.
  bool EqualAttrs(AttrSlice other, Scratch* scratch) const;

  // If this AttrSlice has an attached NodeDef, summarize it.  This is for
  // error messages only: we intentionally do not provide direct access to the
  // NodeDef, since it is not always there.
  std::string SummarizeNode() const;

  // Iteration over all attrs
  AttrValueMap::const_iterator begin() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_utilDTh mht_2(mht_2_v, 361, "", "./tensorflow/core/framework/node_def_util.h", "begin");
 return attrs()->begin(); }
  AttrValueMap::const_iterator end() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_utilDTh mht_3(mht_3_v, 365, "", "./tensorflow/core/framework/node_def_util.h", "end");
 return attrs()->end(); }

  std::string DebugString() const;

 private:
  const AttrValueMap* attrs() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_utilDTh mht_4(mht_4_v, 373, "", "./tensorflow/core/framework/node_def_util.h", "attrs");

    return ndef_ != nullptr ? &ndef_->attr() : attrs_;
  }

  const NodeDef* ndef_;
  const AttrValueMap* attrs_;
};

// Return true if the attr with the name attr_name is defined in node_def.
bool HasNodeAttr(const NodeDef& node_def, StringPiece attr_name);

// Look up the attr with name attr_name and set *value to its value.  If no
// attr with attr_name is found in node_def, or the attr does not have
// a matching type, a non-ok status will be returned.
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::string* value);  // type: "string"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   tstring* value);  // type: "tstring"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   int64_t* value);  // type: "int"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   int32* value);  // type: "int"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   float* value);  // type: "float"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   bool* value);  // type: "bool"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   DataType* value);  // type: "type"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   TensorShapeProto* value);  // type: "shape"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   TensorShape* value);  // type: "shape"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   PartialTensorShape* value);  // type: "shape"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   Tensor* value);  // type: "tensor"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<string>* value);  // type "list(string)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<tstring>* value);  // type "list(tstring)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<int64_t>* value);  // type "list(int)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<int32>* value);  // type "list(int)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<float>* value);  // type "list(float)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<bool>* value);  // type "list(bool)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<DataType>* value);  // type "list(type)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   DataTypeVector* value);  // type "list(type)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<TensorShapeProto>* value);  // type "list(shape)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<TensorShape>* value);  // type "list(shape)"
Status GetNodeAttr(
    const AttrSlice& attrs, StringPiece attr_name,
    std::vector<PartialTensorShape>* value);  // type "list(shape)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<Tensor>* value);  // type: "list(tensor)"

// This version avoids copying the TensorProto.
// REQUIRES: Must not use *value beyond the lifetime of node_def.
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   const TensorProto** value);  // type: "tensor"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    const TensorProto** value);  // type: "tensor"

// This version avoids copying the NameAttrList.
// REQUIRES: Must not use *value beyond the lifetime of node_def.
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   const NameAttrList** value);  // type: "func"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    const NameAttrList** value);  // type: "func"

// These versions copies the NameAttrList(s).
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   NameAttrList* value);  // type: "func"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<NameAttrList>* value);  // type: "list(func)"

// Look up the attr with name attr_name and set *value to its value.  If no
// attr with attr_name is found in node_def, or the attr does not have
// a matching type, false is returned.
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::string* value);  // type: "string"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    int64_t* value);  // type: "int"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<int64_t>* value);  // type: "int"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    int32* value);  // type: "int"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    float* value);  // type: "float"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    bool* value);  // type: "bool"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    DataType* value);  // type: "type"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    TensorShape* value);  // type: "shape"

bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<string>* value);  // type: "list(string)"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<tstring>* value);  // type: "list(tstring)"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<int32>* value);  // type: "list(int)"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<float>* value);  // type: "list(float)"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<bool>* value);  // type: "list(bool)"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<DataType>* value);  // type: "list(type)"
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<TensorShape> value);  // type: "shape"

// Overloads of TryGetNodeAttr() that avoid copying the non-POD attribute
// values.
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<const string*>* value);  // type: "list(string)"
bool TryGetNodeAttr(
    const AttrSlice& attrs, StringPiece attr_name,
    std::vector<const TensorShapeProto*>* value);  // type: "list(shape)"

// Look up the attr with name attr_name and return a reference to its value.
// If no attr with attr_name is found in node_def, or the attr does not have
// a matching type, a reference to an empty string is returned.
// REQUIRES: Must not use the returned value beyond the lifetime of node_def.
const std::string& GetNodeAttrString(const AttrSlice& attrs,
                                     StringPiece attr_name);

// Specialization to parse an attribute directly into a Padding enum.
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   Padding* value);

// Computes the input type for a specific node input.
// REQUIRES: ValidateOpDef(op_def).ok()
Status InputTypeForNode(const NodeDef& node_def, const OpDef& op_def,
                        int input_port, DataType* input_type);
// Computes the input types for a specific node.
// REQUIRES: ValidateOpDef(op_def).ok()
Status InputTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                         DataTypeVector* inputs);
// Computes the output type for a specific node output.
// REQUIRES: ValidateOpDef(op_def).ok()
Status OutputTypeForNode(const NodeDef& node_def, const OpDef& op_def,
                         int output_port, DataType* output_type);
// Computes the output types for a specific node.
// REQUIRES: ValidateOpDef(op_def).ok()
Status OutputTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                          DataTypeVector* outputs);
Status OutputTypesForNode(const AttrSlice& attrs, const OpDef& op_def,
                          DataTypeVector* outputs);

// Computes the input and output types for a specific node.
// REQUIRES: ValidateOpDef(op_def).ok()
Status InOutTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                         DataTypeVector* inputs, DataTypeVector* outputs);
// Computes the number of outputs for a specific node.
// REQUIRES: ValidateOpDef(op_def).ok()
Status NumOutputsForNode(const NodeDef& node_def, const OpDef& op_def,
                         int* num_outputs);

// Validates that the NodeDef:
// * Defines all expected attrs from the OpDef.
// * All attrs satisfies constraints from the OpDef.
// * Has a signature matching SignatureForNode().
// etc.
Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def);

// Computes the mapping from input/output argument name to the
// corresponding input/output index range.  For example,
// input "foo" corresponds to input indices
//   [ (*inputs)["foo"].first, (*inputs)["foo"].second ).
// NOTE(mrry): To reduce allocations when the map is used and save
// space, the returned `NameRangeMap` objects borrow the input/output
// argument names from `op_def`. The `op_def` must outlive the
// returned `NameRangeMap` objects.
typedef gtl::FlatMap<StringPiece, std::pair<int, int>, hash<StringPiece>>
    NameRangeMap;
Status NameRangesForNode(const AttrSlice& attrs, const OpDef& op_def,
                         NameRangeMap* inputs, NameRangeMap* outputs);
// Adds default values to *node_def for unspecified attrs from op_def.
void AddDefaultsToNodeDef(const OpDef& op_def, NodeDef* node_def);

// Remove attributes from node_def when the value is the default from the
// op_def.
void StripDefaultsFromNodeDef(const OpDef& op_def, NodeDef* node_def);

// Validates the syntax of a NodeDef provided externally.
//
// The following is an EBNF-style syntax for NodeDef objects. Note that
// Node objects are actually specified as tensorflow::NodeDef protocol buffers,
// which contain many other fields that are not (currently) validated.
//
// Node         = NodeName, Inputs
// Inputs       = ( DataInput * ), ( ControlInput * )
// DataInput    = NodeName, ( ":", [1-9], [0-9] * ) ?
// ControlInput = "^", NodeName
// NodeName     = [A-Za-z0-9.], [A-Za-z0-9_./] *
Status ValidateExternalNodeDefSyntax(const NodeDef& node_def);

// Returns "status" with formatted NodeDef attached as additional text
// in the error message. If 'allow_multiple_formatted_node' is false and there
// is already a formatted NodeDef present in 'status', we simply attach the name
// of the NodeDef instead of the formatted string.
Status AttachDef(const Status& status, const NodeDef& node_def,
                 bool allow_multiple_formatted_node = false);
// Appends the given prefix and suffix to the original node name in order to
// make the name unique. If it's an "Enter" node and uniquify_frame_name is
// true, use the same way to reset attribute "frame_name".
Status AddPrefixAndSuffixToNode(StringPiece prefix, StringPiece suffix,
                                NodeDef* node_def,
                                bool uniquify_frame_name = true);

// Appends the given prefix to the colocation group name if the name exists
// in `to_match`.
Status MaybeAddPrefixToColocationConstraints(
    const std::unordered_set<string>& match, StringPiece prefix,
    NodeDef* node_def);

// Updates the colocation constraint name with the one provided in the map (if
// it exists in the map) for node_def.
Status MaybeUpdateColocationConstraintsWithMap(
    const std::map<absl::string_view, absl::string_view>& node_name_map,
    NodeDef* node_def);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
