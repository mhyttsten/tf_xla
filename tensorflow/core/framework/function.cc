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
class MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc() {
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

#include "tensorflow/core/framework/function.h"

#include <ctype.h>

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {

/* static */ constexpr const char* const FunctionLibraryDefinition::kArgOp;
/* static */ constexpr const char* const
    FunctionLibraryDefinition::kDeviceArgOp;
/* static */ constexpr const char* const FunctionLibraryDefinition::kRetOp;
/* static */ constexpr const char* const
    FunctionLibraryDefinition::kDeviceRetOp;
/* static */ constexpr const char* const
    FunctionLibraryDefinition::kIntsOnDeviceAttr;
/* static */ constexpr const char* const FunctionLibraryDefinition::kGradientOp;
/* static */ constexpr const char* const FunctionLibraryDefinition::kFuncAttr;

// Extracts the actual type from "attr_values" based on its definition
// "arg_def".
//
// If "arg_def" is a N*T type, *is_type_list is set to false, and
// *dtypes is set to be a vector of size N and each element is T.
//
// If "arg_def" is a list(type), *is_type_list is set to true, and
// *dtypes is set to be a vector of types specified in attrs for
// arg_def.
//
// Otherwise (arg_def is a simple type T), *is_type_list is set to
// false, and *dtypes is set to a single element vector, whose only
// element is T.
Status ArgNumType(AttrSlice attrs, const OpDef::ArgDef& arg_def,
                  bool* is_type_list, DataTypeVector* dtypes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_0(mht_0_v, 241, "", "./tensorflow/core/framework/function.cc", "ArgNumType");

  dtypes->clear();
  if (!arg_def.type_list_attr().empty()) {
    const AttrValue* v = attrs.Find(arg_def.type_list_attr());
    if (v == nullptr) {
      return errors::NotFound("type attr not found: ",
                              arg_def.type_list_attr());
    }
    *is_type_list = true;
    for (int i = 0; i < v->list().type_size(); ++i) {
      dtypes->push_back(v->list().type(i));
    }
    return Status::OK();
  }

  *is_type_list = false;
  int num = 1;
  if (!arg_def.number_attr().empty()) {
    const AttrValue* v = attrs.Find(arg_def.number_attr());
    if (v == nullptr) {
      return errors::NotFound("type attr not found: ", arg_def.type_attr());
    }
    num = v->i();
  }

  DataType dtype;
  if (arg_def.type() != DT_INVALID) {
    dtype = arg_def.type();
  } else if (arg_def.type_attr().empty()) {
    dtype = DT_INVALID;
  } else {
    const AttrValue* v = attrs.Find(arg_def.type_attr());
    if (v == nullptr) {
      return errors::NotFound("type attr not found: ", arg_def.type_attr());
    }
    dtype = v->type();
  }
  dtypes->resize(num, dtype);
  return Status::OK();
}

namespace {

template <typename T>
void AddAttr(const string& name, const T& val, NodeDef* ndef) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_1(mht_1_v, 289, "", "./tensorflow/core/framework/function.cc", "AddAttr");

  SetAttrValue(val, &((*ndef->mutable_attr())[name]));
}

Status ValidateSignatureWithAttrs(const OpDef& sig, AttrSlice attr_values) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_2(mht_2_v, 296, "", "./tensorflow/core/framework/function.cc", "ValidateSignatureWithAttrs");

  // attr_values should specify all attrs defined in fdef, except for those
  // which have a default value
  for (const auto& attr : sig.attr()) {
    const AttrValue* attr_value = attr_values.Find(attr.name());
    if (attr_value) {
      Status status = AttrValueHasType(*attr_value, attr.type());
      if (!status.ok()) {
        errors::AppendToMessage(&status, "for attr '", attr.name(), "'");
        return status;
      }
    } else if (!attr.has_default_value()) {
      return errors::NotFound("Attr ", attr.name(), " is not found from ",
                              SummarizeOpDef(sig));
    }
  }

// TODO(josh11b): Enable this code once it works with function gradients.
// Right now the C++ function gradient code assumes it can pass
// all the attrs of the function to the gradient, and any attrs that
// the gradient doesn't care about will be ignored.
#if 0
  if (attr_values.size() != sig.attr_size()) {
    for (const auto& a : attr_values) {
      // TODO(josh11b): Possibly should ignore attrs that start with "_" here?
      bool found = false;
      for (const auto& s : sig.attr()) {
        if (a.first == s.name()) {
          found = true;
          break;
        }
      }
      if (!found) {
        return errors::NotFound("Attr ", a.first, " is not found in ",
                                SummarizeOpDef(sig));
      }
    }
  }
#endif

  return Status::OK();
}

// A helper class for instantiating functions. This contains shared information
// like the resulting graph and node name index.
class FunctionInstantiationHelper {
 public:
  FunctionInstantiationHelper(GetFunctionSignature get_function,
                              InstantiationResult* result)
      : get_function_(std ::move(get_function)), result_(*result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_3(mht_3_v, 348, "", "./tensorflow/core/framework/function.cc", "FunctionInstantiationHelper");

    result_.nodes.clear();
  }

  // Builds index for nodes that can be used as node's input arguments.
  // `resource_arg_unique_id`: if non-negative, will be populated to the
  // "_resource_arg_unique_id" attribute of the arg node.
  Status BuildInputArgIndex(const OpDef::ArgDef& arg_def, AttrSlice attr_values,
                            const FunctionDef::ArgAttrs* arg_attrs,
                            bool ints_on_device,
                            int64_t resource_arg_unique_id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_4(mht_4_v, 361, "", "./tensorflow/core/framework/function.cc", "BuildInputArgIndex");

    bool is_type_list;
    DataTypeVector dtypes;
    TF_RETURN_IF_ERROR(
        ArgNumType(attr_values, arg_def, &is_type_list, &dtypes));
    if (dtypes.size() < size_t{1}) {
      return errors::Internal("Expected a list of at least one dtype");
    }
    int arg_index = result_.nodes.size();
    TF_RETURN_IF_ERROR(
        AddItem(arg_def.name(), {true, arg_index, 0, is_type_list, dtypes}));
    // Creates dtypes.size() nodes in the graph.
    for (size_t i = 0; i < dtypes.size(); ++i) {
      TF_RETURN_IF_ERROR(AddItem(strings::StrCat(arg_def.name(), ":", i),
                                 {true, arg_index, 0, false, {dtypes[i]}}));
      if (arg_index != result_.nodes.size()) {
        return errors::Internal(
            "Expected arg_index to be equal to the number of nodes in result.",
            " Got ", arg_index, " and ", result_.nodes.size());
      }
      string name = arg_def.name();
      if (dtypes.size() > 1) {
        strings::StrAppend(&name, "_", i);
      }
      NodeDef* gnode = AddNode(name);
      if (ints_on_device && dtypes[i] == DataType::DT_INT32) {
        gnode->set_op(FunctionLibraryDefinition::kDeviceArgOp);
      } else {
        gnode->set_op(FunctionLibraryDefinition::kArgOp);
      }
      DataType dtype = arg_def.is_ref() ? MakeRefType(dtypes[i]) : dtypes[i];
      AddAttr("T", dtype, gnode);
      AddAttr("index", arg_index, gnode);
      if (resource_arg_unique_id >= 0) {
        AddAttr("_resource_arg_unique_id", resource_arg_unique_id, gnode);
      }
      if (arg_attrs) {
        for (const auto& arg_attr : arg_attrs->attr()) {
          AddAttr(arg_attr.first, arg_attr.second, gnode->mutable_attr());
        }
      }
      result_.arg_types.push_back(dtypes[i]);
      ++arg_index;
    }
    return Status::OK();
  }

  Status BuildNodeOutputIndex(const NodeDef& node, AttrSlice attrs,
                              const int arg_index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_5(mht_5_v, 412, "", "./tensorflow/core/framework/function.cc", "BuildNodeOutputIndex");

    const OpDef* node_sig = nullptr;
    TF_RETURN_IF_ERROR(get_function_(node.op(), &node_sig));
    if (node_sig->output_arg_size() == 0) {
      return AddItem(node.name(), {false, arg_index, 0, false, {}});
    }
    const int num_retval = node_sig->output_arg_size();
    int start = 0;
    bool is_type_list;
    DataTypeVector dtypes;
    for (int i = 0; i < num_retval; ++i) {
      TF_RETURN_IF_ERROR(
          ArgNumType(attrs, node_sig->output_arg(i), &is_type_list, &dtypes));
      // Note that we rely on the backwards-compatibility test enforcing
      // that output_arg(*).name() doesn't change here.
      const string base_name =
          strings::StrCat(node.name(), ":", node_sig->output_arg(i).name());
      TF_RETURN_IF_ERROR(
          AddItem(base_name, {false, arg_index, start, is_type_list, dtypes}));
      for (int j = 0; j < static_cast<int>(dtypes.size()); ++j) {
        TF_RETURN_IF_ERROR(
            AddItem(strings::StrCat(base_name, ":", j),
                    {false, arg_index, start + j, false, {dtypes[j]}}));
      }
      start += dtypes.size();
    }
    return Status::OK();
  }

  Status InstantiateNode(const NodeDef& fnode, AttrSlice attrs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_6(mht_6_v, 444, "", "./tensorflow/core/framework/function.cc", "InstantiateNode");

    const OpDef* fnode_sig = nullptr;
    TF_CHECK_OK(get_function_(fnode.op(), &fnode_sig));
    NodeDef* gnode = AddNode(fnode.name());
    gnode->set_op(fnode.op());
    gnode->set_device(fnode.device());
    int gnode_idx = nodes_.size() - 1;

    // Input
    const int num_args = fnode_sig->input_arg_size();
    bool is_type_list;  // ignored
    DataTypeVector dtypes;
    int fnode_arg_index = 0;
    for (int i = 0; i < num_args; ++i) {
      TF_RETURN_IF_ERROR(
          ArgNumType(attrs, fnode_sig->input_arg(i), &is_type_list, &dtypes));
      // Consume inputs (indexed by fnode_arg_index) until we have
      // matched each element of dtypes (indexed by j).
      for (size_t j = 0; j < dtypes.size(); ++fnode_arg_index) {
        if (fnode_arg_index >= fnode.input_size()) {
          // Should never happen if we computed dtypes correctly.
          return errors::InvalidArgument(
              "Attempt to access beyond input size: ", fnode_arg_index,
              " >= ", fnode.input_size());
        }
        // Look up the next input.
        const string& input_name = fnode.input(fnode_arg_index);
        const auto* item = GetItemOrNull(input_name);
        if (item == nullptr) {
          return errors::InvalidArgument(
              "input ", input_name,
              " is not found: ", FormatNodeDefForError(fnode));
        }
        if (item->dtypes.size() > dtypes.size() - j) {
          return errors::InvalidArgument("Input ", input_name, " too long for ",
                                         fnode_sig->input_arg(i).name());
        }
        // Match up all the elements of this input (indexed by k) with
        // elements of dtypes (advancing j).
        for (int k = 0; k < item->dtypes.size(); ++k, ++j) {
          if (item->dtypes[k] != dtypes[j]) {
            return errors::InvalidArgument(
                "input ", fnode_sig->input_arg(i).name(), "[", j,
                "] expected type ", DataTypeString(dtypes[j]),
                " != ", DataTypeString(item->dtypes[k]), ", the type of ",
                input_name, "[", k, "]");
          }
          if (item->is_func_arg) {
            AddInput(gnode_idx, item->nid + k, 0);
          } else {
            AddInput(gnode_idx, item->nid, item->idx + k);
          }
        }
      }
    }

    // Control deps.
    for (int i = fnode_arg_index; i < fnode.input_size(); ++i) {
      const string& input = fnode.input(i);
      if (input.empty() || input[0] != '^') {
        return errors::InvalidArgument("Expected input[", i, "] == '", input,
                                       "' to be a control input.");
      }
      int nid = -1;
      const string node_name = input.substr(1);
      const string node_colon = node_name + ":";
      const string node_colon_bound = node_name + ";";
      // index_ is a map sorted lexicographically, so the key we are looking for
      // must lie in the range [node_name, node_colon_bound).
      auto it = index_.lower_bound(node_name);
      while (it != index_.end() && it->first <= node_colon_bound) {
        if (it->first == node_name || absl::StartsWith(it->first, node_colon)) {
          nid = it->second.nid;
          break;
        }
        ++it;
      }
      if (nid == -1) {
        return errors::InvalidArgument("input[", i, "] == '", input,
                                       "', is not found.");
      }
      AddDep(gnode_idx, nid);
    }

    // Attrs.
    for (const auto& p : attrs) {
      (*gnode->mutable_attr())[p.first] = p.second;
    }

    // Experimental_debug_info.
    if (fnode.has_experimental_debug_info()) {
      gnode->mutable_experimental_debug_info()->MergeFrom(
          fnode.experimental_debug_info());
    }

    // Tye info.
    // TODO(mdan): Might this need adjustment at instantiation?
    if (fnode.has_experimental_type()) {
      *gnode->mutable_experimental_type() = fnode.experimental_type();
    }

    return Status::OK();
  }

  Status AddReturnNode(
      const OpDef::ArgDef& ret_def, AttrSlice attrs,
      const ::tensorflow::protobuf::Map<string, string>& ret_map,
      bool ints_on_device, int* ret_index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_7(mht_7_v, 554, "", "./tensorflow/core/framework/function.cc", "AddReturnNode");

    auto ret_iter = ret_map.find(ret_def.name());
    if (ret_iter == ret_map.end()) {
      return errors::InvalidArgument("Return ", ret_def.name(), " missing.");
    }
    bool is_type_list;
    DataTypeVector dtypes;
    TF_RETURN_IF_ERROR(ArgNumType(attrs, ret_def, &is_type_list, &dtypes));
    CHECK_GE(dtypes.size(), size_t{1});
    const auto* item = GetItemOrNull(ret_iter->second);
    if (item == nullptr) {
      return errors::InvalidArgument("Return ", ret_def.name(), " -> ",
                                     ret_iter->second, " is not found.");
    }
    if (dtypes != item->dtypes) {
      return errors::InvalidArgument("Invalid ret types ", ret_def.name(),
                                     " : ", DataTypeVectorString(dtypes),
                                     " vs. ",
                                     DataTypeVectorString(item->dtypes));
    }
    for (size_t i = 0; i < dtypes.size(); ++i) {
      string name = strings::StrCat(ret_def.name(), "_RetVal");
      if (dtypes.size() > 1) {
        strings::StrAppend(&name, "_", i);
      }
      NodeDef* gnode = AddNode(name);
      if (ints_on_device && dtypes[i] == DataType::DT_INT32) {
        gnode->set_op(FunctionLibraryDefinition::kDeviceRetOp);
      } else {
        gnode->set_op(FunctionLibraryDefinition::kRetOp);
      }
      AddInput(nodes_.size() - 1, item->nid, item->idx + i);
      DataType dtype = ret_def.is_ref() ? MakeRefType(dtypes[i]) : dtypes[i];
      AddAttr("T", dtype, gnode);
      AddAttr("index", (*ret_index)++, gnode);
      result_.ret_types.push_back(dtypes[i]);
    }
    return Status::OK();
  }

  // Adds the actual node inputs to the result graph by converting indexes to
  // the node names.
  void AddNodeInputs() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_8(mht_8_v, 599, "", "./tensorflow/core/framework/function.cc", "AddNodeInputs");

    for (int i = 0; i < result_.nodes.size(); i++) {
      NodeInfo& node_info = nodes_[i];
      for (const auto& p : node_info.data_inputs) {
        result_.nodes[i].add_input(Name(p.first, p.second));
      }
      for (int index : node_info.control_inputs) {
        result_.nodes[i].add_input(Dep(index));
      }
    }
  }

 private:
  // This is used to build a small index for all names that can be used as a
  // node's input arguments.
  //
  // If is_func_arg is true, the name is a function's argument.  In
  // this case, the produced graph def has node[nid:nid + dtype.size()].
  //
  // Otherwise, the name is a function body's node return value.  In
  // this case, the produced graph def has one node node[nid] and
  // the node's output index [idx ... idx + num) corresponds to the
  // named outputs.
  //
  // In all cases, "dtype" specifies the data type.
  struct NameInfoItem {
    bool is_func_arg;
    int nid;
    int idx;
    bool is_type_list;
    DataTypeVector dtypes;
  };

  // Adds an item into the input name index.
  Status AddItem(const string& name, const NameInfoItem& item) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_9(mht_9_v, 637, "", "./tensorflow/core/framework/function.cc", "AddItem");

    if (!index_.insert({name, item}).second) {
      return errors::InvalidArgument(
          strings::StrCat("Duplicated ", item.is_func_arg ? "arg" : "ret",
                          " name: "),
          name);
    }
    return Status::OK();
  }

  const NameInfoItem* GetItemOrNull(const string& name) const {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_10(mht_10_v, 651, "", "./tensorflow/core/framework/function.cc", "GetItemOrNull");

    return gtl::FindOrNull(index_, name);
  }

  string Dep(int node_index) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_11(mht_11_v, 658, "", "./tensorflow/core/framework/function.cc", "Dep");

    return strings::StrCat("^", Name(node_index));
  }

  string Name(int node_index) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_12(mht_12_v, 665, "", "./tensorflow/core/framework/function.cc", "Name");

    CHECK_LT(node_index, nodes_.size());
    return nodes_[node_index].name;
  }

  string Name(int node_index, int output_index) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_13(mht_13_v, 673, "", "./tensorflow/core/framework/function.cc", "Name");

    if (output_index == 0) {
      return Name(node_index);
    } else {
      return strings::StrCat(Name(node_index), ":", output_index);
    }
  }

  NodeDef* AddNode(const string& name) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_14(mht_14_v, 685, "", "./tensorflow/core/framework/function.cc", "AddNode");

    result_.nodes.emplace_back();
    NodeDef* gnode = &result_.nodes.back();
    gnode->set_name(name);
    nodes_.push_back({name, {}, {}});
    CHECK_EQ(result_.nodes.size(), nodes_.size());
    return gnode;
  }

  void AddInput(int node_index, int output_node, int output_index) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_15(mht_15_v, 697, "", "./tensorflow/core/framework/function.cc", "AddInput");

    CHECK_LT(node_index, nodes_.size());
    nodes_[node_index].data_inputs.push_back(
        std::make_pair(output_node, output_index));
  }

  void AddDep(int node_index, int dep_index) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_16(mht_16_v, 706, "", "./tensorflow/core/framework/function.cc", "AddDep");

    CHECK_LT(node_index, nodes_.size());
    nodes_[node_index].control_inputs.push_back(dep_index);
  }

  GetFunctionSignature get_function_;
  InstantiationResult& result_;
  // A small index for all names that can be used as a node's input arguments.
  std::map<string, NameInfoItem> index_;
  // This contains information about a node in the new graph including the node
  // names and input nodes' indexes.
  struct NodeInfo {
    string name;
    // Data inputs where <n, k> means arg k of node n.
    std::vector<std::pair<int, int>> data_inputs;
    // Control inputs (dependencies).
    std::vector<int> control_inputs;
  };
  // nodes_[i] is the information about result_.nodes[i].
  std::vector<NodeInfo> nodes_;
};

// Various helpers Print(proto) to print relevant protos to ascii.
string Print(const OpDef::ArgDef& arg) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_17(mht_17_v, 732, "", "./tensorflow/core/framework/function.cc", "Print");

  string out;
  strings::StrAppend(&out, arg.name(), ":");
  if (arg.is_ref()) strings::StrAppend(&out, "Ref(");
  if (!arg.number_attr().empty()) {
    strings::StrAppend(&out, arg.number_attr(), "*");
  }
  if (arg.type() != DT_INVALID) {
    strings::StrAppend(&out, DataTypeString(arg.type()));
  } else {
    strings::StrAppend(&out, arg.type_attr());
  }
  if (arg.is_ref()) strings::StrAppend(&out, ")");
  return out;
}

// TODO(josh11b): Merge this with SummarizeAttrValue().
// When hash_string_attrs = true, string attributes are hashed instead of being
// truncated with ellipses. This is done to reduce the chance of collisions when
// looking up functions using the canonical representation.
string Print(const AttrValue& attr_value,
             const bool hash_string_attrs = false) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_18(mht_18_v, 756, "", "./tensorflow/core/framework/function.cc", "Print");

  if (attr_value.value_case() == AttrValue::kType) {
    return DataTypeString(attr_value.type());
  } else if ((attr_value.value_case() == AttrValue::kList) &&
             (attr_value.list().type_size() > 0)) {
    string ret = "{";
    for (int i = 0; i < attr_value.list().type_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, DataTypeString(attr_value.list().type(i)));
    }
    strings::StrAppend(&ret, "}");
    return ret;
  } else if (attr_value.value_case() == AttrValue::kFunc) {
    if (attr_value.func().attr_size() == 0) {
      return attr_value.func().name();
    }
    std::vector<string> entries;
    for (const auto& p : attr_value.func().attr()) {
      entries.push_back(strings::StrCat(p.first, "=", Print(p.second)));
    }
    std::sort(entries.begin(), entries.end());
    return strings::StrCat(attr_value.func().name(), "[",
                           absl::StrJoin(entries, ", "), "]");
  } else if (attr_value.value_case() == AttrValue::kS && hash_string_attrs) {
    return strings::StrCat(Fingerprint64(attr_value.s()));
  }
  return SummarizeAttrValue(attr_value);
}

// TODO(josh11b): Merge this with SummarizeNodeDef().
string Print(const NodeDef& n) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_19(mht_19_v, 789, "", "./tensorflow/core/framework/function.cc", "Print");

  string out;
  strings::StrAppend(&out, n.name(), " = ", n.op());
  if (n.attr_size() > 0) {
    std::vector<string> entries;
    for (auto& a : n.attr()) {
      entries.push_back(strings::StrCat(a.first, "=", Print(a.second)));
    }
    std::sort(entries.begin(), entries.end());
    // Add a short device string at the end of all attributes.
    if (!n.device().empty()) {
      DeviceNameUtils::ParsedName parsed;
      if (DeviceNameUtils::ParseFullName(n.device(), &parsed)) {
        entries.push_back(
            strings::StrCat("device=", parsed.type, ":", parsed.id));
      } else {
        entries.push_back("device=<FAILED_TO_PARSE>");
      }
    }
    strings::StrAppend(&out, "[", absl::StrJoin(entries, ", "), "]");
  }
  strings::StrAppend(&out, "(");
  std::vector<StringPiece> dat;
  std::vector<string> dep;
  for (StringPiece s : n.input()) {
    if (absl::ConsumePrefix(&s, "^")) {
      dep.emplace_back(s);
    } else {
      dat.push_back(s);
    }
  }
  strings::StrAppend(&out, absl::StrJoin(dat, ", "), ")");
  if (!dep.empty()) {
    strings::StrAppend(&out, " @ ", absl::StrJoin(dep, ", "));
  }
  return out;
}

string Print(const FunctionDef& fdef) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_20(mht_20_v, 830, "", "./tensorflow/core/framework/function.cc", "Print");

  string out;
  const OpDef& sig = fdef.signature();
  strings::StrAppend(&out, "\n", sig.name());
  if (sig.attr_size() > 0) {
    strings::StrAppend(&out, "[");
    for (int i = 0; i < sig.attr_size(); ++i) {
      const auto& a = sig.attr(i);
      if (i > 0) strings::StrAppend(&out, ", ");
      if (a.type() == "type") {
        strings::StrAppend(&out, a.name(), ":", Print(a.allowed_values()));
      } else {
        strings::StrAppend(&out, a.name(), ":", a.type());
      }
    }
    strings::StrAppend(&out, "]");
  }
  strings::StrAppend(&out, "(");
  for (int i = 0; i < sig.input_arg_size(); ++i) {
    if (i > 0) strings::StrAppend(&out, ", ");
    strings::StrAppend(&out, Print(sig.input_arg(i)));
  }
  strings::StrAppend(&out, ") -> (");
  for (int i = 0; i < sig.output_arg_size(); ++i) {
    if (i > 0) strings::StrAppend(&out, ", ");
    strings::StrAppend(&out, Print(sig.output_arg(i)));
  }
  strings::StrAppend(&out, ") {\n");
  for (const auto& n : fdef.node_def()) {
    strings::StrAppend(&out, "  ", Print(n), "\n");
  }
  for (const auto& cr : fdef.control_ret()) {
    strings::StrAppend(&out, "  @return ", cr.first, " = ", cr.second, "\n");
  }
  for (const auto& r : fdef.ret()) {
    strings::StrAppend(&out, "  return ", r.first, " = ", r.second, "\n");
  }
  strings::StrAppend(&out, "}\n");
  return out;
}

string Print(gtl::ArraySlice<const NodeDef*> nodes) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_21(mht_21_v, 874, "", "./tensorflow/core/framework/function.cc", "Print");

  std::vector<const NodeDef*> arg;
  std::vector<const NodeDef*> ret;
  std::vector<const NodeDef*> body;
  for (const NodeDef* n : nodes) {
    if (n->op() == FunctionLibraryDefinition::kArgOp ||
        n->op() == FunctionLibraryDefinition::kDeviceArgOp) {
      arg.push_back(n);
    } else if (n->op() == FunctionLibraryDefinition::kRetOp ||
               n->op() == FunctionLibraryDefinition::kDeviceRetOp) {
      ret.push_back(n);
    } else {
      body.push_back(n);
    }
  }
  auto comp = [](const NodeDef* x, const NodeDef* y) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_22(mht_22_v, 892, "", "./tensorflow/core/framework/function.cc", "lambda");

    int xi;
    TF_CHECK_OK(GetNodeAttr(*x, "index", &xi));
    int yi;
    TF_CHECK_OK(GetNodeAttr(*y, "index", &yi));
    return xi < yi;
  };
  std::sort(arg.begin(), arg.end(), comp);
  std::sort(ret.begin(), ret.end(), comp);
  string out;
  strings::StrAppend(&out, "\n(");
  auto get_type_and_device = [](const NodeDef& n) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_23(mht_23_v, 906, "", "./tensorflow/core/framework/function.cc", "lambda");

    DataType dt;
    if (!TryGetNodeAttr(n, "T", &dt)) {
      dt = DT_INVALID;
    }
    if (!n.device().empty()) {
      DeviceNameUtils::ParsedName parsed;
      if (DeviceNameUtils::ParseFullName(n.device(), &parsed)) {
        return strings::StrCat(DataTypeString(dt), "@", parsed.type, ":",
                               parsed.id);
      } else {
        LOG(WARNING) << "Failed to parse device \"" << n.device() << "\" in "
                     << n.op() << ":" << n.name();
        return strings::StrCat(DataTypeString(dt), "@",
                               "<FAILED_TO_PARSE_DEVICE>");
      }
    }
    return DataTypeString(dt);
  };
  for (size_t i = 0; i < arg.size(); ++i) {
    const NodeDef* n = arg[i];
    if (i > 0) strings::StrAppend(&out, ", ");
    CHECK_GE(n->attr_size(), 2);
    strings::StrAppend(&out, n->name(), ":", get_type_and_device(*n));
  }
  strings::StrAppend(&out, ") -> (");
  for (size_t i = 0; i < ret.size(); ++i) {
    const NodeDef* n = ret[i];
    if (i > 0) strings::StrAppend(&out, ", ");
    CHECK_LE(2, n->attr_size());

    // The _RetVal op should have a unique non-control input. We assert that
    // here and add it to the output.
    bool found_non_control_input = false;
    for (const string& input : n->input()) {
      if (!input.empty() && input[0] != '^') {
        DCHECK_EQ(found_non_control_input, false)
            << "RetVal node has more than one non-control input: "
            << absl::StrJoin(n->input(), ", ");
        strings::StrAppend(&out, n->input(0), ":", get_type_and_device(*n));
        found_non_control_input = true;
      }
    }
    DCHECK_EQ(found_non_control_input, true)
        << "RetVal did not have any non-control inputs: "
        << absl::StrJoin(n->input(), ", ");
  }
  strings::StrAppend(&out, ") {\n");
  for (size_t i = 0; i < body.size(); ++i) {
    strings::StrAppend(&out, "  ", Print(*body[i]), "\n");
  }
  strings::StrAppend(&out, "}\n");
  return out;
}

Status AddDefaultAttrs(const string& op,
                       const GetFunctionSignature& get_function,
                       AttrValueMap* attrs) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_24(mht_24_v, 967, "", "./tensorflow/core/framework/function.cc", "AddDefaultAttrs");

  const OpDef* op_def = nullptr;
  TF_RETURN_IF_ERROR(get_function(op, &op_def));
  AttrSlice attr_slice(attrs);
  for (const auto& attr_def : op_def->attr()) {
    if (attr_def.has_default_value() && !attr_slice.Find(attr_def.name())) {
      if (!attrs->insert({attr_def.name(), attr_def.default_value()}).second) {
        return errors::Internal("Somehow duplicated: ", attr_def.name());
      }
    }
  }
  return Status::OK();
}

}  // end namespace

Status InstantiateFunction(const FunctionDef& fdef, AttrSlice attr_values,
                           GetFunctionSignature get_function,
                           InstantiationResult* result) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_25(mht_25_v, 988, "", "./tensorflow/core/framework/function.cc", "InstantiateFunction");

  if (VLOG_IS_ON(5)) {
    const auto& signature = fdef.signature();
    VLOG(5) << "Instantiate function definition: name=" << signature.name()
            << " #input_args=" << signature.input_arg_size()
            << " #output_args=" << signature.output_arg_size()
            << " #control_output=" << signature.control_output_size();
    for (const auto& line : str_util::Split(Print(fdef), '\n')) {
      VLOG(5) << "|| " << line;
    }
  }

  const OpDef& sig = fdef.signature();
  TF_RETURN_IF_ERROR(ValidateSignatureWithAttrs(sig, attr_values));

  bool ints_on_device =
      fdef.attr().count(FunctionLibraryDefinition::kIntsOnDeviceAttr) != 0 &&
      fdef.attr().at(FunctionLibraryDefinition::kIntsOnDeviceAttr).b();

  FunctionInstantiationHelper helper(get_function, result);
  Status s;
  for (int i = 0, e = sig.input_arg_size(); i < e; ++i) {
    const OpDef::ArgDef& arg_def = sig.input_arg(i);
    auto it = fdef.arg_attr().find(i);
    const FunctionDef::ArgAttrs* arg_attrs =
        it != fdef.arg_attr().end() ? &it->second : nullptr;
    auto resource_id_it = fdef.resource_arg_unique_id().find(i);
    int64_t resource_arg_unique_id =
        resource_id_it != fdef.resource_arg_unique_id().end()
            ? resource_id_it->second
            : -1LL;
    s = helper.BuildInputArgIndex(arg_def, attr_values, arg_attrs,
                                  ints_on_device, resource_arg_unique_id);

    if (!s.ok()) {
      errors::AppendToMessage(&s, "In ", Print(arg_def));
      return s;
    }
  }

  auto substitute = [attr_values, &sig](StringPiece name, AttrValue* val) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_26(mht_26_v, 1031, "", "./tensorflow/core/framework/function.cc", "lambda");

    // Look for a specified value...
    if (const AttrValue* v = attr_values.Find(name)) {
      *val = *v;
      return true;
    }
    // .. and if not, then check for a default value.
    if (const OpDef::AttrDef* attr = FindAttr(name, sig)) {
      if (attr->has_default_value()) {
        *val = attr->default_value();
        return true;
      }
    }
    // No luck finding a substitution.
    return false;
  };

  // Makes a copy of all attrs in fdef and substitutes placeholders.
  // After this step, every attr is bound to a concrete value.
  std::vector<AttrValueMap> node_attrs;
  node_attrs.resize(fdef.node_def_size());
  for (int i = 0; i < fdef.node_def_size(); ++i) {
    for (auto attr : fdef.node_def(i).attr()) {
      if (!SubstitutePlaceholders(substitute, &attr.second)) {
        return errors::InvalidArgument("Failed to bind all placeholders in ",
                                       SummarizeAttrValue(attr.second));
      }
      if (!node_attrs[i].insert(attr).second) {
        return errors::Internal("Somehow duplicated: ", attr.first);
      }
    }
    TF_RETURN_IF_ERROR(
        AddDefaultAttrs(fdef.node_def(i).op(), get_function, &node_attrs[i]));
  }

  for (int i = 0; i < fdef.node_def_size(); ++i) {
    s = helper.BuildNodeOutputIndex(fdef.node_def(i), AttrSlice(&node_attrs[i]),
                                    result->nodes.size() + i);
    if (!s.ok()) {
      errors::AppendToMessage(&s, "In ",
                              FormatNodeDefForError(fdef.node_def(i)));
      return s;
    }
  }
  // Emits one node for each fdef.node_def.
  for (int i = 0; i < fdef.node_def_size(); ++i) {
    s = helper.InstantiateNode(fdef.node_def(i), AttrSlice(&node_attrs[i]));
    if (!s.ok()) {
      errors::AppendToMessage(&s, "In ",
                              FormatNodeDefForError(fdef.node_def(i)));
      return s;
    }
  }

  // Emits nodes for the function's return values.
  int ret_index = 0;
  for (const OpDef::ArgDef& ret_def : sig.output_arg()) {
    s = helper.AddReturnNode(ret_def, attr_values, fdef.ret(), ints_on_device,
                             &ret_index);
    if (!s.ok()) {
      errors::AppendToMessage(&s, "In function output ", Print(ret_def));
      return s;
    }
  }

  // Adds the actual node inputs using the input indexes.
  helper.AddNodeInputs();

  return Status::OK();
}

string DebugString(const FunctionDef& func_def) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_27(mht_27_v, 1105, "", "./tensorflow/core/framework/function.cc", "DebugString");
 return Print(func_def); }

string DebugString(const GraphDef& instantiated_func_def) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_28(mht_28_v, 1110, "", "./tensorflow/core/framework/function.cc", "DebugString");

  std::vector<const NodeDef*> ptrs;
  for (const NodeDef& n : instantiated_func_def.node()) {
    ptrs.push_back(&n);
  }
  return Print(ptrs);
}

string DebugString(gtl::ArraySlice<NodeDef> instantiated_func_nodes) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_29(mht_29_v, 1121, "", "./tensorflow/core/framework/function.cc", "DebugString");

  std::vector<const NodeDef*> ptrs;
  for (const NodeDef& n : instantiated_func_nodes) {
    ptrs.push_back(&n);
  }
  return Print(ptrs);
}

string DebugStringWhole(const GraphDef& gdef) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_30(mht_30_v, 1132, "", "./tensorflow/core/framework/function.cc", "DebugStringWhole");

  string ret;
  for (const auto& fdef : gdef.library().function()) {
    strings::StrAppend(&ret, Print(fdef));
  }
  strings::StrAppend(&ret, "\n");
  for (const auto& ndef : gdef.node()) {
    strings::StrAppend(&ret, Print(ndef), "\n");
  }
  return ret;
}

namespace {

// Returns the name -> attr mapping of fdef's attrs that have a value set. In
// Python, it's possible to access unset attrs, which returns a default value
// and adds an unset attr to the map.
std::map<string, AttrValue> GetSetAttrs(const FunctionDef& fdef) {
  std::map<string, AttrValue> set_attrs;
  for (const auto& pair : fdef.attr()) {
    if (pair.second.value_case() != AttrValue::VALUE_NOT_SET) {
      set_attrs[pair.first] = pair.second;
    }
  }
  return set_attrs;
}

}  // end namespace

bool FunctionDefsEqual(const FunctionDef& f1, const FunctionDef& f2) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_31(mht_31_v, 1164, "", "./tensorflow/core/framework/function.cc", "FunctionDefsEqual");

  if (!OpDefEqual(f1.signature(), f2.signature())) return false;

  std::map<string, AttrValue> f1_attrs = GetSetAttrs(f1);
  std::map<string, AttrValue> f2_attrs = GetSetAttrs(f2);
  if (f1_attrs.size() != f2_attrs.size()) return false;
  for (const auto& iter1 : f1_attrs) {
    auto iter2 = f2_attrs.find(iter1.first);
    if (iter2 == f2_attrs.end()) return false;
    if (!AreAttrValuesEqual(iter1.second, iter2->second)) return false;
  }

  if (!EqualRepeatedNodeDef(f1.node_def(), f2.node_def(), nullptr)) {
    return false;
  }

  std::map<string, string> ret1(f1.ret().begin(), f1.ret().end());
  std::map<string, string> ret2(f2.ret().begin(), f2.ret().end());
  if (ret1 != ret2) return false;

  std::map<string, string> control_ret1(f1.control_ret().begin(),
                                        f1.control_ret().end());
  std::map<string, string> control_ret2(f2.control_ret().begin(),
                                        f2.control_ret().end());
  if (control_ret1 != control_ret2) return false;

  return true;
}

uint64 FunctionDefHash(const FunctionDef& fdef) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_32(mht_32_v, 1196, "", "./tensorflow/core/framework/function.cc", "FunctionDefHash");

  // signature
  uint64 h = OpDefHash(fdef.signature());

  // attrs
  std::map<string, AttrValue> attrs = GetSetAttrs(fdef);
  for (const auto& p : attrs) {
    h = Hash64(p.first.data(), p.first.size(), h);
    h = Hash64Combine(AttrValueHash(p.second), h);
  }

  // node defs
  h = Hash64Combine(RepeatedNodeDefHash(fdef.node_def()), h);

  // output names
  std::map<string, string> ret(fdef.ret().begin(), fdef.ret().end());
  for (const auto& p : ret) {
    h = Hash64(p.first.data(), p.first.size(), h);
    h = Hash64(p.second.data(), p.second.size(), h);
  }

  // control output names
  std::map<string, string> control_ret(fdef.control_ret().begin(),
                                       fdef.control_ret().end());
  for (const auto& p : control_ret) {
    h = Hash64(p.first.data(), p.first.size(), h);
    h = Hash64(p.second.data(), p.second.size(), h);
  }

  return h;
}

static constexpr const char* const kExecutorAttr = "_executor";

/* static */
string FunctionLibraryRuntime::ExecutorType(const InstantiateOptions& options,
                                            AttrSlice attrs) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_33(mht_33_v, 1235, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryRuntime::ExecutorType");

  if (!options.executor_type.empty()) {
    return options.executor_type;
  } else if (const AttrValue* executor_attr = attrs.Find(kExecutorAttr)) {
    return executor_attr->s();
  } else {
    return string();
  }
}

namespace {
class AttrKeyAndValue {
 public:
  enum ValueRepresentationOp {
    kRaw,
    kCEscape,
  };
  AttrKeyAndValue(absl::string_view key_name, int key_suffix, string value,
                  ValueRepresentationOp value_op = kRaw)
      : key_name_(key_name),
        key_suffix_(key_suffix),
        value_op_(value_op),
        value_(std::move(value)) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("key_name: \"" + std::string(key_name.data(), key_name.size()) + "\"");
   mht_34_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_34(mht_34_v, 1262, "", "./tensorflow/core/framework/function.cc", "AttrKeyAndValue");
}

  bool operator<(const AttrKeyAndValue& b) const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_35(mht_35_v, 1267, "", "./tensorflow/core/framework/function.cc", "operator<");

    if (key_name_ != b.key_name_) {
      return key_name_ < b.key_name_;
    } else if (key_suffix_ != b.key_suffix_) {
      return key_suffix_ < b.key_suffix_;
    } else {
      return value_ < b.value_;
    }
  }

  void AppendTo(bool first, string* s) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_36(mht_36_v, 1280, "", "./tensorflow/core/framework/function.cc", "AppendTo");

    absl::string_view v;
    bool add_escaped = false;
    if ((value_op_ == kCEscape) && NeedsEscaping(value_)) {
      // Use CEscape call below
      add_escaped = true;
    } else {
      // Add raw value contents directly
      v = value_;
    }
    if (key_suffix_ >= 0) {
      strings::StrAppend(s, first ? "" : ",", key_name_, key_suffix_, "=", v);
    } else {
      strings::StrAppend(s, first ? "" : ",", key_name_, "=", v);
    }
    if (add_escaped) {
      strings::StrAppend(s, absl::CEscape(value_));
    }
  }

 private:
  static bool NeedsEscaping(const string& s) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_37(mht_37_v, 1305, "", "./tensorflow/core/framework/function.cc", "NeedsEscaping");

    for (auto c : s) {
      if (!isalnum(c) && (c != ' ')) {
        return true;
      }
    }
    return false;
  }

  absl::string_view key_name_;
  int key_suffix_;  // -1 if missing
  ValueRepresentationOp value_op_;
  string value_;
};
}  // namespace

string GetFunctionResourceInputDevice(
    const Tensor& input, const int arg_index, const FunctionDef& function_def,
    absl::flat_hash_map<string, std::vector<string>>* composite_devices) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_38(mht_38_v, 1326, "", "./tensorflow/core/framework/function.cc", "GetFunctionResourceInputDevice");

  const auto& handles = input.flat<ResourceHandle>();
  const ResourceHandle& handle0 = handles(0);
  string composite_device;
  auto iter = function_def.arg_attr().find(arg_index);
  if (iter != function_def.arg_attr().end()) {
    auto arg_attr = iter->second.attr().find("_composite_device");
    if (arg_attr != iter->second.attr().end()) {
      composite_device = arg_attr->second.s();
    }
  }
  if (!composite_device.empty()) {
    if (composite_devices->find(composite_device) == composite_devices->end()) {
      for (int i = 0; i < handles.size(); ++i) {
        (*composite_devices)[composite_device].push_back(handles(i).device());
      }
    }
    return composite_device;
  } else {
    return handle0.device();
  }
}

string Canonicalize(const string& funcname, AttrSlice attrs,
                    const FunctionLibraryRuntime::InstantiateOptions& options) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("funcname: \"" + funcname + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_39(mht_39_v, 1354, "", "./tensorflow/core/framework/function.cc", "Canonicalize");

  absl::InlinedVector<AttrKeyAndValue, 8> entries;
  entries.reserve(attrs.size() + static_cast<int>(!options.target.empty()) +
                  options.input_devices.size());
  for (const auto& p : attrs) {
    if (p.first != kExecutorAttr) {
      entries.push_back(AttrKeyAndValue(
          p.first, -1, Print(p.second, /*hash_string_attrs=*/true)));
    }
  }
  if (!options.target.empty()) {
    entries.push_back(AttrKeyAndValue("_target", -1, options.target,
                                      AttrKeyAndValue::kCEscape));
  }
  for (int i = 0; i < options.input_devices.size(); ++i) {
    entries.push_back(AttrKeyAndValue("_input_dev", i, options.input_devices[i],
                                      AttrKeyAndValue::kCEscape));
  }
  for (int i = 0; i < options.output_devices.size(); ++i) {
    entries.push_back(AttrKeyAndValue("_output_dev", i,
                                      options.output_devices[i],
                                      AttrKeyAndValue::kCEscape));
  }
  for (const auto& iter : options.input_resource_dtypes_and_shapes) {
    entries.push_back(AttrKeyAndValue("_input_resource_dtype", iter.first,
                                      DataTypeString(iter.second.dtype)));
    entries.push_back(AttrKeyAndValue("_input_resource_shape", iter.first,
                                      iter.second.shape.DebugString(),
                                      AttrKeyAndValue::kCEscape));
  }
  if (options.lib_def) {
    entries.push_back(AttrKeyAndValue(
        "_lib_def", -1,
        absl::StrCat("", reinterpret_cast<uintptr_t>(options.lib_def))));
  }
  if (!options.state_handle.empty()) {
    entries.push_back(
        AttrKeyAndValue("_state_handle", -1, options.state_handle));
  }
  string executor_type = FunctionLibraryRuntime::ExecutorType(options, attrs);
  if (!executor_type.empty()) {
    entries.push_back(AttrKeyAndValue(kExecutorAttr, -1, executor_type));
  }
  if (options.config_proto.ByteSize() > 0) {
    string config_proto_serialized;
    SerializeToStringDeterministic(options.config_proto,
                                   &config_proto_serialized);
    entries.push_back(AttrKeyAndValue("_config_proto", -1,
                                      config_proto_serialized,
                                      AttrKeyAndValue::kCEscape));
  }
  std::sort(entries.begin(), entries.end());
  string result = strings::StrCat(funcname, "[");
  bool first = true;
  for (const auto& entry : entries) {
    entry.AppendTo(first, &result);
    first = false;
  }
  result += "]";
  return result;
}

string Canonicalize(const string& funcname, AttrSlice attrs) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("funcname: \"" + funcname + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_40(mht_40_v, 1420, "", "./tensorflow/core/framework/function.cc", "Canonicalize");

  static const FunctionLibraryRuntime::InstantiateOptions* kEmptyOptions =
      new FunctionLibraryRuntime::InstantiateOptions;
  return Canonicalize(funcname, attrs, *kEmptyOptions);
}

FunctionCallFrame::FunctionCallFrame(DataTypeSlice arg_types,
                                     DataTypeSlice ret_types)
    : arg_types_(arg_types.begin(), arg_types.end()),
      ret_types_(ret_types.begin(), ret_types.end()) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_41(mht_41_v, 1432, "", "./tensorflow/core/framework/function.cc", "FunctionCallFrame::FunctionCallFrame");

  args_.resize(arg_types_.size());
  rets_.resize(ret_types_.size());
}

FunctionCallFrame::~FunctionCallFrame() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_42(mht_42_v, 1440, "", "./tensorflow/core/framework/function.cc", "FunctionCallFrame::~FunctionCallFrame");
}

Status FunctionCallFrame::SetArgs(gtl::ArraySlice<Tensor> args) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_43(mht_43_v, 1445, "", "./tensorflow/core/framework/function.cc", "FunctionCallFrame::SetArgs");

  // Input type checks.
  if (args.size() != arg_types_.size()) {
    return errors::InvalidArgument("Expects ", arg_types_.size(),
                                   " arguments, but ", args.size(),
                                   " is provided");
  }
  for (size_t i = 0; i < args.size(); ++i) {
    if (arg_types_[i] != args[i].dtype()) {
      return errors::InvalidArgument(
          "Expects arg[", i, "] to be ", DataTypeString(arg_types_[i]), " but ",
          DataTypeString(args[i].dtype()), " is provided");
    }
    args_[i] = args[i];
  }
  return Status::OK();
}

Status FunctionCallFrame::GetRetvals(std::vector<Tensor>* rets) const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_44(mht_44_v, 1466, "", "./tensorflow/core/framework/function.cc", "FunctionCallFrame::GetRetvals");

  rets->clear();
  rets->reserve(rets_.size());
  for (size_t i = 0; i < rets_.size(); ++i) {
    const auto& item = rets_[i];
    if (item.has_val) {
      rets->push_back(item.val);
    } else {
      return errors::Internal("Retval[", i, "] does not have value");
    }
  }
  return Status::OK();
}

Status FunctionCallFrame::ConsumeRetvals(std::vector<Tensor>* rets,
                                         bool allow_dead_tensors) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_45(mht_45_v, 1484, "", "./tensorflow/core/framework/function.cc", "FunctionCallFrame::ConsumeRetvals");

  rets->clear();
  rets->reserve(rets_.size());
  for (size_t i = 0; i < rets_.size(); ++i) {
    if (rets_[i].has_val) {
      rets->emplace_back(std::move(rets_[i].val));
    } else if (allow_dead_tensors) {
      rets->emplace_back();
    } else {
      return errors::Internal("Retval[", i, "] does not have value");
    }
  }
  return Status::OK();
}

Status FunctionCallFrame::GetArg(int index, const Tensor** val) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_46(mht_46_v, 1502, "", "./tensorflow/core/framework/function.cc", "FunctionCallFrame::GetArg");

  if (index < 0 || static_cast<size_t>(index) >= args_.size()) {
    return errors::InvalidArgument("GetArg ", index, " is not within [0, ",
                                   args_.size(), ")");
  }
  *val = &args_[index];
  return Status::OK();
}

Status FunctionCallFrame::SetRetval(int index, const Tensor& val) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_47(mht_47_v, 1514, "", "./tensorflow/core/framework/function.cc", "FunctionCallFrame::SetRetval");

  if (index < 0 || static_cast<size_t>(index) >= rets_.size()) {
    return errors::InvalidArgument("SetRetval ", index, " is not within [0, ",
                                   rets_.size(), ")");
  }
  if (val.dtype() != ret_types_[index]) {
    return errors::InvalidArgument(
        "Expects ret[", index, "] to be ", DataTypeString(ret_types_[index]),
        ", but ", DataTypeString(val.dtype()), " is provided.");
  }
  Retval* item = &rets_[index];
  if (!item->has_val) {
    item->has_val = true;
    item->val = val;
  } else {
    return errors::Internal("Retval[", index, "] has already been set.");
  }
  return Status::OK();
}

FunctionLibraryDefinition::FunctionDefAndOpRegistration::
    FunctionDefAndOpRegistration(const FunctionDef& fdef_in,
                                 const StackTracesMap& stack_traces)
    : fdef(fdef_in),
      // Exact shape inference for functions is handled by ShapeRefiner.
      // Here we pass a dummy shape inference function for legacy code paths.
      op_registration_data(fdef.signature(), shape_inference::UnknownShape,
                           true /* is_function */),
      stack_traces(stack_traces) {}

FunctionLibraryDefinition::FunctionLibraryDefinition(
    const FunctionLibraryDefinition& other)
    : default_registry_(other.default_registry_) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_48(mht_48_v, 1549, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::FunctionLibraryDefinition");

  tf_shared_lock l(other.mu_);
  function_defs_ = other.function_defs_;
  func_grad_ = other.func_grad_;
}

FunctionLibraryDefinition::FunctionLibraryDefinition(
    const OpRegistryInterface* default_registry,
    const FunctionDefLibrary& def_lib)
    : default_registry_(default_registry),
      function_defs_(def_lib.function_size()) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_49(mht_49_v, 1562, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::FunctionLibraryDefinition");

  for (const auto& fdef : def_lib.function()) {
    // The latter function definition wins.
    auto& ptr = function_defs_[fdef.signature().name()];
    ptr.reset(new FunctionDefAndOpRegistration(fdef));
  }
  for (const auto& grad : def_lib.gradient()) {
    func_grad_[grad.function_name()] = grad.gradient_func();
  }
}

FunctionLibraryDefinition::~FunctionLibraryDefinition() {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_50(mht_50_v, 1576, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::~FunctionLibraryDefinition");
}

bool FunctionLibraryDefinition::Contains(const string& func) const {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_51(mht_51_v, 1582, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::Contains");

  tf_shared_lock l(mu_);
  return function_defs_.find(func) != function_defs_.end();
}

const FunctionDef* FunctionLibraryDefinition::Find(const string& func) const {
   std::vector<std::string> mht_52_v;
   mht_52_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_52(mht_52_v, 1591, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::Find");

  tf_shared_lock l(mu_);
  auto result = FindHelper(func);
  if (result) {
    return &result->fdef;
  } else {
    return nullptr;
  }
}

std::shared_ptr<FunctionLibraryDefinition::FunctionDefAndOpRegistration>
FunctionLibraryDefinition::FindHelper(const string& func) const {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_53(mht_53_v, 1606, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::FindHelper");

  auto iter = function_defs_.find(func);
  if (iter == function_defs_.end()) {
    return nullptr;
  } else {
    return iter->second;
  }
}

Status FunctionLibraryDefinition::AddFunctionDef(
    const FunctionDef& fdef, const StackTracesMap& stack_traces) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_54(mht_54_v, 1619, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::AddFunctionDef");

  mutex_lock l(mu_);
  bool added;
  return AddFunctionDefHelper(fdef, stack_traces, &added);
}

Status FunctionLibraryDefinition::AddFunctionDefHelper(
    const FunctionDef& fdef, const StackTracesMap& stack_traces, bool* added) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_55(mht_55_v, 1629, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::AddFunctionDefHelper");

  *added = false;
  std::shared_ptr<FunctionDefAndOpRegistration>& entry =
      function_defs_[fdef.signature().name()];
  if (entry) {
    if (!FunctionDefsEqual(entry->fdef, fdef)) {
      return errors::InvalidArgument(
          "Cannot add function '", fdef.signature().name(),
          "' because a different function with the same name already "
          "exists.");
    }
    // Ignore duplicate FunctionDefs.
    return Status::OK();
  }
  const OpDef* op_def;
  if (default_registry_->LookUpOpDef(fdef.signature().name(), &op_def).ok()) {
    return errors::InvalidArgument(
        "Cannot add function '", fdef.signature().name(),
        "' because an op with the same name already exists.");
  }
  entry = std::make_shared<FunctionDefAndOpRegistration>(fdef, stack_traces);
  *added = true;
  return Status::OK();
}

Status FunctionLibraryDefinition::AddHelper(
    std::shared_ptr<FunctionDefAndOpRegistration> registration, bool* added) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_56(mht_56_v, 1658, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::AddHelper");

  *added = false;
  std::shared_ptr<FunctionDefAndOpRegistration>& entry =
      function_defs_[registration->fdef.signature().name()];
  if (entry) {
    if (!FunctionDefsEqual(entry->fdef, registration->fdef)) {
      return errors::InvalidArgument(
          "Cannot add function '", registration->fdef.signature().name(),
          "' because a different function with the same name already "
          "exists.");
    }
    // Ignore duplicate FunctionDefs.
    return Status::OK();
  }
  const OpDef* op_def;
  if (default_registry_
          ->LookUpOpDef(registration->fdef.signature().name(), &op_def)
          .ok()) {
    return errors::InvalidArgument(
        "Cannot add function '", registration->fdef.signature().name(),
        "' because an op with the same name already exists.");
  }
  entry = std::move(registration);
  *added = true;
  return Status::OK();
}

Status FunctionLibraryDefinition::CopyFunctionDefFrom(
    const string& func, const FunctionLibraryDefinition& other) {
   std::vector<std::string> mht_57_v;
   mht_57_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_57(mht_57_v, 1690, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::CopyFunctionDefFrom");

  if (default_registry_ != other.default_registry_) {
    return errors::InvalidArgument(
        "Cannot copy function '", func,
        "' because CopyFunctionDefFrom() requires that both libraries have the "
        "same default registry.");
  }
  std::shared_ptr<FunctionDefAndOpRegistration> function_def;
  {
    tf_shared_lock l(other.mu_);
    function_def = other.FindHelper(func);
  }
  if (!function_def) {
    return errors::InvalidArgument(
        "Cannot copy function '", func,
        "' because no function with that name exists in the other library.");
  }
  {
    mutex_lock l(mu_);
    std::shared_ptr<FunctionDefAndOpRegistration>& entry = function_defs_[func];
    if (entry) {
      if (!FunctionDefsEqual(entry->fdef, function_def->fdef)) {
        return errors::InvalidArgument(
            "Cannot copy function '", func,
            "' because a different function with the same name already "
            "exists.");
      }
    } else {
      entry = std::move(function_def);
    }
  }
  return Status::OK();
}

Status FunctionLibraryDefinition::AddGradientDef(const GradientDef& grad) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_58(mht_58_v, 1727, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::AddGradientDef");

  mutex_lock l(mu_);
  bool added;
  return AddGradientDefHelper(grad, &added);
}

Status FunctionLibraryDefinition::AddGradientDefHelper(const GradientDef& grad,
                                                       bool* added) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_59(mht_59_v, 1737, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::AddGradientDefHelper");

  *added = false;
  string* entry = &func_grad_[grad.function_name()];
  if (!entry->empty()) {
    if (*entry != grad.gradient_func()) {
      return errors::InvalidArgument(
          "Cannot assign gradient function '", grad.gradient_func(), "' to '",
          grad.function_name(), "' because it already has gradient function ",
          "'", *entry, "'");
    }
    // Ignore duplicate GradientDefs
    return Status::OK();
  }
  *entry = grad.gradient_func();
  *added = true;
  return Status::OK();
}

Status FunctionLibraryDefinition::AddLibrary(
    const FunctionLibraryDefinition& other) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_60(mht_60_v, 1759, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::AddLibrary");

  // Clone `other` to ensure thread-safety (grabbing `other`'s lock for
  // the duration of the function could lead to deadlock).
  FunctionLibraryDefinition clone(other);
  mutex_lock l(mu_);
  mutex_lock l2(clone.mu_);
  // Remember the funcs and grads that we added successfully so that
  // we can roll them back on error.
  std::vector<string> funcs;
  std::vector<string> funcs_with_grads;
  Status s;
  bool added;
  for (auto iter : clone.function_defs_) {
    s = AddHelper(iter.second, &added);
    if (!s.ok()) {
      Status remove_status = Remove(funcs, funcs_with_grads);
      if (!remove_status.ok()) {
        return remove_status;
      }
      return s;
    }
    if (added) {
      funcs.push_back(iter.second->fdef.signature().name());
    }
  }
  for (auto iter : clone.func_grad_) {
    GradientDef grad;
    grad.set_function_name(iter.first);
    grad.set_gradient_func(iter.second);
    s = AddGradientDefHelper(grad, &added);
    if (!s.ok()) {
      Status remove_status = Remove(funcs, funcs_with_grads);
      if (!remove_status.ok()) {
        return remove_status;
      }
      return s;
    }
    if (added) {
      funcs_with_grads.push_back(grad.function_name());
    }
  }
  return Status::OK();
}

Status FunctionLibraryDefinition::AddLibrary(
    const FunctionDefLibrary& lib_def) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_61(mht_61_v, 1807, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::AddLibrary");

  // Remember the funcs and grads that we added successfully so that
  // we can roll them back on error.
  mutex_lock l(mu_);
  std::vector<string> funcs;
  std::vector<string> funcs_with_grads;
  Status s;
  bool added;
  for (const FunctionDef& fdef : lib_def.function()) {
    s = AddFunctionDefHelper(fdef, /*stack_traces=*/{}, &added);
    if (!s.ok()) {
      Status remove_status = Remove(funcs, funcs_with_grads);
      if (!remove_status.ok()) {
        return remove_status;
      }
      return s;
    }
    if (added) {
      funcs.push_back(fdef.signature().name());
    }
  }
  for (const GradientDef& grad : lib_def.gradient()) {
    s = AddGradientDefHelper(grad, &added);
    if (!s.ok()) {
      Status remove_status = Remove(funcs, funcs_with_grads);
      if (!remove_status.ok()) {
        return remove_status;
      }
      return s;
    }
    if (added) {
      funcs_with_grads.push_back(grad.function_name());
    }
  }
  return Status::OK();
}

Status FunctionLibraryDefinition::ReplaceFunction(
    const string& func, const FunctionDef& fdef,
    const StackTracesMap& stack_traces) {
   std::vector<std::string> mht_62_v;
   mht_62_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_62(mht_62_v, 1850, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::ReplaceFunction");

  mutex_lock l(mu_);
  bool added;
  TF_RETURN_IF_ERROR(RemoveFunctionHelper(func));
  TF_RETURN_IF_ERROR(AddFunctionDefHelper(fdef, stack_traces, &added));
  return Status::OK();
}

Status FunctionLibraryDefinition::ReplaceGradient(const GradientDef& grad) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_63(mht_63_v, 1861, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::ReplaceGradient");

  mutex_lock l(mu_);
  bool added;
  TF_RETURN_IF_ERROR(RemoveGradient(grad.function_name()));
  TF_RETURN_IF_ERROR(AddGradientDefHelper(grad, &added));
  return Status::OK();
}

Status FunctionLibraryDefinition::RemoveFunction(const string& func) {
   std::vector<std::string> mht_64_v;
   mht_64_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_64(mht_64_v, 1873, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::RemoveFunction");

  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(RemoveFunctionHelper(func));
  return Status::OK();
}

Status FunctionLibraryDefinition::RemoveFunctionHelper(const string& func) {
   std::vector<std::string> mht_65_v;
   mht_65_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_65(mht_65_v, 1883, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::RemoveFunctionHelper");

  const auto& i = function_defs_.find(func);
  if (i == function_defs_.end()) {
    return errors::InvalidArgument("Tried to remove non-existent function '",
                                   func, "'.");
  }
  function_defs_.erase(i);
  return Status::OK();
}

void FunctionLibraryDefinition::Clear() {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_66(mht_66_v, 1896, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::Clear");

  mutex_lock l(mu_);
  function_defs_.clear();
  func_grad_.clear();
}

Status FunctionLibraryDefinition::RemoveGradient(const string& func) {
   std::vector<std::string> mht_67_v;
   mht_67_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_67(mht_67_v, 1906, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::RemoveGradient");

  const auto& i = func_grad_.find(func);
  if (i == func_grad_.end()) {
    return errors::InvalidArgument("Tried to remove non-existent gradient '",
                                   func, "'.");
  }
  func_grad_.erase(i);
  return Status::OK();
}

Status FunctionLibraryDefinition::Remove(
    const std::vector<string>& funcs,
    const std::vector<string>& funcs_with_grads) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_68(mht_68_v, 1921, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::Remove");

  Status s;
  for (const string& f : funcs) {
    s = RemoveFunctionHelper(f);
    if (!s.ok()) {
      return s;
    }
  }
  for (const string& f : funcs_with_grads) {
    s = RemoveGradient(f);
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

string FunctionLibraryDefinition::FindGradient(const string& func) const {
   std::vector<std::string> mht_69_v;
   mht_69_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_69(mht_69_v, 1942, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::FindGradient");

  tf_shared_lock l(mu_);
  return gtl::FindWithDefault(func_grad_, func, "");
}

string FunctionLibraryDefinition::FindGradientHelper(const string& func) const {
   std::vector<std::string> mht_70_v;
   mht_70_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_70(mht_70_v, 1951, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::FindGradientHelper");

  return gtl::FindWithDefault(func_grad_, func, "");
}

Status FunctionLibraryDefinition::LookUp(
    const string& op, const OpRegistrationData** op_reg_data) const {
   std::vector<std::string> mht_71_v;
   mht_71_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_71(mht_71_v, 1960, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::LookUp");

  tf_shared_lock l(mu_);
  auto iter = function_defs_.find(op);
  if (iter != function_defs_.end()) {
    *op_reg_data = &iter->second->op_registration_data;
    return Status::OK();
  }
  return default_registry_->LookUp(op, op_reg_data);
}

string FunctionLibraryDefinition::UniqueFunctionName(StringPiece prefix) const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_72(mht_72_v, 1973, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::UniqueFunctionName");

  tf_shared_lock l(mu_);
  int index = 0;
  string name = strings::StrCat(prefix, index);
  while (function_defs_.find(name) != function_defs_.end()) {
    ++index;
    name = strings::StrCat(prefix, index);
  }
  return name;
}

const FunctionDef* FunctionLibraryDefinition::GetAttrImpl(
    const NodeDef& ndef) const {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_73(mht_73_v, 1988, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::GetAttrImpl");

  if (ndef.op() != kGradientOp) {
    // If 'ndef' calls a function and the function's def has the attr,
    // returns it.
    return Find(ndef.op());
  }

  // If ndef is SymbolicGradient[f=Foo], we use Foo's gradient or
  // Foo's attributes.
  const NameAttrList* forward_func_attrs;
  if (!TryGetNodeAttr(ndef, kFuncAttr, &forward_func_attrs)) {
    return nullptr;
  }
  const string& func_name = forward_func_attrs->name();
  {
    tf_shared_lock l(mu_);
    const string& grad_name = FindGradientHelper(func_name);
    // If 'func' has a user-defined gradient function, uses the grad
    // function's attrs to see if noinline is specified. Otherwise,
    // uses func's attrs.
    if (!grad_name.empty()) {
      if (const auto helper = FindHelper(grad_name)) {
        return &(helper->fdef);
      } else {
        return nullptr;
      }
    }
    if (const auto helper = FindHelper(func_name)) {
      return &(helper->fdef);
    } else {
      return nullptr;
    }
  }
}

std::vector<string> FunctionLibraryDefinition::ListFunctionNames() const {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_74(mht_74_v, 2026, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::ListFunctionNames");

  std::vector<string> function_names;
  tf_shared_lock l(mu_);
  function_names.reserve(function_defs_.size());
  for (const auto& it : function_defs_) {
    function_names.emplace_back(it.first);
  }
  return function_names;
}

FunctionDefLibrary FunctionLibraryDefinition::ToProto() const {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_75(mht_75_v, 2039, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::ToProto");

  FunctionDefLibrary lib;
  tf_shared_lock l(mu_);
  for (const auto& f : function_defs_) {
    *lib.add_function() = f.second->fdef;
  }
  for (const auto& g : func_grad_) {
    GradientDef* gd = lib.add_gradient();
    gd->set_function_name(g.first);
    gd->set_gradient_func(g.second);
  }
  return lib;
}

template <typename T>
Status FunctionLibraryDefinition::GetAttr(const NodeDef& ndef,
                                          const string& attr, T* value) const {
   std::vector<std::string> mht_76_v;
   mht_76_v.push_back("attr: \"" + attr + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_76(mht_76_v, 2059, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::GetAttr");

  const FunctionDef* fdef = GetAttrImpl(ndef);
  if (fdef && TryGetNodeAttr(AttrSlice(&fdef->attr()), attr, value)) {
    return Status::OK();
  }
  return errors::InvalidArgument("Attr ", attr, " is not defined.");
}

template <typename T>
Status FunctionLibraryDefinition::GetAttr(const Node& node, const string& attr,
                                          T* value) const {
   std::vector<std::string> mht_77_v;
   mht_77_v.push_back("attr: \"" + attr + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_77(mht_77_v, 2073, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::GetAttr");

  return GetAttr(node.def(), attr, value);
}

#define GET_ATTR(T)                                                            \
  template Status FunctionLibraryDefinition::GetAttr(const Node&,              \
                                                     const string&, T*) const; \
  template Status FunctionLibraryDefinition::GetAttr(const NodeDef&,           \
                                                     const string&, T*) const;
GET_ATTR(string)
GET_ATTR(bool)
#undef GET_ATTR

namespace {

constexpr char kApiImplements[] = "api_implements";

std::set<string> ReachableFunctions(
    const FunctionLibraryDefinition& flib,
    const protobuf::RepeatedPtrField<NodeDef>& nodes) {
  // Functions that are reachable from the graph.
  std::set<string> reachable_funcs;

  // For any functions, if it has attribute "api_implements" =
  // "some_interface" and it is reachable, then it means any other
  // function with same attribute name and value could also be potentially
  // reachable, eg via implementation_selector swapping the nodedef.
  absl::flat_hash_set<string> reachable_api_interface;

  // Functions might be reachable from the nested function calls, so we keep a
  // queue of functions that we have to check.
  gtl::InlinedVector<const FunctionDef*, 4> func_queue;

  // Add reachable and not already processed functions to the functions queue.
  const auto add_to_func_queue = [&](const string& func_name) {
   std::vector<std::string> mht_78_v;
   mht_78_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_78(mht_78_v, 2111, "", "./tensorflow/core/framework/function.cc", "lambda");

    const FunctionDef* func = flib.Find(func_name);
    if (func && reachable_funcs.find(func_name) == reachable_funcs.end()) {
      func_queue.push_back(func);
    }
  };

  // If any function with certain API name is reachable, all the other functions
  // with same API name should also be checked.
  const auto add_function_with_api_interface = [&](const string& api_name) {
   std::vector<std::string> mht_79_v;
   mht_79_v.push_back("api_name: \"" + api_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_79(mht_79_v, 2124, "", "./tensorflow/core/framework/function.cc", "lambda");

    if (!reachable_api_interface.contains(api_name)) {
      reachable_api_interface.insert(api_name);
      for (const auto& func_name : flib.ListFunctionNames()) {
        const auto& func_def = flib.Find(func_name);
        const auto attr_it = func_def->attr().find(kApiImplements);
        if (attr_it != func_def->attr().end() &&
            attr_it->second.s() == api_name) {
          add_to_func_queue(func_name);
        }
      }
    }
  };

  // Add all the functions that are reachable from the given node to the queue.
  const auto process_node = [&](const NodeDef& node) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_80(mht_80_v, 2142, "", "./tensorflow/core/framework/function.cc", "lambda");

    // Node itself can be a call to the function.
    add_to_func_queue(node.op());

    // Or node can have an attribute referencing a function.
    for (const auto& attr : node.attr()) {
      const auto& attr_value = attr.second;

      // 1. AttrValue.func
      if (attr_value.has_func()) {
        add_to_func_queue(attr_value.func().name());
      }

      // 2. AttrValue.ListValue.func
      if (attr_value.has_list()) {
        for (const auto& func : attr_value.list().func()) {
          add_to_func_queue(func.name());
        }
      }
    }
  };

  // Add all functions that are directly called from the optimized graph.
  std::for_each(nodes.begin(), nodes.end(), process_node);

  // Process all reachable functions.
  while (!func_queue.empty()) {
    const FunctionDef* func = func_queue.back();
    func_queue.pop_back();

    const string& func_name = func->signature().name();
    reachable_funcs.insert(func_name);

    const auto attr_it = func->attr().find(kApiImplements);
    if (attr_it != func->attr().end()) {
      add_function_with_api_interface(attr_it->second.s());
    }

    // Find all the functions called from the function body.
    const auto& func_body = func->node_def();
    std::for_each(func_body.begin(), func_body.end(), process_node);

    // Check if the function has a registered gradient.
    const string grad_func_name = flib.FindGradient(func_name);
    if (!grad_func_name.empty()) add_to_func_queue(grad_func_name);
  }

  return reachable_funcs;
}

FunctionLibraryDefinition ReachableFunctionLibraryDefinition(
    const FunctionLibraryDefinition& flib,
    const protobuf::RepeatedPtrField<NodeDef>& nodes) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_81(mht_81_v, 2197, "", "./tensorflow/core/framework/function.cc", "ReachableFunctionLibraryDefinition");

  std::set<string> reachable_funcs = ReachableFunctions(flib, nodes);

  FunctionLibraryDefinition reachable_flib(flib.default_registry(),
                                           FunctionDefLibrary());

  for (const string& func_name : reachable_funcs) {
    // This should never fail, because we copy functions from a valid flib and
    // use the same default registry.
    Status added = reachable_flib.CopyFunctionDefFrom(func_name, flib);
    TF_DCHECK_OK(added);

    const string grad_func_name = flib.FindGradient(func_name);
    if (!grad_func_name.empty()) {
      GradientDef grad;
      grad.set_function_name(func_name);
      grad.set_gradient_func(grad_func_name);
      // It can only fail if function already has a gradient function.
      const Status added_grad = reachable_flib.AddGradientDef(grad);
      TF_DCHECK_OK(added_grad);
    }
  }

  return reachable_flib;
}

string AllocatorAttributesToString(
    const std::vector<AllocatorAttributes>& attrs) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_82(mht_82_v, 2227, "", "./tensorflow/core/framework/function.cc", "AllocatorAttributesToString");

  string result("[");
  // AllocatorAttribute::DebugString produces around 85 bytes now.
  result.reserve(100 * attrs.size());
  for (const AllocatorAttributes& attr : attrs) {
    result.append(attr.DebugString());
    result.append(", ");
  }
  if (!attrs.empty()) {
    result.resize(result.size() - 2);
  }
  result.append("]");
  return result;
}

const char* IsSet(void* ptr) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_83(mht_83_v, 2245, "", "./tensorflow/core/framework/function.cc", "IsSet");
 return ptr == nullptr ? "unset" : "set"; }

}  // namespace

FunctionLibraryDefinition FunctionLibraryDefinition::ReachableDefinitions(
    const GraphDef& graph) const {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_84(mht_84_v, 2253, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::ReachableDefinitions");

  return ReachableFunctionLibraryDefinition(*this, graph.node());
}

FunctionLibraryDefinition FunctionLibraryDefinition::ReachableDefinitions(
    const FunctionDef& func) const {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_85(mht_85_v, 2261, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryDefinition::ReachableDefinitions");

  return ReachableFunctionLibraryDefinition(*this, func.node_def());
}

string FunctionLibraryRuntime::Options::DebugString() const {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_86(mht_86_v, 2268, "", "./tensorflow/core/framework/function.cc", "FunctionLibraryRuntime::Options::DebugString");

  return absl::StrCat(
      "FLR::Options(step_id=", step_id, " rendezvous=", IsSet(rendezvous),
      " cancellation_manager=", IsSet(cancellation_manager),
      " collective_executor=", IsSet(collective_executor),
      " step_container=", IsSet(step_container),
      " stats_collector=", IsSet(stats_collector), " runner=", IsSet(runner),
      " remote_execution=", remote_execution, " source_device=", source_device,
      " create_rendezvous=", create_rendezvous,
      " allow_dead_tensors=", allow_dead_tensors,
      " args_alloc_attrs=", AllocatorAttributesToString(args_alloc_attrs),
      " rets_alloc_attrs=", AllocatorAttributesToString(rets_alloc_attrs), ")");
}

void FunctionDefHelper::AttrValueWrapper::InitFromString(StringPiece val) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_87(mht_87_v, 2285, "", "./tensorflow/core/framework/function.cc", "FunctionDefHelper::AttrValueWrapper::InitFromString");

  if (val.size() >= 2 && val[0] == '$') {
    proto.set_placeholder(val.data() + 1, val.size() - 1);
  } else {
    SetAttrValue(val, &proto);
  }
}

FunctionDefHelper::AttrValueWrapper FunctionDefHelper::FunctionRef(
    const string& name,
    gtl::ArraySlice<std::pair<string, AttrValueWrapper>> attrs) {
   std::vector<std::string> mht_88_v;
   mht_88_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_88(mht_88_v, 2299, "", "./tensorflow/core/framework/function.cc", "FunctionDefHelper::FunctionRef");

  AttrValueWrapper ret;
  ret.proto.mutable_func()->set_name(name);
  for (const auto& a : attrs) {
    ret.proto.mutable_func()->mutable_attr()->insert({a.first, a.second.proto});
  }
  return ret;
}

NodeDef FunctionDefHelper::Node::ToNodeDef() const {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_89(mht_89_v, 2311, "", "./tensorflow/core/framework/function.cc", "FunctionDefHelper::Node::ToNodeDef");

  NodeDef n;
  n.set_op(this->op);
  n.set_name(GetName());
  for (const auto& a : this->attr) {
    n.mutable_attr()->insert({a.first, a.second.proto});
  }
  for (const string& a : this->arg) {
    n.add_input(a);
  }
  for (const string& d : this->dep) {
    n.add_input(strings::StrCat("^", d));
  }
  if (!this->device.empty()) {
    n.set_device(this->device);
  }
  if (!this->original_node_names.empty()) {
    *n.mutable_experimental_debug_info()->mutable_original_node_names() = {
        this->original_node_names.begin(), this->original_node_names.end()};
  }
  if (!this->original_func_names.empty()) {
    *n.mutable_experimental_debug_info()->mutable_original_func_names() = {
        this->original_func_names.begin(), this->original_func_names.end()};
  }
  return n;
}

/* static */
FunctionDef FunctionDefHelper::Create(
    const string& function_name, gtl::ArraySlice<string> in_def,
    gtl::ArraySlice<string> out_def, gtl::ArraySlice<string> attr_def,
    gtl::ArraySlice<Node> node_def,
    gtl::ArraySlice<std::pair<string, string>> ret_def,
    gtl::ArraySlice<std::pair<string, string>> control_ret_def) {
   std::vector<std::string> mht_90_v;
   mht_90_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_90(mht_90_v, 2348, "", "./tensorflow/core/framework/function.cc", "FunctionDefHelper::Create");

  FunctionDef fdef;

  // Signature
  OpDefBuilder b(function_name);
  for (const auto& i : in_def) b.Input(i);
  for (const auto& o : out_def) b.Output(o);
  for (const auto& a : attr_def) b.Attr(a);
  for (const auto& c : control_ret_def) b.ControlOutput(c.first);

  OpRegistrationData op_reg_data;
  TF_CHECK_OK(b.Finalize(&op_reg_data));
  fdef.mutable_signature()->Swap(&op_reg_data.op_def);

  // Function body
  for (const auto& n : node_def) {
    *(fdef.add_node_def()) = n.ToNodeDef();
  }

  // Returns
  for (const auto& r : ret_def) {
    fdef.mutable_ret()->insert({r.first, r.second});
  }

  // Control returns
  for (const auto& cr : control_ret_def) {
    fdef.mutable_control_ret()->insert({cr.first, cr.second});
  }

  auto* op_def_registry = OpRegistry::Global();
  // Check if any op is stateful.
  for (const auto& n : node_def) {
    const OpDef* op_def = nullptr;
    auto status = op_def_registry->LookUpOpDef(n.op, &op_def);
    // Lookup can fail if e.g. we are calling a function that was not yet
    // defined.  If it happens, conservatively assume the op is stateful.
    if (!status.ok() || op_def->is_stateful()) {
      fdef.mutable_signature()->set_is_stateful(true);
    }
  }

  return fdef;
}

/* static */
FunctionDef FunctionDefHelper::Create(
    const string& function_name, gtl::ArraySlice<string> in_def,
    gtl::ArraySlice<string> out_def, gtl::ArraySlice<string> attr_def,
    gtl::ArraySlice<Node> node_def,
    gtl::ArraySlice<std::pair<string, string>> ret_def) {
   std::vector<std::string> mht_91_v;
   mht_91_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_91(mht_91_v, 2401, "", "./tensorflow/core/framework/function.cc", "FunctionDefHelper::Create");

  return Create(function_name, in_def, out_def, attr_def, node_def, ret_def,
                /*control_ret_def=*/{});
}

/* static */
FunctionDef FunctionDefHelper::Define(const string& name,
                                      gtl::ArraySlice<string> arg_def,
                                      gtl::ArraySlice<string> ret_def,
                                      gtl::ArraySlice<string> attr_def,
                                      gtl::ArraySlice<Node> node_def) {
   std::vector<std::string> mht_92_v;
   mht_92_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_92(mht_92_v, 2415, "", "./tensorflow/core/framework/function.cc", "FunctionDefHelper::Define");

  FunctionDef fdef;
  OpDefBuilder b(name);
  for (const auto& a : arg_def) b.Input(a);
  for (const auto& r : ret_def) b.Output(r);
  for (const auto& a : attr_def) b.Attr(a);

  OpRegistrationData op_reg_data;
  TF_CHECK_OK(b.Finalize(&op_reg_data));
  fdef.mutable_signature()->Swap(&op_reg_data.op_def);

  // Mapping from legacy output names to NodeDef outputs.
  std::unordered_map<string, string> ret_index;
  for (const auto& a : fdef.signature().input_arg()) {
    ret_index[a.name()] = a.name();
  }

  // For looking up OpDefs
  auto* op_def_registry = OpRegistry::Global();

  // Function body
  for (const auto& src : node_def) {
    NodeDef* n = fdef.add_node_def();
    n->set_op(src.op);
    n->set_name(src.GetName());
    for (const auto& a : src.attr) {
      n->mutable_attr()->insert({a.first, a.second.proto});
    }
    for (const string& a : src.arg) {
      const auto iter = ret_index.find(a);
      CHECK(iter != ret_index.end())
          << "Node input '" << a << "' in '" << n->name() << "' of " << name;
      n->add_input(iter->second);
    }
    for (const string& d : src.dep) {
      n->add_input(strings::StrCat("^", d));
    }

    // Add the outputs of this node to ret_index.
    const OpDef* op_def = nullptr;
    TF_CHECK_OK(op_def_registry->LookUpOpDef(n->op(), &op_def)) << n->op();
    CHECK(op_def != nullptr) << n->op();
    NameRangeMap output_names;
    TF_CHECK_OK(NameRangesForNode(*n, *op_def, nullptr, &output_names));
    for (const auto& o : output_names) {
      CHECK_LE(o.second.second, src.ret.size())
          << "Missing ret for output '" << o.first << "' in '" << n->name()
          << "' of " << name;
      for (int i = o.second.first; i < o.second.second; ++i) {
        ret_index[src.ret[i]] =
            strings::StrCat(n->name(), ":", o.first, ":", i - o.second.first);
      }
    }
    if (op_def->is_stateful()) fdef.mutable_signature()->set_is_stateful(true);
  }

  // Returns
  for (const auto& r : fdef.signature().output_arg()) {
    const auto iter = ret_index.find(r.name());
    CHECK(iter != ret_index.end()) << "Return '" << r.name() << "' in " << name;
    fdef.mutable_ret()->insert({r.name(), iter->second});
  }
  return fdef;
}

FunctionDef FunctionDefHelper::Define(gtl::ArraySlice<string> arg_def,
                                      gtl::ArraySlice<string> ret_def,
                                      gtl::ArraySlice<string> attr_def,
                                      gtl::ArraySlice<Node> node_def) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_93(mht_93_v, 2486, "", "./tensorflow/core/framework/function.cc", "FunctionDefHelper::Define");

  return Define("_", arg_def, ret_def, attr_def, node_def);
}

namespace gradient {

typedef std::unordered_map<string, Creator> OpGradFactory;

OpGradFactory* GetOpGradFactory() {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_94(mht_94_v, 2497, "", "./tensorflow/core/framework/function.cc", "GetOpGradFactory");

  static OpGradFactory* factory = new OpGradFactory;
  return factory;
}

bool RegisterOp(const string& op, Creator func) {
   std::vector<std::string> mht_95_v;
   mht_95_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_95(mht_95_v, 2506, "", "./tensorflow/core/framework/function.cc", "RegisterOp");

  CHECK(GetOpGradFactory()->insert({op, func}).second)
      << "Duplicated gradient for " << op;
  return true;
}

Status GetOpGradientCreator(const string& op, Creator* creator) {
   std::vector<std::string> mht_96_v;
   mht_96_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunctionDTcc mht_96(mht_96_v, 2516, "", "./tensorflow/core/framework/function.cc", "GetOpGradientCreator");

  auto fac = GetOpGradFactory();
  auto iter = fac->find(op);
  if (iter == fac->end()) {
    return errors::NotFound("No gradient defined for op: ", op);
  }
  *creator = iter->second;
  return Status::OK();
}

}  // end namespace gradient

}  // namespace tensorflow
