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
class MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc() {
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

#include "tensorflow/core/framework/node_def_builder.h"

#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

NodeDefBuilder::NodeOut::NodeOut(StringPiece n, int i, DataType dt)
    : node(n), index(i), data_type(dt) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::NodeOut::NodeOut");
}

NodeDefBuilder::NodeOut::NodeOut() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::NodeOut::NodeOut");

  // uninitialized, call Reset() before use.
}

void NodeDefBuilder::NodeOut::Reset(StringPiece n, int i, DataType dt) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_2(mht_2_v, 209, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::NodeOut::Reset");

  node = string(n);
  index = i;
  data_type = dt;
}

NodeDefBuilder::NodeDefBuilder(StringPiece name, StringPiece op_name,
                               const OpRegistryInterface* op_registry,
                               const NodeDebugInfo* debug) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_3(mht_3_v, 220, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::NodeDefBuilder");

  node_def_.set_name(string(name));
  const Status status = op_registry->LookUpOpDef(string(op_name), &op_def_);
  if (status.ok()) {
    Initialize();
  } else {
    errors_.push_back(status.error_message());
    inputs_specified_ = 0;
  }
  if (debug != nullptr) MergeDebugInfo(*debug, &node_def_);
}

NodeDefBuilder::NodeDefBuilder(StringPiece name, StringPiece op_name,
                               const NodeDebugInfo& debug)
    : NodeDefBuilder(name, op_name) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_4(mht_4_v, 237, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::NodeDefBuilder");

  MergeDebugInfo(debug, &node_def_);
}

NodeDefBuilder::NodeDefBuilder(StringPiece name, const OpDef* op_def)
    : op_def_(op_def) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_5(mht_5_v, 245, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::NodeDefBuilder");

  node_def_.set_name(string(name));
  Initialize();
}

void NodeDefBuilder::Initialize() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_6(mht_6_v, 253, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::Initialize");

  inputs_specified_ = 0;
  node_def_.set_op(op_def_->name());
}

const OpDef::ArgDef* NodeDefBuilder::NextArgDef() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_7(mht_7_v, 261, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::NextArgDef");

  if (!NextArgAvailable()) return nullptr;
  return &op_def_->input_arg(inputs_specified_++);
}

bool NodeDefBuilder::NextArgAvailable() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_8(mht_8_v, 269, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::NextArgAvailable");

  if (op_def_ == nullptr) {
    return false;
  } else if (inputs_specified_ >= op_def_->input_arg_size()) {
    errors_.push_back(strings::StrCat("More Input() calls than the ",
                                      op_def_->input_arg_size(),
                                      " input_args"));
    return false;
  }
  return true;
}

NodeDefBuilder& NodeDefBuilder::Input(FakeInputFunctor fake_input) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_9(mht_9_v, 284, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::Input");

  if (NextArgAvailable()) {
    Status status = fake_input(*op_def_, inputs_specified_, node_def_, this);
    if (!status.ok()) errors_.push_back(status.error_message());
  }
  return *this;
}

NodeDefBuilder& NodeDefBuilder::Input(StringPiece src_node, int src_index,
                                      DataType dt) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_10(mht_10_v, 296, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::Input");

  const OpDef::ArgDef* arg = NextArgDef();
  if (arg != nullptr) SingleInput(arg, src_node, src_index, dt);
  return *this;
}

NodeDefBuilder& NodeDefBuilder::Input(const NodeOut& src) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_11(mht_11_v, 305, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::Input");

  Input(src.node, src.index, src.data_type);
  return *this;
}

// For inputs that take a list of tensors.
NodeDefBuilder& NodeDefBuilder::Input(gtl::ArraySlice<NodeOut> src_list) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_12(mht_12_v, 314, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::Input");

  const OpDef::ArgDef* arg = NextArgDef();
  if (arg != nullptr) ListInput(arg, src_list);
  return *this;
}

void NodeDefBuilder::SingleInput(const OpDef::ArgDef* input_arg,
                                 StringPiece src_node, int src_index,
                                 DataType dt) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_13(mht_13_v, 325, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::SingleInput");

  AddInput(src_node, src_index);

  if (!input_arg->number_attr().empty() ||
      !input_arg->type_list_attr().empty()) {
    errors_.push_back(strings::StrCat("Single tensor passed to '",
                                      input_arg->name(), "', expected list"));
    return;
  }

  if (input_arg->type() != DT_INVALID) {
    const DataType expected = MaybeAddRef(input_arg, input_arg->type());
    VerifyInputType(input_arg, expected, dt);
  } else {
    VerifyInputRef(input_arg, dt);
    Attr(input_arg->type_attr(), BaseType(dt));
  }
}

void NodeDefBuilder::ListInput(const OpDef::ArgDef* input_arg,
                               gtl::ArraySlice<NodeOut> src_list) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_14(mht_14_v, 348, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::ListInput");

  for (const auto& node_out : src_list) {
    AddInput(node_out.node, node_out.index);
  }

  if (!input_arg->number_attr().empty()) {
    Attr(input_arg->number_attr(), static_cast<int64_t>(src_list.size()));
    if (input_arg->type() != DT_INVALID) {
      const DataType expected = MaybeAddRef(input_arg, input_arg->type());
      for (const auto& node_out : src_list) {
        VerifyInputType(input_arg, expected, node_out.data_type);
      }
    } else if (!src_list.empty()) {
      const DataType base = BaseType(src_list[0].data_type);
      Attr(input_arg->type_attr(), base);
      const DataType expected = MaybeAddRef(input_arg, base);
      for (const auto& node_out : src_list) {
        VerifyInputType(input_arg, expected, node_out.data_type);
      }
    }
  } else if (!input_arg->type_list_attr().empty()) {
    DataTypeVector type_vec;
    type_vec.reserve(src_list.size());
    for (const auto& node_out : src_list) {
      const DataType dt = node_out.data_type;
      VerifyInputRef(input_arg, dt);
      type_vec.push_back(BaseType(dt));
    }
    Attr(input_arg->type_list_attr(), type_vec);
  } else {
    errors_.push_back(strings::StrCat("List provided to input '",
                                      input_arg->name(),
                                      "' when single Tensor expected"));
  }
}

void NodeDefBuilder::AddInput(StringPiece src_node, int src_index) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_15(mht_15_v, 387, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::AddInput");

  if (src_node.empty()) {
    errors_.push_back("Empty input node name");
  } else if (src_node[0] == '^') {
    errors_.push_back(
        strings::StrCat("Non-control input starting with ^: ", src_node));
  } else if (src_index > 0) {
    node_def_.add_input(strings::StrCat(src_node, ":", src_index));
  } else {
    node_def_.add_input(string(src_node));
  }
}

void NodeDefBuilder::VerifyInputType(const OpDef::ArgDef* input_arg,
                                     DataType expected, DataType dt) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_16(mht_16_v, 404, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::VerifyInputType");

  if (!TypesCompatible(expected, dt)) {
    errors_.push_back(strings::StrCat("Input '", input_arg->name(), "' passed ",
                                      DataTypeString(dt), " expected ",
                                      DataTypeString(expected)));
  }
}

void NodeDefBuilder::VerifyInputRef(const OpDef::ArgDef* input_arg,
                                    DataType dt) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_17(mht_17_v, 416, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::VerifyInputRef");

  if (input_arg->is_ref() && !IsRefType(dt)) {
    errors_.push_back(strings::StrCat("Input '", input_arg->name(), "' passed ",
                                      DataTypeString(dt),
                                      " expected ref type"));
  }
}

NodeDefBuilder& NodeDefBuilder::ControlInput(StringPiece src_node) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_18(mht_18_v, 427, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::ControlInput");

  control_inputs_.emplace_back(src_node);
  return *this;
}

NodeDefBuilder& NodeDefBuilder::Device(StringPiece device_spec) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_19(mht_19_v, 435, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::Device");

  node_def_.set_device(string(device_spec));
  return *this;
}

Status NodeDefBuilder::Finalize(NodeDef* node_def, bool consume) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_20(mht_20_v, 443, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::Finalize");

  const std::vector<string>* errors_ptr = &errors_;
  std::vector<string> errors_storage;
  if (op_def_ != nullptr && inputs_specified_ < op_def_->input_arg_size()) {
    // Since this is a const method, to add an error, we have to make
    // a copy of the existing errors.
    errors_storage = errors_;
    errors_storage.push_back(
        strings::StrCat(inputs_specified_, " inputs specified of ",
                        op_def_->input_arg_size(), " inputs in Op"));
    errors_ptr = &errors_storage;
  }

  if (!errors_ptr->empty()) {
    if (errors_ptr->size() == 1) {
      if (op_def_ == nullptr) {
        return errors::InvalidArgument((*errors_ptr)[0],
                                       " while building NodeDef '",
                                       node_def_.name(), "'");
      }
      return errors::InvalidArgument(
          (*errors_ptr)[0], " while building NodeDef '", node_def_.name(),
          "' using ", SummarizeOpDef(*op_def_));
    } else {
      return errors::InvalidArgument(
          errors_ptr->size(), " errors while building NodeDef '",
          node_def_.name(), "' using ", SummarizeOpDef(*op_def_), ":\n",
          absl::StrJoin(*errors_ptr, "\n"));
    }
  } else {
    NodeDef node_def_backup;
    if (node_def == nullptr) node_def = &node_def_backup;
    if (consume) {
      *node_def = std::move(node_def_);
    } else {
      *node_def = node_def_;
    }

    // Add control inputs after the regular inputs.
    for (const auto& control_input : control_inputs_) {
      node_def->add_input(strings::StrCat("^", control_input));
    }

    // Add default values for unspecified attrs.
    AddDefaultsToNodeDef(*op_def_, node_def);

    return Status::OK();
  }
}

bool NodeDefBuilder::AttrValueAlreadyPresent(StringPiece name,
                                             const AttrValue& value) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_21(mht_21_v, 497, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::AttrValueAlreadyPresent");

  if (const AttrValue* found = AttrSlice(node_def_).Find(name)) {
    if (!AreAttrValuesEqual(*found, value)) {
      errors_.push_back(strings::StrCat("Inconsistent values for attr '", name,
                                        "' ", SummarizeAttrValue(*found),
                                        " vs. ", SummarizeAttrValue(value)));
    }
    return true;
  }
  return false;
}

NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, const AttrValue& value) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_22(mht_22_v, 512, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::Attr");

  if (!AttrValueAlreadyPresent(name, value)) {
    AddNodeAttr(name, value, &node_def_);
  }
  return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, AttrValue&& value) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_builderDTcc mht_23(mht_23_v, 522, "", "./tensorflow/core/framework/node_def_builder.cc", "NodeDefBuilder::Attr");

  if (!AttrValueAlreadyPresent(name, value)) {
    AddNodeAttr(name, std::move(value), &node_def_);
  }
  return *this;
}

#define ATTR(T)                                                     \
  NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, T value) { \
    AttrValue attr_value;                                           \
    SetAttrValue(value, &attr_value);                               \
    return Attr(name, attr_value);                                  \
  }
ATTR(StringPiece)
ATTR(const char*)
ATTR(int32_t)
ATTR(int64_t)
ATTR(float)
ATTR(double)
ATTR(bool)
ATTR(DataType)
ATTR(const PartialTensorShape&)
ATTR(const Tensor&)
ATTR(const TensorProto&)
ATTR(const NameAttrList&)
ATTR(gtl::ArraySlice<StringPiece>)
ATTR(gtl::ArraySlice<const char*>)
ATTR(gtl::ArraySlice<string>)
ATTR(gtl::ArraySlice<tstring>)
ATTR(gtl::ArraySlice<int32>)
ATTR(gtl::ArraySlice<int64_t>)
ATTR(gtl::ArraySlice<float>)
ATTR(gtl::ArraySlice<bool>)
ATTR(const std::vector<bool>&)
ATTR(gtl::ArraySlice<DataType>)
ATTR(gtl::ArraySlice<TensorShape>)
ATTR(gtl::ArraySlice<PartialTensorShape>)
ATTR(gtl::ArraySlice<TensorShapeProto>)
ATTR(gtl::ArraySlice<Tensor>)
ATTR(gtl::ArraySlice<NameAttrList>)
#undef ATTR

}  // namespace tensorflow
