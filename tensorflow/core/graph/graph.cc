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
class MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc() {
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

#include "tensorflow/core/graph/graph.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/while_context.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

const int Graph::kControlSlot = -1;

// Node
Node::NodeClass Node::GetNodeClassForOp(const std::string& ts) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("ts: \"" + ts + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/graph/graph.cc", "Node::GetNodeClassForOp");

  static const absl::flat_hash_map<std::string, Node::NodeClass>*
      kNodeClassTable =
#define REF_CLASS(key, value) \
  {key, value}, { "Ref" key, value }
          new absl::flat_hash_map<std::string, Node::NodeClass>({
              // Keep in same order as NodeClass values
              REF_CLASS("Switch", NC_SWITCH),
              REF_CLASS("_SwitchN", NC_SWITCH),
              REF_CLASS("Merge", NC_MERGE),
              REF_CLASS("Enter", NC_ENTER),
              REF_CLASS("Exit", NC_EXIT),
              REF_CLASS("NextIteration", NC_NEXT_ITERATION),
              {"LoopCond", NC_LOOP_COND},
              {"ControlTrigger", NC_CONTROL_TRIGGER},
              {"_Send", NC_SEND},
              {"_HostSend", NC_HOST_SEND},
              {"_Recv", NC_RECV},
              {"_HostRecv", NC_HOST_RECV},
              {"Const", NC_CONSTANT},
              {"HostConst", NC_CONSTANT},
              {"Variable", NC_VARIABLE},
              {"VariableV2", NC_VARIABLE},
              REF_CLASS("Identity", NC_IDENTITY),
              {"GetSessionHandle", NC_GET_SESSION_HANDLE},
              {"GetSessionHandleV2", NC_GET_SESSION_HANDLE},
              {"GetSessionTensor", NC_GET_SESSION_TENSOR},
              {"DeleteSessionTensor", NC_DELETE_SESSION_TENSOR},
              {"Size", NC_METADATA},
              {"Shape", NC_METADATA},
              {"Rank", NC_METADATA},
              {"_ScopedAllocator", NC_SCOPED_ALLOCATOR},
              {"CollectiveReduce", NC_COLLECTIVE},
              {"CollectiveBcastSend", NC_COLLECTIVE},
              {"CollectiveBcastRecv", NC_COLLECTIVE},
              {"CollectiveGather", NC_COLLECTIVE},
              {"FakeParam", NC_FAKE_PARAM},
              {"PartitionedCall", NC_PARTITIONED_CALL},
              {"StatefulPartitionedCall", NC_PARTITIONED_CALL},
              {"SymbolicGradient", NC_SYMBOLIC_GRADIENT},
              {"If", NC_IF},
              {"StatelessIf", NC_IF},
              {"While", NC_WHILE},
              {"StatelessWhile", NC_WHILE},
              {"Case", NC_CASE},
              {"StatelessCase", NC_CASE},
              // Not using the constants defined in FunctionLibraryDefinition
              // for the
              // 4 ops below because android inference library does not link
              // tf.function related files.
              {"_Arg", NC_ARG},
              {"_DeviceArg", NC_ARG},
              {"_Retval", NC_RETVAL},
              {"_DeviceRetval", NC_RETVAL},
              {"_XlaMerge", NC_MERGE},
          });
#undef REF_CLASS

  auto it = kNodeClassTable->find(ts);
  if (it != kNodeClassTable->end()) {
    return it->second;
  } else {
    return NC_OTHER;
  }
}

std::string Node::DebugString() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_1(mht_1_v, 284, "", "./tensorflow/core/graph/graph.cc", "Node::DebugString");

  std::string ret = strings::StrCat("{name:'", name(), "' id:", id_);
  if (IsSource()) {
    strings::StrAppend(&ret, " source}");
  } else if (IsSink()) {
    strings::StrAppend(&ret, " sink}");
  } else {
    strings::StrAppend(&ret, " op device:", "{requested: '", requested_device(),
                       "', assigned: '", assigned_device_name(), "'}", " def:{",
                       SummarizeNode(*this), "}}");
  }
  return ret;
}

Node::Node()
    : id_(-1),
      cost_id_(-1),
      class_(NC_UNINITIALIZED),
      props_(nullptr),
      assigned_device_name_index_(0),
      while_ctx_(nullptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_2(mht_2_v, 307, "", "./tensorflow/core/graph/graph.cc", "Node::Node");
}

void Node::Initialize(int id, int cost_id,
                      std::shared_ptr<NodeProperties> props,
                      Node::NodeClass node_class) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_3(mht_3_v, 314, "", "./tensorflow/core/graph/graph.cc", "Node::Initialize");

  DCHECK_EQ(id_, -1);
  DCHECK(in_edges_.empty());
  DCHECK(out_edges_.empty());
  id_ = id;
  cost_id_ = cost_id;

  props_ = std::move(props);
  class_ = node_class;
}

void Node::Clear() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_4(mht_4_v, 328, "", "./tensorflow/core/graph/graph.cc", "Node::Clear");

  in_edges_.clear();
  out_edges_.clear();
  id_ = -1;
  cost_id_ = -1;
  class_ = NC_UNINITIALIZED;
  props_.reset();
  assigned_device_name_index_ = 0;
}

void Node::UpdateProperties() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_5(mht_5_v, 341, "", "./tensorflow/core/graph/graph.cc", "Node::UpdateProperties");

  DataTypeVector inputs;
  DataTypeVector outputs;
  Status status =
      InOutTypesForNode(props_->node_def, *(props_->op_def), &inputs, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Failed at updating node: " << status;
    return;
  }
  if (props_->input_types != inputs || props_->output_types != outputs) {
    if (TF_PREDICT_TRUE(props_.use_count() == 1)) {
      props_->input_types = inputs;
      props_->input_types_slice = props_->input_types;
      props_->output_types = outputs;
      props_->output_types_slice = props_->output_types;
    } else {
      props_ = std::make_shared<NodeProperties>(
          props_->op_def, std::move(props_->node_def), inputs, outputs);
    }
  }
}

void Node::ClearTypeInfo() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_6(mht_6_v, 366, "", "./tensorflow/core/graph/graph.cc", "Node::ClearTypeInfo");

  if (props_->node_def.has_experimental_type()) {
    MaybeCopyOnWrite();
    props_->node_def.clear_experimental_type();
  }
}

void Node::RunForwardTypeInference() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_7(mht_7_v, 376, "", "./tensorflow/core/graph/graph.cc", "Node::RunForwardTypeInference");

  VLOG(4) << "Forward type inference: " << props_->node_def.DebugString();

  if (props_->fwd_type_fn == nullptr) {
    return;
  }

  std::vector<Node*> input_nodes(props_->input_types.size(), nullptr);
  std::vector<int> input_idx(props_->input_types.size(), 0);
  for (const auto& edge : in_edges_) {
    if (edge->IsControlEdge()) {
      continue;
    }
    DCHECK(edge->dst_input() < input_nodes.size()) << DebugString();
    int i = edge->dst_input();
    input_nodes.at(i) = edge->src();
    input_idx.at(i) = edge->src_output();
  }

  // Note: technically, we could use a very generic type when some of the inputs
  // are unknown. But there is an expectation that a node will have complete
  // inputs soon, so updating intermediate types is largely unnecessary.

  for (const auto* node : input_nodes) {
    if (node == nullptr) {
      // Incomplete inputs, bail.
      ClearTypeInfo();
      return;
    }
  }

  static FullTypeDef* no_type = new FullTypeDef();

  std::vector<std::reference_wrapper<const FullTypeDef>> input_types;
  for (int i = 0; i < input_nodes.size(); i++) {
    const auto* node = input_nodes[i];
    if (node->def().has_experimental_type()) {
      const auto& node_t = node->def().experimental_type();
      if (node_t.type_id() != TFT_UNSET) {
        int ix = input_idx[i];
        if (ix >= node_t.args_size()) {
          LOG(WARNING) << name() << " has bad type information: input " << i
                       << " should have an output " << ix
                       << " but instead only has " << node_t.args_size()
                       << " outputs: " << node_t.DebugString()
                       << "\nThis indicates either "
                          "a bug in op registration or a corrupted graph.";
          ClearTypeInfo();
          return;
        }
        input_types.emplace_back(node_t.args(ix));
      } else {
        input_types.emplace_back(*no_type);
      }
    } else {
      // Incomplete inputs, bail.
      ClearTypeInfo();
      return;
    }
  }

  // TODO(b/224775462): Populate with types from function references.
  TypeRefMap type_vars;

  const auto infer_type = props_->fwd_type_fn(input_types, type_vars);
  if (!infer_type.ok()) {
    // TODO(mdan): Turn this into an error, once all offenders are clean.
    LOG(WARNING) << name()
                 << " failed type inference; this is likely caused by"
                    " a graph in which inconsistent types went "
                    "undetected. This will become an error in the "
                    "future.\nNode information:\n"
                 << props_->node_def.DebugString()
                 << "\nType inference error:\n"
                 << infer_type.status().ToString();
    props_->node_def.clear_experimental_type();
    return;
  }
  const FullTypeDef infer_typedef = infer_type.ValueOrDie();
  if (infer_typedef.type_id() != TFT_UNSET) {
    MaybeCopyOnWrite();
    *(props_->node_def.mutable_experimental_type()) = infer_typedef;
  } else {
    props_->node_def.clear_experimental_type();
  }
}

const std::string& Node::name() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_8(mht_8_v, 466, "", "./tensorflow/core/graph/graph.cc", "Node::name");
 return props_->node_def.name(); }
const std::string& Node::type_string() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_9(mht_9_v, 470, "", "./tensorflow/core/graph/graph.cc", "Node::type_string");
 return props_->node_def.op(); }
const NodeDef& Node::def() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_10(mht_10_v, 474, "", "./tensorflow/core/graph/graph.cc", "Node::def");
 return props_->node_def; }
const OpDef& Node::op_def() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_11(mht_11_v, 478, "", "./tensorflow/core/graph/graph.cc", "Node::op_def");
 return *props_->op_def; }

NodeDef* Node::mutable_def() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_12(mht_12_v, 483, "", "./tensorflow/core/graph/graph.cc", "Node::mutable_def");
 return &props_->node_def; }

int32 Node::num_inputs() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_13(mht_13_v, 488, "", "./tensorflow/core/graph/graph.cc", "Node::num_inputs");
 return props_->input_types.size(); }
DataType Node::input_type(int32_t i) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_14(mht_14_v, 492, "", "./tensorflow/core/graph/graph.cc", "Node::input_type");
 return props_->input_types[i]; }
const DataTypeVector& Node::input_types() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_15(mht_15_v, 496, "", "./tensorflow/core/graph/graph.cc", "Node::input_types");
 return props_->input_types; }

int32 Node::num_outputs() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_16(mht_16_v, 501, "", "./tensorflow/core/graph/graph.cc", "Node::num_outputs");
 return props_->output_types.size(); }
DataType Node::output_type(int32_t o) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_17(mht_17_v, 505, "", "./tensorflow/core/graph/graph.cc", "Node::output_type");
 return props_->output_types[o]; }
const DataTypeVector& Node::output_types() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_18(mht_18_v, 509, "", "./tensorflow/core/graph/graph.cc", "Node::output_types");

  return props_->output_types;
}

AttrSlice Node::attrs() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_19(mht_19_v, 516, "", "./tensorflow/core/graph/graph.cc", "Node::attrs");
 return AttrSlice(def()); }

const protobuf::RepeatedPtrField<std::string>& Node::requested_inputs() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_20(mht_20_v, 521, "", "./tensorflow/core/graph/graph.cc", "Node::requested_inputs");

  return def().input();
}

const std::string& Node::requested_device() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_21(mht_21_v, 528, "", "./tensorflow/core/graph/graph.cc", "Node::requested_device");
 return def().device(); }

gtl::iterator_range<NeighborIter> Node::out_nodes() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_22(mht_22_v, 533, "", "./tensorflow/core/graph/graph.cc", "Node::out_nodes");

  return gtl::make_range(NeighborIter(out_edges_.begin(), false),
                         NeighborIter(out_edges_.end(), false));
}

gtl::iterator_range<NeighborIter> Node::in_nodes() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_23(mht_23_v, 541, "", "./tensorflow/core/graph/graph.cc", "Node::in_nodes");

  return gtl::make_range(NeighborIter(in_edges_.begin(), true),
                         NeighborIter(in_edges_.end(), true));
}

void Node::MaybeCopyOnWrite() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_24(mht_24_v, 549, "", "./tensorflow/core/graph/graph.cc", "Node::MaybeCopyOnWrite");

  // TODO(mdan): As nodes become more dynamic, this may not be worth the cost.
  // NodeProperties may be shared between Nodes. Make a copy if so.
  if (!props_.unique()) {
    props_ = std::make_shared<NodeProperties>(*props_);
  }
}

AttrValue* Node::AddAttrHelper(const std::string& name) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_25(mht_25_v, 561, "", "./tensorflow/core/graph/graph.cc", "Node::AddAttrHelper");

  MaybeCopyOnWrite();
  return &((*props_->node_def.mutable_attr())[name]);
}

void Node::ClearAttr(const std::string& name) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_26(mht_26_v, 570, "", "./tensorflow/core/graph/graph.cc", "Node::ClearAttr");

  MaybeCopyOnWrite();
  (*props_->node_def.mutable_attr()).erase(name);
}

void Node::set_name(std::string name) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_27(mht_27_v, 579, "", "./tensorflow/core/graph/graph.cc", "Node::set_name");

  MaybeCopyOnWrite();
  props_->node_def.set_name(std::move(name));
}

void Node::set_requested_device(const std::string& device) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_28(mht_28_v, 588, "", "./tensorflow/core/graph/graph.cc", "Node::set_requested_device");

  MaybeCopyOnWrite();
  props_->node_def.set_device(device);
}

void Node::set_original_node_names(const std::vector<std::string>& names) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_29(mht_29_v, 596, "", "./tensorflow/core/graph/graph.cc", "Node::set_original_node_names");

  MaybeCopyOnWrite();
  props_->node_def.mutable_experimental_debug_info()
      ->clear_original_node_names();
  if (!names.empty()) {
    *props_->node_def.mutable_experimental_debug_info()
         ->mutable_original_node_names() = {names.begin(), names.end()};
  }
}

void Node::set_original_func_names(const std::vector<std::string>& names) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_30(mht_30_v, 609, "", "./tensorflow/core/graph/graph.cc", "Node::set_original_func_names");

  MaybeCopyOnWrite();
  props_->node_def.mutable_experimental_debug_info()
      ->clear_original_func_names();
  if (!names.empty()) {
    *props_->node_def.mutable_experimental_debug_info()
         ->mutable_original_func_names() = {names.begin(), names.end()};
  }
}

Status Node::input_edge(int idx, const Edge** e) const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_31(mht_31_v, 622, "", "./tensorflow/core/graph/graph.cc", "Node::input_edge");

  if (idx < 0 || idx >= num_inputs()) {
    return errors::InvalidArgument("Invalid input_edge index: ", idx, ", Node ",
                                   name(), " only has ", num_inputs(),
                                   " inputs.");
  }

  // This does a linear search over the edges.  In the common case,
  // the number of elements is small enough that this search isn't
  // expensive.  Should it become a bottleneck, one can make an
  // optimization where, if the number of edges is small, we use
  // linear iteration, and if the number of edges is large, we perform
  // an indexing step during construction that keeps an array of Edges
  // indexed by pointer.  This would keep the size of each Node small
  // in the common case but make this function faster when the number
  // of edges is large.
  for (const Edge* edge : in_edges()) {
    if (edge->dst_input() == idx) {
      *e = edge;
      return Status::OK();
    }
  }

  return errors::NotFound("Could not find input edge ", idx, " for ", name());
}

// Returns a vector of the non-control input edges to a node, indexed by ID.
Status Node::input_edges(std::vector<const Edge*>* input_edges) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_32(mht_32_v, 652, "", "./tensorflow/core/graph/graph.cc", "Node::input_edges");

  input_edges->clear();
  input_edges->resize(num_inputs(), nullptr);

  for (const Edge* edge : in_edges()) {
    if (edge->IsControlEdge()) continue;
    if (edge->dst_input() < 0 || edge->dst_input() >= num_inputs()) {
      return errors::Internal("Invalid edge input number ", edge->dst_input());
    }
    if ((*input_edges)[edge->dst_input()] != nullptr) {
      return errors::Internal("Duplicate edge input number: ",
                              edge->dst_input());
    }
    (*input_edges)[edge->dst_input()] = edge;
  }

  for (int i = 0; i < num_inputs(); ++i) {
    if ((*input_edges)[i] == nullptr) {
      return errors::InvalidArgument("Missing edge input number: ", i);
    }
  }
  return Status::OK();
}

Status Node::input_node(int idx, Node** n) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_33(mht_33_v, 679, "", "./tensorflow/core/graph/graph.cc", "Node::input_node");

  const Edge* e;
  TF_RETURN_IF_ERROR(input_edge(idx, &e));
  if (e == nullptr) {
    *n = nullptr;
  } else {
    *n = e->src();
  }
  return Status::OK();
}

Status Node::input_node(int idx, const Node** const_n) const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_34(mht_34_v, 693, "", "./tensorflow/core/graph/graph.cc", "Node::input_node");

  Node* n;
  TF_RETURN_IF_ERROR(input_node(idx, &n));
  *const_n = n;
  return Status::OK();
}

Status Node::input_tensor(int idx, OutputTensor* t) const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_35(mht_35_v, 703, "", "./tensorflow/core/graph/graph.cc", "Node::input_tensor");

  const Edge* e;
  TF_RETURN_IF_ERROR(input_edge(idx, &e));
  DCHECK(e != nullptr);
  *t = OutputTensor(e->src(), e->src_output());
  return Status::OK();
}

// NodeDebugInfo

NodeDebugInfo::NodeDebugInfo(const Node& n) : NodeDebugInfo(n.def()) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_36(mht_36_v, 716, "", "./tensorflow/core/graph/graph.cc", "NodeDebugInfo::NodeDebugInfo");
}
NodeDebugInfo::NodeDebugInfo(const NodeDef& ndef)
    : NodeDebugInfo(ndef.name(), ndef.has_experimental_debug_info(),
                    ndef.experimental_debug_info()) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_37(mht_37_v, 722, "", "./tensorflow/core/graph/graph.cc", "NodeDebugInfo::NodeDebugInfo");
}
NodeDebugInfo::NodeDebugInfo(
    StringPiece node_name, bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info)
    : name(node_name) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_38(mht_38_v, 729, "", "./tensorflow/core/graph/graph.cc", "NodeDebugInfo::NodeDebugInfo");

  if (has_experimental_debug_info) {
    const auto& node_names = experimental_debug_info.original_node_names();
    original_node_names.assign(node_names.begin(), node_names.end());
    const auto& func_names = experimental_debug_info.original_func_names();
    original_func_names.assign(func_names.begin(), func_names.end());
  }
}
// InputTensor

bool InputTensor::operator==(const InputTensor& other) const {
  return node == other.node && index == other.index;
}

uint64 InputTensor::Hash::operator()(InputTensor const& s) const {
  return Hash64Combine(std::hash<const Node*>()(s.node),
                       std::hash<int>()(s.index));
}

// OutputTensor

bool OutputTensor::operator==(const OutputTensor& other) const {
  return node == other.node && index == other.index;
}

uint64 OutputTensor::Hash::operator()(OutputTensor const& s) const {
  return Hash64Combine(std::hash<const Node*>()(s.node),
                       std::hash<int>()(s.index));
}

// Graph

Graph::Graph(const OpRegistryInterface* ops)
    : ops_(ops, FunctionDefLibrary()),
      versions_(new VersionDef),
      arena_(8 << 10 /* 8kB */) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_39(mht_39_v, 767, "", "./tensorflow/core/graph/graph.cc", "Graph::Graph");

  versions_->set_producer(TF_GRAPH_DEF_VERSION);
  versions_->set_min_consumer(TF_GRAPH_DEF_VERSION_MIN_CONSUMER);

  // Initialize the name interning table for assigned_device_name.
  device_names_.push_back("");
  DCHECK_EQ(0, InternDeviceName(""));

  // Source and sink have no endpoints, just control edges.
  NodeDef def;
  def.set_name("_SOURCE");
  def.set_op("NoOp");
  Status status;
  Node* source = AddNode(def, &status);
  TF_CHECK_OK(status);
  CHECK_EQ(source->id(), kSourceId);

  def.set_name("_SINK");
  Node* sink = AddNode(def, &status);
  TF_CHECK_OK(status);
  CHECK_EQ(sink->id(), kSinkId);

  AddControlEdge(source, sink);
}

Graph::Graph(const FunctionLibraryDefinition& flib_def)
    : Graph(flib_def.default_registry()) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_40(mht_40_v, 796, "", "./tensorflow/core/graph/graph.cc", "Graph::Graph");

  // Need a new-enough consumer to support the functions we add to the graph.
  if (flib_def.num_functions() > 0 && versions_->min_consumer() < 12) {
    versions_->set_min_consumer(12);
  }
  Status s = ops_.AddLibrary(flib_def);
  CHECK(s.ok()) << s.error_message();
}

Graph::~Graph() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_41(mht_41_v, 808, "", "./tensorflow/core/graph/graph.cc", "Graph::~Graph");

  // Manually call the destructors for all the Nodes we constructed using
  // placement new.
  for (Node* node : nodes_) {
    if (node != nullptr) {
      node->~Node();
    }
  }
  for (Node* node : free_nodes_) {
    node->~Node();
  }
  // Edges have no destructor, and we arena-allocated them, so no need to
  // destroy them.
}

std::unique_ptr<Graph> Graph::Clone() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_42(mht_42_v, 826, "", "./tensorflow/core/graph/graph.cc", "Graph::Clone");

  std::unique_ptr<Graph> new_graph(new Graph(flib_def()));
  new_graph->Copy(*this);
  return new_graph;
}

void Graph::Clear() {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_43(mht_43_v, 835, "", "./tensorflow/core/graph/graph.cc", "Graph::Clear");

  // Do a direct iteration clearing nodes removing the RemoveNode helper method.
  // This could avoid this helper and clear directly if it becomes performance
  // sensitive.
  for (Node* n : nodes()) {
    if (!n->IsSource() && !n->IsSink()) RemoveNode(n);
  }
}

const VersionDef& Graph::versions() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_44(mht_44_v, 847, "", "./tensorflow/core/graph/graph.cc", "Graph::versions");
 return *versions_; }
void Graph::set_versions(const VersionDef& versions) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_45(mht_45_v, 851, "", "./tensorflow/core/graph/graph.cc", "Graph::set_versions");
 *versions_ = versions; }

void Graph::Copy(const Graph& src) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_46(mht_46_v, 856, "", "./tensorflow/core/graph/graph.cc", "Graph::Copy");

  SetConstructionContext(src.GetConstructionContextInternal());
  for (Node* n : nodes()) {
    CHECK(n->IsSource() || n->IsSink()) << "*dest must be empty";
  }

  // Copy GraphDef versions
  set_versions(src.versions());

  // Copy the nodes.
  // "Node in src" -> "Node in *dest"
  gtl::FlatMap<const Node*, Node*> node_map;
  node_map.reserve(src.num_nodes());
  node_map[src.source_node()] = source_node();
  node_map[src.sink_node()] = sink_node();
  for (Node* n : src.op_nodes()) {
    auto copy = CopyNode(n);
    copy->in_edges_.reserve(n->in_edges().size());
    copy->out_edges_.reserve(n->out_edges().size());
    node_map[n] = copy;
  }

  // Copy the edges
  edges_.reserve(src.num_edges());
  for (const Edge* e : src.edges()) {
    Node* src_copy = node_map[e->src()];
    Node* dst_copy = node_map[e->dst()];
    AddEdge(src_copy, e->src_output(), dst_copy, e->dst_input());
  }
}

StatusOr<Node*> Graph::AddNode(NodeDef node_def) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_47(mht_47_v, 890, "", "./tensorflow/core/graph/graph.cc", "Graph::AddNode");

  Status s;
  Node* out = AddNode(std::move(node_def), &s);
  TF_RETURN_IF_ERROR(s);
  return out;
}

Node* Graph::AddNode(NodeDef node_def, Status* status) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_48(mht_48_v, 900, "", "./tensorflow/core/graph/graph.cc", "Graph::AddNode");

  const OpRegistrationData* op_reg_data;
  status->Update(ops_.LookUp(node_def.op(), &op_reg_data));
  if (!status->ok()) return nullptr;

  DataTypeVector inputs;
  DataTypeVector outputs;
  status->Update(
      InOutTypesForNode(node_def, op_reg_data->op_def, &inputs, &outputs));
  if (!status->ok()) {
    *status = AttachDef(*status, node_def);
    return nullptr;
  }

  Node::NodeClass node_class = op_reg_data->is_function_op
                                   ? Node::NC_FUNCTION_OP
                                   : Node::GetNodeClassForOp(node_def.op());

  if (node_def.has_experimental_type()) {
    VLOG(3) << "AddNode: node has type set, skipping type constructor "
            << node_def.name();
  } else {
    if (op_reg_data->type_ctor != nullptr) {
      VLOG(3) << "AddNode: found type constructor for " << node_def.name();
      Status s =
          full_type::SpecializeType(AttrSlice(node_def), op_reg_data->op_def,
                                    *(node_def.mutable_experimental_type()));
      if (!s.ok()) {
        *status = errors::InvalidArgument("type error: ", s.ToString());
        VLOG(3) << "AddNode: type inference failed for " << node_def.name()
                << ": " << s;
        return nullptr;
      }
    } else {
      VLOG(3) << "AddNode: no type constructor for " << node_def.name();
    }
  }

  Node* node = AllocateNode(std::make_shared<NodeProperties>(
                                &op_reg_data->op_def, std::move(node_def),
                                inputs, outputs, op_reg_data->fwd_type_fn),
                            nullptr, node_class);
  return node;
}

Node* Graph::CopyNode(const Node* node) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_49(mht_49_v, 948, "", "./tensorflow/core/graph/graph.cc", "Graph::CopyNode");

  DCHECK(!node->IsSource());
  DCHECK(!node->IsSink());
  Node* copy = AllocateNode(node->props_, node, node->class_);
  copy->set_assigned_device_name(node->assigned_device_name());

  // Since the OpDef of a function may be owned by the Graph that owns 'node',
  // relookup the OpDef in the target graph. If it differs, then clone the
  // node properties with the updated OpDef.
  const OpDef* op_def;
  TF_CHECK_OK(ops_.LookUpOpDef(node->type_string(), &op_def));
  if (op_def != node->props_->op_def) {
    copy->MaybeCopyOnWrite();
    copy->props_->op_def = op_def;
  }
  copy->SetStackTrace(node->GetStackTrace());

  return copy;
}

void Graph::RemoveNode(Node* node) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_50(mht_50_v, 971, "", "./tensorflow/core/graph/graph.cc", "Graph::RemoveNode");

  TF_DCHECK_OK(IsValidNode(node)) << node->DebugString();
  DCHECK(!node->IsSource());
  DCHECK(!node->IsSink());

  // Remove any edges involving this node.
  for (const Edge* e : node->in_edges_) {
    CHECK_EQ(e->src_->out_edges_.erase(e), size_t{1});
    edges_[e->id_] = nullptr;
    RecycleEdge(e);
    --num_edges_;
  }
  node->in_edges_.clear();
  for (const Edge* e : node->out_edges_) {
    CHECK_EQ(e->dst_->in_edges_.erase(e), size_t{1});
    edges_[e->id_] = nullptr;
    RecycleEdge(e);
    --num_edges_;
  }
  node->out_edges_.clear();
  ReleaseNode(node);
}

const Edge* Graph::AddEdge(Node* source, int x, Node* dest, int y) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_51(mht_51_v, 997, "", "./tensorflow/core/graph/graph.cc", "Graph::AddEdge");

  TF_DCHECK_OK(IsValidNode(source)) << source->DebugString();
  TF_DCHECK_OK(IsValidNode(dest)) << dest->DebugString();

  // source/sink must only be linked via control slots, and
  // control slots must only be linked to control slots.
  if (source == source_node() || dest == sink_node() || x == kControlSlot ||
      y == kControlSlot) {
    DCHECK_EQ(x, kControlSlot) << source->DebugString();
    DCHECK_EQ(y, kControlSlot) << dest->DebugString();
  }

  Edge* e = nullptr;
  if (free_edges_.empty()) {
    e = new (arena_.Alloc(sizeof(Edge))) Edge;  // placement new
  } else {
    e = free_edges_.back();
    free_edges_.pop_back();
  }
  e->id_ = edges_.size();
  e->src_ = source;
  e->dst_ = dest;
  e->src_output_ = x;
  e->dst_input_ = y;
  CHECK(source->out_edges_.insert(e).second);
  CHECK(dest->in_edges_.insert(e).second);
  edges_.push_back(e);
  ++num_edges_;

  return e;
}

void Graph::RemoveEdge(const Edge* e) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_52(mht_52_v, 1032, "", "./tensorflow/core/graph/graph.cc", "Graph::RemoveEdge");

  TF_DCHECK_OK(IsValidNode(e->src_)) << e->src_->DebugString();
  TF_DCHECK_OK(IsValidNode(e->dst_)) << e->dst_->DebugString();
  CHECK_EQ(e->src_->out_edges_.erase(e), size_t{1});
  CHECK_EQ(e->dst_->in_edges_.erase(e), size_t{1});
  CHECK_EQ(e, edges_[e->id_]);
  CHECK_GT(num_edges_, 0);

  edges_[e->id_] = nullptr;
  RecycleEdge(e);
  --num_edges_;
}

void Graph::RecycleEdge(const Edge* e) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_53(mht_53_v, 1048, "", "./tensorflow/core/graph/graph.cc", "Graph::RecycleEdge");

  free_edges_.push_back(const_cast<Edge*>(e));
}

const Edge* Graph::AddControlEdge(Node* source, Node* dest,
                                  bool allow_duplicates) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_54(mht_54_v, 1056, "", "./tensorflow/core/graph/graph.cc", "Graph::AddControlEdge");

  if (!allow_duplicates) {
    for (const Edge* edge : dest->in_edges()) {
      if (edge->IsControlEdge() && edge->src() == source) {
        // The requested edge already exists.
        return nullptr;
      }
    }
  }
  // Modify dest's NodeDef if necessary.
  if (!source->IsSource() && !dest->IsSink() && !allow_duplicates) {
    // Check if this input is already in dest's NodeDef.
    const std::string new_input = strings::StrCat("^", source->name());
    bool input_exists = false;
    for (const std::string& input : dest->props_->node_def.input()) {
      if (input == new_input) {
        input_exists = true;
        break;
      }
    }
    if (!input_exists) {
      dest->MaybeCopyOnWrite();
      dest->props_->node_def.add_input(new_input);
    }
  }
  return AddEdge(source, kControlSlot, dest, kControlSlot);
}

void Graph::RemoveControlEdge(const Edge* e) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_55(mht_55_v, 1087, "", "./tensorflow/core/graph/graph.cc", "Graph::RemoveControlEdge");

  if (!e->src_->IsSource() && !e->dst_->IsSink()) {
    e->dst_->MaybeCopyOnWrite();
    std::string e_src_name = strings::StrCat("^", e->src_->name());
    auto* inputs = e->dst_->props_->node_def.mutable_input();
    for (auto it = inputs->begin(); it != inputs->end(); ++it) {
      if (*it == e_src_name) {
        inputs->erase(it);
        break;
      }
    }
  }
  RemoveEdge(e);
}

namespace {
const Edge* FindEdge(const Node* dst, int index) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_56(mht_56_v, 1106, "", "./tensorflow/core/graph/graph.cc", "FindEdge");

  for (const Edge* e : dst->in_edges()) {
    if (e->dst_input() == index) return e;
  }
  return nullptr;
}
}  // namespace

Status Graph::UpdateEdge(Node* new_src, int new_src_index, Node* dst,
                         int dst_index) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_57(mht_57_v, 1118, "", "./tensorflow/core/graph/graph.cc", "Graph::UpdateEdge");

  TF_RETURN_IF_ERROR(IsValidOutputTensor(new_src, new_src_index));
  TF_RETURN_IF_ERROR(IsValidInputTensor(dst, dst_index));
  const Edge* e = FindEdge(dst, dst_index);
  if (e == nullptr) {
    return errors::InvalidArgument("Couldn't find edge to ",
                                   FormatNodeForError(*dst));
  }
  RemoveEdge(e);
  AddEdge(new_src, new_src_index, dst, dst_index);
  dst->MaybeCopyOnWrite();
  (*dst->props_->node_def.mutable_input())[dst_index] =
      strings::StrCat(new_src->name(), ":", new_src_index);
  return Status::OK();
}

Status Graph::AddWhileInputHack(Node* new_src, int new_src_index, Node* dst) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_58(mht_58_v, 1137, "", "./tensorflow/core/graph/graph.cc", "Graph::AddWhileInputHack");

  if (!dst->IsWhileNode()) {
    return errors::Internal(
        "dst argument to AddWhileEdgeHack should be a While op, got: ",
        dst->DebugString());
  }
  TF_RETURN_IF_ERROR(IsValidOutputTensor(new_src, new_src_index));
  // Find the current number of data inputs. We'll add the new edge to the next
  // missing data input.
  int dst_index = 0;
  for (const Edge* edge : dst->in_edges()) {
    if (edge->IsControlEdge()) continue;
    ++dst_index;
  }
  TF_RETURN_IF_ERROR(IsValidInputTensor(dst, dst_index));
  AddEdge(new_src, new_src_index, dst, dst_index);
  dst->MaybeCopyOnWrite();
  dst->props_->node_def.add_input(
      strings::StrCat(new_src->name(), ":", new_src_index));
  return Status::OK();
}

Status Graph::AddFunctionLibrary(const FunctionDefLibrary& fdef_lib) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_59(mht_59_v, 1162, "", "./tensorflow/core/graph/graph.cc", "Graph::AddFunctionLibrary");

  // Need a new-enough consumer to support the functions we add to the graph.
  if (fdef_lib.function_size() > 0 && versions_->min_consumer() < 12) {
    versions_->set_min_consumer(12);
  }
  return ops_.AddLibrary(fdef_lib);
}

namespace {

void AddInput(NodeDef* dst, StringPiece src_name, int src_slot) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_60(mht_60_v, 1175, "", "./tensorflow/core/graph/graph.cc", "AddInput");

  if (src_slot == Graph::kControlSlot) {
    dst->add_input(strings::StrCat("^", src_name));
  } else if (src_slot == 0) {
    dst->add_input(src_name.data(), src_name.size());
  } else {
    dst->add_input(strings::StrCat(src_name, ":", src_slot));
  }
}

}  // namespace

void Graph::ToGraphDef(GraphDef* graph_def) const {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_61(mht_61_v, 1190, "", "./tensorflow/core/graph/graph.cc", "Graph::ToGraphDef");

  ToGraphDefSubRange(graph_def, 0);
}

GraphDef Graph::ToGraphDefDebug() const {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_62(mht_62_v, 1197, "", "./tensorflow/core/graph/graph.cc", "Graph::ToGraphDefDebug");

  GraphDef ret;
  ToGraphDef(&ret);
  return ret;
}

void Graph::ToGraphDefSubRange(GraphDef* graph_def, int from_node_id) const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_63(mht_63_v, 1206, "", "./tensorflow/core/graph/graph.cc", "Graph::ToGraphDefSubRange");

  graph_def->Clear();
  *graph_def->mutable_versions() = versions();
  *graph_def->mutable_library() = ops_.ToProto();

  graph_def->mutable_node()->Reserve(std::max(1, num_nodes() - from_node_id));

  std::vector<const Edge*>
      inputs;  // Construct this outside the loop for speed.
  for (auto id = from_node_id; id < num_node_ids(); ++id) {
    const Node* node = FindNodeId(id);
    if (node == nullptr || !node->IsOp()) continue;
    NodeDef* node_def = graph_def->add_node();
    *node_def = node->def();

    // Use the node's assigned device, if any, instead of the device requested
    // in the NodeDef.
    if (!node->assigned_device_name().empty()) {
      node_def->set_device(node->assigned_device_name());
    }

    // Get the inputs for this Node.  We make sure control inputs are
    // after data inputs, as required by GraphDef.
    inputs.clear();
    inputs.resize(node->num_inputs(), nullptr);
    for (const Edge* edge : node->in_edges()) {
      if (edge->IsControlEdge()) {
        inputs.push_back(edge);
      } else {
        DCHECK(edge->dst_input() < inputs.size())
            << "Edge " << edge->DebugString()
            << " is overflowing the expected number of inputs ("
            << node->num_inputs() << ") for node " << node->DebugString();
        CHECK(inputs[edge->dst_input()] == nullptr)
            << "Edge " << edge->src()->name() << "->" << edge->dst()->name()
            << " conflicts with pre-existing input edge "
            << inputs[edge->dst_input()]->src()->name() << "->"
            << inputs[edge->dst_input()]->dst()->name();

        inputs[edge->dst_input()] = edge;
      }
    }
    // Sort the control inputs for more predictable serialization.
    std::sort(inputs.begin() + node->num_inputs(), inputs.end(),
              [](const Edge* a, const Edge* b) -> bool {
                return a->src()->name() < b->src()->name();
              });
    node_def->clear_input();
    node_def->mutable_input()->Reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
      const Edge* edge = inputs[i];
      if (edge == nullptr) {
        if (i < node->requested_inputs().size()) {
          node_def->add_input(node->requested_inputs()[i]);
        } else {
          node_def->add_input("");
        }
      } else {
        const Node* src = edge->src();
        if (!src->IsOp()) continue;
        AddInput(node_def, src->name(), edge->src_output());
      }
    }
  }
}

std::string Graph::NewName(StringPiece prefix) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_64(mht_64_v, 1276, "", "./tensorflow/core/graph/graph.cc", "Graph::NewName");

  return strings::StrCat(prefix, "/_", name_counter_++);
}

Status Graph::IsValidNode(const Node* node) const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_65(mht_65_v, 1283, "", "./tensorflow/core/graph/graph.cc", "Graph::IsValidNode");

  if (node == nullptr) {
    return errors::InvalidArgument("Node is null");
  }
  const int id = node->id();
  if (id < 0) {
    return errors::InvalidArgument("node id ", id, " is less than zero");
  }
  if (static_cast<size_t>(id) >= nodes_.size()) {
    return errors::InvalidArgument(
        "node id ", id, " is >= than number of nodes in graph ", nodes_.size());
  }
  if (nodes_[id] != node) {
    return errors::InvalidArgument("Node with id ", id,
                                   " is different from the passed in node. "
                                   "Does it belong to a different graph?");
  }
  return Status::OK();
}

Status Graph::IsValidOutputTensor(const Node* node, int idx) const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_66(mht_66_v, 1306, "", "./tensorflow/core/graph/graph.cc", "Graph::IsValidOutputTensor");

  TF_RETURN_IF_ERROR(IsValidNode(node));
  if (idx >= node->num_outputs() || idx < 0) {
    return errors::OutOfRange("Node '", node->name(), "' (type: '",
                              node->op_def().name(),
                              "', num of outputs: ", node->num_outputs(),
                              ") does not have ", "output ", idx);
  }
  return Status::OK();
}

Status Graph::IsValidInputTensor(const Node* node, int idx) const {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_67(mht_67_v, 1320, "", "./tensorflow/core/graph/graph.cc", "Graph::IsValidInputTensor");

  TF_RETURN_IF_ERROR(IsValidNode(node));
  if (idx >= node->num_inputs() || idx < 0) {
    return errors::OutOfRange("Node '", node->name(), "' (type: '",
                              node->op_def().name(),
                              "', num of inputs: ", node->num_inputs(),
                              ") does not have ", "input ", idx);
  }
  return Status::OK();
}

Node* Graph::AllocateNode(std::shared_ptr<NodeProperties> props,
                          const Node* cost_node, Node::NodeClass node_class) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_68(mht_68_v, 1335, "", "./tensorflow/core/graph/graph.cc", "Graph::AllocateNode");

  Node* node = nullptr;
  if (free_nodes_.empty()) {
    node = new (arena_.Alloc(sizeof(Node))) Node;  // placement new
  } else {
    node = free_nodes_.back();
    free_nodes_.pop_back();
  }
  node->graph_ = this;
  const int id = nodes_.size();
  int cost_id = cost_node ? cost_node->cost_id() : id;
  node->Initialize(id, cost_id, std::move(props), node_class);
  nodes_.push_back(node);
  ++num_nodes_;
  return node;
}

void Graph::ReleaseNode(Node* node) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_69(mht_69_v, 1355, "", "./tensorflow/core/graph/graph.cc", "Graph::ReleaseNode");

  TF_DCHECK_OK(IsValidNode(node)) << node->DebugString();
  nodes_[node->id()] = nullptr;
  free_nodes_.push_back(node);
  --num_nodes_;
  node->Clear();
}

// Ensures that 'device_name' is present in the device name table, and returns
// the index of that device name. The index is stable, and can be used in
// calls to Node::set_assigned_device_name_index().
int Graph::InternDeviceName(const std::string& device_name) {
   std::vector<std::string> mht_70_v;
   mht_70_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_70(mht_70_v, 1370, "", "./tensorflow/core/graph/graph.cc", "Graph::InternDeviceName");

  // Special case, very common.  Also, this allows us to use a single map
  // lookup below, instead of two.  The 'if (index_cell > 0)' test below
  // relies on this check.
  if (device_name.empty()) {
    return 0;
  }

  int& index_cell = device_names_map_[device_name];
  if (index_cell > 0) {
    return index_cell;
  }

  const int index = device_names_map_.size();
  index_cell = index;
  device_names_.push_back(device_name);
  return index;
}

Status Graph::AddWhileContext(StringPiece frame_name,
                              std::vector<Node*> enter_nodes,
                              std::vector<Node*> exit_nodes,
                              OutputTensor cond_output,
                              std::vector<OutputTensor> body_inputs,
                              std::vector<OutputTensor> body_outputs,
                              WhileContext** result) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_71(mht_71_v, 1398, "", "./tensorflow/core/graph/graph.cc", "Graph::AddWhileContext");

  auto pair = while_ctxs_.insert(std::pair<std::string, WhileContext>(
      std::string(frame_name),
      WhileContext(frame_name, std::move(enter_nodes), std::move(exit_nodes),
                   cond_output, std::move(body_inputs),
                   std::move(body_outputs))));
  if (!pair.second) {
    *result = nullptr;
    return errors::InvalidArgument("WhileContext with frame name '", frame_name,
                                   "' already exists");
  }
  *result = &pair.first->second;
  return Status::OK();
}

std::unordered_map<std::string, Node*> Graph::BuildNodeNameIndex() const {
  std::unordered_map<std::string, Node*> result;
  for (Node* n : nodes()) {
    result[n->name()] = n;
  }
  return result;
}

std::string Edge::DebugString() const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTcc mht_72(mht_72_v, 1424, "", "./tensorflow/core/graph/graph.cc", "Edge::DebugString");

  return strings::Printf("[id=%d %s:%d -> %s:%d]", id_, src_->name().c_str(),
                         src_output_, dst_->name().c_str(), dst_input_);
}

}  // namespace tensorflow
