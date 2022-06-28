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
class MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc() {
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

#include "tensorflow/compiler/jit/node_matchers.h"

#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/graph_node_util.h"

namespace tensorflow {
namespace testing {
namespace matchers {
namespace {

using impl::NodeMatcherProperties;
using impl::OutEdge;

string IndentAllButFirstLine(absl::string_view text) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("text: \"" + std::string(text.data(), text.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/jit/node_matchers.cc", "IndentAllButFirstLine");

  std::vector<std::string> lines = absl::StrSplit(text, '\n');
  for (int i = 1; i < lines.size(); i++) {
    lines[i].insert(0, "  ");
  }
  return absl::StrJoin(lines, "\n");
}

template <typename T>
bool CompareTensor(const Tensor& actual, const Tensor& expected,
                   ::testing::MatchResultListener* listener) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_1(mht_1_v, 221, "", "./tensorflow/compiler/jit/node_matchers.cc", "CompareTensor");

  if (actual.NumElements() != expected.NumElements()) {
    if (listener->IsInterested()) {
      *listener << "\nwas looking for tensor with " << expected.NumElements()
                << " elements, found tensor with " << actual.NumElements()
                << " elements";
      return false;
    }
  }

  for (int64_t i = 0, e = actual.NumElements(); i < e; i++) {
    if (actual.flat<T>()(i) != expected.flat<T>()(i)) {
      *listener << "\nmismatch in constant tensor at index " << i
                << " expected = " << expected.flat<T>()(i)
                << " actual = " << actual.flat<T>()(i);
      return false;
    }
  }

  return true;
}

bool MatchAndExplainTensor(const Tensor& tensor, const Tensor& expected_tensor,
                           ::testing::MatchResultListener* listener) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_2(mht_2_v, 247, "", "./tensorflow/compiler/jit/node_matchers.cc", "MatchAndExplainTensor");

  if (tensor.dtype() != expected_tensor.dtype()) {
    if (listener->IsInterested()) {
      *listener << "\nexpected tensor of type "
                << DataType_Name(expected_tensor.dtype())
                << " but found one of type " << DataType_Name(tensor.dtype());
      return false;
    }
  }

  switch (tensor.dtype()) {
    case DT_HALF:
      return CompareTensor<Eigen::half>(tensor, expected_tensor, listener);
    case DT_FLOAT:
      return CompareTensor<float>(tensor, expected_tensor, listener);
    case DT_DOUBLE:
      return CompareTensor<double>(tensor, expected_tensor, listener);
    case DT_INT8:
      return CompareTensor<int8>(tensor, expected_tensor, listener);
    case DT_INT16:
      return CompareTensor<int16>(tensor, expected_tensor, listener);
    case DT_INT32:
      return CompareTensor<int32>(tensor, expected_tensor, listener);
    case DT_INT64:
      return CompareTensor<int64_t>(tensor, expected_tensor, listener);
    case DT_UINT8:
      return CompareTensor<uint8>(tensor, expected_tensor, listener);
    case DT_UINT16:
      return CompareTensor<uint16>(tensor, expected_tensor, listener);
    case DT_UINT32:
      return CompareTensor<uint32>(tensor, expected_tensor, listener);
    case DT_UINT64:
      return CompareTensor<uint64>(tensor, expected_tensor, listener);
    default:
      LOG(FATAL) << "Unsupported dtype "  // Crash ok: testonly.
                 << DataType_Name(tensor.dtype());
  }
}

struct NodeMatcher : public ::testing::MatcherInterface<const Node*> {
  bool MatchAndExplain(
      const Node* node,
      ::testing::MatchResultListener* listener) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_3(mht_3_v, 292, "", "./tensorflow/compiler/jit/node_matchers.cc", "MatchAndExplain");

    if (op && node->type_string() != *op) {
      if (listener->IsInterested()) {
        *listener << "\nexpected op " << *op << " but found "
                  << node->type_string();
      }
      return false;
    }

    if (assigned_device && node->assigned_device_name() != *assigned_device) {
      if (listener->IsInterested()) {
        *listener << "\nexpected assigned_device " << *assigned_device
                  << " but found \"" << node->assigned_device_name() << "\"";
      }
      return false;
    }

    if (name && node->name() != *name) {
      if (listener->IsInterested()) {
        *listener << "\nexpected name " << *name << " but found "
                  << node->name();
      }
      return false;
    }

    if (constant_value) {
      const TensorProto* proto = nullptr;
      if (!TryGetNodeAttr(node->def(), "value", &proto)) {
        if (listener->IsInterested()) {
          *listener << "\ncould not find \"value\" attribute in node";
        }
        return false;
      }

      Tensor tensor(proto->dtype());
      if (!tensor.FromProto(*proto)) {
        if (listener->IsInterested()) {
          *listener << "\ncould not convert TensorProto in \"value\" attribute "
                       "to Tensor";
        }
        return false;
      }

      if (!MatchAndExplainTensor(/*tensor=*/tensor,
                                 /*expected_tensor=*/*constant_value,
                                 listener)) {
        return false;
      }
    }

    if (input_matchers) {
      if (input_matchers->size() != node->num_inputs()) {
        if (listener->IsInterested()) {
          *listener << "\nexpected " << input_matchers->size()
                    << " inputs but node has " << node->num_inputs();
        }
        return false;
      }

      for (int input_idx = 0, e = input_matchers->size(); input_idx < e;
           input_idx++) {
        if (!MatchAndExplainInput(node, input_idx, listener)) {
          return false;
        }
      }
    }

    std::vector<const Node*> control_deps;
    for (const Edge* e : node->in_edges()) {
      if (e->IsControlEdge()) {
        control_deps.push_back(e->src());
      }
    }

    ::testing::StringMatchResultListener inner_listener;
    if (control_dep_set &&
        !control_dep_set->MatchAndExplain(control_deps, &inner_listener)) {
      if (listener->IsInterested()) {
        string explanation = inner_listener.str();
        if (!explanation.empty()) {
          explanation = absl::StrCat(", ", explanation, ",");
        }
        *listener << "ctrl_deps" << explanation << " does not match expected: ";
        control_dep_set->DescribeTo(listener->stream());
      }
      return false;
    }

    const AttrValueMap attr_value_map = node->def().attr();
    for (const auto& attr_kv_pair : attrs) {
      auto it = attr_value_map.find(attr_kv_pair.first);
      if (it == attr_value_map.end()) {
        if (listener->IsInterested()) {
          *listener << "did not find attribute named \"" << attr_kv_pair.first
                    << "\" in node";
        }
        return false;
      }
      if (attr_kv_pair.second &&
          !AreAttrValuesEqual(it->second, *attr_kv_pair.second)) {
        if (listener->IsInterested()) {
          *listener << "attribute named " << attr_kv_pair.first
                    << " does not match value; expected: \""
                    << SummarizeAttrValue(*attr_kv_pair.second)
                    << "\", found: \"" << SummarizeAttrValue(it->second)
                    << "\"";
        }
        return false;
      }
    }

    return true;
  }

  void DescribeTo(::std::ostream* os) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_4(mht_4_v, 409, "", "./tensorflow/compiler/jit/node_matchers.cc", "DescribeTo");

    std::vector<string> predicates;

    if (name) {
      predicates.push_back(absl::StrCat("name: ", *name));
    }

    if (op) {
      predicates.push_back(absl::StrCat("op: ", *op));
    }

    if (assigned_device) {
      predicates.push_back(absl::StrCat("assigned device: ", *assigned_device));
    }

    bool printed_something = !predicates.empty();

    *os << absl::StrJoin(predicates, ", ");

    if (constant_value) {
      printed_something = true;
      *os << "constant value: " << constant_value->DebugString();
    }

    if (input_matchers) {
      if (!input_matchers->empty()) {
        printed_something = true;
        *os << " with " << (input_matchers->size() == 1 ? "only " : "")
            << "input" << (input_matchers->size() == 1 ? "" : "s") << " ";
      }

      if (input_matchers->size() == 1) {
        ::std::stringstream ss;
        input_matchers->front().DescribeTo(&ss);
        printed_something = true;
        *os << "matching " << ss.str();
      } else {
        int edge_idx = 0;
        for (const ::testing::Matcher<OutEdge>& matcher : (*input_matchers)) {
          *os << "\n  [" << edge_idx << "] matching (";
          ::std::stringstream ss;
          matcher.DescribeTo(&ss);
          printed_something = true;
          *os << IndentAllButFirstLine(ss.str());
          *os << ")";
          edge_idx++;
        }
      }
    }

    if (control_dep_set) {
      printed_something = true;
      *os << " and control deps ";
      control_dep_set->DescribeTo(os);
    }

    if (!attrs.empty()) {
      printed_something = true;
      std::vector<string> attrs_str;
      absl::c_transform(
          attrs, std::back_inserter(attrs_str),
          [](const std::pair<string, absl::optional<AttrValue>>& attr_kv_pair) {
            return absl::StrCat(attr_kv_pair.first, "->",
                                attr_kv_pair.second
                                    ? SummarizeAttrValue(*attr_kv_pair.second)
                                    : "*");
          });
      *os << " and attr values matching [" << absl::StrJoin(attrs_str, ", ")
          << "]";
    }

    if (!printed_something) {
      *os << "is any node";
    }
  }

  bool MatchAndExplainInput(const Node* node, int input_idx,
                            ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_5(mht_5_v, 489, "", "./tensorflow/compiler/jit/node_matchers.cc", "MatchAndExplainInput");

    const Edge* edge;
    if (!node->input_edge(input_idx, &edge).ok()) {
      if (listener->IsInterested()) {
        *listener << "\ncould not find incoming edge for input " << input_idx;
      }
      return false;
    }

    ::testing::StringMatchResultListener inner_listener;
    OutEdge input = {edge->src(), edge->src_output()};
    if ((*input_matchers)[input_idx].MatchAndExplain(input, &inner_listener)) {
      return true;
    }

    if (listener->IsInterested()) {
      *listener << "\ninput " << input_idx << " does not match expected:\n";
      (*input_matchers)[input_idx].DescribeTo(listener->stream());
      string explanation = inner_listener.str();
      if (!explanation.empty()) {
        *listener << ", " << explanation;
      }
    }
    return false;
  }

  absl::optional<string> op;
  absl::optional<string> name;
  absl::optional<string> assigned_device;
  absl::optional<Tensor> constant_value;
  absl::optional<std::vector<::testing::Matcher<OutEdge>>> input_matchers;
  absl::optional<::testing::Matcher<absl::Span<const Node* const>>>
      control_dep_set;
  std::map<string, absl::optional<AttrValue>> attrs;
};

// Matches a dst and dst_output on an input edge.  Today we only use this with
// dst_output=0 but we will eventually need to support multi-output operations.
class OutEdgeMatcher : public ::testing::MatcherInterface<OutEdge> {
 public:
  OutEdgeMatcher(::testing::Matcher<const Node*> src_matcher, int src_oidx)
      : src_matcher_(std::move(src_matcher)), src_oidx_(src_oidx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_6(mht_6_v, 533, "", "./tensorflow/compiler/jit/node_matchers.cc", "OutEdgeMatcher");
}

  bool MatchAndExplain(
      OutEdge out_edge,
      ::testing::MatchResultListener* listener) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_7(mht_7_v, 540, "", "./tensorflow/compiler/jit/node_matchers.cc", "MatchAndExplain");

    ::testing::StringMatchResultListener inner_listener;
    if (!src_matcher_.MatchAndExplain(out_edge.first, &inner_listener)) {
      if (listener->IsInterested()) {
        *listener << "\nsource does not match expected ";
        src_matcher_.DescribeTo(listener->stream());
        string explanation = inner_listener.str();
        if (!explanation.empty()) {
          *listener << "\n\t" << explanation;
        }
      }
      return false;
    }
    if (out_edge.second != src_oidx_) {
      if (listener->IsInterested()) {
        *listener << "\nexpected output slot to be " << src_oidx_
                  << " but found " << out_edge.second;
      }
      return false;
    }

    return true;
  }

  void DescribeTo(::std::ostream* os) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_8(mht_8_v, 567, "", "./tensorflow/compiler/jit/node_matchers.cc", "DescribeTo");

    if (src_oidx_) {
      *os << "output slot: " << src_oidx_ << ", source: (";
    }

    src_matcher_.DescribeTo(os);

    if (src_oidx_) {
      *os << ")";
    }
  }

 private:
  ::testing::Matcher<const Node*> src_matcher_;
  int src_oidx_;
};
}  // namespace

::testing::Matcher<const Node*> impl::NodeWith(
    absl::Span<const NodeMatcherProperties> props) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_9(mht_9_v, 589, "", "./tensorflow/compiler/jit/node_matchers.cc", "impl::NodeWith");

  NodeMatcher* matcher = new NodeMatcher();
  for (const NodeMatcherProperties& prop : props) {
    if (prop.name()) {
      DCHECK(!matcher->name);
      matcher->name = prop.name();
    }

    if (prop.op()) {
      DCHECK(!matcher->op);
      matcher->op = prop.op();
    }

    if (prop.constant_value()) {
      DCHECK(!matcher->constant_value);
      matcher->constant_value = prop.constant_value();
    }

    if (prop.assigned_device()) {
      DCHECK(!matcher->assigned_device);
      matcher->assigned_device = prop.assigned_device();
    }

    if (prop.inputs()) {
      DCHECK(!matcher->input_matchers);
      matcher->input_matchers = *prop.inputs();
    }

    if (prop.control_deps()) {
      DCHECK(!matcher->control_dep_set);
      matcher->control_dep_set =
          ::testing::UnorderedElementsAreArray(*prop.control_deps());
    }

    if (prop.attr()) {
      auto insert_result = matcher->attrs.insert(*prop.attr());
      DCHECK(insert_result.second);
    }
  }

  return ::testing::MakeMatcher(matcher);
}

impl::NodeMatcherProperties Name(string name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_10(mht_10_v, 636, "", "./tensorflow/compiler/jit/node_matchers.cc", "Name");

  impl::NodeMatcherProperties props;
  props.set_name(std::move(name));
  return props;
}

// Matches a node with op `op`.
impl::NodeMatcherProperties Op(string op) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_11(mht_11_v, 647, "", "./tensorflow/compiler/jit/node_matchers.cc", "Op");

  impl::NodeMatcherProperties props;
  props.set_op(std::move(op));
  return props;
}

// Matches a node with assigned device `assigned_device`.
impl::NodeMatcherProperties AssignedDevice(string assigned_device) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("assigned_device: \"" + assigned_device + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_12(mht_12_v, 658, "", "./tensorflow/compiler/jit/node_matchers.cc", "AssignedDevice");

  impl::NodeMatcherProperties props;
  props.set_assigned_device(std::move(assigned_device));
  return props;
}

impl::NodeMatcherProperties impl::Inputs(
    absl::Span<const ::testing::Matcher<OutEdge>> inputs) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_13(mht_13_v, 668, "", "./tensorflow/compiler/jit/node_matchers.cc", "impl::Inputs");

  std::vector<::testing::Matcher<OutEdge>> inputs_vector;
  absl::c_copy(inputs, std::back_inserter(inputs_vector));

  impl::NodeMatcherProperties props;
  props.set_inputs(std::move(inputs_vector));
  return props;
}

impl::NodeMatcherProperties impl::CtrlDeps(
    absl::Span<const ::testing::Matcher<const Node*>> control_deps) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_14(mht_14_v, 681, "", "./tensorflow/compiler/jit/node_matchers.cc", "impl::CtrlDeps");

  std::vector<::testing::Matcher<const Node*>> control_deps_vector;
  absl::c_copy(control_deps, std::back_inserter(control_deps_vector));

  impl::NodeMatcherProperties props;
  props.set_control_deps(std::move(control_deps_vector));
  return props;
}

std::pair<string, AttrValue> impl::AttrLiteralHelper(
    const std::pair<string, bool>& bool_attr) {
  AttrValue attr_value;
  attr_value.set_b(bool_attr.second);
  return {bool_attr.first, attr_value};
}

std::pair<string, AttrValue> impl::AttrLiteralHelper(
    const std::pair<string, absl::Span<const int>>& int_list_attr) {
  AttrValue attr_value;
  AttrValue::ListValue* list = attr_value.mutable_list();
  for (int i : int_list_attr.second) {
    list->add_i(i);
  }
  return {int_list_attr.first, attr_value};
}

std::pair<string, AttrValue> impl::AttrLiteralHelper(
    const std::pair<string, absl::Span<const string>>& string_list_attr) {
  AttrValue attr_value;
  AttrValue::ListValue* list = attr_value.mutable_list();
  for (const string& s : string_list_attr.second) {
    list->add_s(s);
  }
  return {string_list_attr.first, attr_value};
}

impl::NodeMatcherProperties impl::Attr(std::pair<string, AttrValue> attr) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_15(mht_15_v, 720, "", "./tensorflow/compiler/jit/node_matchers.cc", "impl::Attr");

  impl::NodeMatcherProperties props;
  props.set_attr(std::move(attr));
  return props;
}

impl::NodeMatcherProperties impl::Attr(string name) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_16(mht_16_v, 730, "", "./tensorflow/compiler/jit/node_matchers.cc", "impl::Attr");

  impl::NodeMatcherProperties props;
  props.set_attr({std::move(name), absl::nullopt});
  return props;
}

NodeMatcherProperties ConstantValue(
    const ::tensorflow::Input::Initializer& val) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_17(mht_17_v, 740, "", "./tensorflow/compiler/jit/node_matchers.cc", "ConstantValue");

  TF_CHECK_OK(val.status);
  NodeMatcherProperties props;
  props.set_constant_value(val.tensor);
  return props;
}

::testing::Matcher<impl::OutEdge> Const(
    const ::tensorflow::Input::Initializer& val) {
  return Out(NodeWith(ConstantValue(val)));
}
::testing::Matcher<impl::OutEdge> Out(
    int oidx, ::testing::Matcher<const Node*> node_matcher) {
  return ::testing::MakeMatcher(new OutEdgeMatcher(node_matcher, oidx));
}
}  // namespace matchers

Node* FindNodeByName(Graph* g, absl::string_view name) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_18(mht_18_v, 761, "", "./tensorflow/compiler/jit/node_matchers.cc", "FindNodeByName");

  for (Node* n : g->nodes()) {
    if (n->name() == name) {
      return n;
    }
  }

  return nullptr;
}
}  // namespace testing

void PrintTo(const Node* n, ::std::ostream* os) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_19(mht_19_v, 775, "", "./tensorflow/compiler/jit/node_matchers.cc", "PrintTo");
 *os << SummarizeNode(*n); }
void PrintTo(Node* n, ::std::ostream* os) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTcc mht_20(mht_20_v, 779, "", "./tensorflow/compiler/jit/node_matchers.cc", "PrintTo");
 *os << SummarizeNode(*n); }
}  // namespace tensorflow
