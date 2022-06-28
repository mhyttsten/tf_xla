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

// Provides a set of matchers for tensorflow nodes.
//
// Example usage:
//
//  tensorflow::Node* node = ...;
//  EXPECT_THAT(node, NodeWith(Name("name"), Op("op"),
//                             Inputs(Out(3, NodeWith(Name("input"))))))
//
// Matchable node properties (the expressions that go inside NodeWith(...))
// are:
//
//  - Name(string): matches the node name exactly.  We will probably need to
//    have this take a string matcher soon in the future.
//
//  - Op(string): matches the op exactly.
//
//  - AssignedDevice(string): matches the assigned device exactly.
//
//  - Inputs(<ordered list>): matches the list of non-control inputs to the node
//    exactly (i.e. does not match a suffix or a prefix) where each element
//    matches an output of a node (see Out(idx, node) below).
//
//  - CtrlDeps(<unordered list>): matches the list of control dependences on the
//    node exactly but in any order.
//
//  - ConstantValue(tensorflow::Input::Initializer init): matches a Const node
//    with the constant value `init`.  Implies Op("Const").
//
//  - Attr(name, value): Matches a single attribute with name `name` and value
//    `value`.  Right now only boolean values are supported.
//
// Overlapping node properties may not be repeated in a single NodeWith(...)
// matcher.  E.g. NodeWith(Op("Foo"), Op("Bar")) will CHECK-fail.  Since
// ConstantValue implies Op("Const"), a single NodeWith matcher can't have both
// ConstantValue(...) and Op(...).  Multiple Attr() values can be combined as
// long as the attribute names are different.
//
// Out(idx, node) matches the `idx`'th output of a node that matches `node`.

#ifndef TENSORFLOW_COMPILER_JIT_NODE_MATCHERS_H_
#define TENSORFLOW_COMPILER_JIT_NODE_MATCHERS_H_
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
class MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh() {
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


#include <array>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace testing {
namespace matchers {

namespace impl {

using OutEdge = std::pair<const Node*, int>;

// -----------------------------------------------------------------------------
// Implementation details.

// Properties that we match on for a particular Node.  If a particular property
// is nullopt then any value for it is allowed.
class NodeMatcherProperties {
 public:
  using NodeSeqMatcher = std::vector<::testing::Matcher<const Node*>>;
  using InputSeqMatcher = std::vector<::testing::Matcher<OutEdge>>;
  using AttrKeyValuePair = std::pair<string, absl::optional<AttrValue>>;

  const absl::optional<string>& name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_0(mht_0_v, 258, "", "./tensorflow/compiler/jit/node_matchers.h", "name");
 return name_; }
  const absl::optional<string>& op() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_1(mht_1_v, 262, "", "./tensorflow/compiler/jit/node_matchers.h", "op");
 return op_; }
  const absl::optional<string>& assigned_device() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_2(mht_2_v, 266, "", "./tensorflow/compiler/jit/node_matchers.h", "assigned_device");

    return assigned_device_;
  }
  const absl::optional<Tensor>& constant_value() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_3(mht_3_v, 272, "", "./tensorflow/compiler/jit/node_matchers.h", "constant_value");

    return constant_value_;
  }
  const absl::optional<InputSeqMatcher>& inputs() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_4(mht_4_v, 278, "", "./tensorflow/compiler/jit/node_matchers.h", "inputs");

    return input_matchers_;
  }
  const absl::optional<NodeSeqMatcher>& control_deps() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_5(mht_5_v, 284, "", "./tensorflow/compiler/jit/node_matchers.h", "control_deps");

    return control_deps_;
  }
  const absl::optional<AttrKeyValuePair>& attr() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_6(mht_6_v, 290, "", "./tensorflow/compiler/jit/node_matchers.h", "attr");
 return attr_; }

  void set_name(string name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_7(mht_7_v, 296, "", "./tensorflow/compiler/jit/node_matchers.h", "set_name");

    DCHECK(IsEmpty());
    name_ = std::move(name);
  }

  void set_op(string op) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_8(mht_8_v, 305, "", "./tensorflow/compiler/jit/node_matchers.h", "set_op");

    DCHECK(IsEmpty());
    op_ = std::move(op);
  }

  void set_assigned_device(string assigned_device) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("assigned_device: \"" + assigned_device + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_9(mht_9_v, 314, "", "./tensorflow/compiler/jit/node_matchers.h", "set_assigned_device");

    DCHECK(IsEmpty());
    assigned_device_ = std::move(assigned_device);
  }

  void set_constant_value(Tensor constant_value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_10(mht_10_v, 322, "", "./tensorflow/compiler/jit/node_matchers.h", "set_constant_value");

    DCHECK(IsEmpty());
    constant_value_ = std::move(constant_value);
    op_ = "Const";
  }

  void set_inputs(InputSeqMatcher inputs) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_11(mht_11_v, 331, "", "./tensorflow/compiler/jit/node_matchers.h", "set_inputs");

    DCHECK(IsEmpty());
    input_matchers_ = std::move(inputs);
  }

  void set_control_deps(NodeSeqMatcher control_deps) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_12(mht_12_v, 339, "", "./tensorflow/compiler/jit/node_matchers.h", "set_control_deps");

    DCHECK(IsEmpty());
    control_deps_ = std::move(control_deps);
  }

  void set_attr(AttrKeyValuePair attr) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_13(mht_13_v, 347, "", "./tensorflow/compiler/jit/node_matchers.h", "set_attr");

    DCHECK(IsEmpty());
    attr_ = std::move(attr);
  }

  bool IsEmpty() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_14(mht_14_v, 355, "", "./tensorflow/compiler/jit/node_matchers.h", "IsEmpty");

    return !name().has_value() && !op().has_value() && !inputs().has_value() &&
           !control_deps().has_value() && !attr().has_value();
  }

 private:
  absl::optional<string> name_;
  absl::optional<string> op_;
  absl::optional<string> assigned_device_;
  absl::optional<Tensor> constant_value_;
  absl::optional<InputSeqMatcher> input_matchers_;
  absl::optional<NodeSeqMatcher> control_deps_;
  absl::optional<AttrKeyValuePair> attr_;
};

::testing::Matcher<const Node*> NodeWith(
    absl::Span<const NodeMatcherProperties> props);

impl::NodeMatcherProperties Inputs(
    absl::Span<const ::testing::Matcher<OutEdge>> inputs);

impl::NodeMatcherProperties CtrlDeps(
    absl::Span<const ::testing::Matcher<const Node*>> control_deps);

impl::NodeMatcherProperties Attr(std::pair<string, AttrValue> attrs);
impl::NodeMatcherProperties Attr(string name);

std::pair<string, AttrValue> AttrLiteralHelper(
    const std::pair<string, bool>& bool_attr);

std::pair<string, AttrValue> AttrLiteralHelper(
    const std::pair<string, absl::Span<const int>>& int_list_attr);

std::pair<string, AttrValue> AttrLiteralHelper(
    const std::pair<string, absl::Span<const string>>& string_list_attr);
}  // namespace impl

// -----------------------------------------------------------------------------
// Public interface.

// Matches a node with name `name`.
impl::NodeMatcherProperties Name(string name);

// Matches a node with op `op`.
impl::NodeMatcherProperties Op(string op);

// Matches a node with assigned device `assigned_device`.
impl::NodeMatcherProperties AssignedDevice(string assigned_device);

// Matches a node with a boolean typed attribute named `name` and with value
// `value`.
template <typename ValueTy>
impl::NodeMatcherProperties Attr(const string& name, ValueTy value) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_15(mht_15_v, 411, "", "./tensorflow/compiler/jit/node_matchers.h", "Attr");

  return impl::Attr({impl::AttrLiteralHelper({name, value})});
}

inline impl::NodeMatcherProperties Attr(const string& name) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_16(mht_16_v, 419, "", "./tensorflow/compiler/jit/node_matchers.h", "Attr");

  return impl::Attr(name);
}

// Matches a node with inputs `inputs`.
//
// `inputs` are ordered; `inputs`[i] must match input i.
template <typename... Ts>
impl::NodeMatcherProperties Inputs(Ts... inputs) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_17(mht_17_v, 430, "", "./tensorflow/compiler/jit/node_matchers.h", "Inputs");

  return impl::Inputs({inputs...});
}

// Matches the `idx`'th output of a node that matches `node`.
::testing::Matcher<impl::OutEdge> Out(int oidx,
                                      ::testing::Matcher<const Node*> node);

// Matches the first output of a node that matches `node`.
inline ::testing::Matcher<impl::OutEdge> Out(
    ::testing::Matcher<const Node*> node) {
  return Out(0, node);
}

// Matches a node with control dependences `control_deps`.
//
// `control_deps` are unordered and will match the control deps of a node in any
// order.
template <typename... Ts>
impl::NodeMatcherProperties CtrlDeps(Ts... control_deps) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSnode_matchersDTh mht_18(mht_18_v, 452, "", "./tensorflow/compiler/jit/node_matchers.h", "CtrlDeps");

  return impl::CtrlDeps({control_deps...});
}

// Matches a constant node with value `val`.
impl::NodeMatcherProperties ConstantValue(
    const ::tensorflow::Input::Initializer& val);

// The main gmock matcher.  See file comment for example usage.
template <typename... Ts>
::testing::Matcher<const Node*> NodeWith(Ts... args) {
  std::array<impl::NodeMatcherProperties, sizeof...(Ts)> array = {args...};
  return impl::NodeWith(array);
}

::testing::Matcher<impl::OutEdge> Const(
    const ::tensorflow::Input::Initializer& val);
}  // namespace matchers

// If `g` has a node named `name` returns it, otherwise returns null.
Node* FindNodeByName(Graph* g, absl::string_view name);
}  // namespace testing

void PrintTo(const Node* n, ::std::ostream* os);
void PrintTo(Node* n, ::std::ostream* os);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_NODE_MATCHERS_H_
