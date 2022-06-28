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
class MHTracer_DTPStensorflowPScorePSutilPSequal_graph_defDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_defDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSequal_graph_defDTcc() {
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

#include "tensorflow/core/util/equal_graph_def.h"

#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

bool EqualGraphDef(const GraphDef& actual, const GraphDef& expected,
                   string* diff, const EqualGraphDefOptions& options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_defDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/util/equal_graph_def.cc", "EqualGraphDef");

  // Intentionally do not check that versions match so that this routine can
  // be used for less brittle golden file tests.
  return EqualRepeatedNodeDef(actual.node(), expected.node(), diff, options);
}

uint64 GraphDefHash(const GraphDef& gdef, const EqualGraphDefOptions& options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_defDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/util/equal_graph_def.cc", "GraphDefHash");

  return RepeatedNodeDefHash(gdef.node(), options);
}

bool EqualRepeatedNodeDef(const protobuf::RepeatedPtrField<NodeDef>& actual,
                          const protobuf::RepeatedPtrField<NodeDef>& expected,
                          string* diff, const EqualGraphDefOptions& options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_defDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/util/equal_graph_def.cc", "EqualRepeatedNodeDef");

  std::unordered_map<string, const NodeDef*> actual_index;
  for (const NodeDef& node : actual) {
    actual_index[node.name()] = &node;
  }

  for (const NodeDef& expected_node : expected) {
    auto actual_iter = actual_index.find(expected_node.name());
    if (actual_iter == actual_index.end()) {
      if (diff != nullptr) {
        *diff = strings::StrCat("Did not find expected node '",
                                SummarizeNodeDef(expected_node), "'");
      }
      return false;
    }

    if (!EqualNodeDef(*actual_iter->second, expected_node, diff, options)) {
      return false;
    }

    actual_index.erase(actual_iter);
  }

  if (!actual_index.empty()) {
    if (diff != nullptr) {
      *diff =
          strings::StrCat("Found unexpected node '",
                          SummarizeNodeDef(*actual_index.begin()->second), "'");
    }
    return false;
  }

  return true;
}

uint64 RepeatedNodeDefHash(const protobuf::RepeatedPtrField<NodeDef>& ndefs,
                           const EqualGraphDefOptions& options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_defDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/util/equal_graph_def.cc", "RepeatedNodeDefHash");

  uint64 h = 0xDECAFCAFFE;
  // Insert NodeDefs into map to deterministically sort by name
  std::map<string, const NodeDef*> nodes;
  for (const NodeDef& node : ndefs) {
    nodes[node.name()] = &node;
  }
  for (const auto& pair : nodes) {
    h = Hash64(pair.first.data(), pair.first.size(), h);
    h = Hash64Combine(NodeDefHash(*pair.second, options), h);
  }
  return h;
}

namespace {

string JoinStringField(const protobuf::RepeatedPtrField<string>& f) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_defDTcc mht_4(mht_4_v, 278, "", "./tensorflow/core/util/equal_graph_def.cc", "JoinStringField");

  string ret;
  for (int i = 0; i < f.size(); ++i) {
    if (i > 0) strings::StrAppend(&ret, ", ");
    strings::StrAppend(&ret, f.Get(i));
  }
  return ret;
}

}  // namespace

bool EqualNodeDef(const NodeDef& actual, const NodeDef& expected, string* diff,
                  const EqualGraphDefOptions& options) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_defDTcc mht_5(mht_5_v, 293, "", "./tensorflow/core/util/equal_graph_def.cc", "EqualNodeDef");

  if (actual.name() != expected.name()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Actual node name '", actual.name(),
                              "' is not expected '", expected.name(), "'");
    }
    return false;
  }

  if (actual.op() != expected.op()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Node named '", actual.name(), "' has op '",
                              actual.op(), "' that is not expected '",
                              expected.op(), "'");
    }
    return false;
  }

  if (actual.device() != expected.device()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Node named '", actual.name(), "' has device '",
                              actual.device(), "' that is not expected '",
                              expected.device(), "'");
    }
    return false;
  }

  if (actual.input_size() != expected.input_size()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Node named '", actual.name(), "' has inputs '",
                              JoinStringField(actual.input()),
                              "' that don't match expected '",
                              JoinStringField(expected.input()), "'");
    }
    return false;
  }

  int first_control_input = actual.input_size();
  for (int i = 0; i < actual.input_size(); ++i) {
    if (absl::StartsWith(actual.input(i), "^")) {
      first_control_input = i;
      break;
    }
    // Special case for inputs: "tensor" is equivalent to "tensor:0"
    if (actual.input(i) != expected.input(i) &&
        actual.input(i) != strings::StrCat(expected.input(i), ":0") &&
        strings::StrCat(actual.input(i), ":0") != expected.input(i)) {
      if (diff != nullptr) {
        *diff = strings::StrCat("Node named '", actual.name(), "' has input ",
                                i, " '", actual.input(i),
                                "' that doesn't match expected '",
                                expected.input(i), "'");
      }
      return false;
    }
  }

  std::unordered_set<string> actual_control;
  std::unordered_set<string> expected_control;
  for (int i = first_control_input; i < actual.input_size(); ++i) {
    actual_control.insert(actual.input(i));
    expected_control.insert(expected.input(i));
  }
  for (const auto& e : expected_control) {
    if (actual_control.erase(e) == 0) {
      if (diff != nullptr) {
        *diff = strings::StrCat("Node named '", actual.name(),
                                "' missing expected control input '", e, "'");
      }
      return false;
    }
  }
  if (!actual_control.empty()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Node named '", actual.name(),
                              "' has unexpected control input '",
                              *actual_control.begin(), "'");
    }
    return false;
  }

  std::unordered_set<string> actual_attr;
  for (const auto& a : actual.attr()) {
    if (options.ignore_internal_attrs && !a.first.empty() &&
        a.first[0] == '_') {
      continue;
    }
    actual_attr.insert(a.first);
  }
  for (const auto& e : expected.attr()) {
    if (options.ignore_internal_attrs && !e.first.empty() &&
        e.first[0] == '_') {
      continue;
    }

    if (actual_attr.erase(e.first) == 0) {
      if (diff != nullptr) {
        *diff = strings::StrCat("Node named '", actual.name(),
                                "' missing expected attr '", e.first,
                                "' with value: ", SummarizeAttrValue(e.second));
      }
      return false;
    }
    auto iter = actual.attr().find(e.first);
    if (!AreAttrValuesEqual(e.second, iter->second)) {
      if (diff != nullptr) {
        *diff = strings::StrCat(
            "Node named '", actual.name(), "' has attr '", e.first,
            "' with value: ", SummarizeAttrValue(iter->second),
            " that does not match expected: ", SummarizeAttrValue(e.second));
      }
      return false;
    }
  }
  if (!actual_attr.empty()) {
    if (diff != nullptr) {
      *diff = strings::StrCat(
          "Node named '", actual.name(), "' has unexpected attr '",
          *actual_attr.begin(), "' with value: ",
          SummarizeAttrValue(actual.attr().find(*actual_attr.begin())->second));
    }
    return false;
  }

  return true;
}

uint64 NodeDefHash(const NodeDef& ndef, const EqualGraphDefOptions& options) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_defDTcc mht_6(mht_6_v, 423, "", "./tensorflow/core/util/equal_graph_def.cc", "NodeDefHash");

  uint64 h = Hash64(ndef.name());
  h = Hash64(ndef.op().data(), ndef.op().size(), h);
  h = Hash64(ndef.device().data(), ndef.device().size(), h);

  // Normal inputs. Order important.
  int first_control_input = ndef.input_size();
  for (int i = 0; i < ndef.input_size(); ++i) {
    if (absl::StartsWith(ndef.input(i), "^")) {
      first_control_input = i;
      break;
    }
    h = Hash64(ndef.input(i).data(), ndef.input(i).size(), h);
  }

  // Control inputs. Order irrelevant.
  std::set<string> ndef_control;
  for (int i = first_control_input; i < ndef.input_size(); ++i) {
    ndef_control.insert(ndef.input(i));
  }
  for (const string& s : ndef_control) {
    h = Hash64(s.data(), s.size(), h);
  }

  // Attributes
  std::map<string, AttrValue> ndef_attr;
  for (const auto& a : ndef.attr()) {
    if (options.ignore_internal_attrs && !a.first.empty() &&
        a.first[0] == '_') {
      continue;
    }
    ndef_attr[a.first] = a.second;
  }
  for (const auto& a : ndef_attr) {
    h = Hash64(a.first.data(), a.first.size(), h);
    h = Hash64Combine(AttrValueHash(a.second), h);
  }

  return h;
}

}  // namespace tensorflow
