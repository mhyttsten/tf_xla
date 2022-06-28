/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_RESOURCE_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_RESOURCE_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_utilDTh() {
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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
class ResourceUsageAnalysis {
 public:
  // NodeInfo is a triple of function_name:node_name:op to uniquely identity a
  // node in graph. ResourceUsageAnalysis uses it to represent resource sources
  // and users.
  class NodeInfo {
   public:
    absl::optional<std::string> function_name_;
    std::string node_name_;
    std::string op_;

    NodeInfo() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_utilDTh mht_0(mht_0_v, 211, "", "./tensorflow/compiler/tf2xla/resource_util.h", "NodeInfo");
}

    NodeInfo(const absl::optional<std::string>& function_name,
             std::string node_name, std::string op)
        : function_name_(function_name),
          node_name_(std::move(node_name)),
          op_(std::move(op)) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("node_name: \"" + node_name + "\"");
   mht_1_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_utilDTh mht_1(mht_1_v, 222, "", "./tensorflow/compiler/tf2xla/resource_util.h", "NodeInfo");
}

    std::string DebugString() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_utilDTh mht_2(mht_2_v, 227, "", "./tensorflow/compiler/tf2xla/resource_util.h", "DebugString");

      return absl::StrJoin({function_name_.value_or(""), node_name_, op_}, ":");
    }

    bool operator==(const NodeInfo& o) const {
      return function_name_ == o.function_name_ && node_name_ == o.node_name_ &&
             op_ == o.op_;
    }

    template <typename H>
    friend H AbslHashValue(H h, const NodeInfo& o) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_utilDTh mht_3(mht_3_v, 240, "", "./tensorflow/compiler/tf2xla/resource_util.h", "AbslHashValue");

      return H::combine(std::move(h), o.function_name_, o.node_name_, o.op_);
    }
  };

  // This method analyzes a Tensorflow graph and finds all operations that
  // create Stack/TensorArray resources and all the operations that consume
  // resource created by them.
  //
  // Note that _Arg nodes that introduce resources are not considered sources.
  // Note again that Control Flow v1 nodes
  // (Enter/Exit/Switch/Merge/NextIteration) are not supported. Graphs contain
  // these nodes cause analysis failures. However Control Flow v2 nodes
  // (While/If) will be supported.
  //
  // TODO(b/135628319): Support analyzing functional while/if as pass-through
  // ops.
  //
  // For example, consider following subgraph:
  //
  // TensorArrayOp -> Identity -> TensorArrayWriteOp
  //
  // It should be able to tell that TensorArrayWriteOp actually operates on the
  // resource created by TensorArrayOp even though there might be
  // non-resource-specific operations like Identity (or other pass-through
  // operations).
  //
  // source_to_path maps the nodes that creates resources to all nodes that
  // operate on the corresponding resource, not including sources themselves. It
  // is cleared upon calling this method.
  static Status Analyze(
      const Graph* graph, FunctionLibraryRuntime* lib_runtime,
      absl::flat_hash_map<NodeInfo, absl::flat_hash_set<NodeInfo>>*
          source_to_path);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_COMPILER_TF2XLA_RESOURCE_UTIL_H_
