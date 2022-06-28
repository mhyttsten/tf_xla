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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_TRAVERSAL_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_TRAVERSAL_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTh() {
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

#include "tensorflow/core/grappler/graph_topology_view.h"

namespace tensorflow {
namespace grappler {

enum class TraversalDirection { kFollowInputs, kFollowOutputs };

// Encapsulate DFS callbacks that will be called during the graph traversal.
//
// If non-empty, the `pre_order` and `post_order` functors will be called on
// each reachable node (including the `from` nodes) in pre and post order. If
// loops are found, the `on_back_edge` functor will be called on the
// corresponding back edges. Moreover, the pre and post order will assume that
// these back edges will be cut.
struct DfsCallbacks {
  DfsCallbacks() = default;
  DfsCallbacks(std::function<void(const NodeDef*)> pre,
               std::function<void(const NodeDef*)> post,
               std::function<void(const NodeDef*, const NodeDef*)> back_edge)
      : pre_order(std::move(pre)),
        post_order(std::move(post)),
        on_back_edge(std::move(back_edge)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTh mht_0(mht_0_v, 211, "", "./tensorflow/core/grappler/utils/traversal.h", "DfsCallbacks");
}

  static DfsCallbacks PreOrder(std::function<void(const NodeDef*)> pre) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTh mht_1(mht_1_v, 216, "", "./tensorflow/core/grappler/utils/traversal.h", "PreOrder");

    return DfsCallbacks(std::move(pre), nullptr, nullptr);
  }

  static DfsCallbacks PostOrder(std::function<void(const NodeDef*)> post) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTh mht_2(mht_2_v, 223, "", "./tensorflow/core/grappler/utils/traversal.h", "PostOrder");

    return DfsCallbacks(nullptr, std::move(post), nullptr);
  }

  std::function<void(const NodeDef*)> pre_order;
  std::function<void(const NodeDef*)> post_order;
  std::function<void(const NodeDef*, const NodeDef*)> on_back_edge;
};

// Encapsulate DFS predicates for traversing the graph.
//
// The `enter` predicate decides if traversal should enter the node, and the
// `advance` predicate decides if the traversal should follow inputs/outputs
// from the node.
//
// If predicates are empty (default initialized), it's assumed that we can enter
// into any node and advance from any node respectively.
struct DfsPredicates {
  DfsPredicates() = default;
  DfsPredicates(std::function<bool(const NodeDef*)> enter,
                std::function<bool(const NodeDef*)> advance)
      : enter(std::move(enter)), advance(std::move(advance)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTh mht_3(mht_3_v, 247, "", "./tensorflow/core/grappler/utils/traversal.h", "DfsPredicates");
}

  static DfsPredicates Enter(std::function<bool(const NodeDef*)> enter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTh mht_4(mht_4_v, 252, "", "./tensorflow/core/grappler/utils/traversal.h", "Enter");

    return DfsPredicates(std::move(enter), nullptr);
  }

  static DfsPredicates Advance(std::function<bool(const NodeDef*)> advance) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTh mht_5(mht_5_v, 259, "", "./tensorflow/core/grappler/utils/traversal.h", "Advance");

    return DfsPredicates(nullptr, std::move(advance));
  }

  std::function<bool(const NodeDef*)> enter;
  std::function<bool(const NodeDef*)> advance;
};

// Traverse the graph in DFS order in the given direction, starting from the
// list of nodes specified in the `from` argument. Use `predicates` to decide if
// traversal should enter/advance to/from the graph node. These predicates also
// applied to the `from` nodes. Call corresponding callbacks for each visited
// node.
void DfsTraversal(const GraphTopologyView& graph_view,
                  absl::Span<const NodeDef* const> from,
                  TraversalDirection direction, const DfsPredicates& predicates,
                  const DfsCallbacks& callbacks);

// Traverse the graph in DFS order in the given direction, starting from the
// list of nodes specified in the `from` argument. Call corresponding callbacks
// for each visited node.
void DfsTraversal(const GraphTopologyView& graph_view,
                  absl::Span<const NodeDef* const> from,
                  TraversalDirection direction, const DfsCallbacks& callbacks);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_TRAVERSAL_H_
