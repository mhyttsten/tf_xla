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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframeDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframeDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/frame.h"

#include <deque>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {

namespace {}  // namespace

template <typename GraphViewT>
inline Status FrameView::InferFromGraphViewT(const GraphViewT& graph_view) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframeDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/grappler/utils/frame.cc", "FrameView::InferFromGraphViewT");

  if (is_inferred_) {
    return errors::Internal("FrameView was already inferred from the graph");
  }
  is_inferred_ = true;

  std::deque<int> ready_node_indices;

  // All nodes without inputs are automatically added to the ready queue.
  for (const auto& node : graph_view.GetNodes()) {
    if (node.NumRegularFanins() + node.NumControllingFanins() == 0) {
      ready_node_indices.push_back(node.node_index());
      node_to_frames_[node.node()] = node_has_no_frames_;
    }
  }

  const auto* graph = graph_view.graph();

  // We assign unique int id to each frame, and use this map to track what
  // frames we've already seen in the graph.
  absl::flat_hash_map<string, int> frame_name_to_id;

  auto process_fanout = [this, graph](
                            absl::flat_hash_map<string, int>* frame_name_to_id,
                            std::deque<int>* ready_node_indices,
                            const NodeDef* ready_node, int fanout_node_index) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframeDTcc mht_1(mht_1_v, 228, "", "./tensorflow/core/grappler/utils/frame.cc", "lambda");

    const NodeDef* fanout_node = &graph->node(fanout_node_index);
    if (!node_to_frames_.contains(fanout_node)) {
      // If we have never seen this node before, we add all frames from the
      // incoming node (and pop/push frames if coming from Exit/Enter nodes).
      std::vector<int> frame_ids = node_to_frames_[ready_node];

      if (IsExit(*ready_node)) {
        frame_ids.pop_back();
      }

      if (IsEnter(*fanout_node)) {
        const AttrValue* frame_name_attr =
            AttrSlice(*fanout_node).Find("frame_name");

        if (!frame_name_attr) {
          return errors::InvalidArgument(
              "Missing frame name for the Enter node: ",
              SummarizeNodeDef(*fanout_node));
        }

        const string& frame_name = frame_name_attr->s();
        int frame_id;

        if (frame_name_to_id->contains(frame_name)) {
          frame_id = (*frame_name_to_id)[frame_name];
        } else {
          frame_id = static_cast<int>(frame_name_to_id->size());
          (*frame_name_to_id)[frame_name] = frame_id;
        }

        frame_ids.push_back(frame_id);
      }

      ready_node_indices->push_back(fanout_node_index);
      node_to_frames_[fanout_node] = std::move(frame_ids);

    } else {
      // If we've already seen this node before, we need to make sure that graph
      // is correct and same nodes doesn't have incoming edges with conflicting
      // frames (all inputs must be produces in the same frame).

      std::vector<int> frame_ids_fanout = node_to_frames_[fanout_node];
      std::vector<int> frame_ids_node = node_to_frames_[ready_node];

      if (IsEnter(*fanout_node)) {
        frame_ids_fanout.pop_back();
      }
      if (IsExit(*ready_node)) {
        frame_ids_node.pop_back();
      }

      if (frame_ids_node != frame_ids_fanout) {
        return errors::InvalidArgument(
            "Invalid graph: Frame ids for node ", ready_node->name(),
            " does not match frame ids for it's fanout ", fanout_node->name());
      }
    }
    return Status::OK();
  };

  while (!ready_node_indices.empty()) {
    const int ready_node_index = ready_node_indices.front();
    ready_node_indices.pop_front();
    const auto* ready_node_view = graph_view.GetNode(ready_node_index);
    const NodeDef* ready_node_def = ready_node_view->node();

    for (const auto& regular_fanouts_port_i :
         ready_node_view->GetRegularFanouts()) {
      for (const auto& regular_fanout : regular_fanouts_port_i) {
        TF_RETURN_IF_ERROR(process_fanout(&frame_name_to_id,
                                          &ready_node_indices, ready_node_def,
                                          regular_fanout.node_index()));
      }
    }

    for (const auto& controlled_fanout :
         ready_node_view->GetControlledFanouts()) {
      TF_RETURN_IF_ERROR(process_fanout(&frame_name_to_id, &ready_node_indices,
                                        ready_node_def,
                                        controlled_fanout.node_index()));
    }
  }

  num_frames_ = static_cast<int>(frame_name_to_id.size());
  return Status::OK();
}

Status FrameView::InferFromGraphView(const utils::GraphView& graph_view) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframeDTcc mht_2(mht_2_v, 319, "", "./tensorflow/core/grappler/utils/frame.cc", "FrameView::InferFromGraphView");

  return InferFromGraphViewT(graph_view);
}

Status FrameView::InferFromGraphView(
    const utils::MutableGraphView& graph_view) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframeDTcc mht_3(mht_3_v, 327, "", "./tensorflow/core/grappler/utils/frame.cc", "FrameView::InferFromGraphView");

  return InferFromGraphViewT(graph_view);
}

Status FrameView::InferFromGraph(const GraphDef& graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframeDTcc mht_4(mht_4_v, 334, "", "./tensorflow/core/grappler/utils/frame.cc", "FrameView::InferFromGraph");

  Status status;
  utils::GraphView graph_view(&graph, &status);
  TF_RETURN_IF_ERROR(status);
  return InferFromGraphViewT(graph_view);
}

const std::vector<int>& FrameView::Frames(const NodeDef& node) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframeDTcc mht_5(mht_5_v, 344, "", "./tensorflow/core/grappler/utils/frame.cc", "FrameView::Frames");

  DCHECK(is_inferred_) << "FrameView is not initialized";
  auto frames = node_to_frames_.find(&node);
  if (frames == node_to_frames_.end()) {
    LOG(WARNING) << "Node '" << node.name()
                 << "' doesn't belong to the graph used for initialization";
    return node_has_no_frames_;
  } else {
    return frames->second;
  }
}

bool FrameView::IsInFrame(const NodeDef& node) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframeDTcc mht_6(mht_6_v, 359, "", "./tensorflow/core/grappler/utils/frame.cc", "FrameView::IsInFrame");

  return !Frames(node).empty();
}

}  // namespace grappler
}  // namespace tensorflow
