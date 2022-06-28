/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_GROUP_EVENTS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_GROUP_EVENTS_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh() {
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


#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

// Information required to connect events across threads. The first two fields
// specify the event types of parent and child events. In addition to matching
// the event types, both events should have stats of the stat types specified
// in stat_types and their values should be the same.
struct InterThreadConnectInfo {
  int64_t parent_event_type;
  int64_t child_event_type;
  std::vector<int64_t> parent_stat_types;
  std::vector<int64_t> child_stat_types;
};

struct ContextInfo {
  ContextInfo(int type, uint64 id) : type(type), id(id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_0(mht_0_v, 218, "", "./tensorflow/core/profiler/utils/group_events.h", "ContextInfo");
}
  int type;
  uint64 id;
};

struct GroupMetadata {
  std::string name;
  std::string model_id;  // inference only.
  absl::flat_hash_set<int64_t> parents;
  absl::flat_hash_set<int64_t> children;
};

using GroupMetadataMap =
    absl::flat_hash_map<int64_t /*group_id*/, GroupMetadata>;

// A wrapper for XEvent with parent and children pointers. Through these
// pointers, a tree of EventNode is formed.
class EventNode {
 public:
  // REQUIRED: all inputs should not be nullptr.
  EventNode(const XPlaneVisitor* plane, XLine* raw_line, XEvent* raw_event);

  EventNode(const EventNode& event_node);

  const std::vector<EventNode*>& GetParents() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_1(mht_1_v, 245, "", "./tensorflow/core/profiler/utils/group_events.h", "GetParents");
 return parents_; }

  const std::vector<EventNode*>& GetChildren() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_2(mht_2_v, 250, "", "./tensorflow/core/profiler/utils/group_events.h", "GetChildren");
 return children_; }

  void AddChild(EventNode* child) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_3(mht_3_v, 255, "", "./tensorflow/core/profiler/utils/group_events.h", "AddChild");

    children_.push_back(child);
    child->parents_.push_back(this);
  }

  absl::optional<int64_t> GetGroupId() const { return group_id_; }

  std::string GetGroupName() const;

  void SetGroupId(int64_t group_id);

  // Sets group_id for this node and its descendants.
  void PropagateGroupId(int64_t group_id, GroupMetadataMap* group_metadata_map);

  const XPlaneVisitor& GetPlaneVisitor() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_4(mht_4_v, 272, "", "./tensorflow/core/profiler/utils/group_events.h", "GetPlaneVisitor");
 return *plane_; }

  const XEventVisitor& GetEventVisitor() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_5(mht_5_v, 277, "", "./tensorflow/core/profiler/utils/group_events.h", "GetEventVisitor");
 return visitor_; }

  absl::optional<XStatVisitor> GetContextStat(int64_t stat_type) const;

  void AddStepName(absl::string_view step_name);

  // Add a helper stat, "selected_group_ids", with group_ids of the groups
  // connected to this event's group.
  void AddSelectedGroupIds(const GroupMetadataMap& group_metadata_map);

  void SetIsEager(bool is_eager);

  // Returns true if this event is part of eagerly executed op.
  bool IsEager();

  bool IsNestedIn(EventNode* parent);

  // Returns the closest parent (including itself) of the given event type.
  const EventNode* FindParent(int64_t event_type) const;

  absl::optional<ContextInfo> GetProducerContext() const {
    return producer_context_;
  }

  absl::optional<ContextInfo> GetConsumerContext() const {
    return consumer_context_;
  }

  void SetRootLevel(int root_level) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_6(mht_6_v, 308, "", "./tensorflow/core/profiler/utils/group_events.h", "SetRootLevel");
 root_level_ = root_level; }

  int RootLevel() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_7(mht_7_v, 313, "", "./tensorflow/core/profiler/utils/group_events.h", "RootLevel");
 return root_level_; }

  bool IsAsync() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_8(mht_8_v, 318, "", "./tensorflow/core/profiler/utils/group_events.h", "IsAsync");
 return is_async_; }
  bool IsCompiledFunc() const;

  // Compare two EventNodes based on start timestamp.
  bool operator<(const EventNode& other) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_9(mht_9_v, 325, "", "./tensorflow/core/profiler/utils/group_events.h", "operator<");

    return GetEventVisitor().TimestampPs() <
           other.GetEventVisitor().TimestampPs();
  }

 private:
  XStat* FindOrAddStatByType(int64_t stat_type);

  const XPlaneVisitor* plane_;
  XEventVisitor visitor_;
  XLine* raw_line_;
  XEvent* raw_event_;
  std::vector<EventNode*> parents_;
  std::vector<EventNode*> children_;
  absl::optional<int64_t> group_id_;
  absl::optional<ContextInfo> producer_context_;
  absl::optional<ContextInfo> consumer_context_;
  // Root event level.
  // By default root_level_ is set to 0, which means it is not a root event.
  // Events with root_level_ greater than 0 are considered as root events.
  int root_level_ = 0;
  bool is_async_ = false;
};

using EventNodeMap =
    absl::flat_hash_map<int64_t /*event_type*/,
                        std::vector<std::unique_ptr<EventNode>>>;

using EventList = std::vector<EventNode*>;

struct ContextGroup {
  std::vector<EventNode*> producers;
  std::vector<EventNode*> consumers;
};

using ContextGroupMap = absl::flat_hash_map<
    int /*context_type*/,
    absl::flat_hash_map<uint64 /*context_id*/, ContextGroup>>;

// EventForest augments the input XSpace with the trace context. The trace
// context is created by stitching XEvents (1) using the nesting relationship
// within the same thread and (2) comparing the semantic arguments or using
// connect_info_list across threads. It also groups the events by the root
// events specified in root_event_types or marked by the semantic argument.
class EventForest {
 public:
  void AddSpace(
      const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
      XSpace* space);

  void AddPlanes(
      const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
      const std::vector<XPlane*>& planes);

  void ConnectEvents(
      const std::vector<InterThreadConnectInfo>& connect_info_list = {});

  void ConnectTfDataEvents();

  void GroupEvents();

  const EventNodeMap& GetEventNodeMap() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_10(mht_10_v, 389, "", "./tensorflow/core/profiler/utils/group_events.h", "GetEventNodeMap");
 return event_node_map_; }

  const GroupMetadataMap& GetGroupMetadataMap() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSgroup_eventsDTh mht_11(mht_11_v, 394, "", "./tensorflow/core/profiler/utils/group_events.h", "GetGroupMetadataMap");

    return group_metadata_map_;
  }

 private:
  void AddPlane(
      const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
      XPlane* plane);

  // Creates an EventNode for each event in event_node_map and connect events
  // according to the nesting relationship within the thread.
  void ConnectIntraThread(XPlane* plane, XPlaneVisitor* visitor,
                          ContextGroupMap* context_groups);

  // Connects events across threads according to connect_info_list.
  void ConnectInterThread(
      const std::vector<InterThreadConnectInfo>& connect_info_list);

  // Creates event groups and populates group_metadata_map. If a TF loop is
  // used, each TF loop iteration becomes a root. Otherwise, top root events
  // (i.e., none of their ancestors is a root event) are used as roots. A new
  // group is created with all events reachable from a root.
  void CreateEventGroups();

  // Sets the is_eager stat to true for the eagerly executed GPU kernel events.
  void MarkEagerlyExecutedGpuKernels();

  // Sets the is_eager stat to true for the eagerly executed CPU TF op events.
  void MarkEagerlyExecutedCpuTfOps();

  // Populate all the step ids that associated with tf.data pipeline.
  // Because FunctionRun is considered as root, but we want to exclude those
  // FunctionRuns from tf.data.
  void ProcessTfDataSteps();

  // Processes the TF loops and registers the first TF executor event of each
  // iteraton to `tf_loop_root_events_`.
  void ProcessTensorFlowLoop();

  // Processes the worker thread by connecting a FunctionRun with the following
  // eager ops (e.g., for Keras callback).
  void ProcessWorker();

  // Adds model ids to group_metadata_map for inference profiles.
  void ProcessModelIds();

  EventNodeMap event_node_map_;
  std::vector<XPlaneVisitor> visitors_;
  // std::deque for pointer stability.
  std::deque<std::pair<XPlane*, XPlaneVisitor>> planes_;
  // The "step" id (actually it is "function" id that are associated with
  // the tf.data pipeline.
  absl::flat_hash_set<int64_t> tf_data_step_ids_;
  EventList tf_loop_root_events_;
  GroupMetadataMap group_metadata_map_;
};

std::vector<InterThreadConnectInfo> CreateInterThreadConnectInfoList();

// Calls GroupEvents with connect_info_list and root_event_types specific to
// TensorFlow.
void GroupTfEvents(XSpace* space, EventForest* event_forest);
void GroupTfEvents(XSpace* space);

// Returns true if the given space has TF's loop ops.
bool CheckLoopOp(const XSpace& space);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_GROUP_EVENTS_H_
