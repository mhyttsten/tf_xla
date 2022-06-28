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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc() {
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

#include "tensorflow/core/profiler/convert/xplane_to_tf_data_stats.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/protobuf/tf_data_stats.pb.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/html_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

// 50 us from https://www.tensorflow.org/guide/data_performance_analysis
const int64_t kSlowCallThresholdPs = 50 * 1000000;

namespace {

// Returns true if the given iterator event is for a root iterator.
bool IsRootIteratorEvent(const XEventVisitor& iterator_event) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "IsRootIteratorEvent");

  std::vector<absl::string_view> split_result =
      absl::StrSplit(iterator_event.Name(), "::");
  // The root iterator's name contains only its own name (no parent
  // information).
  return split_result.size() == 2;
}

// Returns true if the given iterator event name is for an async iterator.
bool IsAsyncIterator(absl::string_view iterator_event_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("iterator_event_name: \"" + std::string(iterator_event_name.data(), iterator_event_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "IsAsyncIterator");

  static auto* kAsyncIterators = new absl::flat_hash_set<absl::string_view>(
      {"Prefetch", "ParallelInterleave", "ParallelMap", "ParseExample",
       "MapAndBatch", "DataService", "LegacyParallelInterleave",
       "ParallelBatch"});
  return kAsyncIterators->contains(iterator_event_name);
}

void SetIteratorMetadata(int64_t id, const XEventVisitor& event,
                         IteratorMetadata* metadata) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "SetIteratorMetadata");

  metadata->set_id(id);
  auto parent_id_stat = event.GetStat(StatType::kParentId);
  if (parent_id_stat.has_value()) {
    metadata->set_parent_id(parent_id_stat->IntValue());
  }
  metadata->set_name(IteratorName(event.Name()));
  metadata->set_long_name(event.Name().data(), event.Name().size());
  metadata->set_is_async(IsAsyncIterator(metadata->name()));
  // TODO(b/161831651): Set params.
}

// Returns the parent iterator's id if it is a root of a device input
// pipeline.
absl::optional<int64_t> FindDeviceInputPipeline(const XEventVisitor& event) {
  if (event.Type() == HostEventType::kDeviceInputPipelineSecondIterator) {
    auto parent_id_stat = event.GetStat(StatType::kParentId);
    if (parent_id_stat.has_value()) return parent_id_stat->IntValue();
  }
  return absl::nullopt;
}

// Processes EventForest to do the following:
// (1) set iterator metadata
// (2) find root iterator events
// (3) find device input pipeline ids
void ProcessEventForest(const EventForest& event_forest,
                        absl::flat_hash_set<int64_t>* device_input_pipeline_ids,
                        absl::flat_hash_map<int64_t, std::vector<EventNode*>>*
                            root_iterator_event_map,
                        TfDataStats* tf_data_stats) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_3(mht_3_v, 270, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "ProcessEventForest");

  const EventNodeMap& event_node_map = event_forest.GetEventNodeMap();
  auto iterator_event_list =
      gtl::FindOrNull(event_node_map, HostEventType::kIterator);
  if (!iterator_event_list) return;
  for (const auto& iterator_event : *iterator_event_list) {
    const XEventVisitor& iterator_event_visitor =
        iterator_event->GetEventVisitor();
    auto iterator_id_stat = iterator_event_visitor.GetStat(StatType::kStepId);
    if (!iterator_id_stat.has_value()) continue;
    int64_t iterator_id = iterator_id_stat->IntValue();
    auto result = tf_data_stats->mutable_iterator_metadata()->insert(
        {iterator_id, IteratorMetadata()});
    IteratorMetadata& metadata = result.first->second;
    if (result.second) {
      // First time processing this iterator.
      SetIteratorMetadata(iterator_id, iterator_event_visitor, &metadata);
    }
    if (IsRootIteratorEvent(iterator_event_visitor)) {
      // Record root iterator events.
      (*root_iterator_event_map)[iterator_id].push_back(iterator_event.get());
    }
  }
  auto device_input_pipeline_second_iterator_events = gtl::FindOrNull(
      event_node_map, HostEventType::kDeviceInputPipelineSecondIterator);
  if (!device_input_pipeline_second_iterator_events) return;
  for (const auto& iterator_event :
       *device_input_pipeline_second_iterator_events) {
    const XEventVisitor& iterator_event_visitor =
        iterator_event->GetEventVisitor();
    auto iterator_id_stat = iterator_event_visitor.GetStat(StatType::kStepId);
    if (!iterator_id_stat.has_value()) continue;
    int64_t iterator_id = iterator_id_stat->IntValue();
    auto result = tf_data_stats->mutable_iterator_metadata()->insert(
        {iterator_id, IteratorMetadata()});
    IteratorMetadata& metadata = result.first->second;
    if (result.second) {
      // First time processing this iterator.
      SetIteratorMetadata(iterator_id, iterator_event_visitor, &metadata);
      // Find and record device input pipeline ids.
      absl::optional<int64_t> device_input_pipeline_id =
          FindDeviceInputPipeline(iterator_event_visitor);
      if (device_input_pipeline_id.has_value()) {
        device_input_pipeline_ids->insert(*device_input_pipeline_id);
      }
    }
  }
}

void SetInputPipelineMetadata(int64_t id, int64_t name_id,
                              bool is_device_input_pipeline,
                              InputPipelineMetadata* metadata) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_4(mht_4_v, 324, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "SetInputPipelineMetadata");

  constexpr absl::string_view kHostInputPipelinePrefix = "Host:";
  constexpr absl::string_view kDeviceInputPipelinePrefix = "Device:";
  metadata->set_id(id);
  if (is_device_input_pipeline) {
    metadata->set_type(InputPipelineMetadata::DEVICE);
    metadata->set_name(absl::StrCat(kDeviceInputPipelinePrefix, name_id));
  } else {
    metadata->set_type(InputPipelineMetadata::HOST);
    metadata->set_name(absl::StrCat(kHostInputPipelinePrefix, name_id));
  }
}

void ProcessIteratorEvent(const EventNode& iterator_event,
                          InputPipelineStat* input_pipeline_stat,
                          bool is_blocking) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_5(mht_5_v, 342, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "ProcessIteratorEvent");

  const XEventVisitor& visitor = iterator_event.GetEventVisitor();
  auto iterator_id_stat = visitor.GetStat(StatType::kStepId);
  if (!iterator_id_stat.has_value()) return;
  int64_t iterator_id = iterator_id_stat->IntValue();
  auto result = input_pipeline_stat->mutable_iterator_stats()->insert(
      {iterator_id, IteratorStat()});
  IteratorStat& iterator_stat = result.first->second;
  if (result.second) {
    iterator_stat.set_id(iterator_id);
    iterator_stat.set_start_time_ps(visitor.TimestampPs());
  }
  iterator_stat.set_duration_ps(iterator_stat.duration_ps() +
                                visitor.DurationPs());
  int64_t self_time_ps = visitor.DurationPs();
  Timespan self_time_span = visitor.GetTimespan();
  for (EventNode* child : iterator_event.GetChildren()) {
    const XEventVisitor& child_visitor = child->GetEventVisitor();
    if (ParseTfOpFullname(child_visitor.Name()).category == Category::kTfData) {
      int64_t overlap_duration_ps =
          self_time_span.OverlappedDurationPs(child_visitor.GetTimespan());
      ProcessIteratorEvent(*child, input_pipeline_stat,
                           is_blocking && overlap_duration_ps);
      // Note: Assume no overlap between child events.
      self_time_ps -= overlap_duration_ps;
    }
  }
  iterator_stat.set_self_time_ps(iterator_stat.self_time_ps() + self_time_ps);
  iterator_stat.set_is_blocking(iterator_stat.is_blocking() || is_blocking);
  iterator_stat.set_num_calls(iterator_stat.num_calls() + 1);
}

void SetBottleneckIteratorId(InputPipelineStat* input_pipeline_stat) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_6(mht_6_v, 377, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "SetBottleneckIteratorId");

  int64_t bottleneck_iterator_id = 0;
  int64_t max_self_time = 0;
  for (const auto& pair : input_pipeline_stat->iterator_stats()) {
    const auto& id = pair.first;
    const auto& iterator_stat = pair.second;
    if (iterator_stat.is_blocking() &&
        iterator_stat.self_time_ps() > max_self_time) {
      bottleneck_iterator_id = id;
      max_self_time = iterator_stat.self_time_ps();
    }
  }
  input_pipeline_stat->set_bottleneck_iterator_id(bottleneck_iterator_id);
  input_pipeline_stat->set_bottleneck_iterator_latency_ps(max_self_time);
}

void ProcessInputPipelines(
    const absl::flat_hash_set<int64_t>& device_input_pipeline_ids,
    absl::flat_hash_map<int64_t, std::vector<EventNode*>>*
        root_iterator_event_map,
    TfDataStats* tf_data_stats) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_7(mht_7_v, 400, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "ProcessInputPipelines");

  auto* input_pipelines = tf_data_stats->mutable_input_pipelines();
  int64_t num_host_input_pipelines = 0;
  int64_t num_device_input_pipelines = 0;
  for (auto& id_and_events : *root_iterator_event_map) {
    auto& root_iterator_id = id_and_events.first;
    auto& root_iterator_events = id_and_events.second;
    absl::c_sort(root_iterator_events,
                 [](const EventNode* lhs, const EventNode* rhs) {
                   return lhs->GetEventVisitor().DurationPs() >
                          rhs->GetEventVisitor().DurationPs();
                 });
    auto result =
        input_pipelines->insert({root_iterator_id, InputPipelineStats()});
    InputPipelineStats& input_pipeline_stats = result.first->second;
    InputPipelineMetadata* metadata = input_pipeline_stats.mutable_metadata();
    if (result.second) {
      bool is_device_input_pipeline =
          device_input_pipeline_ids.contains(root_iterator_id);
      int64_t name_id = is_device_input_pipeline ? num_device_input_pipelines++
                                                 : num_host_input_pipelines++;
      SetInputPipelineMetadata(root_iterator_id, name_id,
                               is_device_input_pipeline, metadata);
    }
    int64_t sum_latency_ps = 0;
    int64_t min_latency_ps = INT64_MAX;
    int64_t max_latency_ps = 0;
    int64_t num_slow_calls = 0;
    for (const EventNode* root_iterator_event : root_iterator_events) {
      InputPipelineStat* stat = input_pipeline_stats.add_stats();
      ProcessIteratorEvent(*root_iterator_event, stat,
                           /*is_blocking*/ true);
      SetBottleneckIteratorId(stat);
      int64_t latency_ps = root_iterator_event->GetEventVisitor().DurationPs();
      sum_latency_ps += latency_ps;
      min_latency_ps = std::min(min_latency_ps, latency_ps);
      max_latency_ps = std::max(max_latency_ps, latency_ps);
      if (latency_ps > kSlowCallThresholdPs) num_slow_calls++;
    }
    input_pipeline_stats.set_avg_latency_ps(sum_latency_ps /
                                            root_iterator_events.size());
    input_pipeline_stats.set_min_latency_ps(min_latency_ps);
    input_pipeline_stats.set_max_latency_ps(max_latency_ps);
    input_pipeline_stats.set_num_slow_calls(num_slow_calls);
  }
}

void SetBottleneckAnalysis(CombinedTfDataStats* combined_tf_data_stats) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_8(mht_8_v, 450, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "SetBottleneckAnalysis");

  struct InputPipeline {
    InputPipeline(absl::string_view host_name,
                  absl::string_view input_pipeline_name, int64_t max_latency_ps,
                  absl::string_view iterator_name,
                  absl::string_view iterator_long_name,
                  int64_t iterator_latency_ps)
        : host_name(host_name),
          input_pipeline_name(input_pipeline_name),
          max_latency_ps(max_latency_ps),
          iterator_name(iterator_name),
          iterator_long_name(iterator_long_name),
          iterator_latency_ps(iterator_latency_ps) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("host_name: \"" + std::string(host_name.data(), host_name.size()) + "\"");
   mht_9_v.push_back("input_pipeline_name: \"" + std::string(input_pipeline_name.data(), input_pipeline_name.size()) + "\"");
   mht_9_v.push_back("iterator_name: \"" + std::string(iterator_name.data(), iterator_name.size()) + "\"");
   mht_9_v.push_back("iterator_long_name: \"" + std::string(iterator_long_name.data(), iterator_long_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_9(mht_9_v, 469, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "InputPipeline");
}
    absl::string_view host_name;
    absl::string_view input_pipeline_name;
    int64_t max_latency_ps;
    absl::string_view iterator_name;
    absl::string_view iterator_long_name;
    int64_t iterator_latency_ps;

    bool operator<(const InputPipeline& rhs) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_10(mht_10_v, 480, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "operator<");

      return max_latency_ps > rhs.max_latency_ps;
    }
  };
  std::vector<InputPipeline> slow_input_pipelines;
  for (const auto& host_name_and_tf_data_stats :
       combined_tf_data_stats->tf_data_stats()) {
    absl::string_view host_name = host_name_and_tf_data_stats.first;
    const TfDataStats& tf_data_stats = host_name_and_tf_data_stats.second;
    for (const auto& id_and_stats : tf_data_stats.input_pipelines()) {
      const InputPipelineStats& input_pipeline_stats = id_and_stats.second;
      if (input_pipeline_stats.metadata().type() ==
          InputPipelineMetadata::DEVICE) {
        // Ignore device input pipelines.
        continue;
      }
      // Choose the slowest execution trace of the input pipeline.
      // `input_pipeline_stats.stats` is already sorted so choose the first one.
      const InputPipelineStat& input_pipeline_stat =
          input_pipeline_stats.stats(0);
      const IteratorMetadata& metadata = tf_data_stats.iterator_metadata().at(
          input_pipeline_stat.bottleneck_iterator_id());
      slow_input_pipelines.emplace_back(
          host_name, input_pipeline_stats.metadata().name(),
          input_pipeline_stats.max_latency_ps(), metadata.name(),
          metadata.long_name(),
          input_pipeline_stat.bottleneck_iterator_latency_ps());
    }
  }
  std::sort(slow_input_pipelines.begin(), slow_input_pipelines.end());
  for (const auto& input_pipeline : slow_input_pipelines) {
    TfDataBottleneckAnalysis* bottleneck_analysis =
        combined_tf_data_stats->add_bottleneck_analysis();
    bottleneck_analysis->set_host(input_pipeline.host_name.data(),
                                  input_pipeline.host_name.size());
    bottleneck_analysis->set_input_pipeline(
        input_pipeline.input_pipeline_name.data(),
        input_pipeline.input_pipeline_name.size());
    bottleneck_analysis->set_max_latency_ps(input_pipeline.max_latency_ps);
    bottleneck_analysis->set_iterator_name(input_pipeline.iterator_name.data(),
                                           input_pipeline.iterator_name.size());
    bottleneck_analysis->set_iterator_long_name(
        input_pipeline.iterator_long_name.data(),
        input_pipeline.iterator_long_name.size());
    bottleneck_analysis->set_iterator_latency_ps(
        input_pipeline.iterator_latency_ps);
  }
}

std::string GetSuggestion(BottleneckType type) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_11(mht_11_v, 532, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "GetSuggestion");

  constexpr absl::string_view kPlaybookLink =
      "https://www.tensorflow.org/guide/data_performance_analysis";
  constexpr absl::string_view kPlaybookSourceDatasetLink =
      "https://www.tensorflow.org/guide/"
      "data_performance_analysis#source_datasets";
  constexpr absl::string_view kPlaybookCpuUtilizationLink =
      "https://www.tensorflow.org/guide/"
      "data_performance_analysis#3_are_you_reaching_high_cpu_utilization";
  constexpr absl::string_view kPlaybookTransformationLink =
      "https://www.tensorflow.org/guide/"
      "data_performance_analysis#transformation_datasets";
  constexpr absl::string_view kTfGuideParallelDataExtractionLink =
      "https://www.tensorflow.org/guide/"
      "data_performance#parallelizing_data_extraction";
  constexpr absl::string_view kTfGuideParallelTransformationLink =
      "https://www.tensorflow.org/guide/"
      "data_performance#parallelizing_data_transformation";
  constexpr absl::string_view kTfGuideCacheLink =
      "https://www.tensorflow.org/guide/data_performance#caching";
  constexpr absl::string_view kTfDataServiceLink =
      "https://www.tensorflow.org/api_docs/python/tf/data/experimental/"
      "service?version=nightly";
  switch (type) {
    case BottleneckType::kSlowSource:
      return absl::StrFormat(
          "1. Check the locality of a host and input data. Ideally, they "
          "should be in the same cell (or very close, like the same "
          "region).<br/>"
          "2. Parallelize reading from this dataset source. See %s and %s for "
          "more details.<br/>",
          AnchorElement(kPlaybookSourceDatasetLink, "here"),
          AnchorElement(kTfGuideParallelDataExtractionLink, "here"));
    case BottleneckType::kSlowDataService:
      return absl::StrFormat(
          "1. Fetching data from tf.data service took a while. Profile the "
          "tf.data service worker to analyze the issue further.<br/>"
          "2. See %s for more details on tf.data service.<br/>"
          "3. See %s for other suggestions.",
          AnchorElement(kTfDataServiceLink, "this"),
          AnchorElement(kPlaybookLink, "this"));
    case BottleneckType::kSlowRemoteSource:
      return absl::StrFormat(
          "1. The remote data source is slow. Profile its host to analyze the "
          "issue further.<br/>"
          "2. See %s for other suggestions.",
          AnchorElement(kPlaybookLink, "this"));
    case BottleneckType::kSlowTransformationWithParallelVersion:
      return absl::StrFormat(
          "1. Parallelize this transformation by setting "
          "<code>num_parallel_calls=tf.data.experimental.AUTOTUNE</code>. See "
          "%s for more details.<br/>"
          "2. Consider adding <code>cache</code> after this transformation if "
          "your data fits into memory and it is appropriate (e.g., there is no "
          "randomness in upstream transformations like <code>shuffle</code>). "
          "See %s for more details.<br/>"
          "3. Find more resources %s.",
          AnchorElement(kTfGuideParallelTransformationLink, "this"),
          AnchorElement(kTfGuideCacheLink, "this"),
          AnchorElement(kPlaybookTransformationLink, "here"));
    case BottleneckType::kSlowTransformationWithoutParallelVersion:
      return absl::StrFormat(
          "1. This transformation is inherently sequential. Add outer "
          "parallelism by running multiple copies of the input pipeline over "
          "sharded inputs and combining the results. See %s for more "
          "details.<br/>"
          "2. Consider adding <code>cache</code> after this transformation if "
          "your data fits into memory and it is appropriate (e.g., there is no "
          "randomness in upstream transformations like <code>shuffle</code>). "
          "See %s for more details.<br/>"
          "3. Find more resources %s.",
          AnchorElement(kPlaybookTransformationLink, "this"),
          AnchorElement(kTfGuideCacheLink, "this"),
          AnchorElement(kPlaybookCpuUtilizationLink, "here"));
    default:
      return absl::StrFormat("See %s for suggestions.",
                             AnchorElement(kPlaybookLink, "this"));
  }
}

void SetSuggestion(CombinedTfDataStats* combined_tf_data_stats) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_12(mht_12_v, 615, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "SetSuggestion");

  for (TfDataBottleneckAnalysis& bottleneck_analysis :
       *combined_tf_data_stats->mutable_bottleneck_analysis()) {
    bottleneck_analysis.set_suggestion(
        GetSuggestion(GetBottleneckType(bottleneck_analysis.iterator_name())));
  }
}

void SetSummary(CombinedTfDataStats* combined_tf_data_stats) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_13(mht_13_v, 626, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "SetSummary");

  int64_t max_latency_ps = 0;
  if (combined_tf_data_stats->bottleneck_analysis_size()) {
    max_latency_ps =
        combined_tf_data_stats->bottleneck_analysis().at(0).max_latency_ps();
  }
  if (max_latency_ps > kSlowCallThresholdPs) {
    combined_tf_data_stats->set_is_input_bound(true);
    combined_tf_data_stats->set_summary(
        "Your profile has a tf.data input pipeline slower than 50 us. For each "
        "slow input pipeline, below shows a bottleneck in the input pipeline "
        "and a suggestion on how to fix it.");
  } else if (max_latency_ps > 0) {
    combined_tf_data_stats->set_is_input_bound(false);
    combined_tf_data_stats->set_summary(
        "Your profile does not have any tf.data input pipeline slower than 50 "
        "us. Your job could be still input bound if this profile didn't "
        "capture all workers.");
  } else {
    combined_tf_data_stats->set_is_input_bound(false);
    combined_tf_data_stats->set_summary(
        "No tf.data activity captured in your profile. If your job uses "
        "tf.data, try to capture a longer profile.");
  }
}

}  // namespace

BottleneckType GetBottleneckType(absl::string_view bottleneck_iterator_name) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("bottleneck_iterator_name: \"" + std::string(bottleneck_iterator_name.data(), bottleneck_iterator_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_14(mht_14_v, 658, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "GetBottleneckType");

  static auto* kBottleneckTypeMap = new absl::flat_hash_map<absl::string_view,
                                                            BottleneckType>(
      {// Read from storage.
       {"TFRecord", BottleneckType::kSlowSource},
       {"SSTable", BottleneckType::kSlowSource},
       {"RecordIO", BottleneckType::kSlowSource},
       {"Spanner", BottleneckType::kSlowSource},
       {"TFColumn", BottleneckType::kSlowSource},
       {"SleepwalkRemoteDataset", BottleneckType::kSlowSource},
       {"TextLine", BottleneckType::kSlowSource},
       {"StitchedTimelineDataset", BottleneckType::kSlowSource},
       {"DateKeyDataset", BottleneckType::kSlowSource},
       {"CapacitorProto", BottleneckType::kSlowSource},
       {"LMDB", BottleneckType::kSlowSource},
       {"ExternalDataset", BottleneckType::kSlowSource},
       {"PearModel", BottleneckType::kSlowSource},
       {"FixedLengthRecordV2", BottleneckType::kSlowSource},
       // Read from local memory.
       {"FromTensor", BottleneckType::kSlowSource},
       {"TensorSlice", BottleneckType::kSlowSource},
       {"Generator", BottleneckType::kSlowSource},
       {"SyntheticDatasetOp", BottleneckType::kSlowSource},
       // tf.data service.
       {"DataService", BottleneckType::kSlowDataService},
       // Read from remote memory.
       {"GuzzlerDataGuzzlerRemoteDataset", BottleneckType::kSlowRemoteSource},
       {"ReverbDataset", BottleneckType::kSlowRemoteSource},
       {"DatasetSampleGame", BottleneckType::kSlowRemoteSource},
       {"Courier", BottleneckType::kSlowRemoteSource},
       {"ReverbEpisodeDataset", BottleneckType::kSlowRemoteSource},
       // Transformations with parallel version.
       {"Map", BottleneckType::kSlowTransformationWithParallelVersion},
       {"Interleave", BottleneckType::kSlowTransformationWithParallelVersion},
       // Transformations without parallel version.
       {"Filter", BottleneckType::kSlowTransformationWithoutParallelVersion},
       {"Batch", BottleneckType::kSlowTransformationWithoutParallelVersion},
       {"Unbatch", BottleneckType::kSlowTransformationWithoutParallelVersion}});
  if (auto type =
          gtl::FindOrNull(*kBottleneckTypeMap, bottleneck_iterator_name)) {
    return *type;
  }
  return BottleneckType::kOther;
}

void CombinedTfDataStatsBuilder::Add(absl::string_view host_name,
                                     XPlane* host_plane) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("host_name: \"" + std::string(host_name.data(), host_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_15(mht_15_v, 708, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "CombinedTfDataStatsBuilder::Add");

  TfDataStats& tf_data_stats =
      (*combined_tf_data_stats_
            ->mutable_tf_data_stats())[std::string(host_name)];
  EventForest event_forest;
  event_forest.AddPlanes(CreateTfXPlaneVisitor, {host_plane});
  event_forest.ConnectEvents();
  event_forest.ConnectTfDataEvents();
  absl::flat_hash_set<int64_t> device_input_pipeline_ids;
  absl::flat_hash_map<int64_t, std::vector<EventNode*>> root_iterator_event_map;
  ProcessEventForest(event_forest, &device_input_pipeline_ids,
                     &root_iterator_event_map, &tf_data_stats);
  ProcessInputPipelines(device_input_pipeline_ids, &root_iterator_event_map,
                        &tf_data_stats);
}

void CombinedTfDataStatsBuilder::Finalize() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_data_statsDTcc mht_16(mht_16_v, 727, "", "./tensorflow/core/profiler/convert/xplane_to_tf_data_stats.cc", "CombinedTfDataStatsBuilder::Finalize");

  SetBottleneckAnalysis(combined_tf_data_stats_);
  if (generate_suggestion_) SetSuggestion(combined_tf_data_stats_);
  SetSummary(combined_tf_data_stats_);
}

}  // namespace profiler
}  // namespace tensorflow
