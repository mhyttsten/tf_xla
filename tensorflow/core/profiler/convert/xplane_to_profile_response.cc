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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_profile_responseDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_profile_responseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_profile_responseDTcc() {
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
#include "tensorflow/core/profiler/convert/xplane_to_profile_response.h"

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"
#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"
#include "tensorflow/core/profiler/convert/trace_events_to_json.h"
#include "tensorflow/core/profiler/convert/xplane_to_memory_profile.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kTraceViewer = "trace_viewer";
const absl::string_view kTensorflowStats = "tensorflow_stats";
const absl::string_view kInputPipeline = "input_pipeline";
const absl::string_view kOverviewPage = "overview_page";
const absl::string_view kKernelStats = "kernel_stats";
const absl::string_view kMemoryProfile = "memory_profile";
const absl::string_view kXPlanePb = "xplane.pb";

template <typename Proto>
void AddToolData(absl::string_view tool_name, const Proto& tool_output,
                 ProfileResponse* response) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("tool_name: \"" + std::string(tool_name.data(), tool_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_profile_responseDTcc mht_0(mht_0_v, 227, "", "./tensorflow/core/profiler/convert/xplane_to_profile_response.cc", "AddToolData");

  auto* tool_data = response->add_tool_data();
  tool_data->set_name(string(tool_name));
  tool_output.SerializeToString(tool_data->mutable_data());
}

// Returns the tool name with extension.
std::string ToolName(absl::string_view tool) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tool: \"" + std::string(tool.data(), tool.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_profile_responseDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/profiler/convert/xplane_to_profile_response.cc", "ToolName");

  if (tool == kTraceViewer) return "trace.json.gz";
  if (tool == kMemoryProfile) return "memory_profile.json.gz";
  return absl::StrCat(tool, ".pb");
}

}  // namespace

Status ConvertXSpaceToProfileResponse(const XSpace& xspace,
                                      const ProfileRequest& req,
                                      ProfileResponse* response) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_profile_responseDTcc mht_2(mht_2_v, 251, "", "./tensorflow/core/profiler/convert/xplane_to_profile_response.cc", "ConvertXSpaceToProfileResponse");

  absl::flat_hash_set<absl::string_view> tools(req.tools().begin(),
                                               req.tools().end());
  if (tools.empty()) return Status::OK();
  if (tools.contains(kXPlanePb)) {
    AddToolData(kXPlanePb, xspace, response);
  }
  if (tools.contains(kTraceViewer)) {
    Trace trace;
    ConvertXSpaceToTraceEvents(xspace, &trace);
    if (trace.trace_events().empty()) {
      response->set_empty_trace(true);
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(SaveGzippedToolData(
        req.repository_root(), req.session_id(), req.host_name(),
        ToolName(kTraceViewer), TraceEventsToJson(trace)));
    // Trace viewer is the only tool, skip OpStats conversion.
    if (tools.size() == 1) return Status::OK();
  }

  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  options.maybe_drop_incomplete_steps = true;
  OpStats op_stats = ConvertXSpaceToOpStats(xspace, options);
  if (tools.contains(kOverviewPage)) {
    OverviewPage overview_page_db = ConvertOpStatsToOverviewPage(op_stats);
    AddToolData(ToolName(kOverviewPage), overview_page_db, response);
    if (tools.contains(kInputPipeline)) {
      AddToolData(ToolName(kInputPipeline), overview_page_db.input_analysis(),
                  response);
    }
  } else if (tools.contains(kInputPipeline)) {
    AddToolData(ToolName(kInputPipeline),
                ConvertOpStatsToInputPipelineAnalysis(op_stats), response);
  }
  if (tools.contains(kTensorflowStats)) {
    TfStatsDatabase tf_stats_db = ConvertOpStatsToTfStats(op_stats);
    AddToolData(ToolName(kTensorflowStats), tf_stats_db, response);
  }
  if (tools.contains(kKernelStats)) {
    AddToolData(ToolName(kKernelStats), op_stats.kernel_stats_db(), response);
  }
  if (tools.contains(kMemoryProfile)) {
    std::string json_output;
    TF_RETURN_IF_ERROR(ConvertXSpaceToMemoryProfileJson(xspace, &json_output));
    TF_RETURN_IF_ERROR(SaveGzippedToolData(
        req.repository_root(), req.session_id(), req.host_name(),
        ToolName(kMemoryProfile), json_output));
  }
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
