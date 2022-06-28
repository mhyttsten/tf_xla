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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_test_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_test_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_test_utilsDTcc() {
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
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

class XStatValueVisitor {
 public:
  XStatValueVisitor(XEventBuilder* event, const XStatMetadata* stat_metadata)
      : event_(event), stat_metadata_(stat_metadata) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_test_utilsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/profiler/utils/xplane_test_utils.cc", "XStatValueVisitor");
}

  template <typename T>
  void operator()(const T& value) {
    event_->AddStatValue(*stat_metadata_, value);
  }

 private:
  XEventBuilder* event_;
  const XStatMetadata* stat_metadata_;
};

}  // namespace

XPlane* GetOrCreateHostXPlane(XSpace* space) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_test_utilsDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/profiler/utils/xplane_test_utils.cc", "GetOrCreateHostXPlane");

  return FindOrAddMutablePlaneWithName(space, kHostThreadsPlaneName);
}

XPlane* GetOrCreateGpuXPlane(XSpace* space, int32_t device_ordinal) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_test_utilsDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/profiler/utils/xplane_test_utils.cc", "GetOrCreateGpuXPlane");

  std::string name = GpuPlaneName(device_ordinal);
  return FindOrAddMutablePlaneWithName(space, name);
}

void CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    absl::string_view event_name, int64_t offset_ps, int64_t duration_ps,
    std::initializer_list<std::pair<StatType, XStatValue>> stats) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("event_name: \"" + std::string(event_name.data(), event_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_test_utilsDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/profiler/utils/xplane_test_utils.cc", "CreateXEvent");

  auto event_builder = line_builder->AddEvent(
      *plane_builder->GetOrCreateEventMetadata(event_name));
  event_builder.SetOffsetPs(offset_ps);
  event_builder.SetDurationPs(duration_ps);
  for (const auto& stat_type_and_value : stats) {
    StatType stat_type = stat_type_and_value.first;
    const XStatValue& stat_value = stat_type_and_value.second;
    XStatValueVisitor stat_value_visitor(
        &event_builder,
        plane_builder->GetOrCreateStatMetadata(GetStatTypeStr(stat_type)));
    absl::visit(stat_value_visitor, stat_value);
  }
}

void CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    HostEventType event_type, int64_t offset_ps, int64_t duration_ps,
    std::initializer_list<std::pair<StatType, XStatValue>> stats) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_test_utilsDTcc mht_4(mht_4_v, 260, "", "./tensorflow/core/profiler/utils/xplane_test_utils.cc", "CreateXEvent");

  CreateXEvent(plane_builder, line_builder, GetHostEventTypeStr(event_type),
               offset_ps, duration_ps, stats);
}

void CreateTfFunctionCallEvent(XPlaneBuilder* plane_builder,
                               XLineBuilder* line_builder,
                               absl::string_view function_name,
                               int64_t offset_ps, int64_t duration_ps,
                               absl::string_view execution_mode,
                               int64_t tracing_count) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("function_name: \"" + std::string(function_name.data(), function_name.size()) + "\"");
   mht_5_v.push_back("execution_mode: \"" + std::string(execution_mode.data(), execution_mode.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_test_utilsDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/profiler/utils/xplane_test_utils.cc", "CreateTfFunctionCallEvent");

  if (tracing_count >= 0) {
    // Adds the tracing_count stats only if tracing_count is valid.
    CreateXEvent(plane_builder, line_builder, function_name, offset_ps,
                 duration_ps,
                 {{StatType::kTfFunctionCall, execution_mode},
                  {StatType::kTfFunctionTracingCount, tracing_count}});
  } else {
    CreateXEvent(plane_builder, line_builder, function_name, offset_ps,
                 duration_ps, {{StatType::kTfFunctionCall, execution_mode}});
  }
}

}  // namespace profiler
}  // namespace tensorflow
