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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc() {
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
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

XStatVisitor::XStatVisitor(const XPlaneVisitor* plane, const XStat* stat)
    : XStatVisitor(plane, stat, plane->GetStatMetadata(stat->metadata_id()),
                   plane->GetStatType(stat->metadata_id())) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XStatVisitor::XStatVisitor");
}

XStatVisitor::XStatVisitor(const XPlaneVisitor* plane, const XStat* stat,
                           const XStatMetadata* metadata,
                           absl::optional<int64_t> type)
    : stat_(stat), metadata_(metadata), plane_(plane), type_(type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XStatVisitor::XStatVisitor");
}

std::string XStatVisitor::ToString() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_2(mht_2_v, 215, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XStatVisitor::ToString");

  switch (stat_->value_case()) {
    case XStat::kInt64Value:
      return absl::StrCat(stat_->int64_value());
    case XStat::kUint64Value:
      return absl::StrCat(stat_->uint64_value());
    case XStat::kDoubleValue:
      return absl::StrCat(stat_->double_value());
    case XStat::kStrValue:
      return stat_->str_value();
    case XStat::kBytesValue:
      return "<opaque bytes>";
    case XStat::kRefValue:
      return plane_->GetStatMetadata(stat_->ref_value())->name();
    case XStat::VALUE_NOT_SET:
      return "";
  }
}

absl::string_view XStatVisitor::StrOrRefValue() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XStatVisitor::StrOrRefValue");

  switch (stat_->value_case()) {
    case XStat::kStrValue:
      return stat_->str_value();
    case XStat::kRefValue:
      return plane_->GetStatMetadata(stat_->ref_value())->name();
    case XStat::kInt64Value:
    case XStat::kUint64Value:
    case XStat::kDoubleValue:
    case XStat::kBytesValue:
    case XStat::VALUE_NOT_SET:
      return absl::string_view();
  }
}

XEventVisitor::XEventVisitor(const XPlaneVisitor* plane, const XLine* line,
                             const XEvent* event)
    : XStatsOwner<XEvent>(plane, event),
      plane_(plane),
      line_(line),
      event_(event),
      metadata_(plane->GetEventMetadata(event_->metadata_id())),
      type_(plane->GetEventType(event_->metadata_id())) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XEventVisitor::XEventVisitor");
}

XPlaneVisitor::XPlaneVisitor(const XPlane* plane,
                             const TypeGetterList& event_type_getter_list,
                             const TypeGetterList& stat_type_getter_list)
    : XStatsOwner<XPlane>(this, plane), plane_(plane) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_5(mht_5_v, 270, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XPlaneVisitor::XPlaneVisitor");

  BuildEventTypeMap(plane, event_type_getter_list);
  BuildStatTypeMap(plane, stat_type_getter_list);
}

void XPlaneVisitor::BuildEventTypeMap(
    const XPlane* plane, const TypeGetterList& event_type_getter_list) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_6(mht_6_v, 279, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XPlaneVisitor::BuildEventTypeMap");

  for (const auto& event_metadata : plane->event_metadata()) {
    uint64 metadata_id = event_metadata.first;
    const auto& metadata = event_metadata.second;
    for (const auto& event_type_getter : event_type_getter_list) {
      absl::optional<int64_t> event_type = event_type_getter(metadata.name());
      if (event_type.has_value()) {
        auto result = event_type_by_id_.emplace(metadata_id, *event_type);
        DCHECK(result.second);  // inserted
        break;
      }
    }
  }
}

const XEventMetadata* XPlaneVisitor::GetEventMetadata(
    int64_t event_metadata_id) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_7(mht_7_v, 298, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XPlaneVisitor::GetEventMetadata");

  const auto& event_metadata_by_id = plane_->event_metadata();
  const auto it = event_metadata_by_id.find(event_metadata_id);
  if (it != event_metadata_by_id.end()) return &it->second;
  return &XEventMetadata::default_instance();
}

absl::optional<int64_t> XPlaneVisitor::GetEventType(
    int64_t event_metadata_id) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_8(mht_8_v, 309, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XPlaneVisitor::GetEventType");

  const auto it = event_type_by_id_.find(event_metadata_id);
  if (it != event_type_by_id_.end()) return it->second;
  return absl::nullopt;
}

void XPlaneVisitor::BuildStatTypeMap(
    const XPlane* plane, const TypeGetterList& stat_type_getter_list) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_9(mht_9_v, 319, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XPlaneVisitor::BuildStatTypeMap");

  for (const auto& stat_metadata : plane->stat_metadata()) {
    uint64 metadata_id = stat_metadata.first;
    const auto& metadata = stat_metadata.second;
    for (const auto& stat_type_getter : stat_type_getter_list) {
      absl::optional<int64_t> stat_type = stat_type_getter(metadata.name());
      if (stat_type.has_value()) {
        auto result = stat_type_by_id_.emplace(metadata_id, *stat_type);
        DCHECK(result.second);  // inserted
        stat_metadata_by_type_.emplace(*stat_type, &metadata);
        break;
      }
    }
  }
}

const XStatMetadata* XPlaneVisitor::GetStatMetadata(
    int64_t stat_metadata_id) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_10(mht_10_v, 339, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XPlaneVisitor::GetStatMetadata");

  const auto& stat_metadata_by_id = plane_->stat_metadata();
  const auto it = stat_metadata_by_id.find(stat_metadata_id);
  if (it != stat_metadata_by_id.end()) return &it->second;
  return &XStatMetadata::default_instance();
}

absl::optional<int64_t> XPlaneVisitor::GetStatType(
    int64_t stat_metadata_id) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_11(mht_11_v, 350, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XPlaneVisitor::GetStatType");

  const auto it = stat_type_by_id_.find(stat_metadata_id);
  if (it != stat_type_by_id_.end()) return it->second;
  return absl::nullopt;
}

const XStatMetadata* XPlaneVisitor::GetStatMetadataByType(
    int64_t stat_type) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTcc mht_12(mht_12_v, 360, "", "./tensorflow/core/profiler/utils/xplane_visitor.cc", "XPlaneVisitor::GetStatMetadataByType");

  const auto it = stat_metadata_by_type_.find(stat_type);
  if (it != stat_metadata_by_type_.end()) return it->second;
  return nullptr;
}

}  // namespace profiler
}  // namespace tensorflow
