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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc() {
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
#include "tensorflow/core/profiler/utils/xplane_builder.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {

XPlaneBuilder::XPlaneBuilder(XPlane* plane)
    : XStatsBuilder<XPlane>(plane, this), plane_(plane) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::XPlaneBuilder");

  for (auto& id_and_metadata : *plane->mutable_event_metadata()) {
    auto& metadata = id_and_metadata.second;
    last_event_metadata_id_ =
        std::max<int64_t>(last_event_metadata_id_, metadata.id());
    if (!metadata.name().empty()) {
      event_metadata_by_name_.try_emplace(metadata.name(), &metadata);
    }
  }
  for (auto& id_and_metadata : *plane->mutable_stat_metadata()) {
    auto& metadata = id_and_metadata.second;
    last_stat_metadata_id_ =
        std::max<int64_t>(last_stat_metadata_id_, metadata.id());
    if (!metadata.name().empty()) {
      stat_metadata_by_name_.try_emplace(metadata.name(), &metadata);
    }
  }
  for (XLine& line : *plane->mutable_lines()) {
    lines_by_id_.try_emplace(line.id(), &line);
  }
}

XEventMetadata* XPlaneBuilder::GetOrCreateEventMetadata(int64_t metadata_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::GetOrCreateEventMetadata");

  XEventMetadata& metadata = (*plane_->mutable_event_metadata())[metadata_id];
  metadata.set_id(metadata_id);
  return &metadata;
}

XEventMetadata* XPlaneBuilder::CreateEventMetadata() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_2(mht_2_v, 235, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::CreateEventMetadata");

  return GetOrCreateEventMetadata(++last_event_metadata_id_);
}

XEventMetadata* XPlaneBuilder::GetOrCreateEventMetadata(
    absl::string_view name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::GetOrCreateEventMetadata");

  XEventMetadata*& metadata = event_metadata_by_name_[name];
  if (metadata == nullptr) {
    metadata = CreateEventMetadata();
    metadata->set_name(std::string(name));
  }
  return metadata;
}

XEventMetadata* XPlaneBuilder::GetOrCreateEventMetadata(std::string&& name) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_4(mht_4_v, 256, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::GetOrCreateEventMetadata");

  XEventMetadata*& metadata = event_metadata_by_name_[name];
  if (metadata == nullptr) {
    metadata = CreateEventMetadata();
    metadata->set_name(std::move(name));
  }
  return metadata;
}

XEventMetadata* XPlaneBuilder::GetEventMetadata(absl::string_view name) const {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_5(mht_5_v, 269, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::GetEventMetadata");

  auto result = event_metadata_by_name_.find(name);
  if (result == event_metadata_by_name_.end()) return nullptr;
  return result->second;
}

XStatMetadata* XPlaneBuilder::GetStatMetadata(absl::string_view name) const {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_6(mht_6_v, 279, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::GetStatMetadata");

  auto result = stat_metadata_by_name_.find(name);
  if (result == stat_metadata_by_name_.end()) return nullptr;
  return result->second;
}

XStatMetadata* XPlaneBuilder::GetOrCreateStatMetadata(int64_t metadata_id) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_7(mht_7_v, 288, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::GetOrCreateStatMetadata");

  XStatMetadata& metadata = (*plane_->mutable_stat_metadata())[metadata_id];
  metadata.set_id(metadata_id);
  return &metadata;
}

const XStatMetadata* XPlaneBuilder::GetStatMetadata(int64_t metadata_id) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_8(mht_8_v, 297, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::GetStatMetadata");

  auto result = plane_->stat_metadata().find(metadata_id);
  if (result == plane_->stat_metadata().end()) return nullptr;
  return &(result->second);
}

XStatMetadata* XPlaneBuilder::CreateStatMetadata() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_9(mht_9_v, 306, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::CreateStatMetadata");

  return GetOrCreateStatMetadata(++last_stat_metadata_id_);
}

XStatMetadata* XPlaneBuilder::GetOrCreateStatMetadata(absl::string_view name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_10(mht_10_v, 314, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::GetOrCreateStatMetadata");

  XStatMetadata*& metadata = stat_metadata_by_name_[name];
  if (metadata == nullptr) {
    metadata = CreateStatMetadata();
    metadata->set_name(std::string(name));
  }
  return metadata;
}

XStatMetadata* XPlaneBuilder::GetOrCreateStatMetadata(std::string&& name) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_11(mht_11_v, 326, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::GetOrCreateStatMetadata");

  XStatMetadata*& metadata = stat_metadata_by_name_[name];
  if (metadata == nullptr) {
    metadata = CreateStatMetadata();
    metadata->set_name(std::move(name));
  }
  return metadata;
}

XLineBuilder XPlaneBuilder::GetOrCreateLine(int64_t line_id) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_12(mht_12_v, 338, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XPlaneBuilder::GetOrCreateLine");

  XLine*& line = lines_by_id_[line_id];
  if (line == nullptr) {
    line = plane_->add_lines();
    line->set_id(line_id);
  }
  return XLineBuilder(line, this);
}

XEventBuilder XLineBuilder::AddEvent(const XEventMetadata& metadata) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_13(mht_13_v, 350, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XLineBuilder::AddEvent");

  XEvent* event = line_->add_events();
  event->set_metadata_id(metadata.id());
  return XEventBuilder(line_, plane_, event);
}

XEventBuilder XLineBuilder::AddEvent(const XEvent& event) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_14(mht_14_v, 359, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XLineBuilder::AddEvent");

  XEvent* new_event = line_->add_events();
  *new_event = event;
  return XEventBuilder(line_, plane_, new_event);
}

void XLineBuilder::SetTimestampNsAndAdjustEventOffsets(int64_t timestamp_ns) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTcc mht_15(mht_15_v, 368, "", "./tensorflow/core/profiler/utils/xplane_builder.cc", "XLineBuilder::SetTimestampNsAndAdjustEventOffsets");

  int64_t offset_ps = NanoToPico(line_->timestamp_ns() - timestamp_ns);
  line_->set_timestamp_ns(timestamp_ns);
  if (offset_ps) {
    for (auto& event : *line_->mutable_events()) {
      event.set_offset_ps(event.offset_ps() + offset_ps);
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow
