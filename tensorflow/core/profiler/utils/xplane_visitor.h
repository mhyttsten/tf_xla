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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_VISITOR_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_VISITOR_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh() {
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


#include <stddef.h>

#include <functional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

class XPlaneVisitor;

class XStatVisitor {
 public:
  // REQUIRED: plane and stat cannot be nullptr.
  XStatVisitor(const XPlaneVisitor* plane, const XStat* stat);

  // REQUIRED: plane, stat and metadata cannot be nullptr.
  XStatVisitor(const XPlaneVisitor* plane, const XStat* stat,
               const XStatMetadata* metadata, absl::optional<int64_t> type);

  int64_t Id() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_0(mht_0_v, 215, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Id");
 return stat_->metadata_id(); }

  absl::string_view Name() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_1(mht_1_v, 220, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Name");
 return metadata_->name(); }

  absl::optional<int64_t> Type() const { return type_; }

  absl::string_view Description() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_2(mht_2_v, 227, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Description");
 return metadata_->description(); }

  XStat::ValueCase ValueCase() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_3(mht_3_v, 232, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "ValueCase");
 return stat_->value_case(); }

  bool BoolValue() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_4(mht_4_v, 237, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "BoolValue");
 return static_cast<bool>(IntValue()); }

  int64_t IntValue() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_5(mht_5_v, 242, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "IntValue");
 return stat_->int64_value(); }

  uint64 UintValue() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_6(mht_6_v, 247, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "UintValue");
 return stat_->uint64_value(); }

  uint64 IntOrUintValue() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_7(mht_7_v, 252, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "IntOrUintValue");

    return ValueCase() == XStat::kUint64Value ? UintValue()
                                              : static_cast<uint64>(IntValue());
  }

  double DoubleValue() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_8(mht_8_v, 260, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "DoubleValue");
 return stat_->double_value(); }

  // Returns a string view.
  // REQUIRED: the value type should be string type or reference type.
  absl::string_view StrOrRefValue() const;

  const XStat& RawStat() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_9(mht_9_v, 269, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "RawStat");
 return *stat_; }

  // Return a string representation of all value type.
  std::string ToString() const;

 private:
  const XStat* stat_;
  const XStatMetadata* metadata_;
  const XPlaneVisitor* plane_;
  absl::optional<int64_t> type_;
};

template <class T>
class XStatsOwner {
 public:
  // REQUIRED: plane and stats_owner cannot be nullptr.
  XStatsOwner(const XPlaneVisitor* plane, const T* stats_owner)
      : plane_(plane), stats_owner_(stats_owner) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_10(mht_10_v, 289, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "XStatsOwner");
}

  // For each stat, call the specified lambda.
  template <typename ForEachStatFunc>
  void ForEachStat(ForEachStatFunc&& for_each_stat) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_11(mht_11_v, 296, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "ForEachStat");

    for (const XStat& stat : stats_owner_->stats()) {
      for_each_stat(XStatVisitor(plane_, &stat));
    }
  }

  // Shortcut to get a specific stat type, nullopt if absent.
  // This function performs a linear search for the requested stat value.
  // Prefer ForEachStat above when multiple stat values are necessary.
  absl::optional<XStatVisitor> GetStat(int64_t stat_type) const;

  // Same as above that skips searching for the stat.
  absl::optional<XStatVisitor> GetStat(
      int64_t stat_type, const XStatMetadata& stat_metadata) const {
    for (const XStat& stat : stats_owner_->stats()) {
      if (stat.metadata_id() == stat_metadata.id()) {
        return XStatVisitor(plane_, &stat, &stat_metadata, stat_type);
      }
    }
    return absl::nullopt;  // type does not exist in this owner.
  }

 protected:
  const XPlaneVisitor* plane() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_12(mht_12_v, 322, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "plane");
 return plane_; }
  const T* stats_owner() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_13(mht_13_v, 326, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "stats_owner");
 return stats_owner_; }

 private:
  const XPlaneVisitor* plane_;
  const T* stats_owner_;
};

class XEventMetadataVisitor : public XStatsOwner<XEventMetadata> {
 public:
  // REQUIRED: plane and metadata cannot be nullptr.
  XEventMetadataVisitor(const XPlaneVisitor* plane,
                        const XEventMetadata* metadata)
      : XStatsOwner(plane, metadata) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_14(mht_14_v, 341, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "XEventMetadataVisitor");
}

  absl::string_view Name() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_15(mht_15_v, 346, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Name");
 return metadata()->name(); }

  bool HasDisplayName() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_16(mht_16_v, 351, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "HasDisplayName");
 return !metadata()->display_name().empty(); }

  absl::string_view DisplayName() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_17(mht_17_v, 356, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "DisplayName");
 return metadata()->display_name(); }

  // For each child event metadata, call the specified lambda.
  template <typename ForEachChildFunc>
  void ForEachChild(ForEachChildFunc&& for_each_child) const;

 private:
  const XEventMetadata* metadata() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_18(mht_18_v, 366, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "metadata");
 return stats_owner(); }
};

class XEventVisitor : public XStatsOwner<XEvent> {
 public:
  // REQUIRED: plane, line and event cannot be nullptr.
  XEventVisitor(const XPlaneVisitor* plane, const XLine* line,
                const XEvent* event);

  int64_t Id() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_19(mht_19_v, 378, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Id");
 return event_->metadata_id(); }

  absl::string_view Name() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_20(mht_20_v, 383, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Name");
 return metadata_->name(); }

  absl::optional<int64_t> Type() const { return type_; }

  bool HasDisplayName() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_21(mht_21_v, 390, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "HasDisplayName");
 return !metadata_->display_name().empty(); }

  absl::string_view DisplayName() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_22(mht_22_v, 395, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "DisplayName");
 return metadata_->display_name(); }

  double OffsetNs() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_23(mht_23_v, 400, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "OffsetNs");
 return PicoToNano(event_->offset_ps()); }

  int64_t OffsetPs() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_24(mht_24_v, 405, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "OffsetPs");
 return event_->offset_ps(); }

  int64_t LineTimestampNs() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_25(mht_25_v, 410, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "LineTimestampNs");
 return line_->timestamp_ns(); }

  double TimestampNs() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_26(mht_26_v, 415, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "TimestampNs");
 return line_->timestamp_ns() + OffsetNs(); }

  int64_t TimestampPs() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_27(mht_27_v, 420, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "TimestampPs");

    return NanoToPico(line_->timestamp_ns()) + event_->offset_ps();
  }

  double DurationNs() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_28(mht_28_v, 427, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "DurationNs");
 return PicoToNano(event_->duration_ps()); }

  int64_t DurationPs() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_29(mht_29_v, 432, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "DurationPs");
 return event_->duration_ps(); }

  int64_t EndOffsetPs() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_30(mht_30_v, 437, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "EndOffsetPs");

    return event_->offset_ps() + event_->duration_ps();
  }
  int64_t EndTimestampPs() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_31(mht_31_v, 443, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "EndTimestampPs");
 return TimestampPs() + DurationPs(); }

  int64_t NumOccurrences() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_32(mht_32_v, 448, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "NumOccurrences");
 return event_->num_occurrences(); }

  bool operator<(const XEventVisitor& other) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_33(mht_33_v, 453, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "operator<");

    return GetTimespan() < other.GetTimespan();
  }

  const XEventMetadata* metadata() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_34(mht_34_v, 460, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "metadata");
 return metadata_; }

  XEventMetadataVisitor Metadata() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_35(mht_35_v, 465, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Metadata");

    return XEventMetadataVisitor(plane_, metadata_);
  }

  Timespan GetTimespan() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_36(mht_36_v, 472, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "GetTimespan");
 return Timespan(TimestampPs(), DurationPs()); }

 private:
  const XPlaneVisitor* plane_;
  const XLine* line_;
  const XEvent* event_;
  const XEventMetadata* metadata_;
  absl::optional<int64_t> type_;
};

class XLineVisitor {
 public:
  // REQUIRED: plane and line cannot be nullptr.
  XLineVisitor(const XPlaneVisitor* plane, const XLine* line)
      : plane_(plane), line_(line) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_37(mht_37_v, 489, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "XLineVisitor");
}

  int64_t Id() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_38(mht_38_v, 494, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Id");
 return line_->id(); }

  int64_t DisplayId() const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_39(mht_39_v, 499, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "DisplayId");

    return line_->display_id() ? line_->display_id() : line_->id();
  }

  absl::string_view Name() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_40(mht_40_v, 506, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Name");
 return line_->name(); }

  absl::string_view DisplayName() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_41(mht_41_v, 511, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "DisplayName");

    return !line_->display_name().empty() ? line_->display_name()
                                          : line_->name();
  }

  double TimestampNs() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_42(mht_42_v, 519, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "TimestampNs");
 return line_->timestamp_ns(); }

  int64_t DurationPs() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_43(mht_43_v, 524, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "DurationPs");
 return line_->duration_ps(); }

  size_t NumEvents() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_44(mht_44_v, 529, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "NumEvents");
 return line_->events_size(); }

  template <typename ForEachEventFunc>
  void ForEachEvent(ForEachEventFunc&& for_each_event) const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_45(mht_45_v, 535, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "ForEachEvent");

    for (const XEvent& event : line_->events()) {
      for_each_event(XEventVisitor(plane_, line_, &event));
    }
  }

 private:
  const XPlaneVisitor* plane_;
  const XLine* line_;
};

using TypeGetter = std::function<absl::optional<int64_t>(absl::string_view)>;
using TypeGetterList = std::vector<TypeGetter>;

class XPlaneVisitor : public XStatsOwner<XPlane> {
 public:
  // REQUIRED: plane cannot be nullptr.
  explicit XPlaneVisitor(
      const XPlane* plane,
      const TypeGetterList& event_type_getter_list = TypeGetterList(),
      const TypeGetterList& stat_type_getter_list = TypeGetterList());

  int64_t Id() const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_46(mht_46_v, 560, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Id");
 return plane_->id(); }

  absl::string_view Name() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_47(mht_47_v, 565, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "Name");
 return plane_->name(); }

  size_t NumLines() const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_48(mht_48_v, 570, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "NumLines");
 return plane_->lines_size(); }

  template <typename ForEachLineFunc>
  void ForEachLine(ForEachLineFunc&& for_each_line) const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_49(mht_49_v, 576, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "ForEachLine");

    for (const XLine& line : plane_->lines()) {
      for_each_line(XLineVisitor(this, &line));
    }
  }

  // Returns event metadata given its id. Returns a default value if not found.
  const XEventMetadata* GetEventMetadata(int64_t event_metadata_id) const;

  // Returns the type of an event given its id.
  absl::optional<int64_t> GetEventType(int64_t event_metadata_id) const;

  // Returns stat metadata given its id. Returns a default value if not found.
  const XStatMetadata* GetStatMetadata(int64_t stat_metadata_id) const;

  // Returns stat metadata given its type. Returns nullptr if not found.
  // Use as an alternative to GetStatMetadata above.
  const XStatMetadata* GetStatMetadataByType(int64_t stat_type) const;

  // Returns the type of an stat given its id.
  absl::optional<int64_t> GetStatType(int64_t stat_metadata_id) const;

 private:
  void BuildEventTypeMap(const XPlane* plane,
                         const TypeGetterList& event_type_getter_list);
  void BuildStatTypeMap(const XPlane* plane,
                        const TypeGetterList& stat_type_getter_list);

  const XPlane* plane_;

  absl::flat_hash_map<int64_t /*metadata_id*/, int64_t /*EventType*/>
      event_type_by_id_;
  absl::flat_hash_map<int64_t /*metadata_id*/, int64_t /*StatType*/>
      stat_type_by_id_;
  absl::flat_hash_map<int64_t /*StatType*/, const XStatMetadata*>
      stat_metadata_by_type_;
};

template <class T>
absl::optional<XStatVisitor> XStatsOwner<T>::GetStat(int64_t stat_type) const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_50(mht_50_v, 618, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "XStatsOwner<T>::GetStat");

  const auto* stat_metadata = plane_->GetStatMetadataByType(stat_type);
  if (stat_metadata != nullptr) {
    return GetStat(stat_type, *stat_metadata);
  }
  return absl::nullopt;  // type does not exist in this owner.
}

template <typename ForEachChildFunc>
void XEventMetadataVisitor::ForEachChild(
    ForEachChildFunc&& for_each_child) const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_visitorDTh mht_51(mht_51_v, 631, "", "./tensorflow/core/profiler/utils/xplane_visitor.h", "XEventMetadataVisitor::ForEachChild");

  for (int64_t child_id : metadata()->child_id()) {
    const auto* event_metadata = plane()->GetEventMetadata(child_id);
    if (event_metadata != nullptr) {
      for_each_child(XEventMetadataVisitor(plane(), event_metadata));
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_VISITOR_H_
