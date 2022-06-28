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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_BUILDER_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_BUILDER_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh() {
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

#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/meta/type_traits.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

class XPlaneBuilder;

template <typename T>
class XStatsBuilder {
 public:
  explicit XStatsBuilder(T* stats_owner, XPlaneBuilder* stats_metadata_owner)
      : stats_owner_(stats_owner),
        stats_metadata_owner_(stats_metadata_owner) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_0(mht_0_v, 215, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "XStatsBuilder");
}

  // NOTE: A stat shouldn't have existed for the given metadata.
  // Adds a stat for the given metadata and sets its value.
  template <typename ValueT>
  void AddStatValue(const XStatMetadata& metadata, ValueT&& value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_1(mht_1_v, 223, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "AddStatValue");

    SetStatValue(std::forward<ValueT>(value), AddStat(metadata));
  }

  // Adds or finds a stat for the given metadata and sets its value.
  template <typename ValueT>
  void SetOrAddStatValue(const XStatMetadata& metadata, ValueT&& value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_2(mht_2_v, 232, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetOrAddStatValue");

    SetStatValue(std::forward<ValueT>(value), FindOrAddStat(metadata));
  }

  // Adds a stat by copying a stat from another XPlane. Does not check if a stat
  // with the same metadata already exists in the event. To avoid duplicated
  // stats, use the variant below.
  void AddStat(const XStatMetadata& metadata, const XStat& src_stat,
               const XPlane& src_plane) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_3(mht_3_v, 243, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "AddStat");

    CopyStatValue(src_stat, src_plane, AddStat(metadata));
  }
  // Same as above but overrides an existing stat with the same metadata.
  void SetOrAddStat(const XStatMetadata& metadata, const XStat& src_stat,
                    const XPlane& src_plane) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_4(mht_4_v, 251, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetOrAddStat");

    CopyStatValue(src_stat, src_plane, FindOrAddStat(metadata));
  }

  void ParseAndAddStatValue(const XStatMetadata& metadata,
                            absl::string_view value) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("value: \"" + std::string(value.data(), value.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_5(mht_5_v, 260, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "ParseAndAddStatValue");

    int64_t int_value;
    uint64 uint_value;
    double double_value;
    if (absl::SimpleAtoi(value, &int_value)) {
      AddStatValue(metadata, int_value);
    } else if (absl::SimpleAtoi(value, &uint_value)) {
      AddStatValue(metadata, uint_value);
    } else if (absl::SimpleAtod(value, &double_value)) {
      AddStatValue(metadata, double_value);
    } else {
      AddStatValue(metadata, GetOrCreateStatMetadata(value));
    }
  }

  void ReserveStats(size_t num_stats) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_6(mht_6_v, 278, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "ReserveStats");

    stats_owner_->mutable_stats()->Reserve(num_stats);
  }

  template <typename ForEachStatFunc>
  void ForEachStat(ForEachStatFunc&& for_each_stat) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_7(mht_7_v, 286, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "ForEachStat");

    for (XStat& stat : *stats_owner_->mutable_stats()) {
      for_each_stat(&stat);
    }
  }

  const XStat* GetStat(int64_t metadata_id) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_8(mht_8_v, 295, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "GetStat");

    for (auto& stat : *stats_owner_->mutable_stats()) {
      if (stat.metadata_id() == metadata_id) {
        return &stat;
      }
    }
    return nullptr;
  }

  static uint64 IntOrUintValue(const XStat& stat) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_9(mht_9_v, 307, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "IntOrUintValue");

    return stat.value_case() == XStat::kUint64Value ? stat.uint64_value()
                                                    : stat.int64_value();
  }

  absl::string_view StrOrRefValue(const XStat& stat);

 private:
  XStat* AddStat(const XStatMetadata& metadata) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_10(mht_10_v, 318, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "AddStat");

    XStat* stat = stats_owner_->add_stats();
    stat->set_metadata_id(metadata.id());
    return stat;
  }

  XStat* FindOrAddStat(const XStatMetadata& metadata) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_11(mht_11_v, 327, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "FindOrAddStat");

    for (auto& stat : *stats_owner_->mutable_stats()) {
      if (stat.metadata_id() == metadata.id()) {
        return &stat;
      }
    }
    return AddStat(metadata);
  }

  static void SetStatValue(bool value, XStat* stat) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_12(mht_12_v, 339, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetStatValue");

    // bool is integral unsigned, but saved in the signed slot for backwards
    // compatibility.
    stat->set_int64_value(value);
  }
  template <typename Int,
            std::enable_if_t<absl::conjunction<std::is_integral<Int>,
                                               std::is_signed<Int>>::value,
                             bool> = true>
  static void SetStatValue(Int value, XStat* stat) {
    stat->set_int64_value(value);
  }
  template <typename UInt,
            std::enable_if_t<
                absl::conjunction<std::is_integral<UInt>,
                                  absl::negation<std::is_signed<UInt>>>::value,
                bool> = true>
  static void SetStatValue(UInt value, XStat* stat) {
    stat->set_uint64_value(value);
  }
  static void SetStatValue(double value, XStat* stat) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_13(mht_13_v, 362, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetStatValue");

    stat->set_double_value(value);
  }
  static void SetStatValue(const char* value, XStat* stat) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_14(mht_14_v, 369, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetStatValue");

    stat->set_str_value(std::string(value));
  }
  static void SetStatValue(absl::string_view value, XStat* stat) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("value: \"" + std::string(value.data(), value.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_15(mht_15_v, 376, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetStatValue");

    stat->set_str_value(std::string(value));
  }
  static void SetStatValue(std::string&& value, XStat* stat) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_16(mht_16_v, 382, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetStatValue");

    stat->set_str_value(std::move(value));
  }
  static void SetStatValue(const XStatMetadata& value, XStat* stat) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_17(mht_17_v, 388, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetStatValue");

    stat->set_ref_value(value.id());
  }
  static void SetStatValue(const protobuf::MessageLite& proto, XStat* stat) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_18(mht_18_v, 394, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetStatValue");

    auto* bytes = stat->mutable_bytes_value();
    proto.SerializeToString(bytes);
  }

  void CopyStatValue(const XStat& src_stat, const XPlane& src_plane,
                     XStat* dst_stat) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_19(mht_19_v, 403, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "CopyStatValue");

    switch (src_stat.value_case()) {
      case XStat::VALUE_NOT_SET:
        break;
      case XStat::kInt64Value:
        dst_stat->set_int64_value(src_stat.int64_value());
        break;
      case XStat::kUint64Value:
        dst_stat->set_uint64_value(src_stat.uint64_value());
        break;
      case XStat::kDoubleValue:
        dst_stat->set_double_value(src_stat.double_value());
        break;
      case XStat::kStrValue:
        dst_stat->set_str_value(src_stat.str_value());
        break;
      case XStat::kRefValue: {
        const auto& stat_metadata_by_id = src_plane.stat_metadata();
        const auto it = stat_metadata_by_id.find(src_stat.ref_value());
        if (TF_PREDICT_TRUE(it != stat_metadata_by_id.end())) {
          absl::string_view value = it->second.name();
          dst_stat->set_ref_value(GetOrCreateStatMetadata(value).id());
        }
        break;
      }
      case XStat::kBytesValue:
        dst_stat->set_bytes_value(src_stat.bytes_value());
        break;
    }
  }

  const XStatMetadata& GetOrCreateStatMetadata(absl::string_view value);

  T* stats_owner_;
  XPlaneBuilder* stats_metadata_owner_;
};

class XEventBuilder : public XStatsBuilder<XEvent> {
 public:
  XEventBuilder(const XLine* line, XPlaneBuilder* plane, XEvent* event)
      : XStatsBuilder<XEvent>(event, plane), line_(line), event_(event) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_20(mht_20_v, 446, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "XEventBuilder");
}

  int64_t OffsetPs() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_21(mht_21_v, 451, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "OffsetPs");
 return event_->offset_ps(); }
  int64_t MetadataId() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_22(mht_22_v, 455, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "MetadataId");
 return event_->metadata_id(); }

  void SetOffsetPs(int64_t offset_ps) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_23(mht_23_v, 460, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetOffsetPs");
 event_->set_offset_ps(offset_ps); }

  void SetOffsetNs(int64_t offset_ns) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_24(mht_24_v, 465, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetOffsetNs");
 SetOffsetPs(NanoToPico(offset_ns)); }

  void SetTimestampNs(int64_t timestamp_ns) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_25(mht_25_v, 470, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetTimestampNs");

    SetOffsetPs(NanoToPico(timestamp_ns - line_->timestamp_ns()));
  }

  void SetNumOccurrences(int64_t num_occurrences) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_26(mht_26_v, 477, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetNumOccurrences");

    event_->set_num_occurrences(num_occurrences);
  }

  void SetDurationPs(int64_t duration_ps) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_27(mht_27_v, 484, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetDurationPs");

    event_->set_duration_ps(duration_ps);
  }
  void SetDurationNs(int64_t duration_ns) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_28(mht_28_v, 490, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetDurationNs");

    SetDurationPs(NanoToPico(duration_ns));
  }

  void SetEndTimestampPs(int64_t end_timestamp_ps) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_29(mht_29_v, 497, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetEndTimestampPs");

    SetDurationPs(end_timestamp_ps - PicoToNano(line_->timestamp_ns()) -
                  event_->offset_ps());
  }
  void SetEndTimestampNs(int64_t end_timestamp_ns) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_30(mht_30_v, 504, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetEndTimestampNs");

    SetDurationPs(NanoToPico(end_timestamp_ns - line_->timestamp_ns()) -
                  event_->offset_ps());
  }

  Timespan GetTimespan() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_31(mht_31_v, 512, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "GetTimespan");

    return Timespan(NanoToPico(line_->timestamp_ns()) + event_->offset_ps(),
                    event_->duration_ps());
  }

 private:
  const XLine* line_;
  XEvent* event_;
};

class XLineBuilder {
 public:
  explicit XLineBuilder(XLine* line, XPlaneBuilder* plane)
      : line_(line), plane_(plane) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_32(mht_32_v, 528, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "XLineBuilder");
}

  // Returns the owner plane.
  XPlaneBuilder* Plane() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_33(mht_33_v, 534, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "Plane");
 return plane_; }

  int64_t Id() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_34(mht_34_v, 539, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "Id");
 return line_->id(); }
  void SetId(int64_t id) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_35(mht_35_v, 543, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetId");
 line_->set_id(id); }

  int64_t NumEvents() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_36(mht_36_v, 548, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "NumEvents");
 return line_->events_size(); }

  absl::string_view Name() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_37(mht_37_v, 553, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "Name");
 return line_->name(); }
  void SetName(absl::string_view name) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_38(mht_38_v, 558, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetName");
 line_->set_name(std::string(name)); }

  void SetNameIfEmpty(absl::string_view name) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_39(mht_39_v, 564, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetNameIfEmpty");

    if (line_->name().empty()) SetName(name);
  }

  int64_t TimestampNs() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_40(mht_40_v, 571, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "TimestampNs");
 return line_->timestamp_ns(); }
  // This will set the line start timestamp.
  // WARNING: The offset_ps of existing events will not be altered.
  void SetTimestampNs(int64_t timestamp_ns) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_41(mht_41_v, 577, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetTimestampNs");

    line_->set_timestamp_ns(timestamp_ns);
  }
  // This will set the line start timestamp to specific time, and adjust
  // the offset_ps of all existing events.
  void SetTimestampNsAndAdjustEventOffsets(int64_t timestamp_ns);

  void SetDurationPs(int64_t duration_ps) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_42(mht_42_v, 587, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetDurationPs");

    line_->set_duration_ps(duration_ps);
  }

  void ReserveEvents(size_t num_events) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_43(mht_43_v, 594, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "ReserveEvents");

    line_->mutable_events()->Reserve(num_events);
  }

  void SetDisplayNameIfEmpty(absl::string_view display_name) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("display_name: \"" + std::string(display_name.data(), display_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_44(mht_44_v, 602, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetDisplayNameIfEmpty");

    if (line_->display_name().empty()) {
      line_->set_display_name(std::string(display_name));
    }
  }

  XEventBuilder AddEvent(const XEventMetadata& metadata);
  XEventBuilder AddEvent(const XEvent& event);

  template <typename ForEachEventFunc>
  void ForEachEvent(ForEachEventFunc&& for_each_event) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_45(mht_45_v, 615, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "ForEachEvent");

    for (XEvent& event : *line_->mutable_events()) {
      for_each_event(XEventBuilder(line_, plane_, &event));
    }
  }

 private:
  XLine* line_;
  XPlaneBuilder* plane_;
};

// Provides methods to build an XPlane.
// NOTE: avoid to use two builders to wrap the same XPlane.
class XPlaneBuilder : public XStatsBuilder<XPlane> {
 public:
  explicit XPlaneBuilder(XPlane* plane);

  int64_t Id() const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_46(mht_46_v, 635, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "Id");
 return plane_->id(); }
  void SetId(int64_t id) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_47(mht_47_v, 639, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetId");
 plane_->set_id(id); }

  absl::string_view Name() const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_48(mht_48_v, 644, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "Name");
 return plane_->name(); }
  void SetName(absl::string_view name) {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_49(mht_49_v, 649, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "SetName");
 plane_->set_name(std::string(name)); }

  void ReserveLines(size_t num_lines) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_50(mht_50_v, 654, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "ReserveLines");

    plane_->mutable_lines()->Reserve(num_lines);
  }

  template <typename ForEachLineFunc>
  void ForEachLine(ForEachLineFunc&& for_each_line) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_51(mht_51_v, 662, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "ForEachLine");

    for (XLine& line : *plane_->mutable_lines()) {
      for_each_line(XLineBuilder(&line, this));
    }
  }

  // Returns a builder for the line with the given id. Creates a new line if the
  // id was unused, otherwise the builder will add events to an existing line.
  XLineBuilder GetOrCreateLine(int64_t line_id);

  // Returns a new event metadata with an automatically generated metadata_id.
  // WARNING: If calling this function, don't call GetOrCreateEventMetadata.
  XEventMetadata* CreateEventMetadata();

  // Returns event metadata with the given id. Creates a new metadata if the id
  // was unused.
  // WARNING: If calling this function, don't call the string overloads below
  // on the same instance.
  XEventMetadata* GetOrCreateEventMetadata(int64_t metadata_id);

  // Returns event metadata with the given name. The id is internally assigned.
  // Creates a new metadata if the name was unused.
  // Using these overloads guarantees names are unique.
  // WARNING: If calling any of these overloads, do not call the integer one
  // above on the same instance.
  XEventMetadata* GetOrCreateEventMetadata(absl::string_view name);
  XEventMetadata* GetOrCreateEventMetadata(std::string&& name);
  XEventMetadata* GetOrCreateEventMetadata(const char* name) {
   std::vector<std::string> mht_52_v;
   mht_52_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_52(mht_52_v, 693, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "GetOrCreateEventMetadata");

    return GetOrCreateEventMetadata(absl::string_view(name));
  }

  // Returns event metadata with the given name. Returns nullptr if not found.
  XEventMetadata* GetEventMetadata(absl::string_view name) const;

  // Returns stat metadata with the given name. Returns nullptr if not found.
  XStatMetadata* GetStatMetadata(absl::string_view name) const;

  // Returns stat metadata given its id. Returns a default value if not found.
  const XStatMetadata* GetStatMetadata(int64_t metadata_id) const;

  // Returns a new stat metadata with an automatically generated metadata_id.
  // WARNING: If calling this function, don't call GetOrCreateEventMetadata.
  XStatMetadata* CreateStatMetadata();

  // Returns stat metadata with the given id. Creates a new metadata if the id
  // was unused.
  // WARNING: If calling this function, don't call the string overloads below
  // on the same instance.
  XStatMetadata* GetOrCreateStatMetadata(int64_t metadata_id);

  // Returns stat metadata with the given name. The id is internally assigned.
  // Creates a new metadata if the name was unused.
  // Using these overloads guarantees names are unique.
  // WARNING: If calling any of these overloads, do not call the integer one
  // above on the same instance.
  XStatMetadata* GetOrCreateStatMetadata(absl::string_view name);
  XStatMetadata* GetOrCreateStatMetadata(std::string&& name);
  XStatMetadata* GetOrCreateStatMetadata(const char* name) {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_53(mht_53_v, 727, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "GetOrCreateStatMetadata");

    return GetOrCreateStatMetadata(absl::string_view(name));
  }

 private:
  XPlane* plane_;

  // Artifacts to accelerate the builders.
  int64_t last_event_metadata_id_ = 0LL;
  int64_t last_stat_metadata_id_ = 0LL;
  absl::flat_hash_map<std::string, XEventMetadata*> event_metadata_by_name_;
  absl::flat_hash_map<std::string, XStatMetadata*> stat_metadata_by_name_;
  absl::flat_hash_map<int64_t, XLine*> lines_by_id_;
};

template <typename T>
const XStatMetadata& XStatsBuilder<T>::GetOrCreateStatMetadata(
    absl::string_view value) {
   std::vector<std::string> mht_54_v;
   mht_54_v.push_back("value: \"" + std::string(value.data(), value.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_54(mht_54_v, 748, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "XStatsBuilder<T>::GetOrCreateStatMetadata");

  return *stats_metadata_owner_->GetOrCreateStatMetadata(value);
}

template <typename T>
absl::string_view XStatsBuilder<T>::StrOrRefValue(const XStat& stat) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_builderDTh mht_55(mht_55_v, 756, "", "./tensorflow/core/profiler/utils/xplane_builder.h", "XStatsBuilder<T>::StrOrRefValue");

  switch (stat.value_case()) {
    case XStat::kStrValue:
      return stat.str_value();
    case XStat::kRefValue: {
      auto* ref_stat = stats_metadata_owner_->GetStatMetadata(stat.ref_value());
      return ref_stat ? ref_stat->name() : absl::string_view();
    }
    case XStat::kInt64Value:
    case XStat::kUint64Value:
    case XStat::kDoubleValue:
    case XStat::kBytesValue:
    case XStat::VALUE_NOT_SET:
      return absl::string_view();
  }
}
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_BUILDER_H_
