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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utils_testDTcc() {
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

#include "tensorflow/core/profiler/utils/xplane_utils.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

XEvent CreateEvent(int64_t offset_ps, int64_t duration_ps) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utils_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/profiler/utils/xplane_utils_test.cc", "CreateEvent");

  XEvent event;
  event.set_offset_ps(offset_ps);
  event.set_duration_ps(duration_ps);
  return event;
}

TEST(XPlaneUtilsTest, AddAndRemovePlanes) {
  XSpace space;

  auto* p1 = FindOrAddMutablePlaneWithName(&space, "p1");
  EXPECT_EQ(p1, FindPlaneWithName(space, "p1"));
  auto* p2 = FindOrAddMutablePlaneWithName(&space, "p2");
  EXPECT_EQ(p2, FindPlaneWithName(space, "p2"));
  auto* p3 = FindOrAddMutablePlaneWithName(&space, "p3");
  EXPECT_EQ(p3, FindPlaneWithName(space, "p3"));

  // Removing a plane does not invalidate pointers to other planes.

  RemovePlane(&space, p2);
  EXPECT_EQ(space.planes_size(), 2);
  EXPECT_EQ(p1, FindPlaneWithName(space, "p1"));
  EXPECT_EQ(p3, FindPlaneWithName(space, "p3"));

  RemovePlane(&space, p1);
  EXPECT_EQ(space.planes_size(), 1);
  EXPECT_EQ(p3, FindPlaneWithName(space, "p3"));

  RemovePlane(&space, p3);
  EXPECT_EQ(space.planes_size(), 0);
}

TEST(XPlaneUtilsTest, RemoveEmptyPlanes) {
  XSpace space;
  RemoveEmptyPlanes(&space);
  EXPECT_EQ(space.planes_size(), 0);

  auto* plane1 = space.add_planes();
  plane1->set_name("p1");
  plane1->add_lines()->set_name("p1l1");
  plane1->add_lines()->set_name("p1l2");

  auto* plane2 = space.add_planes();
  plane2->set_name("p2");

  auto* plane3 = space.add_planes();
  plane3->set_name("p3");
  plane3->add_lines()->set_name("p3l1");

  auto* plane4 = space.add_planes();
  plane4->set_name("p4");

  RemoveEmptyPlanes(&space);
  ASSERT_EQ(space.planes_size(), 2);
  EXPECT_EQ(space.planes(0).name(), "p1");
  EXPECT_EQ(space.planes(1).name(), "p3");
}

TEST(XPlaneUtilsTest, RemoveEmptyLines) {
  XPlane plane;
  RemoveEmptyLines(&plane);
  EXPECT_EQ(plane.lines_size(), 0);

  auto* line1 = plane.add_lines();
  line1->set_name("l1");
  line1->add_events();
  line1->add_events();

  auto* line2 = plane.add_lines();
  line2->set_name("l2");

  auto* line3 = plane.add_lines();
  line3->set_name("l3");
  line3->add_events();

  auto* line4 = plane.add_lines();
  line4->set_name("l4");

  RemoveEmptyLines(&plane);
  ASSERT_EQ(plane.lines_size(), 2);
  EXPECT_EQ(plane.lines(0).name(), "l1");
  EXPECT_EQ(plane.lines(1).name(), "l3");
}

TEST(XPlaneUtilsTest, RemoveLine) {
  XPlane plane;
  const XLine* line1 = plane.add_lines();
  const XLine* line2 = plane.add_lines();
  const XLine* line3 = plane.add_lines();
  RemoveLine(&plane, line2);
  ASSERT_EQ(plane.lines_size(), 2);
  EXPECT_EQ(&plane.lines(0), line1);
  EXPECT_EQ(&plane.lines(1), line3);
}

TEST(XPlaneUtilsTest, RemoveEvents) {
  XLine line;
  const XEvent* event1 = line.add_events();
  const XEvent* event2 = line.add_events();
  const XEvent* event3 = line.add_events();
  const XEvent* event4 = line.add_events();
  RemoveEvents(&line, {event1, event3});
  ASSERT_EQ(line.events_size(), 2);
  EXPECT_EQ(&line.events(0), event2);
  EXPECT_EQ(&line.events(1), event4);
}

TEST(XPlaneUtilsTest, SortXPlaneTest) {
  XPlane plane;
  XLine* line = plane.add_lines();
  *line->add_events() = CreateEvent(200, 100);
  *line->add_events() = CreateEvent(100, 100);
  *line->add_events() = CreateEvent(120, 50);
  *line->add_events() = CreateEvent(120, 30);
  SortXPlane(&plane);
  ASSERT_EQ(plane.lines_size(), 1);
  ASSERT_EQ(plane.lines(0).events_size(), 4);
  EXPECT_EQ(plane.lines(0).events(0).offset_ps(), 100);
  EXPECT_EQ(plane.lines(0).events(0).duration_ps(), 100);
  EXPECT_EQ(plane.lines(0).events(1).offset_ps(), 120);
  EXPECT_EQ(plane.lines(0).events(1).duration_ps(), 50);
  EXPECT_EQ(plane.lines(0).events(2).offset_ps(), 120);
  EXPECT_EQ(plane.lines(0).events(2).duration_ps(), 30);
  EXPECT_EQ(plane.lines(0).events(3).offset_ps(), 200);
  EXPECT_EQ(plane.lines(0).events(3).duration_ps(), 100);
}

namespace {

XLineBuilder CreateXLine(XPlaneBuilder* plane, absl::string_view name,
                         absl::string_view display, int64_t id,
                         int64_t timestamp_ns) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   mht_1_v.push_back("display: \"" + std::string(display.data(), display.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utils_testDTcc mht_1(mht_1_v, 339, "", "./tensorflow/core/profiler/utils/xplane_utils_test.cc", "CreateXLine");

  XLineBuilder line = plane->GetOrCreateLine(id);
  line.SetName(name);
  line.SetTimestampNs(timestamp_ns);
  line.SetDisplayNameIfEmpty(display);
  return line;
}

XEventBuilder CreateXEvent(XPlaneBuilder* plane, XLineBuilder line,
                           absl::string_view event_name,
                           absl::optional<absl::string_view> display,
                           int64_t offset_ns, int64_t duration_ns) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("event_name: \"" + std::string(event_name.data(), event_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utils_testDTcc mht_2(mht_2_v, 354, "", "./tensorflow/core/profiler/utils/xplane_utils_test.cc", "CreateXEvent");

  XEventMetadata* event_metadata = plane->GetOrCreateEventMetadata(event_name);
  if (display) event_metadata->set_display_name(std::string(*display));
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetNs(offset_ns);
  event.SetDurationNs(duration_ns);
  return event;
}

template <typename T, typename V>
void CreateXStats(XPlaneBuilder* plane, T* stats_owner,
                  absl::string_view stats_name, V stats_value) {
  stats_owner->AddStatValue(*plane->GetOrCreateStatMetadata(stats_name),
                            stats_value);
}

void CheckXLine(const XLine& line, absl::string_view name,
                absl::string_view display, int64_t start_time_ns,
                int64_t events_size) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   mht_3_v.push_back("display: \"" + std::string(display.data(), display.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utils_testDTcc mht_3(mht_3_v, 377, "", "./tensorflow/core/profiler/utils/xplane_utils_test.cc", "CheckXLine");

  EXPECT_EQ(line.name(), name);
  EXPECT_EQ(line.display_name(), display);
  EXPECT_EQ(line.timestamp_ns(), start_time_ns);
  EXPECT_EQ(line.events_size(), events_size);
}

void CheckXEvent(const XEvent& event, const XPlane& plane,
                 absl::string_view name, absl::string_view display,
                 int64_t offset_ns, int64_t duration_ns, int64_t stats_size) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   mht_4_v.push_back("display: \"" + std::string(display.data(), display.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utils_testDTcc mht_4(mht_4_v, 391, "", "./tensorflow/core/profiler/utils/xplane_utils_test.cc", "CheckXEvent");

  const XEventMetadata& event_metadata =
      plane.event_metadata().at(event.metadata_id());
  EXPECT_EQ(event_metadata.name(), name);
  EXPECT_EQ(event_metadata.display_name(), display);
  EXPECT_EQ(event.offset_ps(), NanoToPico(offset_ns));
  EXPECT_EQ(event.duration_ps(), NanoToPico(duration_ns));
  EXPECT_EQ(event.stats_size(), stats_size);
}
}  // namespace

TEST(XPlaneUtilsTest, MergeXPlaneTest) {
  XPlane src_plane, dst_plane;
  constexpr int64_t kLineIdOnlyInSrcPlane = 1LL;
  constexpr int64_t kLineIdOnlyInDstPlane = 2LL;
  constexpr int64_t kLineIdInBothPlanes = 3LL;   // src start ts < dst start ts
  constexpr int64_t kLineIdInBothPlanes2 = 4LL;  // src start ts > dst start ts

  {  // Populate the source plane.
    XPlaneBuilder src(&src_plane);
    CreateXStats(&src, &src, "plane_stat1", 1);    // only in source.
    CreateXStats(&src, &src, "plane_stat3", 3.0);  // shared by source/dest.

    auto l1 = CreateXLine(&src, "l1", "d1", kLineIdOnlyInSrcPlane, 100);
    auto e1 = CreateXEvent(&src, l1, "event1", "display1", 1, 2);
    CreateXStats(&src, &e1, "event_stat1", 2.0);
    auto e2 = CreateXEvent(&src, l1, "event2", absl::nullopt, 3, 4);
    CreateXStats(&src, &e2, "event_stat2", 3);

    auto l2 = CreateXLine(&src, "l2", "d2", kLineIdInBothPlanes, 200);
    auto e3 = CreateXEvent(&src, l2, "event3", absl::nullopt, 5, 7);
    CreateXStats(&src, &e3, "event_stat3", 2.0);
    auto e4 = CreateXEvent(&src, l2, "event4", absl::nullopt, 6, 8);
    CreateXStats(&src, &e4, "event_stat4", 3);
    CreateXStats(&src, &e4, "event_stat5", 3);

    auto l5 = CreateXLine(&src, "l5", "d5", kLineIdInBothPlanes2, 700);
    CreateXEvent(&src, l5, "event51", absl::nullopt, 9, 10);
    CreateXEvent(&src, l5, "event52", absl::nullopt, 11, 12);
  }

  {  // Populate the destination plane.
    XPlaneBuilder dst(&dst_plane);
    CreateXStats(&dst, &dst, "plane_stat2", 2);  // only in dest
    CreateXStats(&dst, &dst, "plane_stat3", 4);  // shared but different.

    auto l3 = CreateXLine(&dst, "l3", "d3", kLineIdOnlyInDstPlane, 300);
    auto e5 = CreateXEvent(&dst, l3, "event5", absl::nullopt, 11, 2);
    CreateXStats(&dst, &e5, "event_stat6", 2.0);
    auto e6 = CreateXEvent(&dst, l3, "event6", absl::nullopt, 13, 4);
    CreateXStats(&dst, &e6, "event_stat7", 3);

    auto l2 = CreateXLine(&dst, "l4", "d4", kLineIdInBothPlanes, 400);
    auto e7 = CreateXEvent(&dst, l2, "event7", absl::nullopt, 15, 7);
    CreateXStats(&dst, &e7, "event_stat8", 2.0);
    auto e8 = CreateXEvent(&dst, l2, "event8", "display8", 16, 8);
    CreateXStats(&dst, &e8, "event_stat9", 3);

    auto l6 = CreateXLine(&dst, "l6", "d6", kLineIdInBothPlanes2, 300);
    CreateXEvent(&dst, l6, "event61", absl::nullopt, 21, 10);
    CreateXEvent(&dst, l6, "event62", absl::nullopt, 22, 12);
  }

  MergePlanes(src_plane, &dst_plane);

  XPlaneVisitor plane(&dst_plane);
  EXPECT_EQ(dst_plane.lines_size(), 4);

  // Check plane level stats,
  EXPECT_EQ(dst_plane.stats_size(), 3);
  absl::flat_hash_map<absl::string_view, absl::string_view> plane_stats;
  plane.ForEachStat([&](const tensorflow::profiler::XStatVisitor& stat) {
    if (stat.Name() == "plane_stat1") {
      EXPECT_EQ(stat.IntValue(), 1);
    } else if (stat.Name() == "plane_stat2") {
      EXPECT_EQ(stat.IntValue(), 2);
    } else if (stat.Name() == "plane_stat3") {
      // XStat in src_plane overrides the counter-part in dst_plane.
      EXPECT_EQ(stat.DoubleValue(), 3.0);
    } else {
      EXPECT_TRUE(false);
    }
  });

  // 3 plane level stats, 9 event level stats.
  EXPECT_EQ(dst_plane.stat_metadata_size(), 12);

  {  // Old lines are untouched.
    const XLine& line = dst_plane.lines(0);
    CheckXLine(line, "l3", "d3", 300, 2);
    CheckXEvent(line.events(0), dst_plane, "event5", "", 11, 2, 1);
    CheckXEvent(line.events(1), dst_plane, "event6", "", 13, 4, 1);
  }
  {  // Lines with the same id are merged.
    // src plane start timestamp > dst plane start timestamp
    const XLine& line = dst_plane.lines(1);
    // NOTE: use minimum start time of src/dst.
    CheckXLine(line, "l4", "d4", 200, 4);
    CheckXEvent(line.events(0), dst_plane, "event7", "", 215, 7, 1);
    CheckXEvent(line.events(1), dst_plane, "event8", "display8", 216, 8, 1);
    CheckXEvent(line.events(2), dst_plane, "event3", "", 5, 7, 1);
    CheckXEvent(line.events(3), dst_plane, "event4", "", 6, 8, 2);
  }
  {  // Lines with the same id are merged.
    // src plane start timestamp < dst plane start timestamp
    const XLine& line = dst_plane.lines(2);
    CheckXLine(line, "l6", "d6", 300, 4);
    CheckXEvent(line.events(0), dst_plane, "event61", "", 21, 10, 0);
    CheckXEvent(line.events(1), dst_plane, "event62", "", 22, 12, 0);
    CheckXEvent(line.events(2), dst_plane, "event51", "", 409, 10, 0);
    CheckXEvent(line.events(3), dst_plane, "event52", "", 411, 12, 0);
  }
  {  // Lines only in source are "copied".
    const XLine& line = dst_plane.lines(3);
    CheckXLine(line, "l1", "d1", 100, 2);
    CheckXEvent(line.events(0), dst_plane, "event1", "display1", 1, 2, 1);
    CheckXEvent(line.events(1), dst_plane, "event2", "", 3, 4, 1);
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
