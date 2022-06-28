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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_events_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_events_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_events_testDTcc() {
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

#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

void CreateXSpace(XSpace* space) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_events_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/profiler/convert/xplane_to_trace_events_test.cc", "CreateXSpace");

  XPlaneBuilder host_plane(space->add_planes());
  host_plane.SetName(kHostThreadsPlaneName);
  XLineBuilder thread1 = host_plane.GetOrCreateLine(10);
  thread1.SetName("thread1");
  XEventBuilder event1 =
      thread1.AddEvent(*host_plane.GetOrCreateEventMetadata("event1"));
  event1.SetTimestampNs(150000);
  event1.SetDurationNs(10000);
  event1.AddStatValue(*host_plane.GetOrCreateStatMetadata("tf_op"),
                      *host_plane.GetOrCreateStatMetadata("Relu"));
  XLineBuilder thread2 = host_plane.GetOrCreateLine(20);
  thread2.SetName("thread2");
  XEventBuilder event2 =
      thread2.AddEvent(*host_plane.GetOrCreateEventMetadata("event2"));
  event2.SetTimestampNs(160000);
  event2.SetDurationNs(10000);
  event2.AddStatValue(*host_plane.GetOrCreateStatMetadata("tf_op"),
                      *host_plane.GetOrCreateStatMetadata("Conv2D"));

  XPlaneBuilder device_plane(space->add_planes());
  device_plane.SetName(GpuPlaneName(0));
  device_plane.SetId(0);
  XLineBuilder stream1 = device_plane.GetOrCreateLine(30);
  stream1.SetName("gpu stream 1");
  XEventBuilder event3 =
      stream1.AddEvent(*device_plane.GetOrCreateEventMetadata("kernel1"));
  event3.SetTimestampNs(180000);
  event3.SetDurationNs(10000);
  event3.AddStatValue(*device_plane.GetOrCreateStatMetadata("correlation id"),
                      55);
}

TEST(ConvertXPlaneToTraceEvents, Convert) {
  XSpace xspace;
  CreateXSpace(&xspace);

  Trace trace;
  ConvertXSpaceToTraceEvents(xspace, &trace);

  ASSERT_EQ(trace.devices_size(), 2);
  EXPECT_EQ(trace.devices().at(kHostThreadsDeviceId).resources_size(), 2);
  EXPECT_EQ(trace.devices().at(kFirstDeviceId).resources_size(), 1);
  EXPECT_EQ(trace.trace_events_size(), 3);
}

TEST(ConvertXPlaneToTraceEvents, Drop) {
  Trace trace;
  for (int i = 0; i < 100; i++) {
    trace.add_trace_events()->set_timestamp_ps((100 - i) % 50);
  }

  MaybeDropEventsForTraceViewer(&trace, 150);
  EXPECT_EQ(trace.trace_events_size(), 100);  // No dropping.

  MaybeDropEventsForTraceViewer(&trace, 50);
  EXPECT_EQ(trace.trace_events_size(), 50);
  for (const auto& event : trace.trace_events()) {
    EXPECT_LT(event.timestamp_ps(), 25);
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
