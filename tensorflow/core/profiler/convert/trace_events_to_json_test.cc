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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_json_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_json_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_json_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/trace_events_to_json.h"

#include <string>

#include "json/json.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

std::string ConvertTextFormattedTraceToJson(const std::string& trace_str) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("trace_str: \"" + trace_str + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_json_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/profiler/convert/trace_events_to_json_test.cc", "ConvertTextFormattedTraceToJson");

  Trace trace;
  EXPECT_TRUE(protobuf::TextFormat::ParseFromString(trace_str, &trace));
  return TraceEventsToJson(trace);
}

Json::Value ToJsonValue(const std::string& json_str) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("json_str: \"" + json_str + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_json_testDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/profiler/convert/trace_events_to_json_test.cc", "ToJsonValue");

  Json::Value json;
  Json::Reader reader;
  EXPECT_TRUE(reader.parse(json_str, json));
  return json;
}

TEST(TraceEventsToJson, JsonConversion) {
  std::string json_output = ConvertTextFormattedTraceToJson(R"proto(
    devices {
      key: 2
      value {
        name: 'D2'
        device_id: 2
        resources {
          key: 2
          value { resource_id: 2 name: 'R2.2' }
        }
      }
    }
    devices {
      key: 1
      value {
        name: 'D1'
        device_id: 1
        resources {
          key: 2
          value { resource_id: 1 name: 'R1.2' }
        }
      }
    }
    trace_events {
      device_id: 1
      resource_id: 2
      name: 'E1.2.1'
      timestamp_ps: 100000
      duration_ps: 10000
      args { key: 'long_name' value: 'E1.2.1 long' }
      args { key: 'arg2' value: 'arg2 val' }
    }
    trace_events {
      device_id: 2
      resource_id: 2
      name: 'E2.2.1 # "comment"'
      timestamp_ps: 105000
    }
  )proto");
  Json::Value json = ToJsonValue(json_output);

  Json::Value expected_json = ToJsonValue(R"(
  {
    "displayTimeUnit": "ns",
    "metadata": { "highres-ticks": true },
    "traceEvents": [
      {"ph":"M", "pid":1, "name":"process_name", "args":{"name":"D1"}},
      {"ph":"M", "pid":1, "name":"process_sort_index", "args":{"sort_index":1}},
      {"ph":"M", "pid":1, "tid":2, "name":"thread_name",
       "args":{"name":"R1.2"}},
      {"ph":"M", "pid":1, "tid":2, "name":"thread_sort_index",
       "args":{"sort_index":2}},
      {"ph":"M", "pid":2, "name":"process_name", "args":{"name":"D2"}},
      {"ph":"M", "pid":2, "name":"process_sort_index", "args":{"sort_index":2}},
      {"ph":"M", "pid":2, "tid":2, "name":"thread_name",
       "args":{"name":"R2.2"}},
      {"ph":"M", "pid":2, "tid":2, "name":"thread_sort_index",
       "args":{"sort_index":2}},
      {
        "ph" : "X",
        "pid" : 1,
        "tid" : 2,
        "name" : "E1.2.1",
        "ts" : 0.1,
        "dur" : 0.01,
        "args" : {"arg2": "arg2 val", "long_name": "E1.2.1 long"}
      },
      {
        "ph" : "X",
        "pid" : 2,
        "tid" : 2,
        "name" : "E2.2.1 # \"comment\"",
        "ts" : 0.105,
        "dur" : 1e-6
      },
      {}
    ]
  })");

  EXPECT_EQ(json, expected_json);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
