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
class MHTracer_DTPStensorflowPSlitePSprofilingPSatrace_profilerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSprofilingPSatrace_profilerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSprofilingPSatrace_profilerDTcc() {
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
#include "tensorflow/lite/profiling/atrace_profiler.h"

#include <dlfcn.h>
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

#include <string>
#include <type_traits>

namespace tflite {
namespace profiling {

// Profiler reporting to ATrace.
class ATraceProfiler : public tflite::Profiler {
 public:
  using FpIsEnabled = std::add_pointer<bool()>::type;
  using FpBeginSection = std::add_pointer<void(const char*)>::type;
  using FpEndSection = std::add_pointer<void()>::type;

  ATraceProfiler() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSatrace_profilerDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/profiling/atrace_profiler.cc", "ATraceProfiler");

    handle_ = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
    if (handle_) {
      // Use dlsym() to prevent crashes on devices running Android 5.1
      // (API level 22) or lower.
      atrace_is_enabled_ =
          reinterpret_cast<FpIsEnabled>(dlsym(handle_, "ATrace_isEnabled"));
      atrace_begin_section_ = reinterpret_cast<FpBeginSection>(
          dlsym(handle_, "ATrace_beginSection"));
      atrace_end_section_ =
          reinterpret_cast<FpEndSection>(dlsym(handle_, "ATrace_endSection"));

      if (!atrace_is_enabled_ || !atrace_begin_section_ ||
          !atrace_end_section_) {
        dlclose(handle_);
        handle_ = nullptr;
      }
    }
  }

  ~ATraceProfiler() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSatrace_profilerDTcc mht_1(mht_1_v, 227, "", "./tensorflow/lite/profiling/atrace_profiler.cc", "~ATraceProfiler");

    if (handle_) {
      dlclose(handle_);
    }
  }

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePSprofilingPSatrace_profilerDTcc mht_2(mht_2_v, 239, "", "./tensorflow/lite/profiling/atrace_profiler.cc", "BeginEvent");

    if (handle_ && atrace_is_enabled_()) {
      // Note: When recording an OPERATOR_INVOKE_EVENT, we have recorded the op
      // name
      // as tag, node index as event_metadata1 and subgraph index as
      // event_metadata2. See the macro TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE
      // defined in tensorflow/lite/core/api/profiler.h for details.
      // Regardless the 'event_type', we encode the perfetto event name as
      // tag@event_metadata1/event_metadata2. In case of OPERATOR_INVOKE_EVENT,
      // the perfetto event name will be op_name@node_index/subgraph_index
      std::string trace_event_tag = tag;
      trace_event_tag += "@";
      trace_event_tag += std::to_string(event_metadata1) + "/" +
                         std::to_string(event_metadata2);
      atrace_begin_section_(trace_event_tag.c_str());
    }
    return 0;
  }

  void EndEvent(uint32_t event_handle) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSatrace_profilerDTcc mht_3(mht_3_v, 261, "", "./tensorflow/lite/profiling/atrace_profiler.cc", "EndEvent");

    if (handle_) {
      atrace_end_section_();
    }
  }

 private:
  // Handle to libandroid.so library. Null if not supported.
  void* handle_;
  FpIsEnabled atrace_is_enabled_;
  FpBeginSection atrace_begin_section_;
  FpEndSection atrace_end_section_;
};

std::unique_ptr<tflite::Profiler> MaybeCreateATraceProfiler() {
#if defined(TFLITE_ENABLE_DEFAULT_PROFILER)
  return std::unique_ptr<tflite::Profiler>(new ATraceProfiler());
#else  // TFLITE_ENABLE_DEFAULT_PROFILER
#if defined(__ANDROID__)
  constexpr char kTraceProp[] = "debug.tflite.trace";
  char trace_enabled[PROP_VALUE_MAX] = "";
  int length = __system_property_get(kTraceProp, trace_enabled);
  if (length == 1 && trace_enabled[0] == '1') {
    return std::unique_ptr<tflite::Profiler>(new ATraceProfiler());
  }
#endif  // __ANDROID__
  return nullptr;
#endif  // TFLITE_ENABLE_DEFAULT_PROFILER
}

}  // namespace profiling
}  // namespace tflite
