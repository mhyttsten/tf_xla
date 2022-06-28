/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_CORE_API_PROFILER_H_
#define TENSORFLOW_LITE_CORE_API_PROFILER_H_
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
class MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh {
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
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh() {
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


#include <cstdint>

namespace tflite {

// A simple utility for enabling profiled event tracing in TensorFlow Lite.
class Profiler {
 public:
  // As certain Profiler instance might be only interested in certain event
  // types, we define each event type value to allow a Profiler to use
  // bitmasking bitwise operations to determine whether an event should be
  // recorded or not.
  enum class EventType {
    // Default event type, the metadata field has no special significance.
    DEFAULT = 1,

    // The event is an operator invocation and the event_metadata field is the
    // index of operator node.
    OPERATOR_INVOKE_EVENT = 2,

    // The event is an invocation for an internal operator of a TFLite delegate.
    // The event_metadata field is the index of operator node that's specific to
    // the delegate.
    DELEGATE_OPERATOR_INVOKE_EVENT = 4,

    // The event is a recording of runtime instrumentation such as the overall
    // TFLite runtime status, the TFLite delegate status (if a delegate
    // is applied), and the overall model inference latency etc.
    // Note, the delegate status and overall status are stored as separate
    // event_metadata fields. In particular, the delegate status is encoded
    // as DelegateStatus::full_status().
    GENERAL_RUNTIME_INSTRUMENTATION_EVENT = 8,
  };

  virtual ~Profiler() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_0(mht_0_v, 220, "", "./tensorflow/lite/core/api/profiler.h", "~Profiler");
}

  // Signals the beginning of an event and returns a handle to the profile
  // event. The `event_metadata1` and `event_metadata2` have different
  // interpretations based on the actual Profiler instance and the `event_type`.
  // For example, as for the 'SubgraphAwareProfiler' defined in
  // lite/core/subgraph.h, when the event_type is OPERATOR_INVOKE_EVENT,
  // `event_metadata1` represents the index of a TFLite node, and
  // `event_metadata2` represents the index of the subgraph that this event
  // comes from.
  virtual uint32_t BeginEvent(const char* tag, EventType event_type,
                              int64_t event_metadata1,
                              int64_t event_metadata2) = 0;
  // Similar w/ the above, but `event_metadata2` defaults to 0.
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_1(mht_1_v, 239, "", "./tensorflow/lite/core/api/profiler.h", "BeginEvent");

    return BeginEvent(tag, event_type, event_metadata, /*event_metadata2*/ 0);
  }

  // Signals an end to the specified profile event with 'event_metadata's, This
  // is useful when 'event_metadata's are not available when the event begins
  // or when one wants to overwrite the 'event_metadata's set at the beginning.
  virtual void EndEvent(uint32_t event_handle, int64_t event_metadata1,
                        int64_t event_metadata2) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_2(mht_2_v, 250, "", "./tensorflow/lite/core/api/profiler.h", "EndEvent");
}
  // Signals an end to the specified profile event.
  virtual void EndEvent(uint32_t event_handle) = 0;

  // Appends an event of type 'event_type' with 'tag' and 'event_metadata'
  // which started at 'start' and ended at 'end'
  // Note:
  // In cases were ProfileSimmarizer and tensorflow::StatsCalculator are used
  // they assume the value is in "usec", if in any case subclasses
  // didn't put usec, then the values are not meaningful.
  // TODO karimnosseir: Revisit and make the function more clear.
  void AddEvent(const char* tag, EventType event_type, uint64_t start,
                uint64_t end, int64_t event_metadata) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_3(mht_3_v, 266, "", "./tensorflow/lite/core/api/profiler.h", "AddEvent");

    AddEvent(tag, event_type, start, end, event_metadata,
             /*event_metadata2*/ 0);
  }

  virtual void AddEvent(const char* tag, EventType event_type, uint64_t start,
                        uint64_t end, int64_t event_metadata1,
                        int64_t event_metadata2) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_4(mht_4_v, 277, "", "./tensorflow/lite/core/api/profiler.h", "AddEvent");
}

 protected:
  friend class ScopedProfile;
};

// Adds a profile event to `profiler` that begins with the construction
// of the object and ends when the object goes out of scope.
// The lifetime of tag should be at least the lifetime of `profiler`.
// `profiler` may be null, in which case nothing is profiled.
class ScopedProfile {
 public:
  ScopedProfile(Profiler* profiler, const char* tag,
                Profiler::EventType event_type = Profiler::EventType::DEFAULT,
                int64_t event_metadata = 0)
      : profiler_(profiler), event_handle_(0) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_5(mht_5_v, 296, "", "./tensorflow/lite/core/api/profiler.h", "ScopedProfile");

    if (profiler) {
      event_handle_ = profiler_->BeginEvent(tag, event_type, event_metadata);
    }
  }

  ~ScopedProfile() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_6(mht_6_v, 305, "", "./tensorflow/lite/core/api/profiler.h", "~ScopedProfile");

    if (profiler_) {
      profiler_->EndEvent(event_handle_);
    }
  }

 protected:
  Profiler* profiler_;
  uint32_t event_handle_;
};

class ScopedOperatorProfile : public ScopedProfile {
 public:
  ScopedOperatorProfile(Profiler* profiler, const char* tag, int node_index)
      : ScopedProfile(profiler, tag, Profiler::EventType::OPERATOR_INVOKE_EVENT,
                      static_cast<uint32_t>(node_index)) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_7(mht_7_v, 324, "", "./tensorflow/lite/core/api/profiler.h", "ScopedOperatorProfile");
}
};

class ScopedDelegateOperatorProfile : public ScopedProfile {
 public:
  ScopedDelegateOperatorProfile(Profiler* profiler, const char* tag,
                                int node_index)
      : ScopedProfile(profiler, tag,
                      Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT,
                      static_cast<uint32_t>(node_index)) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_8(mht_8_v, 337, "", "./tensorflow/lite/core/api/profiler.h", "ScopedDelegateOperatorProfile");
}
};

class ScopedRuntimeInstrumentationProfile : public ScopedProfile {
 public:
  ScopedRuntimeInstrumentationProfile(Profiler* profiler, const char* tag)
      : ScopedProfile(
            profiler, tag,
            Profiler::EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT, -1) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_9(mht_9_v, 349, "", "./tensorflow/lite/core/api/profiler.h", "ScopedRuntimeInstrumentationProfile");
}

  void set_runtime_status(int64_t delegate_status, int64_t interpreter_status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_10(mht_10_v, 354, "", "./tensorflow/lite/core/api/profiler.h", "set_runtime_status");

    if (profiler_) {
      delegate_status_ = delegate_status;
      interpreter_status_ = interpreter_status;
    }
  }

  ~ScopedRuntimeInstrumentationProfile() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSprofilerDTh mht_11(mht_11_v, 364, "", "./tensorflow/lite/core/api/profiler.h", "~ScopedRuntimeInstrumentationProfile");

    if (profiler_) {
      profiler_->EndEvent(event_handle_, delegate_status_, interpreter_status_);
    }
  }

 private:
  int64_t delegate_status_;
  int64_t interpreter_status_;
};

}  // namespace tflite

#define TFLITE_VARNAME_UNIQ_IMPL(name, ctr) name##ctr
#define TFLITE_VARNAME_UNIQ(name, ctr) TFLITE_VARNAME_UNIQ_IMPL(name, ctr)

#define TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE(profiler, tag)          \
  tflite::ScopedProfile TFLITE_VARNAME_UNIQ(_profile_, __COUNTER__)( \
      (profiler), (tag))

#define TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE(profiler, tag, node_index)     \
  tflite::ScopedOperatorProfile TFLITE_VARNAME_UNIQ(_profile_, __COUNTER__)( \
      (profiler), (tag), (node_index))

#define TFLITE_SCOPED_DELEGATE_OPERATOR_PROFILE(profiler, tag, node_index) \
  tflite::ScopedDelegateOperatorProfile TFLITE_VARNAME_UNIQ(               \
      _profile_, __COUNTER__)((profiler), (tag), (node_index))

#define TFLITE_ADD_RUNTIME_INSTRUMENTATION_EVENT(                          \
    profiler, tag, event_metadata1, event_metadata2)                       \
  do {                                                                     \
    if (profiler) {                                                        \
      const auto handle = profiler->BeginEvent(                            \
          tag, Profiler::EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT, \
          event_metadata1, event_metadata2);                               \
      profiler->EndEvent(handle);                                          \
    }                                                                      \
  } while (false);

#endif  // TENSORFLOW_LITE_CORE_API_PROFILER_H_
