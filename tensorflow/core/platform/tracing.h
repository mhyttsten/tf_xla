/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_TRACING_H_
#define TENSORFLOW_CORE_PLATFORM_TRACING_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPStracingDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPStracingDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPStracingDTh() {
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


// Tracing interface

#include <array>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tracing {

// This enumeration contains the identifiers of all TensorFlow CPU profiler
// events. It must be kept in sync with the code in GetEventCategoryName().
enum struct EventCategory : unsigned {
  kScheduleClosure = 0,
  kRunClosure = 1,
  kCompute = 2,
  kNumCategories = 3  // sentinel - keep last
};
constexpr unsigned GetNumEventCategories() {
  return static_cast<unsigned>(EventCategory::kNumCategories);
}
const char* GetEventCategoryName(EventCategory);

// Interface for CPU profiler events.
class EventCollector {
 public:
  virtual ~EventCollector() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPStracingDTh mht_0(mht_0_v, 216, "", "./tensorflow/core/platform/tracing.h", "~EventCollector");
}
  virtual void RecordEvent(uint64 arg) const = 0;
  virtual void StartRegion(uint64 arg) const = 0;
  virtual void StopRegion() const = 0;

  // Annotates the current thread with a name.
  static void SetCurrentThreadName(const char* name);
  // Returns whether event collection is enabled.
  static bool IsEnabled();

 private:
  friend void SetEventCollector(EventCategory, const EventCollector*);
  friend const EventCollector* GetEventCollector(EventCategory);

  static std::array<const EventCollector*, GetNumEventCategories()> instances_;
};
// Set the callback for RecordEvent and ScopedRegion of category.
// Not thread safe. Only call while EventCollector::IsEnabled returns false.
void SetEventCollector(EventCategory category, const EventCollector* collector);

// Returns the callback for RecordEvent and ScopedRegion of category if
// EventCollector::IsEnabled(), otherwise returns null.
inline const EventCollector* GetEventCollector(EventCategory category) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPStracingDTh mht_1(mht_1_v, 241, "", "./tensorflow/core/platform/tracing.h", "GetEventCollector");

  if (EventCollector::IsEnabled()) {
    return EventCollector::instances_[static_cast<unsigned>(category)];
  }
  return nullptr;
}

// Returns a unique id to pass to RecordEvent/ScopedRegion. Never returns zero.
uint64 GetUniqueArg();

// Returns an id for name to pass to RecordEvent/ScopedRegion.
uint64 GetArgForName(StringPiece name);

// Records an atomic event through the currently registered EventCollector.
inline void RecordEvent(EventCategory category, uint64 arg) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPStracingDTh mht_2(mht_2_v, 258, "", "./tensorflow/core/platform/tracing.h", "RecordEvent");

  if (auto collector = GetEventCollector(category)) {
    collector->RecordEvent(arg);
  }
}

// Records an event for the duration of the instance lifetime through the
// currently registered EventCollector.
class ScopedRegion {
 public:
  ScopedRegion(ScopedRegion&& other) noexcept  // Move-constructible.
      : collector_(other.collector_) {
    other.collector_ = nullptr;
  }

  ScopedRegion(EventCategory category, uint64 arg)
      : collector_(GetEventCollector(category)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPStracingDTh mht_3(mht_3_v, 277, "", "./tensorflow/core/platform/tracing.h", "ScopedRegion");

    if (collector_) {
      collector_->StartRegion(arg);
    }
  }

  // Same as ScopedRegion(category, GetUniqueArg()), but faster if
  // EventCollector::IsEnabled() returns false.
  explicit ScopedRegion(EventCategory category)
      : collector_(GetEventCollector(category)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPStracingDTh mht_4(mht_4_v, 289, "", "./tensorflow/core/platform/tracing.h", "ScopedRegion");

    if (collector_) {
      collector_->StartRegion(GetUniqueArg());
    }
  }

  // Same as ScopedRegion(category, GetArgForName(name)), but faster if
  // EventCollector::IsEnabled() returns false.
  ScopedRegion(EventCategory category, StringPiece name)
      : collector_(GetEventCollector(category)) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPStracingDTh mht_5(mht_5_v, 301, "", "./tensorflow/core/platform/tracing.h", "ScopedRegion");

    if (collector_) {
      collector_->StartRegion(GetArgForName(name));
    }
  }

  ~ScopedRegion() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPStracingDTh mht_6(mht_6_v, 310, "", "./tensorflow/core/platform/tracing.h", "~ScopedRegion");

    if (collector_) {
      collector_->StopRegion();
    }
  }

  bool IsEnabled() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPStracingDTh mht_7(mht_7_v, 319, "", "./tensorflow/core/platform/tracing.h", "IsEnabled");
 return collector_ != nullptr; }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ScopedRegion);

  const EventCollector* collector_;
};

// Return the pathname of the directory where we are writing log files.
const char* GetLogDir();

}  // namespace tracing
}  // namespace tensorflow

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/tracing_impl.h"
#else
#include "tensorflow/core/platform/default/tracing_impl.h"
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_TRACING_H_
