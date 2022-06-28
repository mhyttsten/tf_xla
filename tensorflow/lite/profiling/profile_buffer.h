/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_PROFILING_PROFILE_BUFFER_H_
#define TENSORFLOW_LITE_PROFILING_PROFILE_BUFFER_H_
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
class MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh {
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
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh() {
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


#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace profiling {

constexpr uint32_t kInvalidEventHandle = static_cast<uint32_t>(~0) - 1;

// A profiling event.
struct ProfileEvent {
  // Describes the type of event.
  // The event_metadata field may contain additional data for interpreting
  // the event.
  using EventType = tflite::Profiler::EventType;

  // Label of the event. This usually describes the event.
  std::string tag;
  // Timestamp in microseconds when the event began.
  uint64_t begin_timestamp_us;
  // Timestamp in microseconds when the event ended.
  uint64_t end_timestamp_us;

  // The memory usage when the event begins.
  memory::MemoryUsage begin_mem_usage;
  // The memory usage when the event ends.
  memory::MemoryUsage end_mem_usage;

  // The field containing the type of event. This must be one of the event types
  // in EventType.
  EventType event_type;
  // Meta data associated w/ the event.
  int64_t event_metadata;
  // Note: if this is an OPERATOR_INVOKE_EVENT, 'extra_event_metadata' will
  // represent the index of the subgraph that this event comes from.
  int64_t extra_event_metadata;
};

// A buffer of profile events. In general, the buffer works like a ring buffer.
// However, when 'allow_dynamic_expansion' is set, a unlimitted number of buffer
// entries is allowed and more profiling overhead could occur.
// This class is *not thread safe*.
class ProfileBuffer {
 public:
  ProfileBuffer(uint32_t max_num_entries, bool enabled,
                bool allow_dynamic_expansion = false)
      : enabled_(enabled),
        current_index_(0),
        event_buffer_(max_num_entries),
        allow_dynamic_expansion_(allow_dynamic_expansion) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh mht_0(mht_0_v, 242, "", "./tensorflow/lite/profiling/profile_buffer.h", "ProfileBuffer");
}

  // Adds an event to the buffer with begin timestamp set to the current
  // timestamp. Returns a handle to event that can be used to call EndEvent. If
  // buffer is disabled this has no affect.
  // The tag of the event should remain valid till the buffer is valid.
  uint32_t BeginEvent(const char* tag, ProfileEvent::EventType event_type,
                      int64_t event_metadata1, int64_t event_metadata2) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh mht_1(mht_1_v, 253, "", "./tensorflow/lite/profiling/profile_buffer.h", "BeginEvent");

    if (!enabled_) {
      return kInvalidEventHandle;
    }
    uint64_t timestamp = time::NowMicros();
    const auto next_index = GetNextEntryIndex();
    if (next_index.second) {
      return next_index.first;
    }
    const int index = next_index.first;
    event_buffer_[index].tag = tag;
    event_buffer_[index].event_type = event_type;
    event_buffer_[index].event_metadata = event_metadata1;
    event_buffer_[index].extra_event_metadata = event_metadata2;
    event_buffer_[index].begin_timestamp_us = timestamp;
    event_buffer_[index].end_timestamp_us = 0;
    if (event_type != Profiler::EventType::OPERATOR_INVOKE_EVENT) {
      event_buffer_[index].begin_mem_usage = memory::GetMemoryUsage();
    }
    current_index_++;
    return index;
  }

  // Sets the enabled state of buffer to |enabled|
  void SetEnabled(bool enabled) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh mht_2(mht_2_v, 280, "", "./tensorflow/lite/profiling/profile_buffer.h", "SetEnabled");
 enabled_ = enabled; }

  // Sets the end timestamp for event for the handle to current time.
  // If the buffer is disabled or previous event has been overwritten this
  // operation has not effect.
  void EndEvent(uint32_t event_handle, const int64_t* event_metadata1 = nullptr,
                const int64_t* event_metadata2 = nullptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh mht_3(mht_3_v, 289, "", "./tensorflow/lite/profiling/profile_buffer.h", "EndEvent");

    if (!enabled_ || event_handle == kInvalidEventHandle ||
        event_handle > current_index_) {
      return;
    }
    const uint32_t max_size = event_buffer_.size();
    if (current_index_ > (max_size + event_handle)) {
      // Ignore, buffer has already overflowed.
      fprintf(stderr, "Warning: Dropping ProfileBuffer event.\n");
      return;
    }

    int event_index = event_handle % max_size;
    event_buffer_[event_index].end_timestamp_us = time::NowMicros();
    if (event_buffer_[event_index].event_type !=
        Profiler::EventType::OPERATOR_INVOKE_EVENT) {
      event_buffer_[event_index].end_mem_usage = memory::GetMemoryUsage();
    }
    if (event_metadata1) {
      event_buffer_[event_index].event_metadata = *event_metadata1;
    }
    if (event_metadata2) {
      event_buffer_[event_index].extra_event_metadata = *event_metadata2;
    }
  }

  void AddEvent(const char* tag, ProfileEvent::EventType event_type,
                uint64_t start, uint64_t end, int64_t event_metadata1,
                int64_t event_metadata2) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh mht_4(mht_4_v, 321, "", "./tensorflow/lite/profiling/profile_buffer.h", "AddEvent");

    if (!enabled_) {
      return;
    }
    const auto next_index = GetNextEntryIndex();
    if (next_index.second) {
      return;
    }
    const int index = next_index.first;
    event_buffer_[index].tag = tag;
    event_buffer_[index].event_type = event_type;
    event_buffer_[index].event_metadata = event_metadata1;
    event_buffer_[index].extra_event_metadata = event_metadata2;
    event_buffer_[index].begin_timestamp_us = start;
    event_buffer_[index].end_timestamp_us = end;
    current_index_++;
  }

  // Returns the size of the buffer.
  size_t Size() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh mht_5(mht_5_v, 343, "", "./tensorflow/lite/profiling/profile_buffer.h", "Size");

    return (current_index_ >= event_buffer_.size()) ? event_buffer_.size()
                                                    : current_index_;
  }

  // Resets the buffer.
  void Reset() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh mht_6(mht_6_v, 352, "", "./tensorflow/lite/profiling/profile_buffer.h", "Reset");

    enabled_ = false;
    current_index_ = 0;
  }

  // Returns the profile event at the given index. If the index is invalid a
  // nullptr is returned. The return event may get overwritten if more events
  // are added to buffer.
  const struct ProfileEvent* At(size_t index) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_bufferDTh mht_7(mht_7_v, 363, "", "./tensorflow/lite/profiling/profile_buffer.h", "At");

    size_t size = Size();
    if (index >= size) {
      return nullptr;
    }
    const uint32_t max_size = event_buffer_.size();
    uint32_t start =
        (current_index_ > max_size) ? current_index_ % max_size : max_size;
    index = (index + start) % max_size;
    return &event_buffer_[index];
  }

 private:
  // Returns a pair of values. The 1st element refers to the next buffer id,
  // the 2nd element refers to whether the buffer reaches its allowed capacity.
  std::pair<int, bool> GetNextEntryIndex() {
    int index = current_index_ % event_buffer_.size();
    if (current_index_ == 0 || index != 0) {
      return std::make_pair(index, false);
    }

    // Current buffer is full
    if (!allow_dynamic_expansion_) {
      fprintf(stderr, "Warning: Dropping ProfileBuffer event.\n");
      return std::make_pair(current_index_, true);
    } else {
      fprintf(stderr, "Warning: Doubling internal profiling buffer.\n");
      event_buffer_.resize(current_index_ * 2);
      return std::make_pair(current_index_, false);
    }
  }

  bool enabled_;
  uint32_t current_index_;
  std::vector<ProfileEvent> event_buffer_;
  const bool allow_dynamic_expansion_;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_PROFILE_BUFFER_H_
