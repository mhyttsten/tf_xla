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
#ifndef TENSORFLOW_LITE_PROFILING_MEMORY_INFO_H_
#define TENSORFLOW_LITE_PROFILING_MEMORY_INFO_H_
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
class MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_infoDTh {
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
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_infoDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_infoDTh() {
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
#include <sstream>

namespace tflite {
namespace profiling {
namespace memory {

struct MemoryUsage {
  static const size_t kValueNotSet;

  // Indicates whether obtaining memory usage is supported on the platform, thus
  // indicating whether the values defined in this struct make sense or not.
  static bool IsSupported();

  MemoryUsage()
      : max_rss_kb(kValueNotSet),
        total_allocated_bytes(kValueNotSet),
        in_use_allocated_bytes(kValueNotSet) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_infoDTh mht_0(mht_0_v, 204, "", "./tensorflow/lite/profiling/memory_info.h", "MemoryUsage");
}

  // The maximum memory size (in kilobytes) occupied by an OS process that is
  // held in main memory (RAM). Such memory usage information is generally
  // referred as resident set size (rss). This is an alias to rusage::ru_maxrss.
  int64_t max_rss_kb;

  // Total non-mmapped space allocated from system in bytes. This is an alias to
  // mallinfo::arena.
  size_t total_allocated_bytes;

  // Total allocated (including mmapped) bytes that's in use (i.e. excluding
  // those are freed). This is an alias to mallinfo::uordblks.
  size_t in_use_allocated_bytes;

  MemoryUsage operator+(MemoryUsage const& obj) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_infoDTh mht_1(mht_1_v, 222, "", "./tensorflow/lite/profiling/memory_info.h", "+");

    MemoryUsage res;
    res.max_rss_kb = max_rss_kb + obj.max_rss_kb;
    res.total_allocated_bytes =
        total_allocated_bytes + obj.total_allocated_bytes;
    res.in_use_allocated_bytes =
        in_use_allocated_bytes + obj.in_use_allocated_bytes;
    return res;
  }

  MemoryUsage operator-(MemoryUsage const& obj) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_infoDTh mht_2(mht_2_v, 235, "", "./tensorflow/lite/profiling/memory_info.h", "-");

    MemoryUsage res;
    res.max_rss_kb = max_rss_kb - obj.max_rss_kb;
    res.total_allocated_bytes =
        total_allocated_bytes - obj.total_allocated_bytes;
    res.in_use_allocated_bytes =
        in_use_allocated_bytes - obj.in_use_allocated_bytes;
    return res;
  }

  void AllStatsToStream(std::ostream* stream) const;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const MemoryUsage& obj) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_infoDTh mht_3(mht_3_v, 251, "", "./tensorflow/lite/profiling/memory_info.h", "operator<<");

    obj.AllStatsToStream(&stream);
    return stream;
  }
};

// Return the memory usage from the system.
// Note: this currently only works on Linux-based systems. Support on other
// systems will be added later.
MemoryUsage GetMemoryUsage();

}  // namespace memory
}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_MEMORY_INFO_H_
