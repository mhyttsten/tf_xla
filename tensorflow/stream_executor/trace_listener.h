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

// This file defines the StreamExecutor trace listener, used for inserting
// non-device-specific instrumentation into the StreamExecutor.
#ifndef TENSORFLOW_STREAM_EXECUTOR_TRACE_LISTENER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TRACE_LISTENER_H_
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
class MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh() {
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


#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/launch_dim.h"
#include "tensorflow/stream_executor/lib/status.h"

namespace stream_executor {

class Stream;

// Traces StreamExecutor PIMPL-level events.
// The few StreamExecutor interfaces that are synchronous have both Begin and
// Complete versions of their trace calls. Asynchronous operations only have
// Submit calls, as execution of the underlying operations is device-specific.
// As all tracing calls mirror StreamExecutor routines, documentation here is
// minimal.
//
// All calls have default implementations that perform no work; subclasses
// should override functionality of interest. Keep in mind that these routines
// are not called on a dedicated thread, so callbacks should execute quickly.
//
// Note: This API is constructed on an as-needed basis. Users should add
// support for further StreamExecutor operations as required. By enforced
// convention (see SCOPED_TRACE in stream_executor_pimpl.cc), synchronous
// tracepoints should be named NameBegin and NameComplete.
class TraceListener {
 public:
  virtual ~TraceListener() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh mht_0(mht_0_v, 216, "", "./tensorflow/stream_executor/trace_listener.h", "~TraceListener");
}

  virtual void LaunchSubmit(Stream* stream, const ThreadDim& thread_dims,
                            const BlockDim& block_dims,
                            const KernelBase& kernel,
                            const KernelArgsArrayBase& args) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh mht_1(mht_1_v, 224, "", "./tensorflow/stream_executor/trace_listener.h", "LaunchSubmit");
}

  virtual void SynchronousMemcpyH2DBegin(int64_t correlation_id,
                                         const void* host_src, int64_t size,
                                         DeviceMemoryBase* gpu_dst) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh mht_2(mht_2_v, 231, "", "./tensorflow/stream_executor/trace_listener.h", "SynchronousMemcpyH2DBegin");
}
  virtual void SynchronousMemcpyH2DComplete(int64_t correlation_id,
                                            const port::Status* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh mht_3(mht_3_v, 236, "", "./tensorflow/stream_executor/trace_listener.h", "SynchronousMemcpyH2DComplete");
}

  virtual void SynchronousMemcpyD2HBegin(int64_t correlation_id,
                                         const DeviceMemoryBase& gpu_src,
                                         int64_t size, void* host_dst) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh mht_4(mht_4_v, 243, "", "./tensorflow/stream_executor/trace_listener.h", "SynchronousMemcpyD2HBegin");
}
  virtual void SynchronousMemcpyD2HComplete(int64_t correlation_id,
                                            const port::Status* result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh mht_5(mht_5_v, 248, "", "./tensorflow/stream_executor/trace_listener.h", "SynchronousMemcpyD2HComplete");
}

  virtual void BlockHostUntilDoneBegin(int64_t correlation_id, Stream* stream) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh mht_6(mht_6_v, 253, "", "./tensorflow/stream_executor/trace_listener.h", "BlockHostUntilDoneBegin");

  }
  virtual void BlockHostUntilDoneComplete(int64_t correlation_id,
                                          const port::Status* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPStrace_listenerDTh mht_7(mht_7_v, 259, "", "./tensorflow/stream_executor/trace_listener.h", "BlockHostUntilDoneComplete");
}
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_TRACE_LISTENER_H_
