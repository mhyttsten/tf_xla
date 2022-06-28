/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_SCOPED_MEMORY_DEBUG_ANNOTATION_H_
#define TENSORFLOW_CORE_PROFILER_LIB_SCOPED_MEMORY_DEBUG_ANNOTATION_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_memory_debug_annotationDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_memory_debug_annotationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_memory_debug_annotationDTh() {
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
#include <functional>
#include <string>
#include <utility>

namespace tensorflow {
namespace profiler {

// Annotations for memory profiling and debugging purpose.
// ScopedMemoryDebugAnnotation will cache the annotations in thread-local
// memory, and some allocators will try to tag allocations with the annotations.
struct MemoryDebugAnnotation {
  const char* pending_op_name = nullptr;
  int64_t pending_step_id = 0;
  const char* pending_region_type = nullptr;
  int32_t pending_data_type = 0;
  // A lambda function, when invoked, it will generate the string that describe
  // the shape of the pending tensor. By default, the TensorShape string is an
  // empty string.
  std::function<std::string()> pending_shape_func = []() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_memory_debug_annotationDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h", "lambda");
 return ""; };
};

// Wrapper class of MemoryDebugAnnotation for RAII.
class ScopedMemoryDebugAnnotation {
 public:
  static const MemoryDebugAnnotation& CurrentAnnotation() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_memory_debug_annotationDTh mht_1(mht_1_v, 215, "", "./tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h", "CurrentAnnotation");

    return *ThreadMemoryDebugAnnotation();
  }

  explicit ScopedMemoryDebugAnnotation(const char* op_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_memory_debug_annotationDTh mht_2(mht_2_v, 223, "", "./tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h", "ScopedMemoryDebugAnnotation");

    MemoryDebugAnnotation* thread_local_annotation =
        ThreadMemoryDebugAnnotation();
    last_annotation_ = *thread_local_annotation;
    *thread_local_annotation = MemoryDebugAnnotation();
    thread_local_annotation->pending_op_name = op_name;
  }

  explicit ScopedMemoryDebugAnnotation(const char* op_name, int64_t step_id) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_memory_debug_annotationDTh mht_3(mht_3_v, 235, "", "./tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h", "ScopedMemoryDebugAnnotation");

    MemoryDebugAnnotation* thread_local_annotation =
        ThreadMemoryDebugAnnotation();
    last_annotation_ = *thread_local_annotation;
    *thread_local_annotation = MemoryDebugAnnotation();
    thread_local_annotation->pending_op_name = op_name;
    thread_local_annotation->pending_step_id = step_id;
  }

  // This constructor keeps the pending_op_name and pending_step_id from parent
  // (if any).  Otherwise it overwrites with op_name.
  explicit ScopedMemoryDebugAnnotation(
      const char* op_name, const char* region_type, int32_t data_type,
      std::function<std::string()>&& pending_shape_func) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   mht_4_v.push_back("region_type: \"" + (region_type == nullptr ? std::string("nullptr") : std::string((char*)region_type)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_memory_debug_annotationDTh mht_4(mht_4_v, 253, "", "./tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h", "ScopedMemoryDebugAnnotation");

    MemoryDebugAnnotation* thread_local_annotation =
        ThreadMemoryDebugAnnotation();
    last_annotation_ = *thread_local_annotation;
    if (!thread_local_annotation->pending_op_name) {
      thread_local_annotation->pending_op_name = op_name;
    }
    thread_local_annotation->pending_region_type = region_type;
    thread_local_annotation->pending_data_type = data_type;
    thread_local_annotation->pending_shape_func = std::move(pending_shape_func);
  }

  explicit ScopedMemoryDebugAnnotation(
      const char* op_name, int64_t step_id, const char* region_type,
      int32_t data_type, std::function<std::string()>&& pending_shape_func) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   mht_5_v.push_back("region_type: \"" + (region_type == nullptr ? std::string("nullptr") : std::string((char*)region_type)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_memory_debug_annotationDTh mht_5(mht_5_v, 272, "", "./tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h", "ScopedMemoryDebugAnnotation");

    MemoryDebugAnnotation* thread_local_annotation =
        ThreadMemoryDebugAnnotation();
    last_annotation_ = *thread_local_annotation;
    thread_local_annotation->pending_op_name = op_name;
    thread_local_annotation->pending_step_id = step_id;
    thread_local_annotation->pending_region_type = region_type;
    thread_local_annotation->pending_data_type = data_type;
    thread_local_annotation->pending_shape_func = std::move(pending_shape_func);
  }

  ~ScopedMemoryDebugAnnotation() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_memory_debug_annotationDTh mht_6(mht_6_v, 286, "", "./tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h", "~ScopedMemoryDebugAnnotation");

    *ThreadMemoryDebugAnnotation() = last_annotation_;
  }

 private:
  // Returns a pointer to the MemoryDebugAnnotation for the current thread.
  static MemoryDebugAnnotation* ThreadMemoryDebugAnnotation();

  // Stores the previous values in case the annotations are nested.
  MemoryDebugAnnotation last_annotation_;

  ScopedMemoryDebugAnnotation(const ScopedMemoryDebugAnnotation&) = delete;
  ScopedMemoryDebugAnnotation& operator=(const ScopedMemoryDebugAnnotation&) =
      delete;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_SCOPED_MEMORY_DEBUG_ANNOTATION_H_
