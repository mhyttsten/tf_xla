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

#ifndef TENSORFLOW_PYTHON_UTIL_STACK_TRACE_H_
#define TENSORFLOW_PYTHON_UTIL_STACK_TRACE_H_
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
class MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh {
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
   MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh() {
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


#include <Python.h>
#include <frameobject.h>

#include <array>
#include <limits>
#include <sstream>
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace tensorflow {

// Assert that Python GIL is held.
// TODO(cheshire): Fix duplication vs. py_util.h
inline void DCheckPyGilStateForStackTrace() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh mht_0(mht_0_v, 209, "", "./tensorflow/python/util/stack_trace.h", "DCheckPyGilStateForStackTrace");

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 4
  DCHECK(PyGILState_Check());
#endif
}

// A class for capturing Python stack trace.
class StackTrace final {
 public:
  static constexpr int kStackTraceInitialSize = 30;

  StackTrace() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh mht_1(mht_1_v, 223, "", "./tensorflow/python/util/stack_trace.h", "StackTrace");
}

  // Returns `StackTrace` object that captures the current Python stack trace.
  // `limit` determines how many stack frames at most are returned: set to -1
  // for "no limit".
  // Python GIL must be acquired beforehand.
  ABSL_MUST_USE_RESULT
  ABSL_ATTRIBUTE_HOT
  static StackTrace Capture(int limit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh mht_2(mht_2_v, 234, "", "./tensorflow/python/util/stack_trace.h", "Capture");

    DCheckPyGilStateForStackTrace();
    if (limit == -1) limit = std::numeric_limits<int>::max();

    StackTrace result;
    const PyFrameObject* frame = PyThreadState_GET()->frame;
    int i = 0;
    for (; i < limit && frame != nullptr; frame = frame->f_back, ++i) {
      PyCodeObject* code_obj = frame->f_code;
      DCHECK(code_obj != nullptr);

      Py_INCREF(code_obj);
      int line_number =
          PyFrame_GetLineNumber(const_cast<PyFrameObject*>(frame));
      result.code_objs_.push_back(std::make_pair(code_obj, line_number));
    }
    return result;
  }

  // Python GIL must be acquired beforehand.
  ABSL_ATTRIBUTE_HOT
  ~StackTrace() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh mht_3(mht_3_v, 258, "", "./tensorflow/python/util/stack_trace.h", "~StackTrace");
 Clear(); }

  StackTrace(StackTrace&& other) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh mht_4(mht_4_v, 263, "", "./tensorflow/python/util/stack_trace.h", "StackTrace");
 std::swap(code_objs_, other.code_objs_); }

  // Python GIL must be acquired beforehand.
  ABSL_ATTRIBUTE_HOT
  StackTrace& operator=(StackTrace&& other) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh mht_5(mht_5_v, 270, "", "./tensorflow/python/util/stack_trace.h", "=");

    Clear();
    std::swap(code_objs_, other.code_objs_);
    return *this;
  }

  // Returns a structured representation of the captured stack trace.
  // `source_map` provides a custom mapping for translating stack frames,
  // `filter` returns `true` for the stack frames which should be omitted.
  //
  // `reverse_traversal` changes the traversal order of the stack trace, and
  // `limit` bounds the number of returned frames (after filtering).
  std::vector<StackFrame> ToStackFrames(const SourceMap& source_map,
                                        const StackTraceFilter& filtered,
                                        bool reverse_traversal = false,
                                        int limit = -1) const;

  // Python GIL must be acquired beforehand.
  ABSL_ATTRIBUTE_HOT
  void Clear() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh mht_6(mht_6_v, 292, "", "./tensorflow/python/util/stack_trace.h", "Clear");

    if (!code_objs_.empty()) DCheckPyGilStateForStackTrace();
    for (const auto& p : code_objs_) Py_DECREF(p.first);
    code_objs_.clear();
  }

 private:
  absl::InlinedVector<std::pair<PyCodeObject*, int>, kStackTraceInitialSize>
      code_objs_;

  StackTrace(const StackTrace&) = delete;
  StackTrace& operator=(const StackTrace&) = delete;
};

// A class that manages Python stack traces in a circular buffer. Users can
// insert stack trace entries and retrive them by ids.
class StackTraceManager {
 public:
  static constexpr int kStackTraceCircularBufferSize = 1024;

  // Captures the current Python stack trace and returns an id.
  // Python GIL must be acquired beforehand.
  ABSL_MUST_USE_RESULT
  ABSL_ATTRIBUTE_HOT
  int Capture(int limit) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh mht_7(mht_7_v, 319, "", "./tensorflow/python/util/stack_trace.h", "Capture");

    DCheckPyGilStateForStackTrace();
    const int id = next_id_++;
    const int index = id & (kStackTraceCircularBufferSize - 1);
    stack_traces_[index] = StackTrace::Capture(limit);
    return id;
  }

  // Retrieve captured Python stack trace by id. Returns `nullptr` if the
  // requested stack trace is evicted from the circular buffer.
  // Python GIL must be acquired beforehand.
  ABSL_MUST_USE_RESULT
  StackTrace* Get(int id);

 private:
  int next_id_ = 0;
  std::array<StackTrace, kStackTraceCircularBufferSize> stack_traces_;
};

// Singleton StackTraceManager.
extern StackTraceManager* const stack_trace_manager;

// Converts the ManagedStackTrace (identified by ID) to a vector of stack
// frames.
inline std::vector<StackFrame> ManagedStackTraceToStackFrames(
    int id, const SourceMap& source_map, const StackTraceFilter& filtered,
    bool reverse_traversal, int limit) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  StackTrace* stack_trace = stack_trace_manager->Get(id);
  if (!stack_trace) {
    // Must have evicted the stack trace by now. Do best effort.
    return {};
  }

  std::vector<StackFrame> result = stack_trace->ToStackFrames(
      source_map, filtered, reverse_traversal, limit);
  PyGILState_Release(gstate);
  return result;
}

// Returns Python stack trace object that can be converted to string.
// Note that the actual stack trace is kept in a circular buffer for string
// conversion could fail if it's evicted before.
// Python GIL must be acquired beforehand.
inline ManagedStackTrace GetStackTrace(int limit) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSstack_traceDTh mht_8(mht_8_v, 366, "", "./tensorflow/python/util/stack_trace.h", "GetStackTrace");

  DCheckPyGilStateForStackTrace();
  return ManagedStackTrace(stack_trace_manager->Capture(limit),
                           &ManagedStackTraceToStackFrames);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_STACK_TRACE_H_
