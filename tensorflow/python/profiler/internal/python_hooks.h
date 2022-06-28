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
#ifndef TENSORFLOW_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
#define TENSORFLOW_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
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
class MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh {
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
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh() {
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


#include <memory>
#include <stack>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

namespace py = ::pybind11;

struct PythonHooksOptions {
  bool enable_trace_python_function = false;
  bool enable_python_traceme = true;
  bool end_to_end_mode = false;
  // Incomplete events are defined as those python calls which we only see
  // either start or end, but not both. If we want to include them in the final
  // result, profiler start, end time are used respectively to the absent
  // timestamps.
  bool include_incomplete_events = true;
};

struct PythonTraceEntry {
  // Capture the source/line information for a PyCodeObject object.
  // In eager mode, keeping a reference to PyCodeObject leaks device memory.
  PythonTraceEntry(uint64 start, uint64 end, PyCodeObject* py_code_object)
      : start_time_ns(start),
        end_time_ns(end),
        co_filename(py_code_object->co_filename),
        co_name(py_code_object->co_name),
        co_firstlineno(py_code_object->co_firstlineno) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh mht_0(mht_0_v, 224, "", "./tensorflow/python/profiler/internal/python_hooks.h", "PythonTraceEntry");

    Py_XINCREF(co_filename);
    Py_XINCREF(co_name);
  }
  // Capture the source/line information for a PyCFunctionObject object.
  // In eager mode, keeping a reference to PyCFunctionObject leaks device
  // memory.
  PythonTraceEntry(uint64 start, uint64 end, PyCFunctionObject* py_c_function)
      : start_time_ns(start),
        end_time_ns(end),
        method_def(py_c_function->m_ml),
        m_module(py_c_function->m_module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh mht_1(mht_1_v, 238, "", "./tensorflow/python/profiler/internal/python_hooks.h", "PythonTraceEntry");

    Py_XINCREF(m_module);
  }

  ~PythonTraceEntry() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh mht_2(mht_2_v, 245, "", "./tensorflow/python/profiler/internal/python_hooks.h", "~PythonTraceEntry");

    Py_XDECREF(co_filename);
    Py_XDECREF(co_name);
    Py_XDECREF(m_module);
  }

  PythonTraceEntry(PythonTraceEntry&& other) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh mht_3(mht_3_v, 254, "", "./tensorflow/python/profiler/internal/python_hooks.h", "PythonTraceEntry");

    start_time_ns = other.start_time_ns;
    end_time_ns = other.end_time_ns;
    co_firstlineno = other.co_firstlineno;
    co_filename = other.co_filename;
    co_name = other.co_name;
    method_def = other.method_def;
    m_module = other.m_module;
    other.co_filename = nullptr;
    other.co_name = nullptr;
    other.method_def = nullptr;
    other.m_module = nullptr;
  }

  std::string Name() const;

  uint64 start_time_ns;
  uint64 end_time_ns;
  PyObject* co_filename = nullptr;
  PyObject* co_name = nullptr;
  int co_firstlineno = 0;
  PyMethodDef* method_def = nullptr;
  PyObject* m_module = nullptr;

  PythonTraceEntry(const PythonTraceEntry& other) = delete;
  void operator=(const PythonTraceEntry&) = delete;
  void operator=(PythonTraceEntry&&) = delete;
};

struct PerThreadEvents {
  std::deque<PythonTraceEntry> completed;
  std::stack<PythonTraceEntry> active;
};

class PythonHooks;

class PythonHookContext {
 public:
  void Finalize(XSpace* space);

  friend class ::tensorflow::profiler::PythonHooks;

 private:
  void Start(const PythonHooksOptions& option);
  void Stop();
  void ProfileFast(PyFrameObject* frame, int what, PyObject* arg);
  void CollectData(XPlane* raw_plane);
  static void EnableTraceMe(bool enable);

  static void SetProfilerInAllThreads();
  static void ClearProfilerInAllThreads();

  void operator=(const PythonHookContext&) = delete;
  void operator=(PythonHookContext&&) = delete;

  absl::flat_hash_map<int64_t, PerThreadEvents> entries_;
  uint64 start_timestamp_ns_;
  PythonHooksOptions options_;
  // In end to end mode, Python get uninitialized before Stop()/Finalize(), we
  // need to buffer the result.
  absl::optional<XPlane> end_to_end_xplane_;
};

// Singleton for tracing python function calls.
class PythonHooks {
 public:
  static PythonHooks* GetSingleton();

  void Start(const PythonHooksOptions& option) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh mht_4(mht_4_v, 325, "", "./tensorflow/python/profiler/internal/python_hooks.h", "Start");

    if (active_context_) return;
    active_context_ = std::make_unique<PythonHookContext>();
    active_context_->Start(option);
  }

  std::unique_ptr<PythonHookContext> Stop() {
    if (e2e_context_) {
      auto* e2e_context = e2e_context_;
      e2e_context_ = nullptr;
      return absl::WrapUnique(e2e_context);
    }

    if (!active_context_) return nullptr;
    active_context_->Stop();
    std::unique_ptr<PythonHookContext> output = std::move(active_context_);
    active_context_.reset();
    return output;
  }

  friend class ::tensorflow::profiler::PythonHookContext;

 private:
  void ProfileSlow(const py::object& frame, const string& event,
                   const py::object& arg);

  void ProfileFast(PyFrameObject* frame, int what, PyObject* arg) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh mht_5(mht_5_v, 354, "", "./tensorflow/python/profiler/internal/python_hooks.h", "ProfileFast");

    if (TF_PREDICT_TRUE(active_context_)) {
      active_context_->ProfileFast(frame, what, arg);
    }
  }

  static void set_e2e_context(PythonHookContext* e2e_context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh mht_6(mht_6_v, 363, "", "./tensorflow/python/profiler/internal/python_hooks.h", "set_e2e_context");

    e2e_context_ = e2e_context;
  }

  static PythonHookContext* e2e_context() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSpython_hooksDTh mht_7(mht_7_v, 370, "", "./tensorflow/python/profiler/internal/python_hooks.h", "e2e_context");
 return e2e_context_; }

  static int ProfileFunction(PyObject* obj, PyFrameObject* frame, int what,
                             PyObject* arg);

  // active_context_ are accessed when GIL is held, therefore no race
  // conditions.
  std::unique_ptr<PythonHookContext> active_context_;
  static PythonHookContext* e2e_context_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
