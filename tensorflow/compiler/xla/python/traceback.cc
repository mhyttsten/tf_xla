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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc() {
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

#include "tensorflow/compiler/xla/python/traceback.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace py = pybind11;

bool Traceback::enabled_ = true;

Traceback::~Traceback() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/python/traceback.cc", "Traceback::~Traceback");

  // We want Traceback objects to be safe to destroy without holding the
  // GIL, so we defer destruction of the strings.
  GlobalPyRefManager()->AddGarbage(frames_);
}

std::string Traceback::Frame::ToString() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/xla/python/traceback.cc", "Traceback::Frame::ToString");

  return absl::StrFormat("%s:%d (%s)", file_name, line_num, function_name);
}

std::string Traceback::ToString() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc mht_2(mht_2_v, 220, "", "./tensorflow/compiler/xla/python/traceback.cc", "Traceback::ToString");

  std::vector<std::string> frame_strs;
  frame_strs.reserve(frames_.size());
  for (const Frame& frame : Frames()) {
    frame_strs.push_back(frame.ToString());
  }
  return absl::StrJoin(frame_strs, "\n");
}

std::vector<Traceback::Frame> Traceback::Frames() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc mht_3(mht_3_v, 232, "", "./tensorflow/compiler/xla/python/traceback.cc", "Traceback::Frames");

  // We require the GIL because we manipulate Python strings.
  CHECK(PyGILState_Check());
  std::vector<Traceback::Frame> frames;
  frames.reserve(frames_.size());
  for (const auto& frame : frames_) {
    frames.push_back(Frame{
        std::string(py::reinterpret_borrow<py::str>(frame.first->co_filename)),
        std::string(py::reinterpret_borrow<py::str>(frame.first->co_name)),
        frame.first->co_firstlineno,
        PyCode_Addr2Line(frame.first, frame.second)});
  }
  return frames;
}

std::shared_ptr<Traceback> Traceback::Get() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc mht_4(mht_4_v, 250, "", "./tensorflow/compiler/xla/python/traceback.cc", "Traceback::Get");

  DCHECK(PyGILState_Check());
  if (!enabled_) {
    return nullptr;
  }
  auto tb = std::make_shared<Traceback>();
  const PyThreadState* thread_state = PyThreadState_GET();
  for (PyFrameObject* py_frame = thread_state->frame; py_frame != nullptr;
       py_frame = py_frame->f_back) {
    Py_INCREF(py_frame->f_code);
    tb->frames_.emplace_back(py_frame->f_code, py_frame->f_lasti);
  }
  return tb;
}

void Traceback::SetEnabled(bool enabled) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc mht_5(mht_5_v, 268, "", "./tensorflow/compiler/xla/python/traceback.cc", "Traceback::SetEnabled");
 enabled_ = enabled; }

py::object Traceback::AsPythonTraceback() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc mht_6(mht_6_v, 273, "", "./tensorflow/compiler/xla/python/traceback.cc", "Traceback::AsPythonTraceback");

  py::object traceback = py::none();
  py::dict globals;
  py::handle traceback_type(reinterpret_cast<PyObject*>(&PyTraceBack_Type));
  for (const std::pair<PyCodeObject*, int>& frame : frames_) {
    PyFrameObject* py_frame = PyFrame_New(PyThreadState_Get(), frame.first,
                                          globals.ptr(), /*locals=*/nullptr);

    traceback = traceback_type(
        /*tb_next=*/std::move(traceback),
        /*tb_frame=*/
        py::reinterpret_steal<py::object>(
            reinterpret_cast<PyObject*>(py_frame)),
        /*tb_lasti=*/frame.second,
        /*tb_lineno=*/PyCode_Addr2Line(frame.first, frame.second));
  }
  return traceback;
}

void BuildTracebackSubmodule(py::module& m) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc mht_7(mht_7_v, 295, "", "./tensorflow/compiler/xla/python/traceback.cc", "BuildTracebackSubmodule");

  py::class_<Traceback::Frame>(m, "Frame")
      .def_readonly("file_name", &Traceback::Frame::file_name)
      .def_readonly("function_name", &Traceback::Frame::function_name)
      .def_readonly("function_start_line",
                    &Traceback::Frame::function_start_line)
      .def_readonly("line_num", &Traceback::Frame::line_num)
      .def("__repr__", [](const Traceback::Frame& frame) {
        return absl::StrFormat("%s;%s:%d", frame.function_name, frame.file_name,
                               frame.line_num);
      });

  py::class_<Traceback, std::shared_ptr<Traceback>> traceback(
      m, "Traceback", "Represents a Python stack trace.");
  traceback.def_property_static(
      "enabled", [](py::object /* cls */) { return Traceback::enabled(); },
      [](py::object /* cls */, bool enabled) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStracebackDTcc mht_8(mht_8_v, 314, "", "./tensorflow/compiler/xla/python/traceback.cc", "lambda");

        return Traceback::SetEnabled(enabled);
      });
  traceback.def_static(
      "get_traceback", []() { return Traceback::Get(); },
      R"doc(
    Returns a :class:`Traceback` for the current thread.

    If ``Traceback.enabled`` is ``True``, returns a :class:`Traceback` object
    that describes the Python stack of the calling thread. Stack trace
    collection has a small overhead, so it is disabled by default. If traceback
    collection is disabled, returns ``None``.
    )doc");
  traceback.def_property_readonly("frames", &Traceback::Frames);
  traceback.def("raw_frames", [](const Traceback& tb) -> py::tuple {
    // We return a tuple of lists, rather than a list of tuples, because it
    // is cheaper to allocate only three Python objects for everything rather
    // than one per frame.
    py::list out_code(tb.raw_frames().size());
    py::list out_lasti(tb.raw_frames().size());
    for (size_t i = 0; i < tb.raw_frames().size(); ++i) {
      const auto& frame = tb.raw_frames()[i];
      out_code[i] = py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(frame.first));
      out_lasti[i] = py::int_(frame.second);
    }
    return py::make_tuple(out_code, out_lasti);
  });
  traceback.def("__str__", &Traceback::ToString);
  traceback.def("__eq__",
                [](const Traceback& a, const Traceback& b) { return a == b; });
  traceback.def("__hash__",
                [](const Traceback& tb) { return absl::HashOf(tb); });
  traceback.def("as_python_traceback", &Traceback::AsPythonTraceback);

  traceback.def_static(
      "code_addr2line",
      [](py::handle code, int lasti) {
        if (!PyCode_Check(code.ptr())) {
          throw std::runtime_error("code argument must be a code object");
        }
        return PyCode_Addr2Line(reinterpret_cast<PyCodeObject*>(code.ptr()),
                                lasti);
      },
      "Python wrapper around the Python C API function PyCode_Addr2Line");

  // This function replaces the exception traceback associated with the current
  // Python thread.
  m.def(
      "replace_thread_exc_traceback",
      [](py::object tb) {
        if (!tb.is_none() && !PyTraceBack_Check(tb.ptr())) {
          throw std::runtime_error(
              "argument must be a traceback object or None");
        }
        PyThreadState* thread_state = PyThreadState_Get();
        if (!thread_state->exc_info->exc_traceback) {
          throw std::runtime_error(
              "Current thread does not have an active "
              "exception traceback");
        }
        PyObject* old_exc_traceback = thread_state->exc_info->exc_traceback;
        PyObject* new_tb = tb.is_none() ? nullptr : tb.release().ptr();
        thread_state->exc_info->exc_traceback = new_tb;
        Py_XDECREF(old_exc_traceback);
      },
      py::arg("traceback"));
}

}  // namespace xla
