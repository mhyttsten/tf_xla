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
class MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc() {
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
*/

// We extract stack traces in Python using the logic in tf_stack.cc, which
// stores a list of PyCodeObject*. Such stack trace extraction is really fast.
//
// We store the retrieved stack trace within the Node object directly. Then
// whenever the graph is instantiated/copies, we copy the stack trace with it.
// Since the graph instantiation goes through the protobuf roundtrip, we store
// the original stack traces mapping attached in FunctionLibraryDefinition.

// clang-format off
// These headers must be at the top, before including Python.h header
// Otherwise, we get C2039 on MSVC due to 'copysign'
#include "pybind11/complex.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
// clang-format on

#include <frameobject.h>

#include <algorithm>
#include <vector>

#include "Python.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/python/util/stack_trace.h"

struct StackFrame;  // Forward declaration.
struct StackTrace;

PYBIND11_MAKE_OPAQUE(std::vector<StackFrame>);
PYBIND11_MAKE_OPAQUE(StackTrace);

namespace tensorflow {

namespace {

namespace py = pybind11;

using StringSet = absl::flat_hash_set<std::string>;

// Python wrapper for a SourceMap.
class PyBindSourceMap {
 public:
  PyBindSourceMap() : source_map_(std::make_shared<SourceMap>()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_0(mht_0_v, 236, "", "./tensorflow/python/util/tf_stack.cc", "PyBindSourceMap");
}

  // Shares ownership with whoever captures traces in the scope of this map.
  std::shared_ptr<SourceMap> source_map_;
};

// Python wrapper for a FileSet.
class PyBindFileSet {
 public:
  PyBindFileSet() : file_set_(std::make_shared<StringSet>()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_1(mht_1_v, 248, "", "./tensorflow/python/util/tf_stack.cc", "PyBindFileSet");
}

  // Shares ownership with whoever captures traces in the scope of this set.
  std::shared_ptr<StringSet> file_set_;
};

// Returns contents of the line corresponding to the given frame.
//
// Precondition: must be holding Python GIL.
py::str LineContents(const StackFrame& frame) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_2(mht_2_v, 260, "", "./tensorflow/python/util/tf_stack.cc", "LineContents");

  DCheckPyGilStateForStackTrace();
  // Pointers are to avoid static destruction of pybind::object, which
  // occurs in uncontrollable states.
  static const auto* inspect = new py::module(py::module::import("inspect"));
  static const auto* getmodule = new py::function(inspect->attr("getmodule"));
  static const auto* linecache =
      new py::module(py::module::import("linecache"));
  static const auto* checkcache =
      new py::function(linecache->attr("checkcache"));
  static const auto* getline = new py::function(linecache->attr("getline"));
  (*checkcache)(py::str(frame.file_name));

  // Here we use the undocumented second argument of inspect.getmodule to look
  // up a module from a filename. It has been unchanged since 2015.
  const auto& module = (*getmodule)(py::none(), py::str(frame.file_name));
  py::object dict = py::none();
  if (!module.is_none()) {
    // module dict is used by getline to resolve import hooks; see the
    // stdlib's inspect module.
    dict = module.attr("__dict__");
  }
  return py::cast<py::str>(
      (*getline)(py::str(frame.file_name), py::int_(frame.line_number), dict)
          .attr("strip")());
}

// Ignores the frames containing this substring for common prefix calculation.
static const char* kFilenameToIgnorePrefix = "<embedded";

// Converts the given stack frame to string, according to options defined in
// `opts`.
std::string StackFrameToString(
    const StackFrame& frame,
    const AbstractStackTrace::TracePrintingOptions& opts,
    int shared_prefix_size = 0) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_3(mht_3_v, 298, "", "./tensorflow/python/util/tf_stack.cc", "StackFrameToString");

  std::string out = absl::StrFormat(
      "File \"%s\", line %d, in %s",
      absl::StrContains(frame.file_name, kFilenameToIgnorePrefix)
          ? frame.file_name
          : frame.file_name.substr(shared_prefix_size),
      frame.line_number, frame.function_name);

  if (opts.show_line_contents) {
    PyGILState_STATE state = PyGILState_Ensure();
    std::string line_contents = std::string(LineContents(frame));
    PyGILState_Release(state);
    if (!line_contents.empty()) {
      absl::StrAppend(&out, "\n  ", line_contents);
    }
  }
  return out;
}

class StackTraceWrapper : public AbstractStackTrace {
 public:
  StackTraceWrapper(StackTrace&& captured,
                    const std::shared_ptr<SourceMap>& source_map,
                    const std::shared_ptr<StringSet>& filter)
      : captured_(std::move(captured)),
        source_map_(source_map),
        filter_(filter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_4(mht_4_v, 327, "", "./tensorflow/python/util/tf_stack.cc", "StackTraceWrapper");
}

  explicit StackTraceWrapper(absl::Span<StackFrame const> stack_frames)
      : stack_frames_cache_(std::vector<StackFrame>(stack_frames.begin(),
                                                    stack_frames.end())) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_5(mht_5_v, 334, "", "./tensorflow/python/util/tf_stack.cc", "StackTraceWrapper");
}

  static StackTraceWrapper ExtractStack(
      const std::shared_ptr<SourceMap>& source_map,
      const std::shared_ptr<StringSet>& filter) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_6(mht_6_v, 341, "", "./tensorflow/python/util/tf_stack.cc", "ExtractStack");

    return StackTraceWrapper{StackTrace::Capture(-1), source_map, filter};
  }

  absl::Span<StackFrame const> ToFrames() const override {
    if (stack_frames_cache_) {
      return *stack_frames_cache_;
    }

    // Grabbing the GIL solves two purposes: 1) makes the class thread-safe,
    // and 2) ToStackFrames and LineContents actually need it.
    PyGILState_STATE state = PyGILState_Ensure();

    stack_frames_cache_ = captured_.ToStackFrames(
        *source_map_, [&](const char* f) { return StackTraceFiltering(f); });
    stack_frames_cache_->pop_back();  // Drop last stack frame.
    PyGILState_Release(state);
    return *stack_frames_cache_;
  }

  std::vector<StackFrame> GetUserFrames(int limit = -1) const {
    PyGILState_STATE state = PyGILState_Ensure();
    std::vector<StackFrame> user_frames = captured_.ToStackFrames(
        *source_map_,
        [&](const char* file_name) {
          return StackTraceFiltering(file_name) ||
                 IsInternalFrameForFilename(file_name);
        },
        /*reverse_traversal=*/true,
        /*limit=*/limit);
    PyGILState_Release(state);
    // ensure we use the original (outermost first) ordering.
    absl::c_reverse(user_frames);
    return user_frames;
  }

  StackFrame LastUserFrame() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_7(mht_7_v, 380, "", "./tensorflow/python/util/tf_stack.cc", "LastUserFrame");

    if (last_stack_frame_cache_) {
      return *last_stack_frame_cache_;
    }

    PyGILState_STATE state = PyGILState_Ensure();
    std::vector<StackFrame> last_frame = GetUserFrames(1);

    if (last_frame.empty()) {
      last_stack_frame_cache_ = StackFrame{"", -1, ""};
    } else {
      DCHECK_EQ(last_frame.size(), 1);
      last_stack_frame_cache_ = last_frame[0];
    }
    PyGILState_Release(state);
    return *last_stack_frame_cache_;
  }

  std::string ToString(const TracePrintingOptions& opts) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_8(mht_8_v, 401, "", "./tensorflow/python/util/tf_stack.cc", "ToString");

    std::vector<std::string> files_to_find_prefix;
    for (const StackFrame& frame : ToFrames()) {
      if (!absl::StrContains(frame.file_name, kFilenameToIgnorePrefix)) {
        files_to_find_prefix.push_back(frame.file_name);
      }
    }
    int shared_prefix_size =
        opts.filter_common_prefix
            ? io::CommonPathPrefix(files_to_find_prefix).size()
            : 0;

    if (!opts.drop_internal_frames) {
      return ToStringHelper(*stack_frames_cache_, opts, shared_prefix_size);
    }

    std::vector<StackFrame> filtered_frames;
    for (const StackFrame& frame : *stack_frames_cache_) {
      if (!IsInternalFrameForFilename(frame.file_name)) {
        filtered_frames.push_back(frame);
      }
    }
    return ToStringHelper(filtered_frames, opts, shared_prefix_size);
  }

  StackTraceWrapper(StackTraceWrapper&&) = default;
  ~StackTraceWrapper() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_9(mht_9_v, 430, "", "./tensorflow/python/util/tf_stack.cc", "~StackTraceWrapper");

    PyGILState_STATE state = PyGILState_Ensure();
    captured_.Clear();
    source_map_.reset();
    filter_.reset();
    PyGILState_Release(state);
  }

 private:
  static std::string ToStringHelper(absl::Span<StackFrame const> stack_frames,
                                    const TracePrintingOptions& opts,
                                    int shared_prefix_size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_10(mht_10_v, 444, "", "./tensorflow/python/util/tf_stack.cc", "ToStringHelper");

    return absl::StrJoin(
        stack_frames, "\n", [&](std::string* out, const StackFrame& frame) {
          absl::StrAppend(out,
                          StackFrameToString(frame, opts, shared_prefix_size));
        });
  }

  bool StackTraceFiltering(const char* file_name) const {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("file_name: \"" + (file_name == nullptr ? std::string("nullptr") : std::string((char*)file_name)) + "\"");
   MHTracer_DTPStensorflowPSpythonPSutilPStf_stackDTcc mht_11(mht_11_v, 456, "", "./tensorflow/python/util/tf_stack.cc", "StackTraceFiltering");

    return filter_->contains(file_name);
  }

  StackTrace captured_;
  std::shared_ptr<SourceMap> source_map_;
  std::shared_ptr<StringSet> filter_;

  // Using optional to force destruction while we hold a GIL.
  mutable absl::optional<std::vector<StackFrame>> stack_frames_cache_;
  mutable absl::optional<StackFrame> last_stack_frame_cache_;
};

}  // namespace

PYBIND11_MODULE(_tf_stack, m) {
  py::class_<PyBindSourceMap>(m, "PyBindSourceMap")
      .def(py::init())
      .def("update_to",
           [](const PyBindSourceMap& self, const py::tuple& source_map) {
             self.source_map_->clear();
             for (const auto& item : source_map) {
               const auto& tuple_item = py::cast<py::tuple>(item);

               const auto& key = py::cast<py::tuple>(tuple_item[0]);
               std::string&& k_filename = py::cast<std::string>(key[0]);
               int k_lineno = py::cast<int>(key[1]);

               const auto& value = py::cast<py::tuple>(tuple_item[1]);
               std::string&& v_filename = py::cast<std::string>(value[0]);
               int v_lineno = py::cast<int>(value[1]);
               const auto& function_name_val = value[2];
               std::string&& v_function_name =
                   function_name_val.is_none()
                       ? ""
                       : py::cast<std::string>(function_name_val);

               self.source_map_->emplace(
                   SourceLoc{k_filename, k_lineno},
                   StackFrame({v_filename, v_lineno, v_function_name}));
             }
           });

  py::class_<PyBindFileSet>(m, "PyBindFileSet")
      .def(py::init())
      .def("update_to", [](const PyBindFileSet& self, const py::set& file_set) {
        self.file_set_->clear();
        for (const auto& item : file_set) {
          self.file_set_->insert(py::cast<std::string>(item));
        }
      });

  py::class_<StackFrame>(m, "StackFrame")
      .def_property_readonly(
          "filename",
          [](const StackFrame& self) { return py::str(self.file_name); })
      .def_property_readonly(
          "lineno",
          [](const StackFrame& self) { return py::int_(self.line_number); })
      .def_property_readonly(
          "name",
          [](const StackFrame& self) { return py::str(self.function_name); })
      .def_property_readonly(
          "line", [](const StackFrame& self) { return LineContents(self); })

      // For compatibility with the traceback module.
      .def("__eq__", &StackFrame::operator==)
      .def("__ne__", &StackFrame::operator!=)
      .def("__hash__",
           [](const StackFrame& self) {
             return absl::Hash<std::tuple<std::string, int, std::string>>()(
                 std::make_tuple(self.file_name, self.line_number,
                                 self.function_name));
           })
      .def("__getitem__",
           [](const StackFrame& self, const py::object& index) -> py::object {
             return py::make_tuple(
                 py::str(self.file_name), py::int_(self.line_number),
                 py::str(self.function_name), LineContents(self))[index];
           })
      .def("__iter__",
           [](const StackFrame& self) {
             return py::iter(py::make_tuple(
                 py::str(self.file_name), py::int_(self.line_number),
                 py::str(self.function_name), LineContents(self))

             );
           })
      .def("__repr__",
           [](const StackFrame& self) { return StackFrameToString(self, {}); })
      .def("__len__", [](const StackFrame&) { return 4; });

  py::class_<StackTraceWrapper>(m, "StackTraceWrapper", py::module_local(true))
      // TODO(slebedev): upstream negative indexing support into pybind11.
      .def(
          "__getitem__",
          [](const StackTraceWrapper& self, ssize_t index) {
            absl::Span<StackFrame const> frames = self.ToFrames();
            const size_t eff_index =
                index < 0 ? frames.size() + index : static_cast<size_t>(index);
            if (eff_index >= frames.size()) {
              throw py::index_error();
            }
            return frames[eff_index];
          },
          py::return_value_policy::reference_internal)
      .def(
          "__getitem__",
          [](const StackTraceWrapper& self, py::slice slice) {
            absl::Span<StackFrame const> frames = self.ToFrames();
            py::ssize_t start, stop, step, slicelength;
            if (!slice.compute(frames.size(), &start, &stop, &step,
                               &slicelength)) {
              throw py::error_already_set();
            }
            if (step == 1) {
              return StackTraceWrapper{frames.subspan(start, slicelength)};
            }
            // TODO(cheshire): Cleanup, use Python slicing logic directly
            // instead.
            std::vector<StackFrame> out;
            out.reserve(slicelength);
            // Python slices allow negative indexing.
            for (int i = start; i != stop; i += step) {
              out.push_back(frames[i]);
            }
            return StackTraceWrapper{out};
          },
          py::return_value_policy::reference_internal)
      .def("__len__",
           [](const StackTraceWrapper& self) { return self.ToFrames().size(); })
      .def("__eq__",
           [](const StackTraceWrapper& self, const StackTraceWrapper& other) {
             return self.ToFrames() == other.ToFrames();
           })
      .def("__hash__",
           [](const StackTraceWrapper& self) {
             return py::hash(py::str(self.ToString({})));
           })
      // NOTE(feyu): consider remove this and use traceback.format_list(tb)
      // to format the trace.
      .def("__repr__",
           [](const StackTraceWrapper& self) {
             return py::str(self.ToString({}));
           })
      .def(
          "get_user_frames",
          [](const StackTraceWrapper& self) {
            return StackTraceWrapper{self.GetUserFrames()};
          },
          "Returns the non-framework frames as a new trace object.")
      .def(
          "last_user_frame",
          [](const StackTraceWrapper& self) { return self.LastUserFrame(); },
          "Returns the last non-framework frame.");

  m.def(
      "extract_stack_for_node",
      [](const PyBindSourceMap& source_map, const PyBindFileSet& file_set,
         TF_Operation* op) -> const AbstractStackTrace& {
        Node* node = reinterpret_cast<Node*>(op);
        DCHECK(!node->GetStackTrace()) << "Should not reset the stack trace";
        node->SetStackTrace(
            std::make_shared<StackTraceWrapper>(StackTraceWrapper::ExtractStack(
                source_map.source_map_, file_set.file_set_)));
        return *node->GetStackTrace();
      },
      py::return_value_policy::reference);

  m.def(
      "extract_stack",
      [](const PyBindSourceMap& source_map, const PyBindFileSet& file_set) {
        return StackTraceWrapper::ExtractStack(source_map.source_map_,
                                               file_set.file_set_);
      },
      py::return_value_policy::move);
}

}  // namespace tensorflow
