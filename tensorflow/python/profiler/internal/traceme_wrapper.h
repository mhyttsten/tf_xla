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
#ifndef TENSORFLOW_PYTHON_PROFILER_INTERNAL_TRACEME_WRAPPER_
#define TENSORFLOW_PYTHON_PROFILER_INTERNAL_TRACEME_WRAPPER_
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
class MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPStraceme_wrapperDTh {
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
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPStraceme_wrapperDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPStraceme_wrapperDTh() {
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


#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "pybind11/pytypes.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace profiler {

// Wraps TraceMe with an interface that takes python types.
class TraceMeWrapper {
 public:
  // pybind11::str and pybind11::kwargs are taken by const reference to avoid
  // python reference-counting overhead.
  TraceMeWrapper(const pybind11::str& name, const pybind11::kwargs& kwargs)
      : traceme_(
            [&]() {
              std::string name_and_metadata(name);
              if (!kwargs.empty()) {
                AppendMetadata(&name_and_metadata, kwargs);
              }
              return name_and_metadata;
            },
            /*level=*/1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPStraceme_wrapperDTh mht_0(mht_0_v, 214, "", "./tensorflow/python/profiler/internal/traceme_wrapper.h", "TraceMeWrapper");
}

  // pybind11::kwargs is taken by const reference to avoid python
  // reference-counting overhead.
  void SetMetadata(const pybind11::kwargs& kwargs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPStraceme_wrapperDTh mht_1(mht_1_v, 221, "", "./tensorflow/python/profiler/internal/traceme_wrapper.h", "SetMetadata");

    if (TF_PREDICT_FALSE(!kwargs.empty())) {
      traceme_.AppendMetadata([&]() {
        std::string metadata;
        AppendMetadata(&metadata, kwargs);
        return metadata;
      });
    }
  }

  void Stop() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPStraceme_wrapperDTh mht_2(mht_2_v, 234, "", "./tensorflow/python/profiler/internal/traceme_wrapper.h", "Stop");
 traceme_.Stop(); }

  static bool IsEnabled() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPStraceme_wrapperDTh mht_3(mht_3_v, 239, "", "./tensorflow/python/profiler/internal/traceme_wrapper.h", "IsEnabled");
 return tensorflow::profiler::TraceMe::Active(); }

 private:
  // Converts kwargs to strings and appends them to name encoded as TraceMe
  // metadata.
  static void AppendMetadata(std::string* name,
                             const pybind11::kwargs& kwargs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPStraceme_wrapperDTh mht_4(mht_4_v, 248, "", "./tensorflow/python/profiler/internal/traceme_wrapper.h", "AppendMetadata");

    name->push_back('#');
    for (const auto& kv : kwargs) {
      absl::StrAppend(name, std::string(pybind11::str(kv.first)), "=",
                      EncodePyObject(kv.second), ",");
    }
    name->back() = '#';
  }

  static std::string EncodePyObject(const pybind11::handle& handle) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPStraceme_wrapperDTh mht_5(mht_5_v, 260, "", "./tensorflow/python/profiler/internal/traceme_wrapper.h", "EncodePyObject");

    if (pybind11::isinstance<pybind11::bool_>(handle)) {
      return handle.cast<bool>() ? "1" : "0";
    }
    return std::string(pybind11::str(handle));
  }

  tensorflow::profiler::TraceMe traceme_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_PROFILER_INTERNAL_TRACEME_WRAPPER_
