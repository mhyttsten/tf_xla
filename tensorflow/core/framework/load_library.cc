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
class MHTracer_DTPStensorflowPScorePSframeworkPSload_libraryDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSload_libraryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSload_libraryDTcc() {
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

#include <memory>
#include <unordered_set>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

namespace {

struct Library {
  void* handle = nullptr;
  OpList op_list;
};

}  // namespace

// Load a dynamic library.
// On success, returns the handle to library in result, copies the serialized
// OpList of OpDefs registered in the library to *buf and the length to *len,
// and returns OK from the function. Otherwise return nullptr in result
// and an error status from the function, leaving buf and len untouched.
//
// If `library_filename` has already been loaded, we return a cached handle
// and OpList. Ops and kernels are registered as globals when a library is
// loaded for the first time. Without caching, every subsequent load would not
// perform initialization again, so the OpList would be empty.
Status LoadDynamicLibrary(const char* library_filename, void** result,
                          const void** buf, size_t* len) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("library_filename: \"" + (library_filename == nullptr ? std::string("nullptr") : std::string((char*)library_filename)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSload_libraryDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/framework/load_library.cc", "LoadDynamicLibrary");

  static mutex mu(LINKER_INITIALIZED);
  static std::unordered_map<string, Library> loaded_libs;
  Env* env = Env::Default();
  Library library;
  std::unordered_set<string> seen_op_names;
  {
    mutex_lock lock(mu);
    if (loaded_libs.find(library_filename) != loaded_libs.end()) {
      library = loaded_libs[library_filename];
    } else {
      Status s = OpRegistry::Global()->ProcessRegistrations();
      if (!s.ok()) {
        return s;
      }
      TF_RETURN_IF_ERROR(OpRegistry::Global()->SetWatcher(
          [&library, &seen_op_names](const Status& s,
                                     const OpDef& opdef) -> Status {
            if (errors::IsAlreadyExists(s)) {
              if (seen_op_names.find(opdef.name()) == seen_op_names.end()) {
                // Over writing a registration of an op not in this custom op
                // library. Treat this as not an error.
                return Status::OK();
              }
            }
            if (s.ok()) {
              *library.op_list.add_op() = opdef;
              seen_op_names.insert(opdef.name());
            }
            return s;
          }));
      OpRegistry::Global()->DeferRegistrations();
      s = env->LoadDynamicLibrary(library_filename, &library.handle);
      if (s.ok()) {
        s = OpRegistry::Global()->ProcessRegistrations();
      }
      if (!s.ok()) {
        OpRegistry::Global()->ClearDeferredRegistrations();
        TF_RETURN_IF_ERROR(OpRegistry::Global()->SetWatcher(nullptr));
        return s;
      }
      TF_RETURN_IF_ERROR(OpRegistry::Global()->SetWatcher(nullptr));

      loaded_libs[library_filename] = library;
    }
  }
  string str;
  library.op_list.SerializeToString(&str);
  char* str_buf = reinterpret_cast<char*>(port::Malloc(str.length()));
  memcpy(str_buf, str.data(), str.length());
  *buf = str_buf;
  *len = str.length();

  *result = library.handle;
  return Status::OK();
}

}  // namespace tensorflow
