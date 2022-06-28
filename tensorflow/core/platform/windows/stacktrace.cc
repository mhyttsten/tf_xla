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
class MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSstacktraceDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSstacktraceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSstacktraceDTcc() {
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
==============================================================================*/

#include "tensorflow/core/platform/windows/stacktrace.h"

// clang-format off
#include <windows.h>  // Windows.h must be declared above dgbhelp.
#include <dbghelp.h>
// clang-format on

#include <string>

#include "tensorflow/core/platform/mutex.h"

#pragma comment(lib, "dbghelp.lib")

namespace tensorflow {

// We initialize the Symbolizer on first call:
// https://docs.microsoft.com/en-us/windows/win32/debug/initializing-the-symbol-handler
static bool SymbolsAreAvailableInit() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSstacktraceDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/platform/windows/stacktrace.cc", "SymbolsAreAvailableInit");

  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
  return SymInitialize(GetCurrentProcess(), NULL, true);
}

static bool SymbolsAreAvailable() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSstacktraceDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/platform/windows/stacktrace.cc", "SymbolsAreAvailable");

  static bool kSymbolsAvailable = SymbolsAreAvailableInit();  // called once
  return kSymbolsAvailable;
}

// Generating stacktraces involve two steps:
// 1. Producing a list of pointers, where each pointer corresponds to the
//    function called at each stack frame (aka stack unwinding).
// 2. Converting each pointer into a human readable string corresponding to
//    the function's name (aka symbolization).
// Windows provides two APIs for stack unwinding: StackWalk
// (https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-stackwalk64)
// and CaptureStackBackTrace
// (https://docs.microsoft.com/en-us/windows/win32/debug/capturestackbacktrace).
// Although StackWalk is more flexible, it does not have any threadsafety
// guarantees. See https://stackoverflow.com/a/17890764
// Windows provides one symbolization API, SymFromAddr:
// https://docs.microsoft.com/en-us/windows/win32/debug/retrieving-symbol-information-by-address
// which is unfortunately not threadsafe. Therefore, we acquire a lock prior to
// calling it, making this function NOT async-signal-safe.
// FYI from m3b@ about signal safety:
// Functions that block when acquiring mutexes are not async-signal-safe
// primarily because the signal might have been delivered to a thread that holds
// the lock. That is, the thread could self-deadlock if a signal is delivered at
// the wrong moment; no other threads are needed.
std::string CurrentStackTrace() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSstacktraceDTcc mht_2(mht_2_v, 238, "", "./tensorflow/core/platform/windows/stacktrace.cc", "CurrentStackTrace");

  // For reference, many stacktrace-related Windows APIs are documented here:
  // https://docs.microsoft.com/en-us/windows/win32/debug/about-dbghelp.
  HANDLE current_process = GetCurrentProcess();
  static constexpr int kMaxStackFrames = 64;
  void* trace[kMaxStackFrames];
  int num_frames = CaptureStackBackTrace(0, kMaxStackFrames, trace, NULL);

  static mutex mu(tensorflow::LINKER_INITIALIZED);

  std::string stacktrace;
  for (int i = 0; i < num_frames; ++i) {
    const char* symbol = "(unknown)";
    if (SymbolsAreAvailable()) {
      char symbol_info_buffer[sizeof(SYMBOL_INFO) +
                              MAX_SYM_NAME * sizeof(TCHAR)];
      SYMBOL_INFO* symbol_ptr =
          reinterpret_cast<SYMBOL_INFO*>(symbol_info_buffer);
      symbol_ptr->SizeOfStruct = sizeof(SYMBOL_INFO);
      symbol_ptr->MaxNameLen = MAX_SYM_NAME;

      // Because SymFromAddr is not threadsafe, we acquire a lock.
      mutex_lock lock(mu);
      if (SymFromAddr(current_process, reinterpret_cast<DWORD64>(trace[i]), 0,
                      symbol_ptr)) {
        symbol = symbol_ptr->Name;
      }
    }

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "0x%p\t%s", trace[i], symbol);
    stacktrace += buffer;
    stacktrace += "\n";
  }

  return stacktrace;
}

}  // namespace tensorflow
