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
class MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc() {
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

#include "tensorflow/core/platform/env.h"

#include <Shlwapi.h>
#include <Windows.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#undef ERROR

#include <string>
#include <thread>
#include <vector>

#include "tensorflow/core/platform/load_library.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/ram_file_system.h"
#include "tensorflow/core/platform/windows/wide_char.h"
#include "tensorflow/core/platform/windows/windows_file_system.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

#pragma comment(lib, "Shlwapi.lib")

namespace tensorflow {

namespace {

mutex name_mutex(tensorflow::LINKER_INITIALIZED);

std::map<std::thread::id, string>& GetThreadNameRegistry()
    TF_EXCLUSIVE_LOCKS_REQUIRED(name_mutex) {
  static auto* thread_name_registry = new std::map<std::thread::id, string>();
  return *thread_name_registry;
}

class StdThread : public Thread {
 public:
  // thread_options is ignored.
  StdThread(const ThreadOptions& thread_options, const string& name,
            std::function<void()> fn)
      : thread_(fn) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_0(mht_0_v, 227, "", "./tensorflow/core/platform/windows/env.cc", "StdThread");

    mutex_lock l(name_mutex);
    GetThreadNameRegistry().emplace(thread_.get_id(), name);
  }

  ~StdThread() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/platform/windows/env.cc", "~StdThread");

    std::thread::id thread_id = thread_.get_id();
    thread_.join();
    mutex_lock l(name_mutex);
    GetThreadNameRegistry().erase(thread_id);
  }

 private:
  std::thread thread_;
};

class WindowsEnv : public Env {
 public:
  WindowsEnv() : GetSystemTimePreciseAsFileTime_(NULL) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_2(mht_2_v, 251, "", "./tensorflow/core/platform/windows/env.cc", "WindowsEnv");

    // GetSystemTimePreciseAsFileTime function is only available in the latest
    // versions of Windows. For that reason, we try to look it up in
    // kernel32.dll at runtime and use an alternative option if the function
    // is not available.
    HMODULE module = GetModuleHandleW(L"kernel32.dll");
    if (module != NULL) {
      auto func = (FnGetSystemTimePreciseAsFileTime)GetProcAddress(
          module, "GetSystemTimePreciseAsFileTime");
      GetSystemTimePreciseAsFileTime_ = func;
    }
  }

  ~WindowsEnv() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/platform/windows/env.cc", "~WindowsEnv");

    LOG(FATAL) << "Env::Default() must not be destroyed";
  }

  bool MatchPath(const string& path, const string& pattern) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("path: \"" + path + "\"");
   mht_4_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_4(mht_4_v, 276, "", "./tensorflow/core/platform/windows/env.cc", "MatchPath");

    std::wstring ws_path(Utf8ToWideChar(path));
    std::wstring ws_pattern(Utf8ToWideChar(pattern));
    return PathMatchSpecW(ws_path.c_str(), ws_pattern.c_str()) == TRUE;
  }

  void SleepForMicroseconds(int64 micros) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_5(mht_5_v, 285, "", "./tensorflow/core/platform/windows/env.cc", "SleepForMicroseconds");
 Sleep(micros / 1000); }

  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      std::function<void()> fn) override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_6(mht_6_v, 292, "", "./tensorflow/core/platform/windows/env.cc", "StartThread");

    return new StdThread(thread_options, name, fn);
  }

  int32 GetCurrentThreadId() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_7(mht_7_v, 299, "", "./tensorflow/core/platform/windows/env.cc", "GetCurrentThreadId");

    return static_cast<int32>(::GetCurrentThreadId());
  }

  bool GetCurrentThreadName(string* name) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_8(mht_8_v, 306, "", "./tensorflow/core/platform/windows/env.cc", "GetCurrentThreadName");

    mutex_lock l(name_mutex);
    auto thread_name = GetThreadNameRegistry().find(std::this_thread::get_id());
    if (thread_name != GetThreadNameRegistry().end()) {
      *name = thread_name->second;
      return true;
    } else {
      return false;
    }
  }

  static VOID CALLBACK SchedClosureCallback(PTP_CALLBACK_INSTANCE Instance,
                                            PVOID Context, PTP_WORK Work) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_9(mht_9_v, 321, "", "./tensorflow/core/platform/windows/env.cc", "SchedClosureCallback");

    CloseThreadpoolWork(Work);
    std::function<void()>* f = (std::function<void()>*)Context;
    (*f)();
    delete f;
  }
  void SchedClosure(std::function<void()> closure) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_10(mht_10_v, 330, "", "./tensorflow/core/platform/windows/env.cc", "SchedClosure");

    PTP_WORK work = CreateThreadpoolWork(
        SchedClosureCallback, new std::function<void()>(std::move(closure)),
        nullptr);
    SubmitThreadpoolWork(work);
  }

  static VOID CALLBACK SchedClosureAfterCallback(PTP_CALLBACK_INSTANCE Instance,
                                                 PVOID Context,
                                                 PTP_TIMER Timer) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_11(mht_11_v, 342, "", "./tensorflow/core/platform/windows/env.cc", "SchedClosureAfterCallback");

    CloseThreadpoolTimer(Timer);
    std::function<void()>* f = (std::function<void()>*)Context;
    (*f)();
    delete f;
  }

  void SchedClosureAfter(int64 micros, std::function<void()> closure) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_12(mht_12_v, 352, "", "./tensorflow/core/platform/windows/env.cc", "SchedClosureAfter");

    PTP_TIMER timer = CreateThreadpoolTimer(
        SchedClosureAfterCallback,
        new std::function<void()>(std::move(closure)), nullptr);
    // in 100 nanosecond units
    FILETIME FileDueTime;
    ULARGE_INTEGER ulDueTime;
    // Negative indicates the amount of time to wait is relative to the current
    // time.
    ulDueTime.QuadPart = (ULONGLONG) - (10 * micros);
    FileDueTime.dwHighDateTime = ulDueTime.HighPart;
    FileDueTime.dwLowDateTime = ulDueTime.LowPart;
    SetThreadpoolTimer(timer, &FileDueTime, 0, 0);
  }

  Status LoadDynamicLibrary(const char* library_filename,
                            void** handle) override {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("library_filename: \"" + (library_filename == nullptr ? std::string("nullptr") : std::string((char*)library_filename)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_13(mht_13_v, 372, "", "./tensorflow/core/platform/windows/env.cc", "LoadDynamicLibrary");

    return tensorflow::internal::LoadDynamicLibrary(library_filename, handle);
  }

  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              void** symbol) override {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("symbol_name: \"" + (symbol_name == nullptr ? std::string("nullptr") : std::string((char*)symbol_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_14(mht_14_v, 381, "", "./tensorflow/core/platform/windows/env.cc", "GetSymbolFromLibrary");

    return tensorflow::internal::GetSymbolFromLibrary(handle, symbol_name,
                                                      symbol);
  }

  string FormatLibraryFileName(const string& name,
                               const string& version) override {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + name + "\"");
   mht_15_v.push_back("version: \"" + version + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_15(mht_15_v, 392, "", "./tensorflow/core/platform/windows/env.cc", "FormatLibraryFileName");

    return tensorflow::internal::FormatLibraryFileName(name, version);
  }

  string GetRunfilesDir() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_16(mht_16_v, 399, "", "./tensorflow/core/platform/windows/env.cc", "GetRunfilesDir");

    string bin_path = this->GetExecutablePath();
    string runfiles_path = bin_path + ".runfiles\\org_tensorflow";
    Status s = this->IsDirectory(runfiles_path);
    if (s.ok()) {
      return runfiles_path;
    } else {
      return bin_path.substr(0, bin_path.find_last_of("/\\"));
    }
  }

 private:
  void GetLocalTempDirectories(std::vector<string>* list) override;

  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  FnGetSystemTimePreciseAsFileTime GetSystemTimePreciseAsFileTime_;
};

}  // namespace

REGISTER_FILE_SYSTEM("", WindowsFileSystem);
REGISTER_FILE_SYSTEM("file", LocalWinFileSystem);
REGISTER_FILE_SYSTEM("ram", RamFileSystem);

Env* Env::Default() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_17(mht_17_v, 426, "", "./tensorflow/core/platform/windows/env.cc", "Env::Default");

  static Env* default_env = new WindowsEnv;
  return default_env;
}

void WindowsEnv::GetLocalTempDirectories(std::vector<string>* list) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_18(mht_18_v, 434, "", "./tensorflow/core/platform/windows/env.cc", "WindowsEnv::GetLocalTempDirectories");

  list->clear();
  // On windows we'll try to find a directory in this order:
  //   C:/Documents & Settings/whomever/TEMP (or whatever GetTempPath() is)
  //   C:/TMP/
  //   C:/TEMP/
  //   C:/WINDOWS/ or C:/WINNT/
  //   .
  char tmp[MAX_PATH];
  // GetTempPath can fail with either 0 or with a space requirement > bufsize.
  // See http://msdn.microsoft.com/en-us/library/aa364992(v=vs.85).aspx
  DWORD n = GetTempPathA(MAX_PATH, tmp);
  if (n > 0 && n <= MAX_PATH) list->push_back(tmp);
  list->push_back("C:\\tmp\\");
  list->push_back("C:\\temp\\");
}

int setenv(const char* name, const char* value, int overwrite) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_19_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_19(mht_19_v, 456, "", "./tensorflow/core/platform/windows/env.cc", "setenv");

  if (!overwrite) {
    char* env_val = getenv(name);
    if (env_val) {
      return 0;
    }
  }
  return _putenv_s(name, value);
}

int unsetenv(const char* name) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSenvDTcc mht_20(mht_20_v, 470, "", "./tensorflow/core/platform/windows/env.cc", "unsetenv");
 return _putenv_s(name, ""); }

}  // namespace tensorflow
