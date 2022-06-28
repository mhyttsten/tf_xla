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
class MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc() {
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

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifdef __FreeBSD__
#include <pthread_np.h>
#endif

#include <thread>
#include <vector>

#include "tensorflow/core/platform/default/posix_file_system.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/load_library.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/ram_file_system.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

namespace {

mutex name_mutex(tensorflow::LINKER_INITIALIZED);

std::map<std::thread::id, string>& GetThreadNameRegistry()
    TF_EXCLUSIVE_LOCKS_REQUIRED(name_mutex) {
  static auto* thread_name_registry = new std::map<std::thread::id, string>();
  return *thread_name_registry;
}

// We use the pthread API instead of std::thread so we can control stack sizes.
class PThread : public Thread {
 public:
  PThread(const ThreadOptions& thread_options, const std::string& name,
          std::function<void()> fn) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_0(mht_0_v, 232, "", "./tensorflow/core/platform/default/env.cc", "PThread");

    ThreadParams* params = new ThreadParams;
    params->name = name;
    params->fn = std::move(fn);
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    if (thread_options.stack_size != 0) {
      pthread_attr_setstacksize(&attributes, thread_options.stack_size);
    }
    int ret = pthread_create(&thread_, &attributes, &ThreadFn, params);
    // There is no mechanism for the thread creation API to fail, so we CHECK.
    CHECK_EQ(ret, 0) << "Thread " << name
                     << " creation via pthread_create() failed.";
    pthread_attr_destroy(&attributes);
  }

  ~PThread() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_1(mht_1_v, 251, "", "./tensorflow/core/platform/default/env.cc", "~PThread");
 pthread_join(thread_, nullptr); }

 private:
  struct ThreadParams {
    std::string name;
    std::function<void()> fn;
  };
  static void* ThreadFn(void* params_arg) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_2(mht_2_v, 261, "", "./tensorflow/core/platform/default/env.cc", "ThreadFn");

    std::unique_ptr<ThreadParams> params(
        reinterpret_cast<ThreadParams*>(params_arg));
    {
      mutex_lock l(name_mutex);
      GetThreadNameRegistry().emplace(std::this_thread::get_id(), params->name);
    }
    params->fn();
    {
      mutex_lock l(name_mutex);
      GetThreadNameRegistry().erase(std::this_thread::get_id());
    }
    return nullptr;
  }

  pthread_t thread_;
};

class PosixEnv : public Env {
 public:
  PosixEnv() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_3(mht_3_v, 284, "", "./tensorflow/core/platform/default/env.cc", "PosixEnv");
}

  ~PosixEnv() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_4(mht_4_v, 289, "", "./tensorflow/core/platform/default/env.cc", "~PosixEnv");
 LOG(FATAL) << "Env::Default() must not be destroyed"; }

  bool MatchPath(const string& path, const string& pattern) override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("path: \"" + path + "\"");
   mht_5_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_5(mht_5_v, 296, "", "./tensorflow/core/platform/default/env.cc", "MatchPath");

    return fnmatch(pattern.c_str(), path.c_str(), FNM_PATHNAME) == 0;
  }

  void SleepForMicroseconds(int64 micros) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_6(mht_6_v, 303, "", "./tensorflow/core/platform/default/env.cc", "SleepForMicroseconds");

    while (micros > 0) {
      timespec sleep_time;
      sleep_time.tv_sec = 0;
      sleep_time.tv_nsec = 0;

      if (micros >= 1e6) {
        sleep_time.tv_sec =
            std::min<int64_t>(micros / 1e6, std::numeric_limits<time_t>::max());
        micros -= static_cast<int64_t>(sleep_time.tv_sec) * 1e6;
      }
      if (micros < 1e6) {
        sleep_time.tv_nsec = 1000 * micros;
        micros = 0;
      }
      while (nanosleep(&sleep_time, &sleep_time) != 0 && errno == EINTR) {
        // Ignore signals and wait for the full interval to elapse.
      }
    }
  }

  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      std::function<void()> fn) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_7(mht_7_v, 329, "", "./tensorflow/core/platform/default/env.cc", "StartThread");

    return new PThread(thread_options, name, fn);
  }

  int32 GetCurrentThreadId() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_8(mht_8_v, 336, "", "./tensorflow/core/platform/default/env.cc", "GetCurrentThreadId");

    static thread_local int32 current_thread_id = GetCurrentThreadIdInternal();
    return current_thread_id;
  }

  bool GetCurrentThreadName(string* name) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_9(mht_9_v, 344, "", "./tensorflow/core/platform/default/env.cc", "GetCurrentThreadName");

    {
      mutex_lock l(name_mutex);
      auto thread_name =
          GetThreadNameRegistry().find(std::this_thread::get_id());
      if (thread_name != GetThreadNameRegistry().end()) {
        *name = strings::StrCat(thread_name->second, "/", GetCurrentThreadId());
        return true;
      }
    }
#if defined(__GLIBC__) || defined(__FreeBSD__)
    char buf[100];
#ifdef __FreeBSD__
    int res = 0;
    pthread_get_name_np(pthread_self(), buf, static_cast<size_t>(100));
#else
    int res = pthread_getname_np(pthread_self(), buf, static_cast<size_t>(100));
#endif
    if (res != 0) {
      return false;
    }
    *name = buf;
    return true;
#else
    return false;
#endif
  }

  void SchedClosure(std::function<void()> closure) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_10(mht_10_v, 375, "", "./tensorflow/core/platform/default/env.cc", "SchedClosure");

    // TODO(b/27290852): Spawning a new thread here is wasteful, but
    // needed to deal with the fact that many `closure` functions are
    // blocking in the current codebase.
    std::thread closure_thread(closure);
    closure_thread.detach();
  }

  void SchedClosureAfter(int64 micros, std::function<void()> closure) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_11(mht_11_v, 386, "", "./tensorflow/core/platform/default/env.cc", "SchedClosureAfter");

    // TODO(b/27290852): Consuming a thread here is wasteful, but this
    // code is (currently) only used in the case where a step fails
    // (AbortStep). This could be replaced by a timer thread
    SchedClosure([this, micros, closure]() {
      SleepForMicroseconds(micros);
      closure();
    });
  }

  Status LoadDynamicLibrary(const char* library_filename,
                            void** handle) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("library_filename: \"" + (library_filename == nullptr ? std::string("nullptr") : std::string((char*)library_filename)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_12(mht_12_v, 401, "", "./tensorflow/core/platform/default/env.cc", "LoadDynamicLibrary");

    return tensorflow::internal::LoadDynamicLibrary(library_filename, handle);
  }

  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              void** symbol) override {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("symbol_name: \"" + (symbol_name == nullptr ? std::string("nullptr") : std::string((char*)symbol_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_13(mht_13_v, 410, "", "./tensorflow/core/platform/default/env.cc", "GetSymbolFromLibrary");

    return tensorflow::internal::GetSymbolFromLibrary(handle, symbol_name,
                                                      symbol);
  }

  string FormatLibraryFileName(const string& name,
                               const string& version) override {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   mht_14_v.push_back("version: \"" + version + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_14(mht_14_v, 421, "", "./tensorflow/core/platform/default/env.cc", "FormatLibraryFileName");

    return tensorflow::internal::FormatLibraryFileName(name, version);
  }

  string GetRunfilesDir() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_15(mht_15_v, 428, "", "./tensorflow/core/platform/default/env.cc", "GetRunfilesDir");

    string bin_path = this->GetExecutablePath();
    string runfiles_suffix = ".runfiles/org_tensorflow";
    std::size_t pos = bin_path.find(runfiles_suffix);

    // Sometimes (when executing under python) bin_path returns the full path to
    // the python scripts under runfiles. Get the substring.
    if (pos != std::string::npos) {
      return bin_path.substr(0, pos + runfiles_suffix.length());
    }

    // See if we have the executable path. if executable.runfiles exists, return
    // that folder.
    string runfiles_path = bin_path + runfiles_suffix;
    Status s = this->IsDirectory(runfiles_path);
    if (s.ok()) {
      return runfiles_path;
    }

    // If nothing can be found, return something close.
    return bin_path.substr(0, bin_path.find_last_of("/\\"));
  }

 private:
  void GetLocalTempDirectories(std::vector<string>* list) override;

  int32 GetCurrentThreadIdInternal() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_16(mht_16_v, 457, "", "./tensorflow/core/platform/default/env.cc", "GetCurrentThreadIdInternal");

#ifdef __APPLE__
    uint64_t tid64;
    pthread_threadid_np(nullptr, &tid64);
    return static_cast<int32>(tid64);
#elif defined(__FreeBSD__)
    return pthread_getthreadid_np();
#elif defined(__NR_gettid)
    return static_cast<int32>(syscall(__NR_gettid));
#else
    return std::hash<std::thread::id>()(std::this_thread::get_id());
#endif
  }
};

}  // namespace

#if defined(PLATFORM_POSIX) || defined(__APPLE__) || defined(__ANDROID__)
REGISTER_FILE_SYSTEM("", PosixFileSystem);
REGISTER_FILE_SYSTEM("file", LocalPosixFileSystem);
REGISTER_FILE_SYSTEM("ram", RamFileSystem);


Env* Env::Default() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_17(mht_17_v, 483, "", "./tensorflow/core/platform/default/env.cc", "Env::Default");

  static Env* default_env = new PosixEnv;
  return default_env;
}
#endif

void PosixEnv::GetLocalTempDirectories(std::vector<string>* list) {
  list->clear();
  // Directories, in order of preference. If we find a dir that
  // exists, we stop adding other less-preferred dirs
  const char* candidates[] = {
    // Non-null only during unittest/regtest
    getenv("TEST_TMPDIR"),

    // Explicitly-supplied temp dirs
    getenv("TMPDIR"),
    getenv("TMP"),

#if defined(__ANDROID__)
    "/data/local/tmp",
#endif

    // If all else fails
    "/tmp",
  };

  for (const char* d : candidates) {
    if (!d || d[0] == '\0') continue;  // Empty env var

    // Make sure we don't surprise anyone who's expecting a '/'
    string dstr = d;
    if (dstr[dstr.size() - 1] != '/') {
      dstr += "/";
    }

    struct stat statbuf;
    if (!stat(d, &statbuf) && S_ISDIR(statbuf.st_mode) &&
        !access(dstr.c_str(), 0)) {
      // We found a dir that exists and is accessible - we're done.
      list->push_back(dstr);
      return;
    }
  }
}

int setenv(const char* name, const char* value, int overwrite) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_18_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_18(mht_18_v, 533, "", "./tensorflow/core/platform/default/env.cc", "setenv");

  return ::setenv(name, value, overwrite);
}

int unsetenv(const char* name) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSenvDTcc mht_19(mht_19_v, 541, "", "./tensorflow/core/platform/default/env.cc", "unsetenv");
 return ::unsetenv(name); }

}  // namespace tensorflow
