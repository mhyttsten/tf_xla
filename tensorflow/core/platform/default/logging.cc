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
class MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc() {
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

#include "tensorflow/core/platform/default/logging.h"

// TODO(b/142492876): Avoid depending on absl internal.
#include "absl/base/internal/cycleclock.h"
#include "absl/base/internal/sysinfo.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

#if defined(PLATFORM_POSIX_ANDROID)
#include <android/log.h>
#include <iostream>
#include <sstream>
#endif

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <queue>
#include <unordered_map>

namespace tensorflow {

namespace internal {
namespace {

// This is an internal singleton class that manages the log sinks. It allows
// adding and removing the log sinks, as well as handling sending log messages
// to all the registered log sinks.
class TFLogSinks {
 public:
  // Gets the TFLogSinks instance. This is the entry point for using this class.
  static TFLogSinks& Instance();

  // Adds a log sink. The sink argument must not be a nullptr. TFLogSinks
  // takes ownership of the pointer, the user must not free the pointer.
  // The pointer will remain valid until the application terminates or
  // until TFLogSinks::Remove is called for the same pointer value.
  void Add(TFLogSink* sink);

  // Removes a log sink. This will also erase the sink object. The pointer
  // to the sink becomes invalid after this call.
  void Remove(TFLogSink* sink);

  // Gets the currently registered log sinks.
  std::vector<TFLogSink*> GetSinks() const;

  // Sends a log message to all registered log sinks.
  //
  // If there are no log sinks are registered:
  //
  // NO_DEFAULT_LOGGER is defined:
  // Up to 128 messages will be queued until a log sink is added.
  // The queue will then be logged to the first added log sink.
  //
  // NO_DEFAULT_LOGGER is not defined:
  // The messages will be logged using the default logger. The default logger
  // will log to stdout on all platforms except for Android. On Androit the
  // default Android logger will be used.
  void Send(const TFLogEntry& entry);

 private:
  TFLogSinks();
  void SendToSink(TFLogSink& sink, const TFLogEntry& entry);

  std::queue<TFLogEntry> log_entry_queue_;
  static const size_t kMaxLogEntryQueueSize = 128;

  mutable tensorflow::mutex mutex_;
  std::vector<TFLogSink*> sinks_;
};

TFLogSinks::TFLogSinks() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_0(mht_0_v, 259, "", "./tensorflow/core/platform/default/logging.cc", "TFLogSinks::TFLogSinks");

#ifndef NO_DEFAULT_LOGGER
  static TFDefaultLogSink* default_sink = new TFDefaultLogSink();
  sinks_.emplace_back(default_sink);
#endif
}

TFLogSinks& TFLogSinks::Instance() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_1(mht_1_v, 269, "", "./tensorflow/core/platform/default/logging.cc", "TFLogSinks::Instance");

  static TFLogSinks* instance = new TFLogSinks();
  return *instance;
}

void TFLogSinks::Add(TFLogSink* sink) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_2(mht_2_v, 277, "", "./tensorflow/core/platform/default/logging.cc", "TFLogSinks::Add");

  assert(sink != nullptr && "The sink must not be a nullptr");

  tensorflow::mutex_lock lock(mutex_);
  sinks_.emplace_back(sink);

  // If this is the only sink log all the queued up messages to this sink
  if (sinks_.size() == 1) {
    while (!log_entry_queue_.empty()) {
      for (const auto& sink : sinks_) {
        SendToSink(*sink, log_entry_queue_.front());
      }
      log_entry_queue_.pop();
    }
  }
}

void TFLogSinks::Remove(TFLogSink* sink) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_3(mht_3_v, 297, "", "./tensorflow/core/platform/default/logging.cc", "TFLogSinks::Remove");

  assert(sink != nullptr && "The sink must not be a nullptr");

  tensorflow::mutex_lock lock(mutex_);
  auto it = std::find(sinks_.begin(), sinks_.end(), sink);
  if (it != sinks_.end()) sinks_.erase(it);
}

std::vector<TFLogSink*> TFLogSinks::GetSinks() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_4(mht_4_v, 308, "", "./tensorflow/core/platform/default/logging.cc", "TFLogSinks::GetSinks");

  tensorflow::mutex_lock lock(mutex_);
  return sinks_;
}

void TFLogSinks::Send(const TFLogEntry& entry) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_5(mht_5_v, 316, "", "./tensorflow/core/platform/default/logging.cc", "TFLogSinks::Send");

  tensorflow::mutex_lock lock(mutex_);

  // If we don't have any sinks registered, queue them up
  if (sinks_.empty()) {
    // If we've exceeded the maximum queue size, drop the oldest entries
    while (log_entry_queue_.size() >= kMaxLogEntryQueueSize) {
      log_entry_queue_.pop();
    }
    log_entry_queue_.push(entry);
    return;
  }

  // If we have items in the queue, push them out first
  while (!log_entry_queue_.empty()) {
    for (const auto& sink : sinks_) {
      SendToSink(*sink, log_entry_queue_.front());
    }
    log_entry_queue_.pop();
  }

  // ... and now we can log the current log entry
  for (const auto& sink : sinks_) {
    SendToSink(*sink, entry);
  }
}

void TFLogSinks::SendToSink(TFLogSink& sink, const TFLogEntry& entry) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_6(mht_6_v, 346, "", "./tensorflow/core/platform/default/logging.cc", "TFLogSinks::SendToSink");

  sink.Send(entry);
  sink.WaitTillSent();
}

// A class for managing the text file to which VLOG output is written.
// If the environment variable TF_CPP_VLOG_FILENAME is set, all VLOG
// calls are redirected from stderr to a file with corresponding name.
class VlogFileMgr {
 public:
  // Determines if the env variable is set and if necessary
  // opens the file for write access.
  VlogFileMgr();
  // Closes the file.
  ~VlogFileMgr();
  // Returns either a pointer to the file or stderr.
  FILE* FilePtr() const;

 private:
  FILE* vlog_file_ptr;
  char* vlog_file_name;
};

VlogFileMgr::VlogFileMgr() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_7(mht_7_v, 372, "", "./tensorflow/core/platform/default/logging.cc", "VlogFileMgr::VlogFileMgr");

  vlog_file_name = getenv("TF_CPP_VLOG_FILENAME");
  vlog_file_ptr =
      vlog_file_name == nullptr ? nullptr : fopen(vlog_file_name, "w");

  if (vlog_file_ptr == nullptr) {
    vlog_file_ptr = stderr;
  }
}

VlogFileMgr::~VlogFileMgr() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_8(mht_8_v, 385, "", "./tensorflow/core/platform/default/logging.cc", "VlogFileMgr::~VlogFileMgr");

  if (vlog_file_ptr != stderr) {
    fclose(vlog_file_ptr);
  }
}

FILE* VlogFileMgr::FilePtr() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_9(mht_9_v, 394, "", "./tensorflow/core/platform/default/logging.cc", "VlogFileMgr::FilePtr");
 return vlog_file_ptr; }

int ParseInteger(const char* str, size_t size) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_10(mht_10_v, 400, "", "./tensorflow/core/platform/default/logging.cc", "ParseInteger");

  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  string integer_str(str, size);
  std::istringstream ss(integer_str);
  int level = 0;
  ss >> level;
  return level;
}

// Parse log level (int64) from environment variable (char*)
int64_t LogLevelStrToInt(const char* tf_env_var_val) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("tf_env_var_val: \"" + (tf_env_var_val == nullptr ? std::string("nullptr") : std::string((char*)tf_env_var_val)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_11(mht_11_v, 416, "", "./tensorflow/core/platform/default/logging.cc", "LogLevelStrToInt");

  if (tf_env_var_val == nullptr) {
    return 0;
  }
  return ParseInteger(tf_env_var_val, strlen(tf_env_var_val));
}

// Using StringPiece breaks Windows build.
struct StringData {
  struct Hasher {
    size_t operator()(const StringData& sdata) const {
      // For dependency reasons, we cannot use hash.h here. Use DBJHash instead.
      size_t hash = 5381;
      const char* data = sdata.data;
      for (const char* top = data + sdata.size; data < top; ++data) {
        hash = ((hash << 5) + hash) + (*data);
      }
      return hash;
    }
  };

  StringData() = default;
  StringData(const char* data, size_t size) : data(data), size(size) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_12(mht_12_v, 442, "", "./tensorflow/core/platform/default/logging.cc", "StringData");
}

  bool operator==(const StringData& rhs) const {
    return size == rhs.size && memcmp(data, rhs.data, size) == 0;
  }

  const char* data = nullptr;
  size_t size = 0;
};

using VmoduleMap = std::unordered_map<StringData, int, StringData::Hasher>;

// Returns a mapping from module name to VLOG level, derived from the
// TF_CPP_VMODULE environment variable; ownership is transferred to the caller.
VmoduleMap* VmodulesMapFromEnv() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_13(mht_13_v, 459, "", "./tensorflow/core/platform/default/logging.cc", "VmodulesMapFromEnv");

  // The value of the env var is supposed to be of the form:
  //    "foo=1,bar=2,baz=3"
  const char* env = getenv("TF_CPP_VMODULE");
  if (env == nullptr) {
    // If there is no TF_CPP_VMODULE configuration (most common case), return
    // nullptr so that the ShouldVlogModule() API can fast bail out of it.
    return nullptr;
  }
  // The memory returned by getenv() can be invalidated by following getenv() or
  // setenv() calls. And since we keep references to it in the VmoduleMap in
  // form of StringData objects, make a copy of it.
  const char* env_data = strdup(env);
  VmoduleMap* result = new VmoduleMap();
  while (true) {
    const char* eq = strchr(env_data, '=');
    if (eq == nullptr) {
      break;
    }
    const char* after_eq = eq + 1;

    // Comma either points at the next comma delimiter, or at a null terminator.
    // We check that the integer we parse ends at this delimiter.
    const char* comma = strchr(after_eq, ',');
    const char* new_env_data;
    if (comma == nullptr) {
      comma = strchr(after_eq, '\0');
      new_env_data = comma;
    } else {
      new_env_data = comma + 1;
    }
    (*result)[StringData(env_data, eq - env_data)] =
        ParseInteger(after_eq, comma - after_eq);
    env_data = new_env_data;
  }
  return result;
}

bool EmitThreadIdFromEnv() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_14(mht_14_v, 500, "", "./tensorflow/core/platform/default/logging.cc", "EmitThreadIdFromEnv");

  const char* tf_env_var_val = getenv("TF_CPP_LOG_THREAD_ID");
  return tf_env_var_val == nullptr
             ? false
             : ParseInteger(tf_env_var_val, strlen(tf_env_var_val)) != 0;
}

}  // namespace

int64_t MinLogLevelFromEnv() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_15(mht_15_v, 512, "", "./tensorflow/core/platform/default/logging.cc", "MinLogLevelFromEnv");

  // We don't want to print logs during fuzzing as that would slow fuzzing down
  // by almost 2x. So, if we are in fuzzing mode (not just running a test), we
  // return a value so that nothing is actually printed. Since LOG uses >=
  // (see ~LogMessage in this file) to see if log messages need to be printed,
  // the value we're interested on to disable printing is the maximum severity.
  // See also http://llvm.org/docs/LibFuzzer.html#fuzzer-friendly-build-mode
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  return tensorflow::NUM_SEVERITIES;
#else
  const char* tf_env_var_val = getenv("TF_CPP_MIN_LOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
#endif
}

int64_t MaxVLogLevelFromEnv() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_16(mht_16_v, 530, "", "./tensorflow/core/platform/default/logging.cc", "MaxVLogLevelFromEnv");

  // We don't want to print logs during fuzzing as that would slow fuzzing down
  // by almost 2x. So, if we are in fuzzing mode (not just running a test), we
  // return a value so that nothing is actually printed. Since VLOG uses <=
  // (see VLOG_IS_ON in logging.h) to see if log messages need to be printed,
  // the value we're interested on to disable printing is 0.
  // See also http://llvm.org/docs/LibFuzzer.html#fuzzer-friendly-build-mode
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  return 0;
#else
  const char* tf_env_var_val = getenv("TF_CPP_MAX_VLOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
#endif
}

LogMessage::LogMessage(const char* fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("fname: \"" + (fname == nullptr ? std::string("nullptr") : std::string((char*)fname)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_17(mht_17_v, 550, "", "./tensorflow/core/platform/default/logging.cc", "LogMessage::LogMessage");
}

LogMessage& LogMessage::AtLocation(const char* fname, int line) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("fname: \"" + (fname == nullptr ? std::string("nullptr") : std::string((char*)fname)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_18(mht_18_v, 556, "", "./tensorflow/core/platform/default/logging.cc", "LogMessage::AtLocation");

  fname_ = fname;
  line_ = line;
  return *this;
}

LogMessage::~LogMessage() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_19(mht_19_v, 565, "", "./tensorflow/core/platform/default/logging.cc", "LogMessage::~LogMessage");

  // Read the min log level once during the first call to logging.
  static int64_t min_log_level = MinLogLevelFromEnv();
  if (severity_ >= min_log_level) {
    GenerateLogMessage();
  }
}

void LogMessage::GenerateLogMessage() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_20(mht_20_v, 576, "", "./tensorflow/core/platform/default/logging.cc", "LogMessage::GenerateLogMessage");

  TFLogSinks::Instance().Send(TFLogEntry(severity_, fname_, line_, str()));
}

int64_t LogMessage::MaxVLogLevel() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_21(mht_21_v, 583, "", "./tensorflow/core/platform/default/logging.cc", "LogMessage::MaxVLogLevel");

  static int64_t max_vlog_level = MaxVLogLevelFromEnv();
  return max_vlog_level;
}

bool LogMessage::VmoduleActivated(const char* fname, int level) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("fname: \"" + (fname == nullptr ? std::string("nullptr") : std::string((char*)fname)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_22(mht_22_v, 592, "", "./tensorflow/core/platform/default/logging.cc", "LogMessage::VmoduleActivated");

  if (level <= MaxVLogLevel()) {
    return true;
  }
  static VmoduleMap* vmodules = VmodulesMapFromEnv();
  if (TF_PREDICT_TRUE(vmodules == nullptr)) {
    return false;
  }
  const char* last_slash = strrchr(fname, '/');
  const char* module_start = last_slash == nullptr ? fname : last_slash + 1;
  const char* dot_after = strchr(module_start, '.');
  const char* module_limit =
      dot_after == nullptr ? strchr(fname, '\0') : dot_after;
  StringData module(module_start, module_limit - module_start);
  auto it = vmodules->find(module);
  return it != vmodules->end() && it->second >= level;
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_23(mht_23_v, 615, "", "./tensorflow/core/platform/default/logging.cc", "LogMessageFatal::LogMessageFatal");
}
LogMessageFatal::~LogMessageFatal() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_24(mht_24_v, 619, "", "./tensorflow/core/platform/default/logging.cc", "LogMessageFatal::~LogMessageFatal");

  // abort() ensures we don't return (we promised we would not via
  // ATTRIBUTE_NORETURN).
  GenerateLogMessage();
  abort();
}

void LogString(const char* fname, int line, int severity,
               const string& message) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("fname: \"" + (fname == nullptr ? std::string("nullptr") : std::string((char*)fname)) + "\"");
   mht_25_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_25(mht_25_v, 632, "", "./tensorflow/core/platform/default/logging.cc", "LogString");

  LogMessage(fname, line, severity) << message;
}

template <>
void MakeCheckOpValueString(std::ostream* os, const char& v) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_26(mht_26_v, 640, "", "./tensorflow/core/platform/default/logging.cc", "MakeCheckOpValueString");

  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << static_cast<int16>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_27(mht_27_v, 652, "", "./tensorflow/core/platform/default/logging.cc", "MakeCheckOpValueString");

  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << static_cast<int16>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_28(mht_28_v, 664, "", "./tensorflow/core/platform/default/logging.cc", "MakeCheckOpValueString");

  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << static_cast<uint16>(v);
  }
}

#if LANG_CXX11
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& v) {
  (*os) << "nullptr";
}
#endif

CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << "Check failed: " << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_29(mht_29_v, 687, "", "./tensorflow/core/platform/default/logging.cc", "CheckOpMessageBuilder::~CheckOpMessageBuilder");
 delete stream_; }

std::ostream* CheckOpMessageBuilder::ForVar2() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_30(mht_30_v, 692, "", "./tensorflow/core/platform/default/logging.cc", "CheckOpMessageBuilder::ForVar2");

  *stream_ << " vs. ";
  return stream_;
}

string* CheckOpMessageBuilder::NewString() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_31(mht_31_v, 700, "", "./tensorflow/core/platform/default/logging.cc", "CheckOpMessageBuilder::NewString");

  *stream_ << ")";
  return new string(stream_->str());
}

namespace {
// The following code behaves like AtomicStatsCounter::LossyAdd() for
// speed since it is fine to lose occasional updates.
// Returns old value of *counter.
uint32 LossyIncrement(std::atomic<uint32>* counter) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_32(mht_32_v, 712, "", "./tensorflow/core/platform/default/logging.cc", "LossyIncrement");

  const uint32 value = counter->load(std::memory_order_relaxed);
  counter->store(value + 1, std::memory_order_relaxed);
  return value;
}
}  // namespace

bool LogEveryNState::ShouldLog(int n) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_33(mht_33_v, 722, "", "./tensorflow/core/platform/default/logging.cc", "LogEveryNState::ShouldLog");

  return n != 0 && (LossyIncrement(&counter_) % n) == 0;
}

bool LogFirstNState::ShouldLog(int n) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_34(mht_34_v, 729, "", "./tensorflow/core/platform/default/logging.cc", "LogFirstNState::ShouldLog");

  const int counter_value =
      static_cast<int>(counter_.load(std::memory_order_relaxed));
  if (counter_value < n) {
    counter_.store(counter_value + 1, std::memory_order_relaxed);
    return true;
  }
  return false;
}

bool LogEveryPow2State::ShouldLog(int ignored) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_35(mht_35_v, 742, "", "./tensorflow/core/platform/default/logging.cc", "LogEveryPow2State::ShouldLog");

  const uint32 new_value = LossyIncrement(&counter_) + 1;
  return (new_value & (new_value - 1)) == 0;
}

bool LogEveryNSecState::ShouldLog(double seconds) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_36(mht_36_v, 750, "", "./tensorflow/core/platform/default/logging.cc", "LogEveryNSecState::ShouldLog");

  LossyIncrement(&counter_);
  const int64_t now_cycles = absl::base_internal::CycleClock::Now();
  int64_t next_cycles = next_log_time_cycles_.load(std::memory_order_relaxed);
  do {
    if (now_cycles <= next_cycles) return false;
  } while (!next_log_time_cycles_.compare_exchange_weak(
      next_cycles,
      now_cycles + seconds * absl::base_internal::CycleClock::Frequency(),
      std::memory_order_relaxed, std::memory_order_relaxed));
  return true;
}

}  // namespace internal

void TFAddLogSink(TFLogSink* sink) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_37(mht_37_v, 768, "", "./tensorflow/core/platform/default/logging.cc", "TFAddLogSink");

  internal::TFLogSinks::Instance().Add(sink);
}

void TFRemoveLogSink(TFLogSink* sink) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_38(mht_38_v, 775, "", "./tensorflow/core/platform/default/logging.cc", "TFRemoveLogSink");

  internal::TFLogSinks::Instance().Remove(sink);
}

std::vector<TFLogSink*> TFGetLogSinks() {
  return internal::TFLogSinks::Instance().GetSinks();
}

void TFDefaultLogSink::Send(const TFLogEntry& entry) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_39(mht_39_v, 786, "", "./tensorflow/core/platform/default/logging.cc", "TFDefaultLogSink::Send");

#ifdef PLATFORM_POSIX_ANDROID
  int android_log_level;
  switch (entry.log_severity()) {
    case absl::LogSeverity::kInfo:
      android_log_level = ANDROID_LOG_INFO;
      break;
    case absl::LogSeverity::kWarning:
      android_log_level = ANDROID_LOG_WARN;
      break;
    case absl::LogSeverity::kError:
      android_log_level = ANDROID_LOG_ERROR;
      break;
    case absl::LogSeverity::kFatal:
      android_log_level = ANDROID_LOG_FATAL;
      break;
    default:
      if (entry.log_severity() < absl::LogSeverity::kInfo) {
        android_log_level = ANDROID_LOG_VERBOSE;
      } else {
        android_log_level = ANDROID_LOG_ERROR;
      }
      break;
  }

  std::stringstream ss;
  const auto& fname = entry.FName();
  auto pos = fname.find("/");
  ss << (pos != std::string::npos ? fname.substr(pos + 1) : fname) << ":"
     << entry.Line() << " " << entry.ToString();
  __android_log_write(android_log_level, "native", ss.str().c_str());

  // Also log to stderr (for standalone Android apps).
  // Don't use 'std::cerr' since it crashes on Android.
  fprintf(stderr, "native : %s\n", ss.str().c_str());

  // Android logging at level FATAL does not terminate execution, so abort()
  // is still required to stop the program.
  if (entry.log_severity() == absl::LogSeverity::kFatal) {
    abort();
  }
#else   // PLATFORM_POSIX_ANDROID
  static const internal::VlogFileMgr vlog_file;
  static bool log_thread_id = internal::EmitThreadIdFromEnv();
  uint64 now_micros = EnvTime::NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32_t micros_remainder = static_cast<int32>(now_micros % 1000000);
  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];
  strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S",
           localtime(&now_seconds));
  const size_t tid_buffer_size = 10;
  char tid_buffer[tid_buffer_size] = "";
  if (log_thread_id) {
    snprintf(tid_buffer, sizeof(tid_buffer), " %7u",
             absl::base_internal::GetTID());
  }

  char sev;
  switch (entry.log_severity()) {
    case absl::LogSeverity::kInfo:
      sev = 'I';
      break;

    case absl::LogSeverity::kWarning:
      sev = 'W';
      break;

    case absl::LogSeverity::kError:
      sev = 'E';
      break;

    case absl::LogSeverity::kFatal:
      sev = 'F';
      break;

    default:
      assert(false && "Unknown logging severity");
      sev = '?';
      break;
  }

  fprintf(vlog_file.FilePtr(), "%s.%06d: %c%s %s:%d] %s\n", time_buffer,
          micros_remainder, sev, tid_buffer, entry.FName().c_str(),
          entry.Line(), entry.ToString().c_str());
#endif  // PLATFORM_POSIX_ANDROID
}

void UpdateLogVerbosityIfDefined(const char* env_var) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("env_var: \"" + (env_var == nullptr ? std::string("nullptr") : std::string((char*)env_var)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTcc mht_40(mht_40_v, 878, "", "./tensorflow/core/platform/default/logging.cc", "UpdateLogVerbosityIfDefined");
}

}  // namespace tensorflow
