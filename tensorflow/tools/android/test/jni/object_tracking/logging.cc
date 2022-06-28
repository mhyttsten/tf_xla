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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/android/test/jni/object_tracking/logging.h"

#ifdef STANDALONE_DEMO_LIB

#include <android/log.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <sstream>

LogMessage::LogMessage(const char* fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

void LogMessage::GenerateLogMessage() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc mht_0(mht_0_v, 198, "", "./tensorflow/tools/android/test/jni/object_tracking/logging.cc", "LogMessage::GenerateLogMessage");

  int android_log_level;
  switch (severity_) {
    case INFO:
      android_log_level = ANDROID_LOG_INFO;
      break;
    case WARNING:
      android_log_level = ANDROID_LOG_WARN;
      break;
    case ERROR:
      android_log_level = ANDROID_LOG_ERROR;
      break;
    case FATAL:
      android_log_level = ANDROID_LOG_FATAL;
      break;
    default:
      if (severity_ < INFO) {
        android_log_level = ANDROID_LOG_VERBOSE;
      } else {
        android_log_level = ANDROID_LOG_ERROR;
      }
      break;
  }

  std::stringstream ss;
  const char* const partial_name = strrchr(fname_, '/');
  ss << (partial_name != nullptr ? partial_name + 1 : fname_) << ":" << line_
     << " " << str();
  __android_log_write(android_log_level, "native", ss.str().c_str());

  // Also log to stderr (for standalone Android apps).
  std::cerr << "native : " << ss.str() << std::endl;

  // Android logging at level FATAL does not terminate execution, so abort()
  // is still required to stop the program.
  if (severity_ == FATAL) {
    abort();
  }
}

namespace {

// Parse log level (int64) from environment variable (char*)
int64_t LogLevelStrToInt(const char* tf_env_var_val) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tf_env_var_val: \"" + (tf_env_var_val == nullptr ? std::string("nullptr") : std::string((char*)tf_env_var_val)) + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc mht_1(mht_1_v, 245, "", "./tensorflow/tools/android/test/jni/object_tracking/logging.cc", "LogLevelStrToInt");

  if (tf_env_var_val == nullptr) {
    return 0;
  }

  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  std::string min_log_level(tf_env_var_val);
  std::istringstream ss(min_log_level);
  int64_t level;
  if (!(ss >> level)) {
    // Invalid vlog level setting, set level to default (0)
    level = 0;
  }

  return level;
}

int64_t MinLogLevelFromEnv() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc mht_2(mht_2_v, 267, "", "./tensorflow/tools/android/test/jni/object_tracking/logging.cc", "MinLogLevelFromEnv");

  const char* tf_env_var_val = getenv("TF_CPP_MIN_LOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
}

int64_t MinVLogLevelFromEnv() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc mht_3(mht_3_v, 275, "", "./tensorflow/tools/android/test/jni/object_tracking/logging.cc", "MinVLogLevelFromEnv");

  const char* tf_env_var_val = getenv("TF_CPP_MIN_VLOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
}

}  // namespace

LogMessage::~LogMessage() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc mht_4(mht_4_v, 285, "", "./tensorflow/tools/android/test/jni/object_tracking/logging.cc", "LogMessage::~LogMessage");

  // Read the min log level once during the first call to logging.
  static int64_t min_log_level = MinLogLevelFromEnv();
  if (TF_PREDICT_TRUE(severity_ >= min_log_level)) GenerateLogMessage();
}

int64_t LogMessage::MinVLogLevel() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc mht_5(mht_5_v, 294, "", "./tensorflow/tools/android/test/jni/object_tracking/logging.cc", "LogMessage::MinVLogLevel");

  static const int64_t min_vlog_level = MinVLogLevelFromEnv();
  return min_vlog_level;
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, ANDROID_LOG_FATAL) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc mht_6(mht_6_v, 304, "", "./tensorflow/tools/android/test/jni/object_tracking/logging.cc", "LogMessageFatal::LogMessageFatal");
}
LogMessageFatal::~LogMessageFatal() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc mht_7(mht_7_v, 308, "", "./tensorflow/tools/android/test/jni/object_tracking/logging.cc", "LogMessageFatal::~LogMessageFatal");

  // abort() ensures we don't return (we promised we would not via
  // ATTRIBUTE_NORETURN).
  GenerateLogMessage();
  abort();
}

void LogString(const char* fname, int line, int severity,
               const std::string& message) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("fname: \"" + (fname == nullptr ? std::string("nullptr") : std::string((char*)fname)) + "\"");
   mht_8_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc mht_8(mht_8_v, 321, "", "./tensorflow/tools/android/test/jni/object_tracking/logging.cc", "LogString");

  LogMessage(fname, line, severity) << message;
}

void LogPrintF(const int severity, const char* format, ...) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSloggingDTcc mht_9(mht_9_v, 329, "", "./tensorflow/tools/android/test/jni/object_tracking/logging.cc", "LogPrintF");

  char message[1024];
  va_list argptr;
  va_start(argptr, format);
  vsnprintf(message, 1024, format, argptr);
  va_end(argptr);
  __android_log_write(severity, "native", message);

  // Also log to stderr (for standalone Android apps).
  std::cerr << "native : " << message << std::endl;
}

#endif
