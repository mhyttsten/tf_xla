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
#ifndef TENSORFLOW_LITE_TOCO_TOCO_PORT_H_
#define TENSORFLOW_LITE_TOCO_TOCO_PORT_H_
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
class MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTh {
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
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTh() {
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


// Portability layer for toco tool. Mainly, abstract filesystem access so we
// can build and use on google internal environments and on OSX.

#include <string>
#include "google/protobuf/text_format.h"
#include "tensorflow/lite/toco/format_port.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#if defined(PLATFORM_GOOGLE)
#include "absl/strings/cord.h"
#endif  // PLATFORM_GOOGLE

#ifdef PLATFORM_GOOGLE
#define TFLITE_PROTO_NS proto2
#else
#define TFLITE_PROTO_NS google::protobuf
#endif

#ifdef __ANDROID__
#include <sstream>
namespace std {

template <typename T>
std::string to_string(T value)
{
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTh mht_0(mht_0_v, 212, "", "./tensorflow/lite/toco/toco_port.h", "to_string");

    std::ostringstream os ;
    os << value ;
    return os.str() ;
}

#ifdef __ARM_ARCH_7A__
double round(double x);
#endif
}
#endif

namespace toco {
namespace port {

// Things like tests use other initialization routines that need control
// of flags. However, for testing we still want to use toco_port.h facilities.
// This function sets initialized flag trivially.
void InitGoogleWasDoneElsewhere();
void InitGoogle(const char* usage, int* argc, char*** argv, bool remove_flags);
void CheckInitGoogleIsDone(const char* message);

namespace file {
class Options {};
inline Options Defaults() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTh mht_1(mht_1_v, 239, "", "./tensorflow/lite/toco/toco_port.h", "Defaults");

  Options o;
  return o;
}
tensorflow::Status GetContents(const std::string& filename,
                               std::string* contents, const Options& options);
tensorflow::Status SetContents(const std::string& filename,
                               const std::string& contents,
                               const Options& options);
std::string JoinPath(const std::string& base, const std::string& filename);
tensorflow::Status Writable(const std::string& filename);
tensorflow::Status Readable(const std::string& filename,
                            const Options& options);
tensorflow::Status Exists(const std::string& filename, const Options& options);
}  // namespace file

// Copy `src` string to `dest`. User must ensure `dest` has enough space.
#if defined(PLATFORM_GOOGLE)
void CopyToBuffer(const ::absl::Cord& src, char* dest);
#endif  // PLATFORM_GOOGLE
void CopyToBuffer(const std::string& src, char* dest);

inline uint32 ReverseBits32(uint32 n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTh mht_2(mht_2_v, 264, "", "./tensorflow/lite/toco/toco_port.h", "ReverseBits32");

  n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1);
  n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2);
  n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4);
  return (((n & 0xFF) << 24) | ((n & 0xFF00) << 8) | ((n & 0xFF0000) >> 8) |
          ((n & 0xFF000000) >> 24));
}
}  // namespace port

inline bool ParseFromStringOverload(const std::string& in,
                                    TFLITE_PROTO_NS::Message* proto) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("in: \"" + in + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTh mht_3(mht_3_v, 278, "", "./tensorflow/lite/toco/toco_port.h", "ParseFromStringOverload");

  return TFLITE_PROTO_NS::TextFormat::ParseFromString(in, proto);
}

template <typename Proto>
bool ParseFromStringEitherTextOrBinary(const std::string& input_file_contents,
                                       Proto* proto) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("input_file_contents: \"" + input_file_contents + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTh mht_4(mht_4_v, 288, "", "./tensorflow/lite/toco/toco_port.h", "ParseFromStringEitherTextOrBinary");

  if (proto->ParseFromString(input_file_contents)) {
    return true;
  }

  if (ParseFromStringOverload(input_file_contents, proto)) {
    return true;
  }

  return false;
}

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TOCO_PORT_H_
