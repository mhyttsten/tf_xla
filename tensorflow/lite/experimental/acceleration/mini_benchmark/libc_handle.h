/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBC_HANDLE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBC_HANDLE_H_
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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibc_handleDTh {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibc_handleDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibc_handleDTh() {
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


#ifdef __ANDROID__
#include <dlfcn.h>
#endif

#include <cstdio>
#include <memory>
#include <utility>

#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

// This class offers a handle to C Standard Library (LibC) shared object
// library on Android. It offers pointers to functions in LibC.
// Fmemopen is available as native API from Android SDK 23 onwards. In order to
// support Android devices from SDK 21 onwards, we load fmemopen dynamically
// from the libc shared object library.
// TODO(b/172544567): Support Apple.
class LibCHandle {
 public:
  // Factory to get an initialised instance of LibCHandle.
  // Loads the libc dynamic library and gets handle to all the
  // required functions. Stores the initialisation status in `status`.
  static LibCHandle Create(Status& status);
  LibCHandle(LibCHandle const&) = delete;
  LibCHandle& operator=(const LibCHandle&) = delete;
  LibCHandle(LibCHandle&& other)
      : libc_(std::exchange(other.libc_, nullptr)),
        fmemopen_(std::exchange(other.fmemopen_, nullptr)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibc_handleDTh mht_0(mht_0_v, 217, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libc_handle.h", "LibCHandle");
}
  LibCHandle& operator=(LibCHandle&& other) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibc_handleDTh mht_1(mht_1_v, 221, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libc_handle.h", "=");

    if (&other != this) {
      CloseHandle();
      libc_ = std::exchange(other.libc_, nullptr);
      fmemopen_ = std::exchange(other.fmemopen_, nullptr);
    }
    return *this;
  }
  ~LibCHandle() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibc_handleDTh mht_2(mht_2_v, 232, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libc_handle.h", "~LibCHandle");
 CloseHandle(); }
  // Definition can be found here
  // https://man7.org/linux/man-pages/man3/fmemopen.3.html
  FILE* fmemopen(void* buf, size_t size, const char* mode) const;

 private:
  using FmemopenPtr = FILE* (*)(void*, size_t, const char*);
  LibCHandle(void* libc, FmemopenPtr ptr) : libc_(libc), fmemopen_(ptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibc_handleDTh mht_3(mht_3_v, 242, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libc_handle.h", "LibCHandle");
}
  // Closes the dynamic library loaded in libc_.
  void CloseHandle() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibc_handleDTh mht_4(mht_4_v, 247, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libc_handle.h", "CloseHandle");

#ifdef __ANDROID__
    if (libc_ != nullptr) {
      dlclose(libc_);
    }
#endif
  }
  void* libc_ = nullptr;
  FmemopenPtr fmemopen_ = nullptr;
};

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBC_HANDLE_H_
