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
class MHTracer_DTPStensorflowPSlitePSallocationDTcc {
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
   MHTracer_DTPStensorflowPSlitePSallocationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSallocationDTcc() {
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

#include "tensorflow/lite/allocation.h"

#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <cstdio>
#include <memory>

#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {

#ifndef TFLITE_MCU
FileCopyAllocation::FileCopyAllocation(const char* filename,
                                       ErrorReporter* error_reporter)
    : Allocation(error_reporter, Allocation::Type::kFileCopy) {
  // Obtain the file size using fstat, or report an error if that fails.
  std::unique_ptr<FILE, decltype(&fclose)> file(fopen(filename, "rb"), fclose);
  if (!file) {
    error_reporter_->Report("Could not open '%s'.", filename);
    return;
  }
  struct stat sb;

// support usage of msvc's posix-like fileno symbol
#ifdef _WIN32
#define FILENO(_x) _fileno(_x)
#else
#define FILENO(_x) fileno(_x)
#endif
  if (fstat(FILENO(file.get()), &sb) != 0) {
    error_reporter_->Report("Failed to get file size of '%s'.", filename);
    return;
  }
#undef FILENO
  buffer_size_bytes_ = sb.st_size;
  std::unique_ptr<char[]> buffer(new char[buffer_size_bytes_]);
  if (!buffer) {
    error_reporter_->Report("Malloc of buffer to hold copy of '%s' failed.",
                            filename);
    return;
  }
  size_t bytes_read =
      fread(buffer.get(), sizeof(char), buffer_size_bytes_, file.get());
  if (bytes_read != buffer_size_bytes_) {
    error_reporter_->Report("Read of '%s' failed (too few bytes read).",
                            filename);
    return;
  }
  // Versions of GCC before 6.2.0 don't support std::move from non-const
  // char[] to const char[] unique_ptrs.
  copied_buffer_.reset(const_cast<char const*>(buffer.release()));
}

FileCopyAllocation::~FileCopyAllocation() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTcc mht_0(mht_0_v, 241, "", "./tensorflow/lite/allocation.cc", "FileCopyAllocation::~FileCopyAllocation");
}

const void* FileCopyAllocation::base() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTcc mht_1(mht_1_v, 246, "", "./tensorflow/lite/allocation.cc", "FileCopyAllocation::base");
 return copied_buffer_.get(); }

size_t FileCopyAllocation::bytes() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTcc mht_2(mht_2_v, 251, "", "./tensorflow/lite/allocation.cc", "FileCopyAllocation::bytes");
 return buffer_size_bytes_; }

bool FileCopyAllocation::valid() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTcc mht_3(mht_3_v, 256, "", "./tensorflow/lite/allocation.cc", "FileCopyAllocation::valid");
 return copied_buffer_ != nullptr; }
#endif

MemoryAllocation::MemoryAllocation(const void* ptr, size_t num_bytes,
                                   ErrorReporter* error_reporter)
    : Allocation(error_reporter, Allocation::Type::kMemory) {
#ifdef __arm__
  if ((reinterpret_cast<uintptr_t>(ptr) & 0x3) != 0) {
    // The flatbuffer schema has alignment requirements of up to 16 bytes to
    // guarantee that data can be correctly accesses by various backends.
    // Therefore, model pointer should also be 16-bytes aligned to preserve this
    // requirement. But this condition only checks 4-bytes alignment which is
    // the mininum requirement to prevent SIGBUS fault on 32bit ARM. Some models
    // could require 8 or 16 bytes alignment which is not checked yet.
    //
    // Note that 64-bit ARM may also suffer a performance impact, but no crash -
    // that case is not checked.
    TF_LITE_REPORT_ERROR(error_reporter,
                         "The supplied buffer is not 4-bytes aligned");
    buffer_ = nullptr;
    buffer_size_bytes_ = 0;
    return;
  }
#endif  // __arm__

  buffer_ = ptr;
  buffer_size_bytes_ = num_bytes;
}

MemoryAllocation::~MemoryAllocation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTcc mht_4(mht_4_v, 288, "", "./tensorflow/lite/allocation.cc", "MemoryAllocation::~MemoryAllocation");
}

const void* MemoryAllocation::base() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTcc mht_5(mht_5_v, 293, "", "./tensorflow/lite/allocation.cc", "MemoryAllocation::base");
 return buffer_; }

size_t MemoryAllocation::bytes() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTcc mht_6(mht_6_v, 298, "", "./tensorflow/lite/allocation.cc", "MemoryAllocation::bytes");
 return buffer_size_bytes_; }

bool MemoryAllocation::valid() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTcc mht_7(mht_7_v, 303, "", "./tensorflow/lite/allocation.cc", "MemoryAllocation::valid");
 return buffer_ != nullptr; }

}  // namespace tflite
