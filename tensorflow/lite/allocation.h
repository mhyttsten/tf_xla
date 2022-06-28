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
/// \file
/// Memory management for TF Lite.
#ifndef TENSORFLOW_LITE_ALLOCATION_H_
#define TENSORFLOW_LITE_ALLOCATION_H_
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
class MHTracer_DTPStensorflowPSlitePSallocationDTh {
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
   MHTracer_DTPStensorflowPSlitePSallocationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSallocationDTh() {
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


#include <stddef.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {

// A memory allocation handle. This could be a mmap or shared memory.
class Allocation {
 public:
  virtual ~Allocation() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTh mht_0(mht_0_v, 202, "", "./tensorflow/lite/allocation.h", "~Allocation");
}

  enum class Type {
    kMMap,
    kFileCopy,
    kMemory,
  };

  // Base pointer of this allocation
  virtual const void* base() const = 0;
  // Size in bytes of the allocation
  virtual size_t bytes() const = 0;
  // Whether the allocation is valid
  virtual bool valid() const = 0;
  // Return the type of the Allocation.
  Type type() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTh mht_1(mht_1_v, 220, "", "./tensorflow/lite/allocation.h", "type");
 return type_; }

 protected:
  Allocation(ErrorReporter* error_reporter, Type type)
      : error_reporter_(error_reporter), type_(type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTh mht_2(mht_2_v, 227, "", "./tensorflow/lite/allocation.h", "Allocation");
}
  ErrorReporter* error_reporter_;

 private:
  const Type type_;
};

// Note that not all platforms support MMAP-based allocation.
// Use `IsSupported()` to check.
class MMAPAllocation : public Allocation {
 public:
  // Loads and maps the provided file to a memory region.
  MMAPAllocation(const char* filename, ErrorReporter* error_reporter);

  // Maps the provided file descriptor to a memory region.
  // Note: The provided file descriptor will be dup'ed for usage; the caller
  // retains ownership of the provided descriptor and should close accordingly.
  MMAPAllocation(int fd, ErrorReporter* error_reporter);

  // Maps the provided file descriptor, with the given offset and length (both
  // in bytes), to a memory region.
  // Note: The provided file descriptor will be dup'ed for usage; the caller
  // retains ownership of the provided descriptor and should close accordingly.
  MMAPAllocation(int fd, size_t offset, size_t length,
                 ErrorReporter* error_reporter);

  virtual ~MMAPAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

  int fd() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSallocationDTh mht_3(mht_3_v, 261, "", "./tensorflow/lite/allocation.h", "fd");
 return mmap_fd_; }

  static bool IsSupported();

 protected:
  // Data required for mmap.
  int mmap_fd_ = -1;  // mmap file descriptor
  const void* mmapped_buffer_;
  size_t buffer_size_bytes_ = 0;
  // Used when the address to mmap is not page-aligned.
  size_t offset_in_buffer_ = 0;

 private:
  // Assumes ownership of the provided `owned_fd` instance.
  MMAPAllocation(ErrorReporter* error_reporter, int owned_fd);

  // Assumes ownership of the provided `owned_fd` instance, and uses the given
  // offset and length (both in bytes) for memory mapping.
  MMAPAllocation(ErrorReporter* error_reporter, int owned_fd, size_t offset,
                 size_t length);
};

class FileCopyAllocation : public Allocation {
 public:
  // Loads the provided file into a heap memory region.
  FileCopyAllocation(const char* filename, ErrorReporter* error_reporter);
  virtual ~FileCopyAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

 private:
  std::unique_ptr<const char[]> copied_buffer_;
  size_t buffer_size_bytes_ = 0;
};

class MemoryAllocation : public Allocation {
 public:
  // Provides a (read-only) view of the provided buffer region as an allocation.
  // Note: The caller retains ownership of `ptr`, and must ensure it remains
  // valid for the lifetime of the class instance.
  MemoryAllocation(const void* ptr, size_t num_bytes,
                   ErrorReporter* error_reporter);
  virtual ~MemoryAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

 private:
  const void* buffer_;
  size_t buffer_size_bytes_ = 0;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_ALLOCATION_H_
