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
class MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc {
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
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <fcntl.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
namespace {

size_t GetFdSizeBytes(int fd) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/mmap_allocation.cc", "GetFdSizeBytes");

  if (fd < 0) {
    return 0;
  }

  struct stat fd_stat;
  if (fstat(fd, &fd_stat) != 0) {
    return 0;
  }

  return fd_stat.st_size;
}

}  // namespace

MMAPAllocation::MMAPAllocation(const char* filename,
                               ErrorReporter* error_reporter)
    : MMAPAllocation(error_reporter, open(filename, O_RDONLY)) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + (filename == nullptr ? std::string("nullptr") : std::string((char*)filename)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_1(mht_1_v, 220, "", "./tensorflow/lite/mmap_allocation.cc", "MMAPAllocation::MMAPAllocation");

  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Could not open '%s'.", filename);
  }
}

MMAPAllocation::MMAPAllocation(int fd, ErrorReporter* error_reporter)
    : MMAPAllocation(error_reporter, dup(fd)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/mmap_allocation.cc", "MMAPAllocation::MMAPAllocation");

  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to dup '%d' file descriptor.",
                         fd);
  }
}

MMAPAllocation::MMAPAllocation(int fd, size_t offset, size_t length,
                               ErrorReporter* error_reporter)
    : MMAPAllocation(error_reporter, dup(fd), offset, length) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_3(mht_3_v, 242, "", "./tensorflow/lite/mmap_allocation.cc", "MMAPAllocation::MMAPAllocation");

  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to dup '%d' file descriptor.",
                         fd);
  }
}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd)
    : MMAPAllocation(error_reporter, owned_fd, /*offset=*/0,
                     /*length=*/GetFdSizeBytes(owned_fd)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_4(mht_4_v, 254, "", "./tensorflow/lite/mmap_allocation.cc", "MMAPAllocation::MMAPAllocation");
}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd,
                               size_t offset, size_t length)
    : Allocation(error_reporter, Allocation::Type::kMMap),
      mmap_fd_(owned_fd),
      mmapped_buffer_(MAP_FAILED),
      buffer_size_bytes_(length) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_5(mht_5_v, 264, "", "./tensorflow/lite/mmap_allocation.cc", "MMAPAllocation::MMAPAllocation");

  if (owned_fd < 0) {
    return;
  }

#ifdef __ANDROID__
  static int pagesize = getpagesize();
#else
  static int pagesize = sysconf(_SC_PAGE_SIZE);
#endif

  offset_in_buffer_ = offset % pagesize;

  size_t file_size = GetFdSizeBytes(mmap_fd_);
  if (length + offset > file_size) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Asked to mmap '%d' bytes from fd '%d' at offset "
                         "'%d'. This is over the length of file '%d'.",
                         length, mmap_fd_, offset, file_size);
    return;
  }

  mmapped_buffer_ =
      mmap(nullptr, /*__len=*/length + offset_in_buffer_, PROT_READ, MAP_SHARED,
           mmap_fd_, /*__offset=*/offset - offset_in_buffer_);
  if (mmapped_buffer_ == MAP_FAILED) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Mmap of '%d' at offset '%d' failed with error '%d'.",
                         mmap_fd_, offset, errno);
    return;
  }
}

MMAPAllocation::~MMAPAllocation() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_6(mht_6_v, 300, "", "./tensorflow/lite/mmap_allocation.cc", "MMAPAllocation::~MMAPAllocation");

  if (valid()) {
    munmap(const_cast<void*>(mmapped_buffer_),
           buffer_size_bytes_ + offset_in_buffer_);
  }
  if (mmap_fd_ >= 0) {
    close(mmap_fd_);
  }
}

const void* MMAPAllocation::base() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_7(mht_7_v, 313, "", "./tensorflow/lite/mmap_allocation.cc", "MMAPAllocation::base");

  return reinterpret_cast<const void*>(
      reinterpret_cast<const char*>(mmapped_buffer_) + offset_in_buffer_);
}

size_t MMAPAllocation::bytes() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_8(mht_8_v, 321, "", "./tensorflow/lite/mmap_allocation.cc", "MMAPAllocation::bytes");
 return buffer_size_bytes_; }

bool MMAPAllocation::valid() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_9(mht_9_v, 326, "", "./tensorflow/lite/mmap_allocation.cc", "MMAPAllocation::valid");
 return mmapped_buffer_ != MAP_FAILED; }

bool MMAPAllocation::IsSupported() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSmmap_allocationDTcc mht_10(mht_10_v, 331, "", "./tensorflow/lite/mmap_allocation.cc", "MMAPAllocation::IsSupported");
 return true; }

}  // namespace tflite
