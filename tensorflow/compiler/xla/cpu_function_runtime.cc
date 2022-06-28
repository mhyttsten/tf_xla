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
class MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTcc() {
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

#include "tensorflow/compiler/xla/cpu_function_runtime.h"

#include "absl/base/dynamic_annotations.h"

namespace xla {
namespace {
// Inline memory allocation routines here, because depending on '//base' brings
// in libraries which use c++ streams, which adds considerable code size on
// android.
void* aligned_malloc(size_t size, int minimum_alignment) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTcc mht_0(mht_0_v, 194, "", "./tensorflow/compiler/xla/cpu_function_runtime.cc", "aligned_malloc");

#if defined(__ANDROID__) || defined(OS_ANDROID) || defined(OS_CYGWIN)
  return memalign(minimum_alignment, size);
#elif defined(_WIN32)
  return _aligned_malloc(size, minimum_alignment);
#else  // !__ANDROID__ && !OS_ANDROID && !OS_CYGWIN
  void* ptr = nullptr;
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return memory
  // aligned to at least the size of a pointer.
  const int required_alignment = sizeof(void*);
  if (minimum_alignment < required_alignment) return malloc(size);
  if (posix_memalign(&ptr, minimum_alignment, size) != 0)
    return nullptr;
  else
    return ptr;
#endif
}

void aligned_free(void* aligned_memory) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/cpu_function_runtime.cc", "aligned_free");

#if defined(_WIN32)
  _aligned_free(aligned_memory);
#else
  free(aligned_memory);
#endif
}

size_t align_to(size_t n, size_t align) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/cpu_function_runtime.cc", "align_to");

  return (((n - 1) / align) + 1) * align;
}
}  // namespace

namespace cpu_function_runtime {
size_t AlignedBufferBytes(const BufferInfo* buffer_infos, size_t n,
                          bool allocate_entry_params) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTcc mht_3(mht_3_v, 237, "", "./tensorflow/compiler/xla/cpu_function_runtime.cc", "AlignedBufferBytes");

  size_t total = 0;
  for (size_t i = 0; i < n; ++i) {
    bool should_allocate =
        buffer_infos[i].is_temp_buffer() ||
        (buffer_infos[i].is_entry_parameter() && allocate_entry_params);

    if (should_allocate) {
      total += align_to(buffer_infos[i].size(), Align());
    }
  }
  return total;
}

void* MallocContiguousBuffers(const BufferInfo* buffer_infos, size_t n,
                              bool allocate_entry_params, void** bufs,
                              bool annotate_initialized) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTcc mht_4(mht_4_v, 256, "", "./tensorflow/compiler/xla/cpu_function_runtime.cc", "MallocContiguousBuffers");

  const size_t total =
      AlignedBufferBytes(buffer_infos, n, allocate_entry_params);
  void* contiguous = nullptr;
  if (total > 0) {
    contiguous = aligned_malloc(total, Align());
    if (annotate_initialized) {
      // Since the memory for temp buffers is written to by JITed code, msan has
      // no way of knowing the memory was initialized, so explicitly mark it.
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(contiguous, total);
    }
  }
  uintptr_t pos = reinterpret_cast<uintptr_t>(contiguous);
  for (size_t i = 0; i < n; ++i) {
    bool should_allocate =
        buffer_infos[i].is_temp_buffer() ||
        (buffer_infos[i].is_entry_parameter() && allocate_entry_params);
    if (should_allocate) {
      bufs[i] = reinterpret_cast<void*>(pos);
      pos += align_to(buffer_infos[i].size(), Align());
    } else {
      bufs[i] = nullptr;
    }
  }
  return contiguous;
}

void FreeContiguous(void* contiguous) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTcc mht_5(mht_5_v, 286, "", "./tensorflow/compiler/xla/cpu_function_runtime.cc", "FreeContiguous");

  if (contiguous != nullptr) {
    aligned_free(contiguous);
  }
}
}  // namespace cpu_function_runtime
}  // namespace xla
