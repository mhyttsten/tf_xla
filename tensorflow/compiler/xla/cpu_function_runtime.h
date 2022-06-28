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

#ifndef TENSORFLOW_COMPILER_XLA_CPU_FUNCTION_RUNTIME_H_
#define TENSORFLOW_COMPILER_XLA_CPU_FUNCTION_RUNTIME_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh() {
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


#include <stdint.h>

#include <cassert>
#include <cstdlib>
#include <utility>

namespace xla {
namespace cpu_function_runtime {
// Stores information about one buffer used by an XLA:CPU compiled function.
// These buffers are used for holding inputs to the computation, outputs from
// the computation and as temporary scratch space.
class BufferInfo {
 public:
  // Creates a BufferInfo from a serialized encoding generated by `Encode`.
  explicit BufferInfo(std::pair<uint64_t, uint64_t> encoding)
      : entry_param_number_(encoding.second) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "BufferInfo");

    Kind kind;
    uint64_t size;
    Unpack(encoding.first, &kind, &size);
    kind_ = kind;
    size_ = size;
  }

  // Returns true if this buffer stores a constant.  These never need to be
  // allocated by the runtime.
  bool is_constant() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "is_constant");
 return kind() == Kind::kConstant; }

  // Returns true if this buffer stores an entry parameter.  These may or may
  // not need to be allocated by the runtime, depending on
  // XlaCompiledCpuFunction::AllocMode.
  bool is_entry_parameter() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_2(mht_2_v, 224, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "is_entry_parameter");
 return kind() == Kind::kEntryParameter; }

  // Returns the entry parameter number of this buffer.
  uint64_t entry_parameter_number() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_3(mht_3_v, 230, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "entry_parameter_number");

    assert(is_entry_parameter());
    return entry_param_number_;
  }

  // Returns true if this buffer is temporary scratch space required by the XLA
  // computations.  These are always allocated by the runtime.
  bool is_temp_buffer() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_4(mht_4_v, 240, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "is_temp_buffer");
 return kind() == Kind::kTempBuffer; }

  // Returns true if this buffer is allocated on the C stack or into registers.
  // These buffers are never allocated by the runtime.
  bool is_on_stack_buffer() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_5(mht_5_v, 247, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "is_on_stack_buffer");
 return kind() == Kind::kOnStackBuffer; }

  // Returns the size for this buffer.
  uint64_t size() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_6(mht_6_v, 253, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "size");
 return size_; }

  // Encodes this BufferInfo into two 64 bit integers that can be used to
  // reconstruct the BufferInfo later using the constructor.  We need this
  // because we use BufferInfo in places where using protocol buffers would
  // negatively impact binary size.
  std::pair<uint64_t, uint64_t> Encode() const {
    static_assert(sizeof(*this) == 16, "");
    uint64_t upper = Pack(kind(), size_);
    uint64_t lower = entry_param_number_;
    return {upper, lower};
  }

  bool operator==(const BufferInfo& buffer_info) const {
    if (kind() != buffer_info.kind() || size() != buffer_info.size()) {
      return false;
    }
    return !is_entry_parameter() ||
           entry_parameter_number() == buffer_info.entry_parameter_number();
  }

  // Factory methods:

  static BufferInfo MakeTempBuffer(uint64_t size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_7(mht_7_v, 279, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "MakeTempBuffer");

    return BufferInfo(Kind::kTempBuffer, /*size=*/size,
                      /*entry_param_number=*/-1);
  }
  static BufferInfo MakeConstant(uint64_t size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_8(mht_8_v, 286, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "MakeConstant");

    return BufferInfo(Kind::kConstant, /*size=*/size,
                      /*entry_param_number=*/-1);
  }
  static BufferInfo MakeEntryParameter(uint64_t size, uint64_t param_number) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_9(mht_9_v, 293, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "MakeEntryParameter");

    return BufferInfo(Kind::kEntryParameter, /*size=*/size,
                      /*entry_param_number=*/param_number);
  }
  static BufferInfo MakeOnStackBuffer(uint64_t size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_10(mht_10_v, 300, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "MakeOnStackBuffer");

    return BufferInfo(Kind::kOnStackBuffer, /*size=*/size,
                      /*entry_param_number=*/-1);
  }

 private:
  BufferInfo() = default;

  enum class Kind : uint64_t {
    kConstant,
    kTempBuffer,
    kEntryParameter,
    kOnStackBuffer
  };

  Kind kind() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_11(mht_11_v, 318, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "kind");
 return static_cast<Kind>(kind_); }

  explicit BufferInfo(Kind kind, uint64_t size, uint64_t entry_param_number)
      : kind_(kind), size_(size), entry_param_number_(entry_param_number) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_12(mht_12_v, 324, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "BufferInfo");
}

  static uint64_t Pack(Kind kind, uint64_t size) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_13(mht_13_v, 329, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "Pack");

    return (static_cast<uint64_t>(size) << 2) | static_cast<uint64_t>(kind);
  }

  static void Unpack(uint64_t packed, Kind* kind, uint64_t* size) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScpu_function_runtimeDTh mht_14(mht_14_v, 336, "", "./tensorflow/compiler/xla/cpu_function_runtime.h", "Unpack");

    *size = packed >> 2;
    *kind = static_cast<Kind>((packed << 62) >> 62);
  }

  Kind kind_ : 2;
  uint64_t size_ : 62;
  int64_t entry_param_number_;
};

// Align to 64-bytes, to mimic tensorflow::Allocator::kAllocatorAlignment.
inline constexpr size_t Align() { return 64; }

// The minimum alignment of buffers passed to XLA:CPU.
inline constexpr size_t MinAlign() { return 16; }

// When declaring variables that will be passed to an XLA instance as input via
// set_arg_data(), be it a regular input or a resource variable in the graph,
// the C++ variables must be aligned.
//
// Example usage:
//   XLA_ALIGN std::array<float, 4> arg_x;
//   XLA_ALIGN float arg_y;
//   xla_instance.set_arg_data(0, arg_x.date());
//   xla_instance.set_arg_data(0, &arg_y);
#define XLA_ALIGN alignas(xla::cpu_function_runtime::Align())

// AlignedBufferBytes returns the sum of the size of each buffer in
// `buffer_infos`, skipping constants, on-stack buffers and, if
// allocate_entry_params is false, entry parameters.  There are `n` entries in
// `buffer_infos`.  Each buffer is aligned to Align() byte boundaries.
size_t AlignedBufferBytes(const BufferInfo* buffer_infos, size_t n,
                          bool allocate_entry_params);

// MallocContiguousBuffers allocates buffers for use by the entry point
// generated by tfcompile.  There are `n` entries in `buffer_infos`.  If
// `annotate_initialized` is set, the allocated memory will be annotated as
// having been initialized - this is useful when allocating temporary buffers.
// If allocate_entry_params is true then allocates temp buffers and entry
// parameters, otherwise allocated only temp buffers.  Slots in `bufs`
// corresponding to unallocated buffers are set to nullptr.
//
// A single contiguous block of memory is allocated, and portions of it are
// parceled out into `bufs`, which must have space for `n` entries.  Returns
// the head of the allocated contiguous block, which should be passed to
// FreeContiguous when the buffers are no longer in use.
void* MallocContiguousBuffers(const BufferInfo* buffer_infos, size_t n,
                              bool allocate_entry_params, void** bufs,
                              bool annotate_initialized);

// FreeContiguous frees the contiguous block of memory allocated by
// MallocContiguousBuffers.
void FreeContiguous(void* contiguous);
}  // namespace cpu_function_runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CPU_FUNCTION_RUNTIME_H_
