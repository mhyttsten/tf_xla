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
class MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc() {
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
#include "tensorflow/lite/optional_debug_tools.h"

#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {
// Just forward declarations.
const char* AllocTypeName(TfLiteAllocationType type);

void PrintIntVector(const std::vector<int>& v,
                    bool collapse_consecutives = true,
                    bool add_newline = false);

// A class to represent the information of a memory arena that's used in TfLite
// runtime for holding allocated memory of tensors. The information includes
// the following:
// 1. The memory allocation type.
// 2. The tensor id of the tensor that has the most amount of memory allocated,
// and the memory size.
// 3. The estimated memory boundary and size of the arena.
class MemoryArenaInfo {
 public:
  explicit MemoryArenaInfo(TfLiteAllocationType type)
      : allocation_type_(type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_0(mht_0_v, 222, "", "./tensorflow/lite/optional_debug_tools.cc", "MemoryArenaInfo");
}

  void Update(size_t tensor_index, const TfLiteTensor& tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_1(mht_1_v, 227, "", "./tensorflow/lite/optional_debug_tools.cc", "Update");

    if (tensor.allocation_type != allocation_type_) return;
    if (tensor.data.data == nullptr) return;
    if (tensor.bytes > max_tensor_mem_bytes_) {
      max_tensor_mem_bytes_ = tensor.bytes;
      max_tensor_id_ = tensor_index;
    }

    size_t current_start_addr = reinterpret_cast<size_t>(tensor.data.data);

    size_t current_end_addr = current_start_addr + tensor.bytes;
    if (current_start_addr < min_tensor_start_addr_) {
      min_tensor_start_addr_ = current_start_addr;
    }
    if (current_end_addr > max_tensor_end_addr_) {
      max_tensor_end_addr_ = current_end_addr;
    }

    TensorAllocInfo info;
    info.tensor_id = tensor_index;
    info.start_addr = current_start_addr;
    info.bytes = tensor.bytes;
    const auto result = alloc_info_.insert(info);
    // Simply check that the insertion succeeds.
    assert(result.second);
    (void)result;  // suppress the "unused variable" compilation error.
  }

  size_t GetArenaStartingAddress() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_2(mht_2_v, 258, "", "./tensorflow/lite/optional_debug_tools.cc", "GetArenaStartingAddress");
 return min_tensor_start_addr_; }

  void Print() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_3(mht_3_v, 263, "", "./tensorflow/lite/optional_debug_tools.cc", "Print");

    printf("%s Info: ", AllocTypeName(allocation_type_));
    if (max_tensor_end_addr_ == 0) {
      printf("not holding any allocation.\n");
      return;
    }
    printf("\nTensor %zu has the max size %zu bytes (%.3f MB).\n",
           max_tensor_id_, max_tensor_mem_bytes_,
           static_cast<float>(max_tensor_mem_bytes_) / (1 << 20));
    const size_t arena_size = max_tensor_end_addr_ - min_tensor_start_addr_;
    printf(
        "This memory arena is estimated as[0x%zx, 0x%zx), taking %zu bytes "
        "(%.3f MB).\n",
        max_tensor_end_addr_, min_tensor_start_addr_, arena_size,
        static_cast<float>(arena_size) / (1 << 20));

    std::vector<const TensorAllocInfo*> arena_increase_trace;
    size_t last_end_addr = 0;
    for (const auto& info : alloc_info_) {
      if (info.start_addr >= last_end_addr) {
        arena_increase_trace.emplace_back(&info);
        last_end_addr = info.start_addr + info.bytes;
      }
    }
    printf(
        "One possible set of tensors that have non-overlapping memory spaces "
        "with each other, and they take up the whole arena:\n");
    printf("Tensor ");
    for (int i = 0; i < arena_increase_trace.size() - 1; ++i) {
      printf("%zu -> ", arena_increase_trace[i]->tensor_id);
    }
    printf("%zu.\n", arena_increase_trace.back()->tensor_id);
  }

 private:
  struct TensorAllocInfo {
    size_t tensor_id;
    size_t start_addr;
    size_t bytes;
  };

  // Compare first according to 'start_addr' in increasing order, then secondly
  // according to 'bytes' in decreasing order and finally according to
  // 'tensor_id' in increasing order.
  struct TensorAllocInfoCompare {
    bool operator()(const TensorAllocInfo& lhs,
                    const TensorAllocInfo& rhs) const {
      if (lhs.start_addr < rhs.start_addr) return true;
      if (lhs.start_addr == rhs.start_addr) {
        if (lhs.bytes > rhs.bytes) return true;
        if (lhs.bytes == rhs.bytes) return lhs.tensor_id < rhs.tensor_id;
        return false;
      }
      return false;
    }
  };

  const TfLiteAllocationType allocation_type_;
  size_t max_tensor_mem_bytes_ = 0;
  // the index of the tensor that has the max memory size.
  size_t max_tensor_id_ = -1;
  size_t min_tensor_start_addr_ = std::numeric_limits<size_t>::max();
  size_t max_tensor_end_addr_ = 0;
  std::set<TensorAllocInfo, TensorAllocInfoCompare> alloc_info_;
};

class DynamicMemoryInfo {
 public:
  void Update(size_t tensor_index, const TfLiteTensor& tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_4(mht_4_v, 334, "", "./tensorflow/lite/optional_debug_tools.cc", "Update");

    if (tensor.allocation_type != kTfLiteDynamic) return;
    if (tensor.data.data == nullptr) return;
    if (tensor.bytes > max_tensor_mem_bytes_) {
      max_tensor_mem_bytes_ = tensor.bytes;
      max_tensor_ids_.clear();
      max_tensor_ids_.push_back(tensor_index);
    } else if (tensor.bytes == max_tensor_mem_bytes_) {
      max_tensor_ids_.push_back(static_cast<int>(tensor_index));
    }
    total_mem_bytes_ += tensor.bytes;
    num_total_tensors_++;
  }

  void Print() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_5(mht_5_v, 351, "", "./tensorflow/lite/optional_debug_tools.cc", "Print");

    printf("kTfLiteDynamic Info: ");
    if (total_mem_bytes_ == 0) {
      printf("not holding any allocation.\n");
      return;
    }
    printf("\n%zu Tensors ", max_tensor_ids_.size());
    PrintIntVector(max_tensor_ids_, /*collapse_consecutives*/ false);
    printf(" have the max size %zu bytes (%.3f MB).\n", max_tensor_mem_bytes_,
           static_cast<float>(max_tensor_mem_bytes_) / (1 << 20));
    printf("There are %d dynamic tensors, taking %zu bytes (%.3f MB).\n",
           num_total_tensors_, total_mem_bytes_,
           static_cast<float>(total_mem_bytes_) / (1 << 20));
  }

 private:
  size_t max_tensor_mem_bytes_ = 0;
  // the index list of the tensor that has the max memory size.
  std::vector<int> max_tensor_ids_;
  size_t total_mem_bytes_ = 0;
  int num_total_tensors_ = 0;
};

class ModelTensorMemoryInfo {
 public:
  ModelTensorMemoryInfo()
      : rw_info_(kTfLiteArenaRw),
        rw_persistent_info_(kTfLiteArenaRwPersistent),
        mmap_info_(kTfLiteMmapRo) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_6(mht_6_v, 382, "", "./tensorflow/lite/optional_debug_tools.cc", "ModelTensorMemoryInfo");
}

  void Update(size_t tensor_index, const TfLiteTensor& tensor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_7(mht_7_v, 387, "", "./tensorflow/lite/optional_debug_tools.cc", "Update");

    rw_info_.Update(tensor_index, tensor);
    rw_persistent_info_.Update(tensor_index, tensor);
    mmap_info_.Update(tensor_index, tensor);
    dynamic_info_.Update(tensor_index, tensor);
  }

  // Get the offset from the beginning address of the memory arena for 'tensor'.
  // Returns -1 if not applicable. Otherwise, returns a non-negative value.
  int64_t GetOffsetFromArenaStart(const TfLiteTensor& tensor) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_8(mht_8_v, 399, "", "./tensorflow/lite/optional_debug_tools.cc", "GetOffsetFromArenaStart");

    if (tensor.data.data == nullptr) return -1;
    size_t tensor_address = reinterpret_cast<size_t>(tensor.data.data);
    if (tensor.allocation_type == kTfLiteArenaRw) {
      return static_cast<int64_t>(tensor_address -
                                  rw_info_.GetArenaStartingAddress());
    }
    if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
      return static_cast<int64_t>(
          tensor_address - rw_persistent_info_.GetArenaStartingAddress());
    }
    if (tensor.allocation_type == kTfLiteMmapRo) {
      return static_cast<int64_t>(tensor_address -
                                  mmap_info_.GetArenaStartingAddress());
    }
    return -1;
  }

  void Print() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_9(mht_9_v, 420, "", "./tensorflow/lite/optional_debug_tools.cc", "Print");

    printf("\n");
    rw_info_.Print();
    printf("\n");
    rw_persistent_info_.Print();
    printf("\n");
    mmap_info_.Print();
    printf("\n");
    dynamic_info_.Print();
    printf("\n");
  }

 private:
  MemoryArenaInfo rw_info_;
  MemoryArenaInfo rw_persistent_info_;
  MemoryArenaInfo mmap_info_;
  DynamicMemoryInfo dynamic_info_;
};

template <typename T>
void PrintTotalBytesOfTensors(const Subgraph& subgraph, const T& tensor_ids,
                              const std::string& prefix = " -> ") {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_10(mht_10_v, 444, "", "./tensorflow/lite/optional_debug_tools.cc", "PrintTotalBytesOfTensors");

  size_t total = 0;
  for (const auto id : tensor_ids) {
    const TfLiteTensor* tensor = subgraph.tensor(id);
    if (tensor == nullptr) continue;
    total += tensor->bytes;
  }
  printf("%s%zuB (%.2fMB)\n", prefix.c_str(), total,
         static_cast<float>(total) / (1 << 20));
}

void PrintIntVector(const std::vector<int>& v, bool collapse_consecutives,
                    bool add_newline) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_11(mht_11_v, 459, "", "./tensorflow/lite/optional_debug_tools.cc", "PrintIntVector");

  if (v.empty()) {
    printf("(null)");
    if (add_newline) {
      printf("\n");
    }
    return;
  }

  int range_start = v[0];
  int range_end = range_start;
  std::function<void(const char*)> print_range = [&](const char* suffix) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("suffix: \"" + (suffix == nullptr ? std::string("nullptr") : std::string((char*)suffix)) + "\"");
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_12(mht_12_v, 474, "", "./tensorflow/lite/optional_debug_tools.cc", "lambda");

    if (range_end == range_start) {
      printf("%d%s", range_start, suffix);
    } else if (range_end == range_start + 1) {
      printf("%d,%d%s", range_start, range_end, suffix);
    } else {
      printf("%d-%d%s", range_start, range_end, suffix);
    }
  };

  printf("[");
  for (int i = 1; i < v.size(); ++i) {
    int current = v[i];
    if (collapse_consecutives && (current == range_end + 1)) {
      range_end = current;
    } else {
      print_range(",");
      range_start = range_end = current;
    }
  }
  print_range("]");
  if (add_newline) {
    printf("\n");
  }
}

void PrintTfLiteIntVector(const TfLiteIntArray* v,
                          bool collapse_consecutives = true,
                          bool add_newline = false) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_13(mht_13_v, 505, "", "./tensorflow/lite/optional_debug_tools.cc", "PrintTfLiteIntVector");

  std::vector<int> tmp;
  if (!v || v->size <= 0) {
    PrintIntVector(tmp, collapse_consecutives, add_newline);
    return;
  }
  tmp.insert(tmp.end(), v->data, v->data + v->size);
  PrintIntVector(tmp, collapse_consecutives, add_newline);
}

const char* TensorTypeName(TfLiteType type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_14(mht_14_v, 518, "", "./tensorflow/lite/optional_debug_tools.cc", "TensorTypeName");

  switch (type) {
    case kTfLiteNoType:
      return "kTfLiteNoType";
    case kTfLiteFloat32:
      return "kTfLiteFloat32";
    case kTfLiteInt32:
      return "kTfLiteInt32";
    case kTfLiteUInt32:
      return "kTfLiteUInt32";
    case kTfLiteUInt8:
      return "kTfLiteUInt8";
    case kTfLiteInt8:
      return "kTfLiteInt8";
    case kTfLiteInt64:
      return "kTfLiteInt64";
    case kTfLiteUInt64:
      return "kTfLiteUInt64";
    case kTfLiteString:
      return "kTfLiteString";
    case kTfLiteBool:
      return "kTfLiteBool";
    case kTfLiteUInt16:
      return "kTfLiteUInt16";
    case kTfLiteInt16:
      return "kTfLiteInt16";
    case kTfLiteComplex64:
      return "kTfLiteComplex64";
    case kTfLiteComplex128:
      return "kTfLiteComplex128";
    case kTfLiteFloat16:
      return "kTfLiteFloat16";
    case kTfLiteFloat64:
      return "kTfLiteFloat64";
    case kTfLiteResource:
      return "kTfLiteResource";
    case kTfLiteVariant:
      return "kTfLiteVariant";
  }
  return "(invalid)";
}

const char* AllocTypeName(TfLiteAllocationType type) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_15(mht_15_v, 563, "", "./tensorflow/lite/optional_debug_tools.cc", "AllocTypeName");

  switch (type) {
    case kTfLiteMemNone:
      return "kTfLiteMemNone";
    case kTfLiteMmapRo:
      return "kTfLiteMmapRo";
    case kTfLiteDynamic:
      return "kTfLiteDynamic";
    case kTfLiteArenaRw:
      return "kTfLiteArenaRw";
    case kTfLiteArenaRwPersistent:
      return "kTfLiteArenaRwPersistent";
    case kTfLitePersistentRo:
      return "kTfLitePersistentRo";
    case kTfLiteCustom:
      return "kTfLiteCustom";
  }
  return "(invalid)";
}

std::string TruncateString(const char* str, int size_limit,
                           bool truncate_at_end = false) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_16(mht_16_v, 588, "", "./tensorflow/lite/optional_debug_tools.cc", "TruncateString");

  if (str == nullptr) return "(nil)";

  std::string truncated(str);
  const size_t length = truncated.size();
  if (length <= size_limit) return truncated;

  if (size_limit <= 3) return std::string(size_limit, '.');

  if (truncate_at_end) {
    truncated.resize(size_limit);
    // Change the the last 3 chars to  "..." to imply truncation.
    truncated.replace(size_limit - 3, 3, "...");
  } else {
    truncated.erase(0, length - size_limit);
    // Change the the first 3 chars to  "..." to imply truncation.
    truncated.replace(0, 3, "...");
  }
  return truncated;
}

}  // namespace

// Prints a dump of what tensors and what nodes are in the interpreter.
void PrintInterpreterState(const Interpreter* interpreter) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSoptional_debug_toolsDTcc mht_17(mht_17_v, 615, "", "./tensorflow/lite/optional_debug_tools.cc", "PrintInterpreterState");

  const size_t num_subgraphs = interpreter->subgraphs_size();
  printf("Interpreter has %zu subgraphs.\n\n", num_subgraphs);

  for (int i = 0; i < num_subgraphs; ++i) {
    const Subgraph& subgraph = *(interpreter->subgraph(i));
    printf("-----------Subgraph-%d has %zu tensors and %zu nodes------------\n",
           i, subgraph.tensors_size(), subgraph.nodes_size());
    printf("%zu Inputs: ", subgraph.inputs().size());
    PrintIntVector(subgraph.inputs());
    PrintTotalBytesOfTensors(subgraph, subgraph.inputs());

    printf("%zu Outputs: ", subgraph.outputs().size());
    PrintIntVector(subgraph.outputs());
    PrintTotalBytesOfTensors(subgraph, subgraph.outputs());
    printf("\n");

    // Collect info about tensor memory allocation.
    ModelTensorMemoryInfo tensor_mem_info;
    for (size_t tensor_index = 0; tensor_index < subgraph.tensors_size();
         tensor_index++) {
      const TfLiteTensor* tensor =
          subgraph.tensor(static_cast<int>(tensor_index));
      tensor_mem_info.Update(tensor_index, *tensor);
    }

    printf("Tensor %3s %-25s %-15s %-18s %-18s %-10s %-16s\n", "ID", "Name",
           "Type", "AllocType", "Size (Bytes/MB)", "Shape", "MemAddr-Offset");
    for (size_t tensor_index = 0; tensor_index < subgraph.tensors_size();
         tensor_index++) {
      const TfLiteTensor* tensor =
          subgraph.tensor(static_cast<int>(tensor_index));
      printf("Tensor %3zu %-25s %-15s %-18s %-8zu / %.2f ", tensor_index,
             TruncateString(tensor->name, 25, /*truncate_at_end*/ true).c_str(),
             TruncateString(TensorTypeName(tensor->type), 15).c_str(),
             TruncateString(AllocTypeName(tensor->allocation_type), 18).c_str(),
             tensor->bytes, (static_cast<float>(tensor->bytes) / (1 << 20)));
      PrintTfLiteIntVector(tensor->dims, /*collapse_consecutives*/ false);
      const int64_t start_offset =
          tensor_mem_info.GetOffsetFromArenaStart(*tensor);
      const int64_t end_offset =
          start_offset == -1
              ? -1
              : start_offset + static_cast<int64_t>(tensor->bytes);
      printf(" [%" PRId64 ", %" PRId64 ")\n", start_offset, end_offset);
    }
    tensor_mem_info.Print();

    // Dumps debugging info provided by the underlying memory planner.
    // Note that this will output nothing unless the
    // ":simple_memory_arena_debug_dump" is added as an extra dependence.
    subgraph.DumpMemoryPlannerDebugInfo();

    // Going to print out all nodes (i.e. op kernels) in this subgraph.
    std::vector<bool> replaced_node_bits;
    std::vector<size_t> replaced_by_node;
    replaced_node_bits.resize(subgraph.nodes_size());
    replaced_by_node.resize(subgraph.nodes_size());
    bool has_delegate_applied = false;
    for (size_t node_index = 0; node_index < subgraph.nodes_size();
         node_index++) {
      replaced_node_bits[node_index] = false;
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          subgraph.node_and_registration(static_cast<int>(node_index));
      const TfLiteNode& node = node_and_reg->first;
      auto* const delegate = node.delegate;
      if (delegate != nullptr) {
        has_delegate_applied = true;
        auto* params = static_cast<TfLiteDelegateParams*>(node.builtin_data);
        for (int nid : TfLiteIntArrayView(params->nodes_to_replace)) {
          replaced_node_bits[nid] = true;
          replaced_by_node[nid] = node_index;
        }
      }
    }
    for (size_t node_index = 0; node_index < subgraph.nodes_size();
         node_index++) {
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          subgraph.node_and_registration(static_cast<int>(node_index));
      const TfLiteNode& node = node_and_reg->first;
      const TfLiteRegistration& reg = node_and_reg->second;

      std::string delegated_status;
      bool is_node_delegated = false;
      TfLiteIntArray empty_int_array;
      empty_int_array.size = 0;
      if (node.delegate == nullptr) {
        if (replaced_node_bits[node_index]) {
          delegated_status = "(delegated by node ";
          delegated_status.append(std::to_string(replaced_by_node[node_index]));
          delegated_status.append(")");
          is_node_delegated = true;
        } else {
          delegated_status = "(not delegated)";
        }
      }

      if (reg.custom_name != nullptr) {
        printf("Node %3zu Operator Custom Name %s %s\n", node_index,
               reg.custom_name, delegated_status.c_str());
      } else {
        printf("Node %3zu Operator Builtin Code %3d %s %s\n", node_index,
               reg.builtin_code, EnumNamesBuiltinOperator()[reg.builtin_code],
               delegated_status.c_str());
      }
      printf("  %d Input Tensors:",
             node.inputs != nullptr ? node.inputs->size : 0);
      PrintTfLiteIntVector(
          node.inputs,
          /*collapse_consecutives=*/(node.delegate != nullptr));
      PrintTotalBytesOfTensors(
          subgraph, is_node_delegated ? TfLiteIntArrayView(&empty_int_array)
                                      : TfLiteIntArrayView(node.inputs));

      printf("  %d Output Tensors:",
             node.outputs != nullptr ? node.outputs->size : 0);
      PrintTfLiteIntVector(node.outputs);
      PrintTotalBytesOfTensors(
          subgraph, is_node_delegated ? TfLiteIntArrayView(&empty_int_array)
                                      : TfLiteIntArrayView(node.outputs));

      if (node.intermediates && node.intermediates->size) {
        printf("  %d Intermediate Tensors:", node.intermediates->size);
        PrintTfLiteIntVector(node.intermediates);
        PrintTotalBytesOfTensors(subgraph,
                                 is_node_delegated
                                     ? TfLiteIntArrayView(&empty_int_array)
                                     : TfLiteIntArrayView(node.intermediates));
      }

      if (node.temporaries && node.temporaries->size) {
        printf("  %d Temporary Tensors:", node.temporaries->size);
        PrintTfLiteIntVector(node.temporaries);
        PrintTotalBytesOfTensors(
            subgraph, is_node_delegated ? TfLiteIntArrayView(&empty_int_array)
                                        : TfLiteIntArrayView(node.temporaries));
      }
    }

    printf("\nExecution plan as the list of %zu nodes invoked in-order: ",
           subgraph.execution_plan().size());
    PrintIntVector(subgraph.execution_plan(), /*collapse_consecutives=*/true,
                   /*add_newline=*/true);
    if (has_delegate_applied) {
      printf("Among these nodes in the execution plan:\n");
      for (int node_id : subgraph.execution_plan()) {
        const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
            subgraph.node_and_registration(node_id);
        const TfLiteNode& node = node_and_reg->first;
        auto* const delegate = node.delegate;
        if (delegate == nullptr) continue;
        const char* delegate_name = node_and_reg->second.custom_name;
        auto* delegate_params =
            static_cast<TfLiteDelegateParams*>(node.builtin_data);
        printf("  Node %d is a %s node (%p), which has delegated %d nodes: ",
               node_id, delegate_name == nullptr ? "[n/a]" : delegate_name,
               delegate, delegate_params->nodes_to_replace->size);
        PrintTfLiteIntVector(delegate_params->nodes_to_replace,
                             /*collapse_consecutives=*/true,
                             /*add_newline=*/true);
      }
    }

    printf("--------------Subgraph-%d dump has completed--------------\n\n", i);
  }
}

}  // namespace tflite
