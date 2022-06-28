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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TYPED_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_TYPED_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh() {
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


#include <limits>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Variant;

// Convenience functions to do typed allocation.  C++ constructors
// and destructors are invoked for complex types if necessary.
class TypedAllocator {
 public:
  // May return NULL if the tensor has too many elements to represent in a
  // single allocation.
  template <typename T>
  static T* Allocate(Allocator* raw_allocator, size_t num_elements,
                     const AllocationAttributes& allocation_attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh mht_0(mht_0_v, 207, "", "./tensorflow/core/framework/typed_allocator.h", "Allocate");

    // TODO(jeff): Do we need to allow clients to pass in alignment
    // requirements?

    if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T))) {
      return nullptr;
    }

    void* p =
        raw_allocator->AllocateRaw(Allocator::kAllocatorAlignment,
                                   sizeof(T) * num_elements, allocation_attr);
    T* typed_p = reinterpret_cast<T*>(p);
    if (typed_p) RunCtor<T>(raw_allocator, typed_p, num_elements);
    return typed_p;
  }

  template <typename T>
  static void Deallocate(Allocator* raw_allocator, T* ptr,
                         size_t num_elements) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh mht_1(mht_1_v, 228, "", "./tensorflow/core/framework/typed_allocator.h", "Deallocate");

    if (ptr) {
      RunDtor<T>(raw_allocator, ptr, num_elements);
      raw_allocator->DeallocateRaw(ptr);
    }
  }

 private:
  // No constructors or destructors are run for simple types
  template <typename T>
  static void RunCtor(Allocator* raw_allocator, T* p, size_t n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh mht_2(mht_2_v, 241, "", "./tensorflow/core/framework/typed_allocator.h", "RunCtor");

    static_assert(is_simple_type<T>::value, "T is not a simple type.");
  }

  template <typename T>
  static void RunDtor(Allocator* raw_allocator, T* p, size_t n) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh mht_3(mht_3_v, 249, "", "./tensorflow/core/framework/typed_allocator.h", "RunDtor");
}

  static void RunVariantCtor(Variant* p, size_t n);

  static void RunVariantDtor(Variant* p, size_t n);
};

template <>
/* static */
inline void TypedAllocator::RunCtor(Allocator* raw_allocator, tstring* p,
                                    size_t n) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh mht_4(mht_4_v, 262, "", "./tensorflow/core/framework/typed_allocator.h", "TypedAllocator::RunCtor");

  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) new (p) tstring();
  }
}

template <>
/* static */
inline void TypedAllocator::RunDtor(Allocator* raw_allocator, tstring* p,
                                    size_t n) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh mht_5(mht_5_v, 274, "", "./tensorflow/core/framework/typed_allocator.h", "TypedAllocator::RunDtor");

  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) p->~tstring();
  }
}

template <>
/* static */
inline void TypedAllocator::RunCtor(Allocator* raw_allocator, ResourceHandle* p,
                                    size_t n) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh mht_6(mht_6_v, 286, "", "./tensorflow/core/framework/typed_allocator.h", "TypedAllocator::RunCtor");

  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) new (p) ResourceHandle();
  }
}

template <>
/* static */
inline void TypedAllocator::RunDtor(Allocator* raw_allocator, ResourceHandle* p,
                                    size_t n) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh mht_7(mht_7_v, 298, "", "./tensorflow/core/framework/typed_allocator.h", "TypedAllocator::RunDtor");

  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) p->~ResourceHandle();
  }
}

template <>
/* static */
inline void TypedAllocator::RunCtor(Allocator* raw_allocator, Variant* p,
                                    size_t n) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh mht_8(mht_8_v, 310, "", "./tensorflow/core/framework/typed_allocator.h", "TypedAllocator::RunCtor");

  if (!raw_allocator->AllocatesOpaqueHandle()) {
    RunVariantCtor(p, n);
  }
}

template <>
/* static */
inline void TypedAllocator::RunDtor(Allocator* raw_allocator, Variant* p,
                                    size_t n) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStyped_allocatorDTh mht_9(mht_9_v, 322, "", "./tensorflow/core/framework/typed_allocator.h", "TypedAllocator::RunDtor");

  if (!raw_allocator->AllocatesOpaqueHandle()) {
    RunVariantDtor(p, n);
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TYPED_ALLOCATOR_H_
