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

// CUDA userspace driver library wrapper functionality.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
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
class MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh() {
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


#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"

namespace stream_executor {
namespace gpu {
// Formats CUresult to output prettified values into a log stream.
static std::string ToString(CUresult result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh mht_0(mht_0_v, 199, "", "./tensorflow/stream_executor/cuda/cuda_driver.h", "ToString");

  const char* error_name;
  if (cuGetErrorName(result, &error_name)) {
    return absl::StrCat("UNKNOWN ERROR (", static_cast<int>(result), ")");
  }
  const char* error_string;
  if (cuGetErrorString(result, &error_string)) {
    return error_name;
  }
  return absl::StrCat(error_name, ": ", error_string);
}

// CUDAContext wraps a cuda CUcontext handle, and includes a unique id. The
// unique id is positive, and ids are not repeated within the process.
class GpuContext {
 public:
  GpuContext(CUcontext context, int64_t id) : context_(context), id_(id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh mht_1(mht_1_v, 218, "", "./tensorflow/stream_executor/cuda/cuda_driver.h", "GpuContext");
}

  CUcontext context() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh mht_2(mht_2_v, 223, "", "./tensorflow/stream_executor/cuda/cuda_driver.h", "context");
 return context_; }
  int64_t id() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh mht_3(mht_3_v, 227, "", "./tensorflow/stream_executor/cuda/cuda_driver.h", "id");
 return id_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  CUcontext const context_;
  const int64_t id_;
};

// Manages the singleton map of contexts that we've created, mapping
// from the CUcontext to the GpuContext* that we pass around internally.
// This also manages assignment of unique ids to GpuContexts, to allow
// for fast comparison of a context against the current context.
//
// CUDA-runtime-created contexts are avoided, if triple angle
// brace launches are required, by using the scoped activations in
// gpu/gpu_activation.h.
class CreatedContexts {
 public:
  // Returns whether context is a member of the live set.
  static bool Has(CUcontext context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh mht_4(mht_4_v, 254, "", "./tensorflow/stream_executor/cuda/cuda_driver.h", "Has");

    absl::ReaderMutexLock lock(&mu_);
    return Live()->find(context) != Live()->end();
  }

  // Adds context to the live set, or returns it if it's already present.
  static GpuContext* Add(CUcontext context, int device_ordinal) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh mht_5(mht_5_v, 263, "", "./tensorflow/stream_executor/cuda/cuda_driver.h", "Add");

    CHECK(context != nullptr);
    absl::MutexLock lock(&mu_);

    auto insert_result = Live()->insert(std::make_pair(context, nullptr));
    auto it = insert_result.first;
    if (insert_result.second) {
      // context was not present in the map.  Add it.
      it->second = absl::make_unique<GpuContext>(context, next_id_++);
      (*LiveOrdinal())[device_ordinal].push_back(context);
    }
    return it->second.get();
  }

  // Removes context from the live set.
  static void Remove(CUcontext context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh mht_6(mht_6_v, 281, "", "./tensorflow/stream_executor/cuda/cuda_driver.h", "Remove");

    CHECK(context != nullptr);
    absl::MutexLock lock(&mu_);
    auto it = Live()->find(context);
    CHECK(it != Live()->end()) << context;
    Live()->erase(it);
    for (auto p : (*LiveOrdinal())) {
      auto it2 = std::find(p.second.begin(), p.second.end(), context);
      if (it2 != p.second.end()) {
        p.second.erase(it2, it2++);
        if (p.second.empty()) {
          LiveOrdinal()->erase(p.first);
        }
        break;
      }
    }
  }

  // Return the context associated to that ptr.
  static CUcontext GetAnyContext(void* ptr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh mht_7(mht_7_v, 303, "", "./tensorflow/stream_executor/cuda/cuda_driver.h", "GetAnyContext");

    absl::ReaderMutexLock lock(&mu_);
    int device_ordinal;
    CUresult result = cuPointerGetAttribute(static_cast<void*>(&device_ordinal),
                                            CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                            reinterpret_cast<CUdeviceptr>(ptr));
    if (result != CUDA_SUCCESS) {
      LOG(FATAL) << "Not able to get the device_ordinal for ptr: " << ptr
                 << ". Error: " << ToString(result);
    }
    CHECK_EQ(LiveOrdinal()->count(device_ordinal), 1);
    CHECK(!LiveOrdinal()->at(device_ordinal).empty())
        << "Need at least one context.";
    return LiveOrdinal()->at(device_ordinal)[0];
  }

 private:
  // Returns the live map singleton.
  static absl::node_hash_map<CUcontext, std::unique_ptr<GpuContext>>* Live() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh mht_8(mht_8_v, 324, "", "./tensorflow/stream_executor/cuda/cuda_driver.h", "Live");

    static auto singleton =
        new absl::node_hash_map<CUcontext, std::unique_ptr<GpuContext>>;
    return singleton;
  }
  static absl::node_hash_map<int, std::vector<CUcontext>>* LiveOrdinal() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTh mht_9(mht_9_v, 332, "", "./tensorflow/stream_executor/cuda/cuda_driver.h", "LiveOrdinal");

    static auto singleton =
        new absl::node_hash_map<int, std::vector<CUcontext>>;
    return singleton;
  }

  // Lock that guards access-to/mutation-of the live set.
  static absl::Mutex mu_;
  static int64_t next_id_;
};
}  // namespace gpu

namespace cuda {

using MemorySpace = gpu::MemorySpace;

using CUDADriver = gpu::GpuDriver;

using ScopedActivateContext = gpu::ScopedActivateContext;

using CudaContext = gpu::GpuContext;

// Returns the current context set in CUDA. This is done by calling the cuda
// driver (e.g., this value is not our cached view of the current context).
CUcontext CurrentContextOrDie();

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
