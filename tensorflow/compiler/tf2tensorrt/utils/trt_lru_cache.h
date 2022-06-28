/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_LRU_CACHE_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_LRU_CACHE_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh() {
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


#include <list>
#include <thread>
#include <unordered_map>

#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_int8_calibrator.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

template <class Key, class Value, class HashFunction>
class LRUCache {
 public:
  typedef Value value_type;
  typedef Key key_type;
  typedef HashFunction hasher;
  typedef typename std::unordered_map<key_type, value_type, hasher> map_type;
  typedef typename map_type::iterator iterator;
  typedef typename map_type::const_iterator const_iterator;

  LRUCache() : capacity_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_0(mht_0_v, 218, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "LRUCache");
}
  explicit LRUCache(size_t capacity) : capacity_(capacity) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_1(mht_1_v, 222, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "LRUCache");
}

  size_t capacity() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_2(mht_2_v, 227, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "capacity");
 return capacity_; }

  void reserve(size_t capacity) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_3(mht_3_v, 232, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "reserve");

    capacity_ = capacity;
    DiscardOld();
  }

  size_t size() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_4(mht_4_v, 240, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "size");
 return objects_.size(); }

  size_t count(const key_type& key) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_5(mht_5_v, 245, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "count");
 return objects_.count(key); }

  value_type& at(const key_type& key) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_6(mht_6_v, 250, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "at");
 return Touch(key); }

  const_iterator begin() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_7(mht_7_v, 255, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "begin");
 return objects_.begin(); }
  const_iterator end() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_8(mht_8_v, 259, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "end");
 return objects_.end(); }

  iterator begin() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_9(mht_9_v, 264, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "begin");
 return objects_.begin(); }
  iterator end() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_10(mht_10_v, 268, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "end");
 return objects_.end(); }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    DiscardOld(1);
    std::pair<iterator, bool> result =
        objects_.emplace(std::forward<Args>(args)...);
    key_type key = result.first->first;
    if (result.second) {
      keys_.push_front(key);
    } else {
      TouchNoCheck(key);  // The key must exist in this case.
    }
    return result;
  }

 private:
  std::unordered_map<key_type, value_type, hasher> objects_;
  std::list<key_type> keys_;
  size_t capacity_;
  value_type not_found_value_;

  value_type& Touch(const key_type& key) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_11(mht_11_v, 293, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "Touch");

    // Check that the key exists, and let it return std::out_of_range error if
    // not.
    value_type& value = objects_.at(key);
    TouchNoCheck(key);
    return value;
  }

  void TouchNoCheck(const key_type& key) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_12(mht_12_v, 304, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "TouchNoCheck");

    auto rank = std::find(keys_.begin(), keys_.end(), key);
    if (rank != keys_.begin()) {
      keys_.erase(rank);
      keys_.push_front(key);
    }
  }

  // Creates n free positions in cache
  void DiscardOld(size_t n = 0) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_13(mht_13_v, 316, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "DiscardOld");

    DCHECK(capacity_ >= n) << "Insufficient capacity in cache (capacity = "
                           << capacity_ << ", requested " << n << ")";
    while (objects_.size() > (capacity_ - n)) {
      key_type discard_key = keys_.back();
      keys_.pop_back();
      objects_.erase(discard_key);
    }
  }
};

#if GOOGLE_CUDA && GOOGLE_TENSORRT

struct EngineContext {
  EngineContext() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_14(mht_14_v, 333, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "EngineContext");
}  // Creates an empty context.
  EngineContext(TrtUniquePtrType<nvinfer1::ICudaEngine>&& cuda_engine,
                ExecutionContext&& execution_context)
      : cuda_engine(std::move(cuda_engine)) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_15(mht_15_v, 339, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "EngineContext");

    execution_contexts.push_back(std::move(execution_context));
  }
  EngineContext(TrtUniquePtrType<nvinfer1::ICudaEngine>&& cuda_engine,
                std::vector<ExecutionContext>&& execution_contexts)
      : cuda_engine(std::move(cuda_engine)),
        execution_contexts(std::move(execution_contexts)) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_16(mht_16_v, 348, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "EngineContext");
}

  mutex mu;
  TrtUniquePtrType<nvinfer1::ICudaEngine> cuda_engine;

  Status GetExecutionContext(int idx, nvinfer1::IExecutionContext** exec_ctx,
                             bool* has_device_memory)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_17(mht_17_v, 358, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "GetExecutionContext");

    if (idx >= execution_contexts.size()) {
      return errors::Internal("Requested engine context with index ", idx,
                              ", but only ", execution_contexts.size(),
                              "contexts are present.");
    }
    *exec_ctx = execution_contexts[idx].get();
    *has_device_memory = execution_contexts[idx].HasDeviceMemory();
    return Status::OK();
  }

  int GetNumContexts() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTh mht_18(mht_18_v, 372, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h", "GetNumContexts");

    mutex_lock lock(mu);
    return execution_contexts.size();
  }

  // In explicit batch mode, we maintain a vector of contexts for each engine,
  // where each context is created for a specific profile. This is because it is
  // either not possible or non-trivial to change the profile of a context for
  // the following reasons:
  // - To switch profiles (from TRT 7), one must first ensure that all inference
  //   calls in that context are finished. This would require an additional
  //   synchronization before we call setOptimizationProfile. To avoid this
  //   extra sync call, we mantain separate execution context for each profile.
  // IExecutionContext object is not thread safe: only one thread should use it
  // for inference at a time therefore we need a mutex. More details at
  // https://docs.nvidia.com/deeplearning/sdk/tensorrt-best-practices/index.html#thread-safety
  // Additional discussion about execution context management and thread safety
  // at https://github.com/tensorflow/tensorflow/issues/36959
  std::vector<ExecutionContext> execution_contexts TF_GUARDED_BY(mu);
};

// Contains the context required to build the calibration data.
class CalibrationContext {
 public:
  string TerminateCalibration();

  // Lookup table for temporary staging areas of input tensors for calibration.
  std::unordered_map<string, std::pair<void*, size_t>> device_buffers_;

  // Temporary staging areas for calibration inputs.
  std::vector<Tensor> device_tensors_;

  std::unique_ptr<TRTInt8Calibrator> calibrator_;
  TrtUniquePtrType<nvinfer1::IBuilder> builder_;
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;
  // TODO(sami): Use threadpool threads!
  std::unique_ptr<std::thread> thr_;

 private:
  mutex mu_;
  bool terminated_ TF_GUARDED_BY(mu_) = false;
  std::string calibration_table_ TF_GUARDED_BY(mu_);
};

ABSL_CONST_INIT extern const absl::string_view kTfTrtContainerName;

class TRTEngineCacheResource : public ResourceBase {
 public:
  // According to the TensorRT API, the logger is considered a singleton by the
  // TensorRT library, and multiple instances of IRuntime and/or IBuilder must
  // all use the same logger. So here we make it a singleton.
  //
  // TODO(laigd): use this logger in all places where conversion happens.
  static Logger& GetLogger();

  TRTEngineCacheResource(OpKernelContext* ctx, size_t capacity);

  ~TRTEngineCacheResource() override;

  string DebugString() const override;

  // Returns the EngineContext that is compatible with input_shapes.
  // Returns nullptr if no compatible EngineContexts is found in cache.
  EngineContext* GetEngineContext(const std::vector<TensorShape>& input_shapes);

  // Returns the EngineContext that is compatible with profile_id.
  // This function should be only called in explicit batch mode where
  // cache size is expected to be at most one.
  // Returns nullptr if no compatible EngineContexts is found in cache.
  EngineContext* GetEngineContext(const int profile_id);

  // Keep device allocator for TRT.
  std::unique_ptr<TRTBaseAllocator> allocator_;

  // Declare cache after allocator so that it is destroyed before allocator is.
  LRUCache<std::vector<TensorShape>, std::unique_ptr<EngineContext>,
           VectorTensorShapeHasher>
      cache_;

  // TODO(hinsu): Use different calibration context for the available shapes and
  // attach it to each item of the cache.
  std::unique_ptr<CalibrationContext> calib_ctx_;

  // This object maintains all the optimization profiles during profile
  // generation and engine build. During runtime the list of profiles is used to
  // look up a matching profile for the input data.
  TrtShapeOptimizationProfile profiles_;
};

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_LRU_CACHE_H_
