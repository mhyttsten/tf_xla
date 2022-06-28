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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTcc() {
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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"

#include <sstream>

#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/mutex.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

string CalibrationContext::TerminateCalibration() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.cc", "CalibrationContext::TerminateCalibration");

  mutex_lock l(mu_);
  if (terminated_) return calibration_table_;

  TRTInt8Calibrator* raw_calibrator = calibrator_.get();
  raw_calibrator->waitAndSetDone();
  terminated_ = true;

  // At this point the calibration thread `thr_` is woken up and can
  // transfer the ownership of `calibrator_` and `engine_` at any time, so
  // it's not safe to use `calibrator_` below, but we can still access it
  // using raw pointer.
  // TODO(laigd): make TRTEngineOp::AllocateCalibrationResources() a member
  // function of this class instead.

  thr_->join();
  calibration_table_ = raw_calibrator->getCalibrationTableAsString();
  return calibration_table_;
}

const absl::string_view kTfTrtContainerName = "TF-TRT";

Logger& TRTEngineCacheResource::GetLogger() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTcc mht_1(mht_1_v, 226, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.cc", "TRTEngineCacheResource::GetLogger");

  static Logger* logger = new Logger();
  return *logger;
}

TRTEngineCacheResource::TRTEngineCacheResource(OpKernelContext* ctx,
                                               size_t capacity)
    : cache_(capacity) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTcc mht_2(mht_2_v, 236, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.cc", "TRTEngineCacheResource::TRTEngineCacheResource");

  auto device = ctx->device();
  auto alloc = device->GetAllocator(AllocatorAttributes());
  if (!alloc) {
    LOG(ERROR) << "Can't find device allocator for gpu device "
               << device->name();
    allocator_ = nullptr;
  } else {
    allocator_.reset(new TRTDeviceAllocator(alloc));
  }
}

TRTEngineCacheResource::~TRTEngineCacheResource() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTcc mht_3(mht_3_v, 251, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.cc", "TRTEngineCacheResource::~TRTEngineCacheResource");

  VLOG(1) << "Destroying TRTEngineCacheResource...";
}

string TRTEngineCacheResource::DebugString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTcc mht_4(mht_4_v, 258, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.cc", "TRTEngineCacheResource::DebugString");

  std::stringstream oss;
  using std::dec;
  using std::endl;
  using std::hex;
  oss << "TRTEngineCacheResource: ";
  oss << "TRTBaseAllocator = " << hex << allocator_.get() << dec << ", ";
  oss << "LRUCache = " << hex << &cache_ << dec << endl;
  oss << "Containing " << cache_.size() << " entries: " << endl;
  for (const auto& item : cache_) {
    mutex_lock lock(item.second->mu);
    oss << TensorShapeUtils::ShapeListString(item.first) << ": " << hex
        << "ICudaEngine: " << item.second->cuda_engine.get() << ", "
        << "IExecutionContext: ";
    absl::c_for_each(
        item.second->execution_contexts,
        [&](const ExecutionContext& ctx) { oss << ctx.get() << ","; });
    oss << dec << endl;
  }
  return oss.str();
}

EngineContext* TRTEngineCacheResource::GetEngineContext(
    const std::vector<TensorShape>& input_shapes) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTcc mht_5(mht_5_v, 284, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.cc", "TRTEngineCacheResource::GetEngineContext");

  EngineContext* engine_context = nullptr;
  int64 min_matched_batch_size = kint64max;
  for (const auto& pair : cache_) {
    const std::vector<TensorShape>& cached_input_shapes = pair.first;
    // This should not happen, but just for safety.
    if (input_shapes.size() != cached_input_shapes.size()) {
      LOG(ERROR) << "Input shape list size mismatch"
                 << ", cached size: " << cached_input_shapes.size()
                 << " vs. input size: " << input_shapes.size();
    }
    if (AreShapesCompatible(input_shapes, cached_input_shapes)) {
      const int cached_batch_size = cached_input_shapes[0].dim_size(0);
      if (min_matched_batch_size > cached_batch_size) {
        min_matched_batch_size = cached_batch_size;
        engine_context = pair.second.get();
      }
    }
  }
  return engine_context;
}

EngineContext* TRTEngineCacheResource::GetEngineContext(const int profile_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_lru_cacheDTcc mht_6(mht_6_v, 309, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.cc", "TRTEngineCacheResource::GetEngineContext");

  if (profiles_.NeedProfiles() && profile_id >= profiles_.GetNumProfiles()) {
    LOG(ERROR) << "Out of range: profile_id " << profile_id
               << " is larger than number of profiles "
               << profiles_.GetNumProfiles();
    return nullptr;
  }
  if (cache_.size() > 1) {
    LOG(ERROR) << "Cache is expected to have at most "
               << "1 engine in explicit batch mode where profiles are used.";
    return nullptr;
  }
  if (cache_.size() == 0) {
    return nullptr;
  }
  return cache_.begin()->second.get();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
