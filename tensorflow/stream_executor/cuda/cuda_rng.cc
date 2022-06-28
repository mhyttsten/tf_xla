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
class MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc() {
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

#include "tensorflow/stream_executor/cuda/cuda_rng.h"

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/rng.h"
// clang-format off
#include "third_party/gpus/cuda/include/curand.h"
// clang-format on

// Formats curandStatus_t to output prettified values into a log stream.
std::ostream &operator<<(std::ostream &in, const curandStatus_t &status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_0(mht_0_v, 203, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "operator<<");

#define OSTREAM_CURAND_STATUS(__name) \
  case CURAND_STATUS_##__name:        \
    in << "CURAND_STATUS_" #__name;   \
    return in;

  switch (status) {
    OSTREAM_CURAND_STATUS(SUCCESS)
    OSTREAM_CURAND_STATUS(VERSION_MISMATCH)
    OSTREAM_CURAND_STATUS(NOT_INITIALIZED)
    OSTREAM_CURAND_STATUS(ALLOCATION_FAILED)
    OSTREAM_CURAND_STATUS(TYPE_ERROR)
    OSTREAM_CURAND_STATUS(OUT_OF_RANGE)
    OSTREAM_CURAND_STATUS(LENGTH_NOT_MULTIPLE)
    OSTREAM_CURAND_STATUS(LAUNCH_FAILURE)
    OSTREAM_CURAND_STATUS(PREEXISTING_FAILURE)
    OSTREAM_CURAND_STATUS(INITIALIZATION_FAILED)
    OSTREAM_CURAND_STATUS(ARCH_MISMATCH)
    OSTREAM_CURAND_STATUS(INTERNAL_ERROR)
    default:
      in << "curandStatus_t(" << static_cast<int>(status) << ")";
      return in;
  }
}

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kGpuRandPlugin);

GpuRng::GpuRng(GpuExecutor* parent) : parent_(parent), rng_(nullptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_1(mht_1_v, 236, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::GpuRng");
}

GpuRng::~GpuRng() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_2(mht_2_v, 241, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::~GpuRng");

  if (rng_ != nullptr) {
    cuda::ScopedActivateExecutorContext sac(parent_);
    curandDestroyGenerator(rng_);
  }
}

bool GpuRng::Init() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_3(mht_3_v, 251, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::Init");

  absl::MutexLock lock(&mu_);
  CHECK(rng_ == nullptr);

  cuda::ScopedActivateExecutorContext sac(parent_);
  curandStatus_t ret = curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT);
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create random number generator: " << ret;
    return false;
  }

  CHECK(rng_ != nullptr);
  return true;
}

bool GpuRng::SetStream(Stream* stream) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_4(mht_4_v, 269, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::SetStream");

  cuda::ScopedActivateExecutorContext sac(parent_);
  curandStatus_t ret = curandSetStream(rng_, AsGpuStreamValue(stream));
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for random generation: " << ret;
    return false;
  }

  return true;
}

// Returns true if std::complex stores its contents as two consecutive
// elements. Tests int, float and double, as the last two are independent
// specializations.
constexpr bool ComplexIsConsecutiveFloats() {
  return sizeof(std::complex<int>) == 8 && sizeof(std::complex<float>) == 8 &&
      sizeof(std::complex<double>) == 16;
}

template <typename T>
bool GpuRng::DoPopulateRandUniformInternal(Stream* stream, DeviceMemory<T>* v) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_5(mht_5_v, 292, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::DoPopulateRandUniformInternal");

  absl::MutexLock lock(&mu_);
  static_assert(ComplexIsConsecutiveFloats(),
                "std::complex values are not stored as consecutive values");

  if (!SetStream(stream)) {
    return false;
  }

  // std::complex<T> is currently implemented as two consecutive T variables.
  uint64_t element_count = v->ElementCount();
  if (std::is_same<T, std::complex<float>>::value ||
      std::is_same<T, std::complex<double>>::value) {
    element_count *= 2;
  }

  cuda::ScopedActivateExecutorContext sac(parent_);
  curandStatus_t ret;
  if (std::is_same<T, float>::value ||
      std::is_same<T, std::complex<float>>::value) {
    ret = curandGenerateUniform(
        rng_, reinterpret_cast<float*>(GpuMemoryMutable(v)), element_count);
  } else {
    ret = curandGenerateUniformDouble(
        rng_, reinterpret_cast<double*>(GpuMemoryMutable(v)), element_count);
  }
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do uniform generation of " << v->ElementCount()
               << " " << TypeString<T>() << "s at " << v->opaque() << ": "
               << ret;
    return false;
  }

  return true;
}

bool GpuRng::DoPopulateRandUniform(Stream* stream, DeviceMemory<float>* v) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_6(mht_6_v, 331, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::DoPopulateRandUniform");

  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream, DeviceMemory<double>* v) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_7(mht_7_v, 338, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::DoPopulateRandUniform");

  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream,
                                   DeviceMemory<std::complex<float>>* v) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_8(mht_8_v, 346, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::DoPopulateRandUniform");

  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream,
                                   DeviceMemory<std::complex<double>>* v) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_9(mht_9_v, 354, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::DoPopulateRandUniform");

  return DoPopulateRandUniformInternal(stream, v);
}

template <typename ElemT, typename FuncT>
bool GpuRng::DoPopulateRandGaussianInternal(Stream* stream, ElemT mean,
                                            ElemT stddev,
                                            DeviceMemory<ElemT>* v,
                                            FuncT func) {
  absl::MutexLock lock(&mu_);

  if (!SetStream(stream)) {
    return false;
  }

  cuda::ScopedActivateExecutorContext sac(parent_);
  uint64_t element_count = v->ElementCount();
  curandStatus_t ret =
      func(rng_, GpuMemoryMutable(v), element_count, mean, stddev);

  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do gaussian generation of " << v->ElementCount()
               << " floats at " << v->opaque() << ": " << ret;
    return false;
  }

  return true;
}

bool GpuRng::DoPopulateRandGaussian(Stream* stream, float mean, float stddev,
                                    DeviceMemory<float>* v) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_10(mht_10_v, 387, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::DoPopulateRandGaussian");

  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        curandGenerateNormal);
}

bool GpuRng::DoPopulateRandGaussian(Stream* stream, double mean, double stddev,
                                    DeviceMemory<double>* v) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_11(mht_11_v, 396, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::DoPopulateRandGaussian");

  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        curandGenerateNormalDouble);
}

bool GpuRng::SetSeed(Stream* stream, const uint8* seed, uint64_t seed_bytes) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_12(mht_12_v, 404, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "GpuRng::SetSeed");

  absl::MutexLock lock(&mu_);
  CHECK(rng_ != nullptr);

  if (!CheckSeed(seed, seed_bytes)) {
    return false;
  }

  if (!SetStream(stream)) {
    return false;
  }

  cuda::ScopedActivateExecutorContext sac(parent_);
  // Requires 8 bytes of seed data; checked in RngSupport::CheckSeed (above)
  // (which itself requires 16 for API consistency with host RNG fallbacks).
  curandStatus_t ret = curandSetPseudoRandomGeneratorSeed(
      rng_, *(reinterpret_cast<const uint64_t*>(seed)));
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set rng seed: " << ret;
    return false;
  }

  ret = curandSetGeneratorOffset(rng_, 0);
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to reset rng position: " << ret;
    return false;
  }
  return true;
}

}  // namespace gpu

void initialize_curand() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_rngDTcc mht_13(mht_13_v, 439, "", "./tensorflow/stream_executor/cuda/cuda_rng.cc", "initialize_curand");

  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::RngFactory>(
          cuda::kCudaPlatformId, gpu::kGpuRandPlugin, "cuRAND",
          [](internal::StreamExecutorInterface* parent) -> rng::RngSupport* {
            gpu::GpuExecutor* cuda_executor =
                dynamic_cast<gpu::GpuExecutor*>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR)
                  << "Attempting to initialize an instance of the cuRAND "
                  << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            gpu::GpuRng* rng = new gpu::GpuRng(cuda_executor);
            if (!rng->Init()) {
              // Note: Init() will log a more specific error.
              delete rng;
              return nullptr;
            }
            return rng;
          });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuRAND factory: "
               << status.error_message();
  }

  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kRng, gpu::kGpuRandPlugin);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_curand,
                            { stream_executor::initialize_curand(); });
