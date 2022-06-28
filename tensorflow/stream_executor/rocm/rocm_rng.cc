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
class MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc() {
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

#include "rocm/include/hiprand/hiprand.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/stream_executor/gpu/gpu_rng.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"

// Formats hiprandStatus_t to output prettified values into a log stream.
std::ostream& operator<<(std::ostream& in, const hiprandStatus_t& status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_0(mht_0_v, 201, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "operator<<");

#define OSTREAM_HIPRAND_STATUS(__name) \
  case HIPRAND_STATUS_##__name:        \
    in << "HIPRAND_STATUS_" #__name;   \
    return in;

  switch (status) {
    OSTREAM_HIPRAND_STATUS(SUCCESS)
    OSTREAM_HIPRAND_STATUS(VERSION_MISMATCH)
    OSTREAM_HIPRAND_STATUS(NOT_INITIALIZED)
    OSTREAM_HIPRAND_STATUS(ALLOCATION_FAILED)
    OSTREAM_HIPRAND_STATUS(TYPE_ERROR)
    OSTREAM_HIPRAND_STATUS(OUT_OF_RANGE)
    OSTREAM_HIPRAND_STATUS(LENGTH_NOT_MULTIPLE)
    OSTREAM_HIPRAND_STATUS(LAUNCH_FAILURE)
    OSTREAM_HIPRAND_STATUS(PREEXISTING_FAILURE)
    OSTREAM_HIPRAND_STATUS(INITIALIZATION_FAILED)
    OSTREAM_HIPRAND_STATUS(ARCH_MISMATCH)
    OSTREAM_HIPRAND_STATUS(INTERNAL_ERROR)
    default:
      in << "hiprandStatus_t(" << static_cast<int>(status) << ")";
      return in;
  }
}

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kGpuRandPlugin);

namespace wrap {

#ifdef PLATFORM_GOOGLE

#define STREAM_EXECUTOR_HIPRAND_WRAP(__name)                        \
  struct WrapperShim__##__name {                                    \
    template <typename... Args>                                     \
    hiprandStatus_t operator()(GpuExecutor* parent, Args... args) { \
      gpu::ScopedActivateExecutorContext sac{parent};               \
      return ::__name(args...);                                     \
    }                                                               \
  } __name;

#else

#define STREAM_EXECUTOR_HIPRAND_WRAP(__name)                              \
  struct DynLoadShim__##__name {                                          \
    static const char* kName;                                             \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;          \
    static void* GetDsoHandle() {                                         \
      auto s = internal::CachedDsoLoader::GetRocrandDsoHandle();          \
      return s.ValueOrDie();                                              \
    }                                                                     \
    static FuncPtrT LoadOrDie() {                                         \
      void* f;                                                            \
      auto s = port::Env::Default()->GetSymbolFromLibrary(GetDsoHandle(), \
                                                          kName, &f);     \
      CHECK(s.ok()) << "could not find " << kName                         \
                    << " in rocrand DSO; dlerror: " << s.error_message(); \
      return reinterpret_cast<FuncPtrT>(f);                               \
    }                                                                     \
    static FuncPtrT DynLoad() {                                           \
      static FuncPtrT f = LoadOrDie();                                    \
      return f;                                                           \
    }                                                                     \
    template <typename... Args>                                           \
    hiprandStatus operator()(GpuExecutor* parent, Args... args) {         \
      gpu::ScopedActivateExecutorContext sac{parent};                     \
      return DynLoad()(args...);                                          \
    }                                                                     \
  } __name;                                                               \
  const char* DynLoadShim__##__name::kName = #__name;

#endif

STREAM_EXECUTOR_HIPRAND_WRAP(hiprandCreateGenerator);
STREAM_EXECUTOR_HIPRAND_WRAP(hiprandDestroyGenerator);
STREAM_EXECUTOR_HIPRAND_WRAP(hiprandSetStream);
STREAM_EXECUTOR_HIPRAND_WRAP(hiprandGenerateUniform);
STREAM_EXECUTOR_HIPRAND_WRAP(hiprandGenerateUniformDouble);
STREAM_EXECUTOR_HIPRAND_WRAP(hiprandSetPseudoRandomGeneratorSeed);
STREAM_EXECUTOR_HIPRAND_WRAP(hiprandSetGeneratorOffset);
STREAM_EXECUTOR_HIPRAND_WRAP(hiprandGenerateNormal);
STREAM_EXECUTOR_HIPRAND_WRAP(hiprandGenerateNormalDouble);

}  // namespace wrap

GpuRng::GpuRng(GpuExecutor* parent) : parent_(parent), rng_(nullptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_1(mht_1_v, 291, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::GpuRng");
}

GpuRng::~GpuRng() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_2(mht_2_v, 296, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::~GpuRng");

  if (rng_ != nullptr) {
    wrap::hiprandDestroyGenerator(parent_, rng_);
  }
}

bool GpuRng::Init() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_3(mht_3_v, 305, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::Init");

  absl::MutexLock lock{&mu_};
  CHECK(rng_ == nullptr);

  hiprandStatus_t ret =
      wrap::hiprandCreateGenerator(parent_, &rng_, HIPRAND_RNG_PSEUDO_DEFAULT);
  if (ret != HIPRAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create random number generator: " << ret;
    return false;
  }

  CHECK(rng_ != nullptr);
  return true;
}

bool GpuRng::SetStream(Stream* stream) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_4(mht_4_v, 323, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::SetStream");

  hiprandStatus_t ret =
      wrap::hiprandSetStream(parent_, rng_, AsGpuStreamValue(stream));
  if (ret != HIPRAND_STATUS_SUCCESS) {
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_5(mht_5_v, 346, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::DoPopulateRandUniformInternal");

  absl::MutexLock lock{&mu_};
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

  hiprandStatus_t ret;
  if (std::is_same<T, float>::value ||
      std::is_same<T, std::complex<float>>::value) {
    ret = wrap::hiprandGenerateUniform(
        parent_, rng_, reinterpret_cast<float*>(GpuMemoryMutable(v)),
        element_count);
  } else {
    ret = wrap::hiprandGenerateUniformDouble(
        parent_, rng_, reinterpret_cast<double*>(GpuMemoryMutable(v)),
        element_count);
  }
  if (ret != HIPRAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do uniform generation of " << v->ElementCount()
               << " " << TypeString<T>() << "s at " << v->opaque() << ": "
               << ret;
    return false;
  }

  return true;
}

bool GpuRng::DoPopulateRandUniform(Stream* stream, DeviceMemory<float>* v) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_6(mht_6_v, 386, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::DoPopulateRandUniform");

  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream, DeviceMemory<double>* v) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_7(mht_7_v, 393, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::DoPopulateRandUniform");

  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream,
                                   DeviceMemory<std::complex<float>>* v) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_8(mht_8_v, 401, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::DoPopulateRandUniform");

  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream,
                                   DeviceMemory<std::complex<double>>* v) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_9(mht_9_v, 409, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::DoPopulateRandUniform");

  return DoPopulateRandUniformInternal(stream, v);
}

template <typename ElemT, typename FuncT>
bool GpuRng::DoPopulateRandGaussianInternal(Stream* stream, ElemT mean,
                                            ElemT stddev,
                                            DeviceMemory<ElemT>* v,
                                            FuncT func) {
  absl::MutexLock lock{&mu_};

  if (!SetStream(stream)) {
    return false;
  }

  uint64_t element_count = v->ElementCount();
  hiprandStatus_t ret =
      func(parent_, rng_, GpuMemoryMutable(v), element_count, mean, stddev);

  if (ret != HIPRAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do gaussian generation of " << v->ElementCount()
               << " floats at " << v->opaque() << ": " << ret;
    return false;
  }

  return true;
}

bool GpuRng::DoPopulateRandGaussian(Stream* stream, float mean, float stddev,
                                    DeviceMemory<float>* v) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_10(mht_10_v, 441, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::DoPopulateRandGaussian");

  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        wrap::hiprandGenerateNormal);
}

bool GpuRng::DoPopulateRandGaussian(Stream* stream, double mean, double stddev,
                                    DeviceMemory<double>* v) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_11(mht_11_v, 450, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::DoPopulateRandGaussian");

  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        wrap::hiprandGenerateNormalDouble);
}

bool GpuRng::SetSeed(Stream* stream, const uint8* seed, uint64_t seed_bytes) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_12(mht_12_v, 458, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "GpuRng::SetSeed");

  absl::MutexLock lock{&mu_};
  CHECK(rng_ != nullptr);

  if (!CheckSeed(seed, seed_bytes)) {
    return false;
  }

  if (!SetStream(stream)) {
    return false;
  }

  // Requires 8 bytes of seed data; checked in RngSupport::CheckSeed (above)
  // (which itself requires 16 for API consistency with host RNG fallbacks).
  hiprandStatus_t ret = wrap::hiprandSetPseudoRandomGeneratorSeed(
      parent_, rng_, *(reinterpret_cast<const uint64_t*>(seed)));
  if (ret != HIPRAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set rng seed: " << ret;
    return false;
  }

  ret = wrap::hiprandSetGeneratorOffset(parent_, rng_, 0);
  if (ret != HIPRAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to reset rng position: " << ret;
    return false;
  }
  return true;
}

}  // namespace gpu

void initialize_rocrand() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_rngDTcc mht_13(mht_13_v, 492, "", "./tensorflow/stream_executor/rocm/rocm_rng.cc", "initialize_rocrand");

  auto rocRandAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      rocm::kROCmPlatformId, PluginKind::kRng, gpu::kGpuRandPlugin);

  if (!rocRandAlreadyRegistered) {
    port::Status status =
        PluginRegistry::Instance()->RegisterFactory<PluginRegistry::RngFactory>(
            rocm::kROCmPlatformId, gpu::kGpuRandPlugin, "rocRAND",
            [](internal::StreamExecutorInterface* parent) -> rng::RngSupport* {
              gpu::GpuExecutor* rocm_executor =
                  dynamic_cast<gpu::GpuExecutor*>(parent);
              if (rocm_executor == nullptr) {
                LOG(ERROR)
                    << "Attempting to initialize an instance of the hipRAND "
                    << "support library with a non-ROCM StreamExecutor";
                return nullptr;
              }

              gpu::GpuRng* rng = new gpu::GpuRng(rocm_executor);
              if (!rng->Init()) {
                // Note: Init() will log a more specific error.
                delete rng;
                return nullptr;
              }
              return rng;
            });

    if (!status.ok()) {
      LOG(ERROR) << "Unable to register rocRAND factory: "
                 << status.error_message();
    }

    PluginRegistry::Instance()->SetDefaultFactory(
        rocm::kROCmPlatformId, PluginKind::kRng, gpu::kGpuRandPlugin);
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_rocrand,
                            { stream_executor::initialize_rocrand(); });
