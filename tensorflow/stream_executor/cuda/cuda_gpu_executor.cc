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
class MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc() {
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

#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"

#include <utility>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#define PATH_MAX MAX_PATH
#else
#include <unistd.h>
#endif
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_event.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/cuda/cuda_timer.h"
#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/mathutil.h"
#include "tensorflow/stream_executor/lib/numbers.h"
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/timer.h"

// LOG(ERROR) uses a const named ERROR, so a macro with the same name is
// always unwanted. This happens on Windows that defines such a macro.
#undef ERROR

#ifdef PLATFORMS_GPUS_CUDA_DYNAMIC_LIBCUDA_DYNAMIC_LIBCUDA_H_
#error \
    "No driver calls in this file, wrap driver functionality in cuda_driver.cc."
#endif

#ifdef __CUDA_RUNTIME_H__
#error \
    "CUDA runtime being included into CUDA GPU executor; should be driver only."
#endif

extern bool FLAGS_check_gpu_leaks;
bool FLAGS_prefer_cubin_to_ptx = true;

namespace stream_executor {
namespace gpu {

// Hook that can be used to CUBIN-ate PTX before it is loaded into the driver.
// It has been observed that loading both PTX and cubins into the driver library
// can cause it to crash, but loading only CUBINs avoids those crashes;
// therefore, it's useful to have this hook to hack in uniform CUBIN-ation of
// PTX code.
//
// As this is an implementation-detail workaround, the usage is to declare this
// variable with extern linkage and populate it from another translation unit.
std::function<std::string(const std::string&)> g_cubinate;

static GpuEvent* AsGpuEvent(Event* event) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_0(mht_0_v, 257, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "AsGpuEvent");

  DCHECK(event != nullptr);
  return static_cast<GpuEvent*>(event->implementation());
}

// Given a platform-independent timer datatype, returns the internal CUDA
// platform implementation pointer.
static GpuTimer* AsGpuTimer(Timer* timer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_1(mht_1_v, 267, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "AsGpuTimer");

  DCHECK(timer != nullptr);
  return static_cast<GpuTimer*>(timer->implementation());
}

// Given const GPU memory, returns a libcuda device pointer datatype, suitable
// for passing directly to libcuda APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// libcuda APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to libcuda functions which will honor constness.
static CUdeviceptr AsCudaDevicePtr(const DeviceMemoryBase& gpu_mem) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_2(mht_2_v, 281, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "AsCudaDevicePtr");

  return reinterpret_cast<CUdeviceptr>(gpu_mem.opaque());
}

// See description on const version above.
static CUdeviceptr AsCudaDevicePtr(DeviceMemoryBase* gpu_mem) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_3(mht_3_v, 289, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "AsCudaDevicePtr");

  return AsCudaDevicePtr(*gpu_mem);
}

GpuContext* ExtractGpuContext(GpuExecutor* cuda_exec) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_4(mht_4_v, 296, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "ExtractGpuContext");

  CHECK(cuda_exec != nullptr);
  return cuda_exec->gpu_context();
}

GpuExecutor::~GpuExecutor() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_5(mht_5_v, 304, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::~GpuExecutor");

  CHECK(kernel_to_gpu_binary_.empty()) << "GpuExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "GpuExecutor has loaded modules.";
  if (context_ != nullptr) {
    GpuDriver::DestroyContext(context_);
  }
}

port::Status GpuExecutor::Init(int device_ordinal,
                               DeviceOptions device_options) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_6(mht_6_v, 316, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::Init");

  device_ordinal_ = device_ordinal;

  auto status = GpuDriver::Init();
  if (!status.ok()) {
    return status;
  }

  status = GpuDriver::GetDevice(device_ordinal_, &device_);
  if (!status.ok()) {
    return status;
  }

  status = GpuDriver::CreateContext(device_ordinal_, device_, device_options,
                                    &context_);
  if (!status.ok()) {
    return status;
  }

  return GpuDriver::GetComputeCapability(&cc_major_, &cc_minor_, device_);
}

bool GpuExecutor::FindOnDiskForComputeCapability(
    absl::string_view filename, absl::string_view canonical_suffix,
    std::string* found_filename) const {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_7_v.push_back("canonical_suffix: \"" + std::string(canonical_suffix.data(), canonical_suffix.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_7(mht_7_v, 345, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::FindOnDiskForComputeCapability");

  if (cc_major_ == 0 && cc_minor_ == 0) {
    return false;
  }

  std::string cc_specific =
      absl::StrCat(filename, ".cc", cc_major_, cc_minor_, canonical_suffix);
  if (port::FileExists(cc_specific).ok()) {
    VLOG(2) << "found compute-capability-specific file, using that: "
            << cc_specific;
    *found_filename = cc_specific;
    return true;
  }

  VLOG(2) << "could not find compute-capability specific file at: "
          << cc_specific;
  if (port::FileExists(std::string(filename)).ok()) {
    *found_filename = std::string(filename);
    return true;
  }

  return false;
}

bool GpuExecutor::FindOnDiskForISAVersion(absl::string_view filename,
                                          absl::string_view canonical_suffix,
                                          std::string* found_filename) const {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_8_v.push_back("canonical_suffix: \"" + std::string(canonical_suffix.data(), canonical_suffix.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_8(mht_8_v, 376, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::FindOnDiskForISAVersion");

  LOG(ERROR)
      << "Feature not supported on CUDA platform (FindOnDiskForISAVersion)";
  return false;
}
// Returns the path to the running executable.
// N.B. Derived from //knowledge/smalltalk/background_kb.cc
// Arg: strip_exe: if true, remove the name of the executable itself from the
//                 returned string. Example: calling this from /usr/bin/foo
//                 would return /usr/bin.
static std::string GetBinaryDir(bool strip_exe) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_9(mht_9_v, 389, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GetBinaryDir");

  std::string exe_path = port::GetExecutablePath();
  if (strip_exe) {
    // The exe is the last component of the path, so remove one component.
    std::vector<std::string> components = absl::StrSplit(exe_path, '/');
    components.pop_back();
    return absl::StrJoin(components, "/");
  }
  return exe_path;
}

port::Status GpuExecutor::LoadModuleFromCuBin(const char* cubin,
                                              CUmodule* module) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("cubin: \"" + (cubin == nullptr ? std::string("nullptr") : std::string((char*)cubin)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_10(mht_10_v, 405, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::LoadModuleFromCuBin");

  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[cubin];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(GpuDriver::LoadCubin(context_, cubin, module));
    module_refcount = 1;
    VLOG(3) << "Loaded CUBIN " << static_cast<const void*>(cubin)
            << " as module " << *module;
  } else {
    ++module_refcount;
    VLOG(3) << "CUBIN " << static_cast<const void*>(cubin)
            << " is already loaded as module " << *module;
  }
  gpu_binary_to_module_[cubin] = {*module, module_refcount};
  return port::Status::OK();
}

port::Status GpuExecutor::LoadModuleFromPtx(const char* ptx, CUmodule* module) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("ptx: \"" + (ptx == nullptr ? std::string("nullptr") : std::string((char*)ptx)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_11(mht_11_v, 427, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::LoadModuleFromPtx");

  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[ptx];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(GpuDriver::LoadPtx(context_, ptx, module));
    VLOG(3) << "Loaded PTX " << static_cast<const void*>(ptx) << " as module "
            << *module;
    module_refcount = 1;
  } else {
    ++module_refcount;
    VLOG(3) << "PTX " << static_cast<const void*>(ptx)
            << " is already loaded as module " << module;
  }
  gpu_binary_to_module_[ptx] = {*module, module_refcount};
  return port::Status::OK();
}

port::Status GpuExecutor::LoadModuleFromHsaco(const char* hsaco,
                                              CUmodule* module) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("hsaco: \"" + (hsaco == nullptr ? std::string("nullptr") : std::string((char*)hsaco)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_12(mht_12_v, 450, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::LoadModuleFromHsaco");

  return port::InternalError(
      "Feature not supported on CUDA platform (LoadModuleFromHsaco)");
}

port::Status GpuExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                                    KernelBase* kernel) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_13(mht_13_v, 459, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::GetKernel");

  GpuKernel* cuda_kernel = AsGpuKernel(kernel);
  CUmodule module;
  const std::string* kernelname;

  VLOG(3) << "GetKernel on kernel " << kernel << " : " << kernel->name();

  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    kernelname = &spec.cuda_cubin_in_memory().kernelname();
    const char* cubin = spec.cuda_cubin_in_memory().bytes();
    TF_RETURN_IF_ERROR(LoadModuleFromCuBin(cubin, &module));
    kernel_to_gpu_binary_[kernel] = cubin;
  } else if (spec.has_cuda_ptx_in_memory()) {
    kernelname = &spec.cuda_ptx_in_memory().kernelname();

    if (cc_major_ == 0 && cc_minor_ == 0) {
      return port::InternalError("Compute capability not set");
    }

    const char* ptx = spec.cuda_ptx_in_memory().text(cc_major_, cc_minor_);
    if (ptx == nullptr) {
      ptx = spec.cuda_ptx_in_memory().default_text();
    }
    if (ptx == nullptr) {
      LOG(FATAL) << "Loader spec has no ptx for kernel " << *kernelname;
    }

    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(LoadModuleFromPtx(ptx, &module));
    kernel_to_gpu_binary_[kernel] = ptx;
  } else {
    return port::InternalError("No method of loading CUDA kernel provided");
  }
  VLOG(2) << "getting function " << *kernelname << " from module " << module;
  if (!GpuDriver::GetModuleFunction(context_, module, kernelname->c_str(),
                                    cuda_kernel->gpu_function_ptr())) {
    return port::InternalError("Could not find the corresponding function");
  }

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the CUDA API.
  cuda_kernel->set_arity(spec.arity());

  KernelMetadata kernel_metadata;
  TF_RETURN_IF_ERROR(GetKernelMetadata(cuda_kernel, &kernel_metadata));
  kernel->set_metadata(kernel_metadata);
  kernel->set_name(*kernelname);
  return port::Status::OK();
}

bool GpuExecutor::UnloadGpuBinary(const void* gpu_binary) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_14(mht_14_v, 513, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::UnloadGpuBinary");

  auto module_it = gpu_binary_to_module_.find(gpu_binary);
  if (gpu_binary_to_module_.end() == module_it) {
    VLOG(3) << "No loaded CUDA module for " << gpu_binary;
    return false;
  }
  auto& module = module_it->second.first;
  auto& refcount = module_it->second.second;
  VLOG(3) << "Found CUDA module " << module << " with refcount " << refcount;
  if (--refcount == 0) {
    VLOG(3) << "Unloading CUDA module " << module;
    GpuDriver::UnloadModule(context_, module);
    gpu_binary_to_module_.erase(module_it);
  }
  return true;
}

void GpuExecutor::UnloadKernel(const KernelBase* kernel) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_15(mht_15_v, 533, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::UnloadKernel");

  VLOG(3) << "Unloading kernel " << kernel << " : " << kernel->name();

  absl::MutexLock lock{&in_memory_modules_mu_};
  auto gpu_binary_it = kernel_to_gpu_binary_.find(kernel);
  if (kernel_to_gpu_binary_.end() == gpu_binary_it) {
    VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
            << " has never been loaded.";
    return;  // We've never seen this kernel.
  }
  VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
          << " has loaded GPU code " << gpu_binary_it->second;
  UnloadGpuBinary(gpu_binary_it->second);
  kernel_to_gpu_binary_.erase(gpu_binary_it);
}

port::Status GpuExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                                     ModuleHandle* module_handle) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_16(mht_16_v, 553, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::LoadModule");

  // In GpuExecutor we store the pointer to the GPU binary (PTX or CUBIN) as
  // ModuleHandle::id().
  CUmodule cu_module;
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(LoadModuleFromCuBin(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()),
        &cu_module));
    *module_handle = ModuleHandle(const_cast<void*>(
        static_cast<const void*>(spec.cuda_cubin_in_memory().data())));
    return port::Status::OK();
  } else if (spec.has_cuda_ptx_in_memory()) {
    if (cc_major_ == 0 && cc_minor_ == 0) {
      return port::InternalError("Compute capability not set");
    }

    if (!spec.cuda_ptx_in_memory()) {
      return port::InternalError("PTX not found in spec");
    }

    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(
        LoadModuleFromPtx(spec.cuda_ptx_in_memory(), &cu_module));
    *module_handle = ModuleHandle(
        const_cast<void*>(static_cast<const void*>(spec.cuda_ptx_in_memory())));
    return port::Status::OK();
  }
  return port::InternalError("No method of loading CUDA module provided");
}

bool GpuExecutor::UnloadModule(ModuleHandle module_handle) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_17(mht_17_v, 587, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::UnloadModule");

  const char* gpu_binary = reinterpret_cast<const char*>(module_handle.id());
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(gpu_binary);
}

namespace {
absl::uint128 Fingerprint128(const absl::string_view s) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("s: \"" + std::string(s.data(), s.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_18(mht_18_v, 598, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "Fingerprint128");

  auto fp = tensorflow::Fingerprint128(s);
  return absl::MakeUint128(fp.high64, fp.low64);
}
}  // namespace

port::StatusOr<std::shared_ptr<DeviceMemoryBase>>
GpuExecutor::CreateOrShareConstant(Stream* stream,
                                   const std::vector<uint8_t>& content) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_19(mht_19_v, 609, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CreateOrShareConstant");

  absl::MutexLock lock{&shared_constants_mu_};
  // We assume all constants are uniquely identified by this hash. In the
  // (highly unlikely) event of a hash collision, the program will likely crash
  // (because the cached constant that will be returned by mistake is unlikely
  // to have the correct size).
  absl::uint128 fingerprint = Fingerprint128(absl::string_view(
      reinterpret_cast<const char*>(content.data()), content.size()));
  // Must insert nullptr first to get an iterator to the insertion point.
  auto insert_result = shared_constants_.insert(
      {fingerprint, std::weak_ptr<DeviceMemoryBase>()});
  auto it = insert_result.first;
  bool was_already_in_cache = !insert_result.second;
  std::shared_ptr<DeviceMemoryBase> shared_constant;

  if (was_already_in_cache) {
    shared_constant = it->second.lock();
  }

  if (shared_constant == nullptr) {
    // Either the constant wasn't found in the cache, or it was but its
    // weak_ptr had expired.
    DeviceMemoryBase* new_constant =
        new DeviceMemoryBase(Allocate(content.size(), /*memory_space=*/0));
    if (new_constant->opaque() == nullptr) {
      return port::InternalError(absl::StrFormat(
          "Failed to allocate %d bytes for new constant", content.size()));
    }

    port::Status status =
        stream->ThenMemcpy(new_constant, content.data(), content.size())
            .BlockHostUntilDone();
    if (!status.ok()) {
      Deallocate(new_constant);
      status.Update(port::InternalError(absl::StrFormat(
          "Memcpy to device address %p failed", new_constant->opaque())));
      return status;
    }

    // Capturing 'this' in the custom deleter means this executor must
    // outlive all shared uses of this constant.
    shared_constant = std::shared_ptr<DeviceMemoryBase>(
        new_constant, [this](DeviceMemoryBase* p) {
          Deallocate(p);
          delete p;
        });
    it->second = std::weak_ptr<DeviceMemoryBase>(shared_constant);
  }

  return shared_constant;
}

port::Status GpuExecutor::GetKernelMetadata(GpuKernel* cuda_kernel,
                                            KernelMetadata* kernel_metadata) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_20(mht_20_v, 665, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::GetKernelMetadata");

  int value;
  TF_RETURN_IF_ERROR(GpuDriver::FuncGetAttribute(
      CU_FUNC_ATTRIBUTE_NUM_REGS, *cuda_kernel->gpu_function_ptr(), &value));
  kernel_metadata->set_registers_per_thread(value);

  TF_RETURN_IF_ERROR(
      GpuDriver::FuncGetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                  *cuda_kernel->gpu_function_ptr(), &value));
  kernel_metadata->set_shared_memory_bytes(value);
  return port::Status::OK();
}

port::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const KernelBase& kernel,
                                 const KernelArgsArrayBase& args) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_21(mht_21_v, 684, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::Launch");

  CHECK_EQ(kernel.Arity(), args.number_of_arguments());
  CUstream custream = AsGpuStreamValue(stream);
  const GpuKernel* cuda_kernel = AsGpuKernel(&kernel);
  CUfunction cufunc = cuda_kernel->AsGpuFunctionHandle();

  // Only perform/print the occupancy check once.  Even just checking to see
  // whether we've done an occupancy check on this kernel before isn't free
  // (because we have to synchronize), so we only do this at -v 2+.
  if (VLOG_IS_ON(2)) {
    absl::MutexLock lock(&launched_kernels_mu_);
    if (!launched_kernels_.count(cufunc)) {
      VlogOccupancyInfo(kernel, thread_dims, block_dims);
      // TODO(rspringer): Remove elements from launched_kernels_...if we ever
      // expose a kernel/module deallocation method.
      launched_kernels_.insert(cufunc);
    }
  }

  if (cuda_kernel->GetPreferredCacheConfig() !=
      KernelCacheConfig::kNoPreference) {
    TF_RETURN_IF_ERROR(GpuDriver::FuncSetCacheConfig(
        cufunc, cuda_kernel->GetGpuCacheConfig()));
  }

  void** kernel_params = const_cast<void**>(args.argument_addresses().data());

  return GpuDriver::LaunchKernel(context_, kernel.name(), cufunc, block_dims.x,
                                 block_dims.y, block_dims.z, thread_dims.x,
                                 thread_dims.y, thread_dims.z,
                                 args.number_of_shared_bytes(), custream,
                                 kernel_params, nullptr /* = extra */);
}

// This is a non-essential operation; if there's a failure, proceed without
// logging an error. It's nearly certain that in case of failures, we'd never
// get here in the first place; these are very low-impact routines.
void GpuExecutor::VlogOccupancyInfo(const KernelBase& kernel,
                                    const ThreadDim& thread_dims,
                                    const BlockDim& block_dims) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_22(mht_22_v, 726, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::VlogOccupancyInfo");

  VLOG(2) << "Computing kernel occupancy for kernel "
          << kernel.demangled_name();
  VLOG(2) << "Thread dimensions (" << thread_dims.x << ", " << thread_dims.y
          << ", " << thread_dims.z << ")";

  int regs_per_thread;
  if (!kernel.metadata().registers_per_thread(&regs_per_thread)) {
    return;
  }

  int smem_per_block;
  if (!kernel.metadata().shared_memory_bytes(&smem_per_block)) {
    return;
  }

  const DeviceDescription& device_description =
      kernel.parent()->GetDeviceDescription();

  const GpuKernel* cuda_kernel = AsGpuKernel(&kernel);
  CUfunction cufunc = cuda_kernel->AsGpuFunctionHandle();

  int blocks_per_sm = CalculateOccupancy(device_description, regs_per_thread,
                                         smem_per_block, thread_dims, cufunc);
  VLOG(2) << "Resident blocks per SM is " << blocks_per_sm;

  int suggested_threads =
      CompareOccupancy(&blocks_per_sm, device_description, regs_per_thread,
                       smem_per_block, thread_dims, cufunc);
  if (suggested_threads != 0) {
    VLOG(2) << "The cuda occupancy calculator recommends using "
            << suggested_threads
            << " threads per block to achieve an occupancy of " << blocks_per_sm
            << " blocks per SM.";
  }
}

// Compute and return maximum blocks per core (occupancy) based on the
// device description, some kernel characteristics and the number of threads per
// block.  If unable to compute occupancy, zero is returned.
int GpuExecutor::CalculateOccupancy(const DeviceDescription& device_description,
                                    uint64_t registers_per_thread,
                                    uint64_t shared_memory_per_block,
                                    const ThreadDim& thread_dims,
                                    CUfunction func) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_23(mht_23_v, 773, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CalculateOccupancy");

  int suggested_blocks = 0;
  int suggested_threads = 0;
  CUresult err = cuOccupancyMaxPotentialBlockSize(
      &suggested_blocks, &suggested_threads, func, nullptr,
      shared_memory_per_block, 0);
  CHECK_EQ(err, CUDA_SUCCESS);
  return suggested_blocks;
}

// Compute and return the suggested thread count to achieve ideal occupancy.
// If the provided thread dimensions match this number, zero is returned.
int GpuExecutor::CompareOccupancy(int* initial_blocks,
                                  const DeviceDescription& device_description,
                                  uint64_t registers_per_thread,
                                  uint64_t shared_memory_per_block,
                                  const ThreadDim& thread_dims,
                                  CUfunction func) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_24(mht_24_v, 793, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CompareOccupancy");

  int suggested_blocks = 0;
  int suggested_threads = 0;
  CUresult err = cuOccupancyMaxPotentialBlockSize(
      &suggested_blocks, &suggested_threads, func, nullptr,
      shared_memory_per_block, 0);
  CHECK_EQ(err, CUDA_SUCCESS);
  if (suggested_blocks > *initial_blocks) {
    *initial_blocks = suggested_blocks;
    return suggested_threads;
  } else {
    return 0;
  }
}

DeviceMemoryBase GpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_25(mht_25_v, 811, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::Allocate");

  CHECK_EQ(memory_space, 0);
  return DeviceMemoryBase(GpuDriver::DeviceAllocate(context_, size), size);
}

void* GpuExecutor::GetSubBuffer(DeviceMemoryBase* mem, uint64_t offset_bytes,
                                uint64_t size_bytes) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_26(mht_26_v, 820, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::GetSubBuffer");

  // offset and size are in bytes, so char* works as the pointer type.
  return reinterpret_cast<char*>(mem->opaque()) + offset_bytes;
}

void GpuExecutor::Deallocate(DeviceMemoryBase* mem) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_27(mht_27_v, 828, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::Deallocate");

  GpuDriver::DeviceDeallocate(context_, mem->opaque());
}

bool GpuExecutor::HostMemoryRegister(void* location, uint64_t size) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_28(mht_28_v, 835, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::HostMemoryRegister");

  if (location == nullptr || size == 0) {
    LOG(WARNING) << "attempting to register null or zero-sized memory: "
                 << location << "; size " << size;
  }
  VLOG(2) << "registering " << location << " size " << size;
  return GpuDriver::HostRegister(context_, location, size);
}

bool GpuExecutor::HostMemoryUnregister(void* location) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_29(mht_29_v, 847, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::HostMemoryUnregister");

  VLOG(2) << "unregistering " << location;
  return GpuDriver::HostUnregister(context_, location);
}

bool GpuExecutor::SynchronizeAllActivity() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_30(mht_30_v, 855, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::SynchronizeAllActivity");

  return GpuDriver::SynchronizeContext(context_);
}

port::Status GpuExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                             uint64_t size) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_31(mht_31_v, 863, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::SynchronousMemZero");

  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return GpuDriver::SynchronousMemsetUint32(
        context_, AsCudaDevicePtr(location), 0x0, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                           0x0, size);
}

port::Status GpuExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                            int value, uint64_t size) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_32(mht_32_v, 877, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::SynchronousMemSet");

  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    // cudaMemset reinterprets "value" as a uint8.
    uint8 byte_value = static_cast<uint8>(value);
    uint32 pattern = (byte_value << 24) | (byte_value << 16) |
                     (byte_value << 8) | byte_value;
    return GpuDriver::SynchronousMemsetUint32(
        context_, AsCudaDevicePtr(location), pattern, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                           value, size);
}

port::Status GpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                            const void* host_src,
                                            uint64_t size) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_33(mht_33_v, 896, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::SynchronousMemcpy");

  return GpuDriver::SynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                                         host_src, size);
}

port::Status GpuExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceMemoryBase& gpu_src,
                                            uint64_t size) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_34(mht_34_v, 906, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::SynchronousMemcpy");

  return GpuDriver::SynchronousMemcpyD2H(context_, host_dst,
                                         AsCudaDevicePtr(gpu_src), size);
}

port::Status GpuExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_35(mht_35_v, 915, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::SynchronousMemcpyDeviceToDevice");

  return GpuDriver::SynchronousMemcpyD2D(context_, AsCudaDevicePtr(gpu_dst),
                                         AsCudaDevicePtr(gpu_src), size);
}

port::Status GpuExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                  uint64_t size) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_36(mht_36_v, 924, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::MemZero");

  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return Memset32(stream, location, 0x0, size);
  } else {
    return Memset(stream, location, 0x0, size);
  }
}

port::Status GpuExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                 uint8 pattern, uint64_t size) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_37(mht_37_v, 937, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::Memset");

  VLOG(2) << "enqueueing memset8 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  return GpuDriver::AsynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                            pattern, size,
                                            AsGpuStreamValue(stream));
}

port::Status GpuExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                   uint32 pattern, uint64_t size) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_38(mht_38_v, 950, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::Memset32");

  VLOG(2) << "enqueueing memset32 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
        size % 4 == 0);
  return GpuDriver::AsynchronousMemsetUint32(
      context_, AsCudaDevicePtr(location), pattern, size / 4,
      AsGpuStreamValue(stream));
}

bool GpuExecutor::Memcpy(Stream* stream, void* host_dst,
                         const DeviceMemoryBase& gpu_src, uint64_t size) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_39(mht_39_v, 965, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::Memcpy");

  return GpuDriver::AsynchronousMemcpyD2H(context_, host_dst,
                                          AsCudaDevicePtr(gpu_src), size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                         const void* host_src, uint64_t size) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_40(mht_40_v, 975, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::Memcpy");

  return GpuDriver::AsynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                                          host_src, size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::MemcpyDeviceToDevice(Stream* stream,
                                       DeviceMemoryBase* gpu_dst,
                                       const DeviceMemoryBase& gpu_src,
                                       uint64_t size) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_41(mht_41_v, 987, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::MemcpyDeviceToDevice");

  return GpuDriver::AsynchronousMemcpyD2D(context_, AsCudaDevicePtr(gpu_dst),
                                          AsCudaDevicePtr(gpu_src), size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::HostCallback(Stream* stream,
                               std::function<port::Status()> callback) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_42(mht_42_v, 997, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::HostCallback");

  auto callback_ptr = new std::function<void()>([callback]() {
    port::Status s = callback();
    if (!s.ok()) {
      LOG(WARNING) << "Host callback failed: " << s;
    }
  });
  return GpuDriver::AddStreamCallback(context_, AsGpuStreamValue(stream),
                                      InternalHostCallback, callback_ptr);
}

/* static */ void GpuExecutor::InternalHostCallback(CUstream stream,
                                                    CUresult status,
                                                    void* data) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_43(mht_43_v, 1013, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::InternalHostCallback");

  std::function<void()>* callback =
      reinterpret_cast<std::function<void()>*>(data);
  (*callback)();
  delete callback;
}

port::Status GpuExecutor::AllocateEvent(Event* event) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_44(mht_44_v, 1023, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::AllocateEvent");

  return AsGpuEvent(event)->Init();
}

port::Status GpuExecutor::DeallocateEvent(Event* event) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_45(mht_45_v, 1030, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::DeallocateEvent");

  return AsGpuEvent(event)->Destroy();
}

port::Status GpuExecutor::RecordEvent(Stream* stream, Event* event) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_46(mht_46_v, 1037, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::RecordEvent");

  return AsGpuEvent(event)->Record(AsGpuStream(stream));
}

port::Status GpuExecutor::WaitForEvent(Stream* stream, Event* event) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_47(mht_47_v, 1044, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::WaitForEvent");

  if (GpuDriver::WaitStreamOnEvent(context_, AsGpuStream(stream)->gpu_stream(),
                                   AsGpuEvent(event)->gpu_event())) {
    return port::Status::OK();
  } else {
    return port::Status(
        port::error::INTERNAL,
        absl::StrFormat("error recording waiting for CUDA event on stream %p",
                        stream));
  }
}

Event::Status GpuExecutor::PollForEventStatus(Event* event) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_48(mht_48_v, 1059, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::PollForEventStatus");

  return AsGpuEvent(event)->PollForStatus();
}

bool GpuExecutor::AllocateStream(Stream* stream) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_49(mht_49_v, 1066, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::AllocateStream");

  return AsGpuStream(stream)->Init();
}

void GpuExecutor::DeallocateStream(Stream* stream) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_50(mht_50_v, 1073, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::DeallocateStream");

  GpuStream* cuda_stream = AsGpuStream(stream);
  if (!cuda_stream->IsIdle()) {
    LOG(ERROR) << "Deallocating stream with pending work";
  }
  cuda_stream->Destroy();
}

bool GpuExecutor::AllocateTimer(Timer* timer) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_51(mht_51_v, 1084, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::AllocateTimer");

  return AsGpuTimer(timer)->Init();
}

void GpuExecutor::DeallocateTimer(Timer* timer) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_52(mht_52_v, 1091, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::DeallocateTimer");

  AsGpuTimer(timer)->Destroy();
}

bool GpuExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_53(mht_53_v, 1098, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CreateStreamDependency");

  CUevent other_completed_event = *AsGpuStream(other)->completed_event();
  bool ok = GpuDriver::RecordEvent(context_, other_completed_event,
                                   AsGpuStreamValue(other))
                .ok();
  if (!ok) {
    LOG(ERROR) << "failed to record completion event; "
                  "therefore, failed to create inter-stream dependency";
    return false;
  }

  return GpuDriver::WaitStreamOnEvent(context_, AsGpuStreamValue(dependent),
                                      other_completed_event);
}

bool GpuExecutor::StartTimer(Stream* stream, Timer* timer) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_54(mht_54_v, 1116, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::StartTimer");

  return AsGpuTimer(timer)->Start(AsGpuStream(stream));
}

bool GpuExecutor::StopTimer(Stream* stream, Timer* timer) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_55(mht_55_v, 1123, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::StopTimer");

  return AsGpuTimer(timer)->Stop(AsGpuStream(stream));
}

port::Status GpuExecutor::BlockHostUntilDone(Stream* stream) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_56(mht_56_v, 1130, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::BlockHostUntilDone");

  return GpuDriver::SynchronizeStream(context_, AsGpuStreamValue(stream));
}

blas::BlasSupport* GpuExecutor::CreateBlas() {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_57(mht_57_v, 1137, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CreateBlas");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(cuda::kCudaPlatformId,
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

dnn::DnnSupport* GpuExecutor::CreateDnn() {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_58(mht_58_v, 1154, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CreateDnn");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(cuda::kCudaPlatformId,
                                                       plugin_config_.dnn());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve DNN factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

fft::FftSupport* GpuExecutor::CreateFft() {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_59(mht_59_v, 1171, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CreateFft");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(cuda::kCudaPlatformId,
                                                       plugin_config_.fft());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

rng::RngSupport* GpuExecutor::CreateRng() {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_60(mht_60_v, 1188, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CreateRng");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(cuda::kCudaPlatformId,
                                                       plugin_config_.rng());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve RNG factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

// TODO(rspringer): Remove in b/18544742.
bool GpuExecutor::SupportsDnn() const {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_61(mht_61_v, 1206, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::SupportsDnn");
 return true; }

bool GpuExecutor::CanEnablePeerAccessTo(StreamExecutorInterface* other) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_62(mht_62_v, 1211, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CanEnablePeerAccessTo");

  GpuExecutor* cuda_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::CanEnablePeerAccess(context_, cuda_other->context_);
}

port::Status GpuExecutor::EnablePeerAccessTo(StreamExecutorInterface* other) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_63(mht_63_v, 1219, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::EnablePeerAccessTo");

  GpuExecutor* cuda_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::EnablePeerAccess(context_, cuda_other->context_);
}

bool GpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_64(mht_64_v, 1227, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::DeviceMemoryUsage");

  return GpuDriver::GetDeviceMemoryInfo(context_, free, total);
}

bool GpuExecutor::GetSymbol(const std::string& symbol_name,
                            ModuleHandle module_handle, void** mem,
                            size_t* bytes) {
   std::vector<std::string> mht_65_v;
   mht_65_v.push_back("symbol_name: \"" + symbol_name + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_65(mht_65_v, 1237, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::GetSymbol");

  CHECK(static_cast<bool>(module_handle));

  auto lookup_in_module = [&](CUmodule module) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_66(mht_66_v, 1243, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "lambda");

    CHECK(module != nullptr);
    return GpuDriver::GetModuleSymbol(context_, module, symbol_name.c_str(),
                                      reinterpret_cast<CUdeviceptr*>(mem),
                                      bytes);
  };

  {  // give limited scope to mutex_lock
    absl::MutexLock lock{&in_memory_modules_mu_};
    auto it = gpu_binary_to_module_.find(module_handle.id());
    CHECK(it != gpu_binary_to_module_.end());
    return lookup_in_module(it->second.first);
  }

  LOG(INFO) << "Failed to find symbol: " << symbol_name;
  return false;
}

bool FillBlockDimLimit(GpuDeviceHandle device, BlockDim* block_dim_limit) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_67(mht_67_v, 1264, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "FillBlockDimLimit");

  // The BlockDim name is a mismatch against these GRID_DIM_* queries because
  // we use BlockDims to express the dimensions of blocks within a grid
  // (as opposed to ThreadDim which expresses the dimensions of threads
  // within a block).
  int x, y, z;
  if (!GpuDriver::GetGridLimits(&x, &y, &z, device)) {
    return false;
  }

  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return true;
}

bool GpuExecutor::SupportsBlas() const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_68(mht_68_v, 1283, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::SupportsBlas");
 return true; }

bool GpuExecutor::SupportsFft() const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_69(mht_69_v, 1288, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::SupportsFft");
 return true; }

bool GpuExecutor::SupportsRng() const {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_70(mht_70_v, 1293, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::SupportsRng");
 return true; }

std::unique_ptr<internal::EventInterface>
GpuExecutor::CreateEventImplementation() {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_71(mht_71_v, 1299, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CreateEventImplementation");

  return std::unique_ptr<internal::EventInterface>(new GpuEvent(this));
}

std::unique_ptr<internal::KernelInterface>
GpuExecutor::CreateKernelImplementation() {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_72(mht_72_v, 1307, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CreateKernelImplementation");

  return std::unique_ptr<internal::KernelInterface>(new GpuKernel());
}

std::unique_ptr<internal::StreamInterface>
GpuExecutor::GetStreamImplementation() {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_73(mht_73_v, 1315, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::GetStreamImplementation");

  return std::unique_ptr<internal::StreamInterface>(new GpuStream(this));
}

std::unique_ptr<internal::TimerInterface>
GpuExecutor::GetTimerImplementation() {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_74(mht_74_v, 1323, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::GetTimerImplementation");

  return std::unique_ptr<internal::TimerInterface>(new GpuTimer(this));
}

void* GpuExecutor::GpuContextHack() {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_75(mht_75_v, 1330, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::GpuContextHack");
 return context_; }

GpuContext* GpuExecutor::gpu_context() {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_76(mht_76_v, 1335, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::gpu_context");
 return context_; }

// Attempts to read the NUMA node corresponding to the GPU device's PCI bus out
// of SysFS. Returns -1 if it cannot.
//
// For anything more complicated/prod-focused than this, you'll likely want to
// turn to gsys' topology modeling.
static int TryToReadNumaNode(const std::string& pci_bus_id,
                             int device_ordinal) {
   std::vector<std::string> mht_77_v;
   mht_77_v.push_back("pci_bus_id: \"" + pci_bus_id + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_77(mht_77_v, 1347, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "TryToReadNumaNode");

#if defined(__APPLE__)
  LOG(INFO) << "OS X does not support NUMA - returning NUMA node zero";
  return 0;
#elif defined(PLATFORM_WINDOWS)
  // Windows support for NUMA is not currently implemented. Return node 0.
  return 0;
#else
  VLOG(2) << "trying to read NUMA node for device ordinal: " << device_ordinal;
  static const int kUnknownNumaNode = -1;

  if (pci_bus_id.empty()) {
    LOG(INFO) << "no PCI bus ID for device ordinal: " << device_ordinal;
    return kUnknownNumaNode;
  }

  std::string filename =
      absl::StrFormat("/sys/bus/pci/devices/%s/numa_node", pci_bus_id);

  // We have to use fopen/fread here so that the device properties can be
  // populated before InitGoogle procedure has been completed (at which point we
  // could use the file::* utilities).
  FILE* file = fopen(filename.c_str(), "r");
  if (file == nullptr) {
    LOG(INFO) << "could not open file to read NUMA node: " << filename
              << "\nYour kernel may have been built without NUMA support.";
    return kUnknownNumaNode;
  }

  std::string content;
  char buf[32];
  size_t did_read = fread(buf, sizeof(buf[0]), sizeof(buf) - 1, file);
  buf[did_read] = '\0';
  content = buf;

  int32_t value;
  if (port::safe_strto32(content, &value)) {
    if (value < 0) {  // See http://b/18228951 for details on this path.
      LOG(INFO) << "successful NUMA node read from SysFS had negative value ("
                << value
                << "), but there must be at least one NUMA node"
                   ", so returning NUMA node zero";
      fclose(file);
      return 0;
    }
    fclose(file);
    return value;
  }

  LOG(WARNING)
      << "could not convert SysFS file contents to integral NUMA node value: "
      << content;

  fclose(file);
  return kUnknownNumaNode;
#endif
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
GpuExecutor::CreateDeviceDescription(int device_ordinal) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_gpu_executorDTcc mht_78(mht_78_v, 1409, "", "./tensorflow/stream_executor/cuda/cuda_gpu_executor.cc", "GpuExecutor::CreateDeviceDescription");

  GpuDeviceHandle device;
  auto status = GpuDriver::GetDevice(device_ordinal, &device);
  if (!status.ok()) {
    return status;
  }

  int cc_major;
  int cc_minor;
  status = GpuDriver::GetComputeCapability(&cc_major, &cc_minor, device);
  if (!status.ok()) {
    return status;
  }

  internal::DeviceDescriptionBuilder builder;

  {
    int driver_version = 0;
    (void)GpuDriver::GetDriverVersion(&driver_version);
    std::string augmented_driver_version = absl::StrFormat(
        "%d (%s)", driver_version,
        cuda::DriverVersionStatusToString(Diagnostician::FindDsoVersion()));
    builder.set_driver_version(augmented_driver_version);
  }

  {
    std::string pci_bus_id = GpuDriver::GetPCIBusID(device);

    // Lower the hex characters to match sysfs.
    pci_bus_id = absl::AsciiStrToLower(pci_bus_id);
    builder.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = TryToReadNumaNode(pci_bus_id, device_ordinal);
    builder.set_numa_node(numa_node);
  }

  {
    builder.set_threads_per_block_limit(
        GpuDriver::GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                      device)
            .ValueOrDie());

    ThreadDim thread_dim_limit;
    thread_dim_limit.x = GpuDriver::GetDeviceAttribute(
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device)
                             .ValueOrDie();
    thread_dim_limit.y = GpuDriver::GetDeviceAttribute(
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device)
                             .ValueOrDie();
    thread_dim_limit.z = GpuDriver::GetDeviceAttribute(
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device)
                             .ValueOrDie();
    builder.set_thread_dim_limit(thread_dim_limit);

    int clock_rate =
        GpuDriver::GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device)
            .ValueOrDie();
    builder.set_clock_rate_ghz(static_cast<float>(clock_rate) / 1e6);
  }

  {
    bool ecc_enabled = false;
    (void)GpuDriver::IsEccEnabled(device, &ecc_enabled);
    builder.set_ecc_enabled(ecc_enabled);
  }

  {
    uint64_t device_memory_size = -1;
    (void)GpuDriver::GetDeviceTotalMemory(device, &device_memory_size);
    builder.set_device_memory_size(device_memory_size);
  }

  port::StatusOr<int> mem_clock_khz = GpuDriver::GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device_ordinal);
  port::StatusOr<int> mem_bus_width_bits = GpuDriver::GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device_ordinal);
  if (mem_clock_khz.ok() && mem_bus_width_bits.ok()) {
    // Times 2 because HBM is DDR memory; it gets two data bits per each data
    // lane.
    builder.set_memory_bandwidth(2 * int64_t{mem_clock_khz.ValueOrDie()} *
                                 1000 *
                                 int64_t{mem_bus_width_bits.ValueOrDie()} / 8);
  }

  {
    BlockDim block_dim_limit;
    FillBlockDimLimit(device, &block_dim_limit);
    builder.set_block_dim_limit(block_dim_limit);
  }

  {
    std::string device_name;
    TF_RETURN_IF_ERROR(GpuDriver::GetDeviceName(device, &device_name));
    builder.set_name(device_name);
  }

  builder.set_platform_version(
      absl::StrCat("Compute Capability ", cc_major, ".", cc_minor));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  builder.set_device_address_bits(64);

  builder.set_device_vendor("NVIDIA Corporation");
  builder.set_cuda_compute_capability(cc_major, cc_minor);
  builder.set_shared_memory_per_core(
      GpuDriver::GetMaxSharedMemoryPerCore(device).ValueOrDie());
  builder.set_shared_memory_per_block(
      GpuDriver::GetMaxSharedMemoryPerBlock(device).ValueOrDie());
  builder.set_core_count(
      GpuDriver::GetMultiprocessorCount(device).ValueOrDie());
  builder.set_threads_per_core_limit(
      GpuDriver::GetMaxThreadsPerMultiprocessor(device).ValueOrDie());
  builder.set_registers_per_block_limit(
      GpuDriver::GetMaxRegistersPerBlock(device).ValueOrDie());
  builder.set_threads_per_warp(
      GpuDriver::GetThreadsPerWarp(device).ValueOrDie());
  builder.set_registers_per_core_limit(
      GpuDriver::GetDeviceAttribute(
          CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device)
          .ValueOrDie());

  return builder.Build();
}

}  // namespace gpu

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(cuda_gpu_executor, {});
