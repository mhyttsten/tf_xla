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
class MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc() {
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

#include <unistd.h>

#include "absl/base/casts.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/gpu/gpu_event.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_timer.h"
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
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/rocm/rocm_diagnostics.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/timer.h"

#ifdef PLATFORMS_GPUS_ROCM_DYNAMIC_LIBROCM_DYNAMIC_LIBROCM_H_
#error \
    "No driver calls in this file, wrap driver functionality in rocm_driver.cc."
#endif

#ifdef __ROCM_RUNTIME_H__
#error \
    "ROCM runtime being included into ROCM GPU executor; should be driver only."
#endif

namespace stream_executor {
namespace gpu {

static GpuEvent* AsGpuEvent(Event* event) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_0(mht_0_v, 231, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "AsGpuEvent");

  DCHECK(event != nullptr);
  return static_cast<GpuEvent*>(event->implementation());
}

// Given a platform-independent timer datatype, returns the internal ROCM
// platform implementation pointer.
static GpuTimer* AsGpuTimer(Timer* timer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_1(mht_1_v, 241, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "AsGpuTimer");

  DCHECK(timer != nullptr);
  return static_cast<GpuTimer*>(timer->implementation());
}

// Given const GPU memory, returns a librocm device pointer datatype, suitable
// for passing directly to librocm APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// librocm APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to librocm functions which will honor constness.
static hipDeviceptr_t AsROCmDevicePtr(const DeviceMemoryBase& gpu_mem) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_2(mht_2_v, 255, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "AsROCmDevicePtr");

  return const_cast<hipDeviceptr_t>(gpu_mem.opaque());
}

// See description on const version above.
static hipDeviceptr_t AsROCmDevicePtr(DeviceMemoryBase* gpu_mem) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_3(mht_3_v, 263, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "AsROCmDevicePtr");

  return AsROCmDevicePtr(*gpu_mem);
}

static GpuContext* GetGpuContext(Stream* stream) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_4(mht_4_v, 270, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GetGpuContext");

  return static_cast<GpuExecutor*>(stream->parent()->implementation())
      ->gpu_context();
}

GpuContext* ExtractGpuContext(GpuExecutor* rocm_exec) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_5(mht_5_v, 278, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "ExtractGpuContext");

  CHECK(rocm_exec != nullptr);
  return rocm_exec->gpu_context();
}

GpuExecutor::~GpuExecutor() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_6(mht_6_v, 286, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::~GpuExecutor");

  for (auto& it : disk_modules_) {
    GpuDriver::UnloadModule(context_, it.second);
  }
  for (auto& it : in_memory_modules_) {
    GpuDriver::UnloadModule(context_, it.second);
  }
  if (context_ != nullptr) {
    GpuDriver::DestroyContext(context_);
  }
  CHECK(kernel_to_gpu_binary_.empty()) << "GpuExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "GpuExecutor has loaded modules.";
}
bool GpuExecutor::UnloadModule(ModuleHandle module_handle) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_7(mht_7_v, 302, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::UnloadModule");

  const char* gpu_binary = reinterpret_cast<const char*>(module_handle.id());
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(gpu_binary);
}

port::StatusOr<std::shared_ptr<DeviceMemoryBase>>
GpuExecutor::CreateOrShareConstant(Stream* stream,
                                   const std::vector<uint8_t>& content) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_8(mht_8_v, 313, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CreateOrShareConstant");

  return port::UnimplementedError("Not implemented for ROCm");
}

bool GpuExecutor::UnloadGpuBinary(const void* gpu_binary) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_9(mht_9_v, 320, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::UnloadGpuBinary");

  auto module_it = gpu_binary_to_module_.find(gpu_binary);
  if (gpu_binary_to_module_.end() == module_it) {
    VLOG(3) << "No loaded  HSACO module for " << gpu_binary;
    return false;
  }
  auto& module = module_it->second.first;
  auto& refcount = module_it->second.second;
  VLOG(3) << "Found HSACO module " << module << " with refcount " << refcount;
  if (--refcount == 0) {
    VLOG(3) << "Unloading  HSACO module " << module;
    GpuDriver::UnloadModule(context_, module);
    gpu_binary_to_module_.erase(module_it);
    const char* mem_it = nullptr;
    for (auto x : in_memory_modules_) {
      if (x.second == module) mem_it = x.first;
    }
    if (mem_it != nullptr) in_memory_modules_.erase(mem_it);
  }
  return true;
}

void GpuExecutor::UnloadKernel(const KernelBase* kernel) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_10(mht_10_v, 345, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::UnloadKernel");

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

port::Status GpuExecutor::Init(int device_ordinal,
                               DeviceOptions device_options) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_11(mht_11_v, 365, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::Init");

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

  return GpuDriver::GetGpuISAVersion(&version_, device_);
}

bool GpuExecutor::FindOnDiskForComputeCapability(
    absl::string_view filename, absl::string_view canonical_suffix,
    string* found_filename) const {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_12_v.push_back("canonical_suffix: \"" + std::string(canonical_suffix.data(), canonical_suffix.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_12(mht_12_v, 394, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::FindOnDiskForComputeCapability");

  LOG(FATAL) << "Feature not supported on ROCM platform "
                "(FindOnDiskForComputeCapability)";
  return false;
}

bool GpuExecutor::FindOnDiskForISAVersion(absl::string_view filename,
                                          absl::string_view canonical_suffix,
                                          string* found_filename) const {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_13_v.push_back("canonical_suffix: \"" + std::string(canonical_suffix.data(), canonical_suffix.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_13(mht_13_v, 407, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::FindOnDiskForISAVersion");

  if (version_ == 0) {
    return false;
  }

  string cc_specific =
      absl::StrCat(filename, ".cc", version_, canonical_suffix);
  if (port::FileExists(cc_specific).ok()) {
    VLOG(2) << "found AMDGPU ISA version-specific file, using that: "
            << cc_specific;
    *found_filename = cc_specific;
    return true;
  }

  VLOG(2) << "could not find AMDGPU ISA version-specific file at: "
          << cc_specific;
  if (port::FileExists(string(filename)).ok()) {
    *found_filename = string(filename);
    return true;
  }

  return false;
}

// Returns the path to the running executable.
// N.B. Derived from //knowledge/smalltalk/background_kb.cc
// Arg: strip_exe: if true, remove the name of the executable itself from the
//                 returned string. Example: calling this from /usr/bin/foo
//                 would return /usr/bin.
static string GetBinaryDir(bool strip_exe) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_14(mht_14_v, 439, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GetBinaryDir");

  char exe_path[PATH_MAX] = {0};
  PCHECK(readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1) != -1);
  // Make sure it's null-terminated:
  exe_path[sizeof(exe_path) - 1] = 0;

  if (strip_exe) {
    // The exe is the last component of the path, so remove one component.
    string ret = exe_path;
    std::vector<string> components = absl::StrSplit(exe_path, '/');
    components.pop_back();
    return absl::StrJoin(components, "/");
  }
  return exe_path;
}

port::Status GpuExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                                    KernelBase* kernel) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_15(mht_15_v, 459, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::GetKernel");

  GpuKernel* rocm_kernel = AsGpuKernel(kernel);
  hipModule_t module = nullptr;
  const string* kernelname;

  const OnDiskKernelLoaderSpec* on_disk_spec = nullptr;
  bool has_cubin = spec.has_cuda_cubin_on_disk();
  if (has_cubin) {
    on_disk_spec = &spec.cuda_cubin_on_disk();
  }

  if (on_disk_spec != nullptr) {
    return port::InternalError(
        "Loading ROCM kernel from disk is not supported");
  } else if (spec.has_cuda_cubin_in_memory()) {
    kernelname = &spec.cuda_cubin_in_memory().kernelname();

    const char* hsaco = spec.cuda_cubin_in_memory().bytes();
    absl::MutexLock lock{&in_memory_modules_mu_};
    module = in_memory_modules_[hsaco];

    if (module == nullptr) {
      TF_RETURN_IF_ERROR(GpuDriver::LoadHsaco(context_, hsaco, &module));
    }
    kernel_to_gpu_binary_[kernel] = hsaco;
  } else {
    return port::InternalError("No method of loading ROCM kernel provided");
  }

  VLOG(2) << "getting function " << *kernelname << " from module " << module;
  if (!GpuDriver::GetModuleFunction(context_, module, kernelname->c_str(),
                                    rocm_kernel->gpu_function_ptr())) {
    return port::InternalError("Failed getting module function");
  }

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the ROCM API.
  rocm_kernel->set_arity(spec.arity());

  KernelMetadata kernel_metadata;
  TF_RETURN_IF_ERROR(GetKernelMetadata(rocm_kernel, &kernel_metadata));
  kernel->set_metadata(kernel_metadata);
  kernel->set_name(*kernelname);
  return port::Status::OK();
}

port::Status GpuExecutor::GetKernelMetadata(GpuKernel* rocm_kernel,
                                            KernelMetadata* kernel_metadata) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_16(mht_16_v, 509, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::GetKernelMetadata");

  int value = 0;
  // TODO(ROCm) implement this feature in HIP
  kernel_metadata->set_registers_per_thread(value);

  // TODO(ROCm) implement this feature in HIP
  kernel_metadata->set_shared_memory_bytes(value);
  return port::Status::OK();
}

port::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const KernelBase& kernel,
                                 const KernelArgsArrayBase& args) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_17(mht_17_v, 525, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::Launch");

  CHECK_EQ(kernel.Arity(), args.number_of_arguments());
  GpuStreamHandle hipstream = AsGpuStreamValue(stream);
  const GpuKernel* rocm_kernel = AsGpuKernel(&kernel);
  hipFunction_t hipfunc = rocm_kernel->AsGpuFunctionHandle();

  // Only perform/print the occupancy check once.  Even just checking to see
  // whether we've done an occupancy check on this kernel before isn't free
  // (because we have to synchronize), so we only do this at -v 2+.
  if (VLOG_IS_ON(2)) {
    absl::MutexLock lock(&launched_kernels_mu_);
    if (!launched_kernels_.count(hipfunc)) {
      VlogOccupancyInfo(kernel, thread_dims, block_dims);
      // TODO(rspringer): Remove elements from launched_kernels_...if we ever
      // expose a kernel/module deallocation method.
      launched_kernels_.insert(hipfunc);
    }
  }

  if (rocm_kernel->GetPreferredCacheConfig() !=
      KernelCacheConfig::kNoPreference) {
    TF_RETURN_IF_ERROR(GpuDriver::FuncSetCacheConfig(
        hipfunc, rocm_kernel->GetGpuCacheConfig()));
  }

  // prepare kernargs
  // KernelArgsArrayBase keeps the pointer of arguments
  // deference them here
  std::vector<void*> kernargs;
  KernelArgIterator iter = args.arg_iterator();
  while (iter.has_next()) {
    KernelArg arg = iter.next();
    VLOG(2) << "*(arg.address): "
            << reinterpret_cast<void*>(
                   *static_cast<const uint64_t*>(arg.address));
    kernargs.push_back(
        reinterpret_cast<void*>(*static_cast<const uint64_t*>(arg.address)));
  }

  size_t size = sizeof(void*) * kernargs.size();
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  return GpuDriver::LaunchKernel(
      GetGpuContext(stream), kernel.name(), hipfunc, block_dims.x, block_dims.y,
      block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
      args.number_of_shared_bytes(), hipstream, nullptr, (void**)&config);
}

int GpuExecutor::CalculateOccupancy(const DeviceDescription& device_description,
                                    uint64_t registers_per_thread,
                                    uint64_t shared_memory_per_block,
                                    const ThreadDim& thread_dims,
                                    GpuFunctionHandle func) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_18(mht_18_v, 581, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CalculateOccupancy");

  LOG(FATAL) << "Feature not supported on ROCM platform (CalculateOccupancy)";
  return 0;
}

int GpuExecutor::CompareOccupancy(int* initial_blocks,
                                  const DeviceDescription& device_description,
                                  uint64_t registers_per_thread,
                                  uint64_t shared_memory_per_block,
                                  const ThreadDim& thread_dims,
                                  GpuFunctionHandle func) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_19(mht_19_v, 594, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CompareOccupancy");

  LOG(FATAL) << "Feature not supported on ROCM platform (CompareOccupancy)";
  return 0;
}

port::Status GpuExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                                     ModuleHandle* module_handle) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_20(mht_20_v, 603, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::LoadModule");

  // In GpuExecutor we store the pointer to the  HSACO binary  as
  // ModuleHandle::id().
  hipModule_t hip_module = nullptr;
  // TODO(ROCm): Need  generic term instead of cubin/cuda/ptx
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(LoadModuleFromHsaco(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()),
        &hip_module));
    *module_handle = ModuleHandle(const_cast<void*>(
        static_cast<const void*>(spec.cuda_cubin_in_memory().data())));
    return port::Status::OK();
  } else {
    return port::InternalError("No HASCO binary found");
  }
}

port::Status GpuExecutor::LoadModuleFromCuBin(const char* cubin,
                                              hipModule_t* module) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("cubin: \"" + (cubin == nullptr ? std::string("nullptr") : std::string((char*)cubin)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_21(mht_21_v, 626, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::LoadModuleFromCuBin");

  LOG(FATAL) << "Feature not supported on ROCM platform (LoadModuleFromCuBin)";
}

port::Status GpuExecutor::LoadModuleFromPtx(const char* ptx,
                                            hipModule_t* module) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("ptx: \"" + (ptx == nullptr ? std::string("nullptr") : std::string((char*)ptx)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_22(mht_22_v, 635, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::LoadModuleFromPtx");

  LOG(FATAL) << "Feature not supported on ROCM platform (LoadModuleFromPtx)";
}

port::Status GpuExecutor::LoadModuleFromHsaco(const char* hsaco,
                                              hipModule_t* module) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("hsaco: \"" + (hsaco == nullptr ? std::string("nullptr") : std::string((char*)hsaco)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_23(mht_23_v, 644, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::LoadModuleFromHsaco");

  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[hsaco];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(GpuDriver::LoadHsaco(context_, hsaco, module));
    module_refcount = 1;
    in_memory_modules_[hsaco] = *module;
    VLOG(3) << "Loaded HSACO " << static_cast<const void*>(hsaco)
            << " as module " << *module;
  } else {
    ++module_refcount;
    VLOG(3) << "HSACO " << static_cast<const void*>(hsaco)
            << " is already loaded as module " << *module;
  }
  gpu_binary_to_module_[hsaco] = {*module, module_refcount};
  return port::Status::OK();
}

// This is a non-essential operation; if there's a failure, proceed without
// logging an error. It's nearly certain that in case of failures, we'd never
// get here in the first place; these are very low-impact routines.
void GpuExecutor::VlogOccupancyInfo(const KernelBase& kernel,
                                    const ThreadDim& thread_dims,
                                    const BlockDim& block_dims) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_24(mht_24_v, 671, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::VlogOccupancyInfo");

  // TODO(ROCm) implement this feature in HIP
}

DeviceMemoryBase GpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_25(mht_25_v, 678, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::Allocate");

  CHECK_EQ(memory_space, 0);
  return DeviceMemoryBase(GpuDriver::DeviceAllocate(context_, size), size);
}

void* GpuExecutor::GetSubBuffer(DeviceMemoryBase* mem, uint64_t offset_bytes,
                                uint64_t size_bytes) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_26(mht_26_v, 687, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::GetSubBuffer");

  // offset and size are in bytes, so char* works as the pointer type.
  return reinterpret_cast<char*>(mem->opaque()) + offset_bytes;
}

void GpuExecutor::Deallocate(DeviceMemoryBase* mem) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_27(mht_27_v, 695, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::Deallocate");

  GpuDriver::DeviceDeallocate(context_, mem->opaque());
}

bool GpuExecutor::HostMemoryRegister(void* location, uint64_t size) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_28(mht_28_v, 702, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::HostMemoryRegister");

  if (location == nullptr || size == 0) {
    LOG(WARNING) << "attempting to register null or zero-sized memory: "
                 << location << "; size " << size;
  }
  VLOG(2) << "registering " << location << " size " << size;
  return GpuDriver::HostRegister(context_, location, size);
}

bool GpuExecutor::HostMemoryUnregister(void* location) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_29(mht_29_v, 714, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::HostMemoryUnregister");

  VLOG(2) << "unregistering " << location;
  return GpuDriver::HostUnregister(context_, location);
}

bool GpuExecutor::SynchronizeAllActivity() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_30(mht_30_v, 722, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::SynchronizeAllActivity");

  return GpuDriver::SynchronizeContext(context_);
}

port::Status GpuExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                             uint64_t size) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_31(mht_31_v, 730, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::SynchronousMemZero");

  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return GpuDriver::SynchronousMemsetUint32(
        context_, AsROCmDevicePtr(location), 0x0, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsROCmDevicePtr(location),
                                           0x0, size);
}

port::Status GpuExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                            int value, uint64_t size) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_32(mht_32_v, 744, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::SynchronousMemSet");

  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    // hipMemset reinterprets "value" as a uint8.
    uint8 byte_value = static_cast<uint8>(value);
    uint32 pattern = (byte_value << 24) | (byte_value << 16) |
                     (byte_value << 8) | byte_value;
    return GpuDriver::SynchronousMemsetUint32(
        context_, AsROCmDevicePtr(location), pattern, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsROCmDevicePtr(location),
                                           value, size);
}

port::Status GpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                            const void* host_src,
                                            uint64_t size) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_33(mht_33_v, 763, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::SynchronousMemcpy");

  return GpuDriver::SynchronousMemcpyH2D(context_, AsROCmDevicePtr(gpu_dst),
                                         host_src, size);
}

port::Status GpuExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceMemoryBase& gpu_src,
                                            uint64_t size) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_34(mht_34_v, 773, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::SynchronousMemcpy");

  return GpuDriver::SynchronousMemcpyD2H(context_, host_dst,
                                         AsROCmDevicePtr(gpu_src), size);
}

port::Status GpuExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_35(mht_35_v, 782, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::SynchronousMemcpyDeviceToDevice");

  return GpuDriver::SynchronousMemcpyD2D(context_, AsROCmDevicePtr(gpu_dst),
                                         AsROCmDevicePtr(gpu_src), size);
}

port::Status GpuExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                  uint64_t size) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_36(mht_36_v, 791, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::MemZero");

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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_37(mht_37_v, 804, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::Memset");

  VLOG(2) << "enqueueing memset8 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  return GpuDriver::AsynchronousMemsetUint8(context_, AsROCmDevicePtr(location),
                                            pattern, size,
                                            AsGpuStreamValue(stream));
}

port::Status GpuExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                   uint32 pattern, uint64_t size) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_38(mht_38_v, 817, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::Memset32");

  VLOG(2) << "enqueueing memset32 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
        size % 4 == 0);
  return GpuDriver::AsynchronousMemsetUint32(
      context_, AsROCmDevicePtr(location), pattern, size / 4,
      AsGpuStreamValue(stream));
}

bool GpuExecutor::Memcpy(Stream* stream, void* host_dst,
                         const DeviceMemoryBase& gpu_src, uint64_t size) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_39(mht_39_v, 832, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::Memcpy");

  return GpuDriver::AsynchronousMemcpyD2H(context_, host_dst,
                                          AsROCmDevicePtr(gpu_src), size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                         const void* host_src, uint64_t size) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_40(mht_40_v, 842, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::Memcpy");

  return GpuDriver::AsynchronousMemcpyH2D(context_, AsROCmDevicePtr(gpu_dst),
                                          host_src, size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::MemcpyDeviceToDevice(Stream* stream,
                                       DeviceMemoryBase* gpu_dst,
                                       const DeviceMemoryBase& gpu_src,
                                       uint64_t size) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_41(mht_41_v, 854, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::MemcpyDeviceToDevice");

  return GpuDriver::AsynchronousMemcpyD2D(context_, AsROCmDevicePtr(gpu_dst),
                                          AsROCmDevicePtr(gpu_src), size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::HostCallback(Stream* stream,
                               std::function<port::Status()> callback) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_42(mht_42_v, 864, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::HostCallback");

  auto callback_ptr = new std::function<void()>([callback]() {
    port::Status s = callback();
    if (!s.ok()) {
      LOG(WARNING) << "Host callback failed: " << s;
    }
  });
  return GpuDriver::AddStreamCallback(context_, AsGpuStreamValue(stream),
                                      InternalHostCallback, callback_ptr);
}

/* static */ void GpuExecutor::InternalHostCallback(GpuStreamHandle stream,
                                                    hipError_t status,
                                                    void* data) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_43(mht_43_v, 880, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::InternalHostCallback");

  std::function<void()>* callback =
      reinterpret_cast<std::function<void()>*>(data);
  (*callback)();
  delete callback;
}

port::Status GpuExecutor::AllocateEvent(Event* event) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_44(mht_44_v, 890, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::AllocateEvent");

  return AsGpuEvent(event)->Init();
}

port::Status GpuExecutor::DeallocateEvent(Event* event) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_45(mht_45_v, 897, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::DeallocateEvent");

  return AsGpuEvent(event)->Destroy();
}

port::Status GpuExecutor::RecordEvent(Stream* stream, Event* event) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_46(mht_46_v, 904, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::RecordEvent");

  return AsGpuEvent(event)->Record(AsGpuStream(stream));
}

port::Status GpuExecutor::WaitForEvent(Stream* stream, Event* event) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_47(mht_47_v, 911, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::WaitForEvent");

  if (GpuDriver::WaitStreamOnEvent(context_, AsGpuStream(stream)->gpu_stream(),
                                   AsGpuEvent(event)->gpu_event())) {
    return port::Status::OK();
  } else {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat("error recording waiting for ROCM event on stream %p",
                        stream)};
  }
}

Event::Status GpuExecutor::PollForEventStatus(Event* event) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_48(mht_48_v, 926, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::PollForEventStatus");

  return AsGpuEvent(event)->PollForStatus();
}

bool GpuExecutor::AllocateStream(Stream* stream) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_49(mht_49_v, 933, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::AllocateStream");

  return AsGpuStream(stream)->Init();
}

void GpuExecutor::DeallocateStream(Stream* stream) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_50(mht_50_v, 940, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::DeallocateStream");

  GpuStream* rocm_stream = AsGpuStream(stream);
  if (!rocm_stream->IsIdle()) {
    LOG(ERROR) << "Deallocating stream with pending work";
  }
  rocm_stream->Destroy();
}

bool GpuExecutor::AllocateTimer(Timer* timer) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_51(mht_51_v, 951, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::AllocateTimer");

  return AsGpuTimer(timer)->Init();
}

void GpuExecutor::DeallocateTimer(Timer* timer) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_52(mht_52_v, 958, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::DeallocateTimer");

  AsGpuTimer(timer)->Destroy();
}

bool GpuExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_53(mht_53_v, 965, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CreateStreamDependency");

  GpuEventHandle other_completed_event = *AsGpuStream(other)->completed_event();
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_54(mht_54_v, 983, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::StartTimer");

  return AsGpuTimer(timer)->Start(AsGpuStream(stream));
}

bool GpuExecutor::StopTimer(Stream* stream, Timer* timer) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_55(mht_55_v, 990, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::StopTimer");

  return AsGpuTimer(timer)->Stop(AsGpuStream(stream));
}

port::Status GpuExecutor::BlockHostUntilDone(Stream* stream) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_56(mht_56_v, 997, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::BlockHostUntilDone");

  return GpuDriver::SynchronizeStream(context_, AsGpuStreamValue(stream));
}

blas::BlasSupport* GpuExecutor::CreateBlas() {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_57(mht_57_v, 1004, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CreateBlas");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(rocm::kROCmPlatformId,
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_58(mht_58_v, 1021, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CreateDnn");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(rocm::kROCmPlatformId,
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_59(mht_59_v, 1038, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CreateFft");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(rocm::kROCmPlatformId,
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_60(mht_60_v, 1055, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CreateRng");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(rocm::kROCmPlatformId,
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_61(mht_61_v, 1073, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::SupportsDnn");
 return true; }

bool GpuExecutor::CanEnablePeerAccessTo(StreamExecutorInterface* other) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_62(mht_62_v, 1078, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CanEnablePeerAccessTo");

  GpuExecutor* rocm_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::CanEnablePeerAccess(context_, rocm_other->context_);
}

port::Status GpuExecutor::EnablePeerAccessTo(StreamExecutorInterface* other) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_63(mht_63_v, 1086, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::EnablePeerAccessTo");

  GpuExecutor* rocm_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::EnablePeerAccess(context_, rocm_other->context_);
}

bool GpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_64(mht_64_v, 1094, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::DeviceMemoryUsage");

  return GpuDriver::GetDeviceMemoryInfo(context_, free, total);
}

bool GpuExecutor::GetSymbol(const string& symbol_name,
                            ModuleHandle module_handle, void** mem,
                            size_t* bytes) {
   std::vector<std::string> mht_65_v;
   mht_65_v.push_back("symbol_name: \"" + symbol_name + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_65(mht_65_v, 1104, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::GetSymbol");

  absl::MutexLock lock{&in_memory_modules_mu_};
  if (static_cast<bool>(module_handle)) {
    auto it = gpu_binary_to_module_.find(module_handle.id());
    CHECK(it != gpu_binary_to_module_.end());
    if (GpuDriver::GetModuleSymbol(
            context_, it->second.first, symbol_name.c_str(),
            reinterpret_cast<hipDeviceptr_t*>(mem), bytes)) {
      return true;
    }
  }

  for (auto& it : gpu_binary_to_module_) {
    if (GpuDriver::GetModuleSymbol(
            context_, it.second.first, symbol_name.c_str(),
            reinterpret_cast<hipDeviceptr_t*>(mem), bytes)) {
      return true;
    }
  }

  LOG(INFO) << "Falied to find symbol in any modules: " << symbol_name;
  return false;
}

bool FillBlockDimLimit(GpuDeviceHandle device, BlockDim* block_dim_limit) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_66(mht_66_v, 1131, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "FillBlockDimLimit");

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
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_67(mht_67_v, 1150, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::SupportsBlas");
 return true; }

bool GpuExecutor::SupportsFft() const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_68(mht_68_v, 1155, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::SupportsFft");
 return true; }

bool GpuExecutor::SupportsRng() const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_69(mht_69_v, 1160, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::SupportsRng");
 return true; }

std::unique_ptr<internal::EventInterface>
GpuExecutor::CreateEventImplementation() {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_70(mht_70_v, 1166, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CreateEventImplementation");

  return std::unique_ptr<internal::EventInterface>(new GpuEvent(this));
}

std::unique_ptr<internal::KernelInterface>
GpuExecutor::CreateKernelImplementation() {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_71(mht_71_v, 1174, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CreateKernelImplementation");

  return std::unique_ptr<internal::KernelInterface>(new GpuKernel());
}

std::unique_ptr<internal::StreamInterface>
GpuExecutor::GetStreamImplementation() {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_72(mht_72_v, 1182, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::GetStreamImplementation");

  return std::unique_ptr<internal::StreamInterface>(new GpuStream(this));
}

std::unique_ptr<internal::TimerInterface>
GpuExecutor::GetTimerImplementation() {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_73(mht_73_v, 1190, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::GetTimerImplementation");

  return std::unique_ptr<internal::TimerInterface>(new GpuTimer(this));
}

void* GpuExecutor::GpuContextHack() {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_74(mht_74_v, 1197, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::GpuContextHack");
 return context_; }

GpuContext* GpuExecutor::gpu_context() {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_75(mht_75_v, 1202, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::gpu_context");
 return context_; }

// Attempts to read the NUMA node corresponding to the GPU device's PCI bus out
// of SysFS. Returns -1 if it cannot.
//
// For anything more complicated/prod-focused than this, you'll likely want to
// turn to gsys' topology modeling.
static int TryToReadNumaNode(const string& pci_bus_id, int device_ordinal) {
   std::vector<std::string> mht_76_v;
   mht_76_v.push_back("pci_bus_id: \"" + pci_bus_id + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_76(mht_76_v, 1213, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "TryToReadNumaNode");

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
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
GpuExecutor::CreateDeviceDescription(int device_ordinal) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_gpu_executorDTcc mht_77(mht_77_v, 1267, "", "./tensorflow/stream_executor/rocm/rocm_gpu_executor.cc", "GpuExecutor::CreateDeviceDescription");

  GpuDeviceHandle device;
  auto status = GpuDriver::GetDevice(device_ordinal, &device);
  if (!status.ok()) {
    return status;
  }

  int version;
  status = GpuDriver::GetGpuISAVersion(&version, device);
  if (!status.ok()) {
    return status;
  }

  std::string gcn_arch_name;
  status = GpuDriver::GetGpuGCNArchName(device, &gcn_arch_name);
  if (!status.ok()) {
    return status;
  }

  internal::DeviceDescriptionBuilder builder;

  {
    int driver_version = 0;
    (void)GpuDriver::GetDriverVersion(&driver_version);
    string augmented_driver_version = absl::StrFormat(
        "%d (%s)", driver_version,
        rocm::DriverVersionStatusToString(Diagnostician::FindDsoVersion())
            .c_str());
    builder.set_driver_version(augmented_driver_version);
  }

  {
    string pci_bus_id = GpuDriver::GetPCIBusID(device);

    // Lower the hex characters to match sysfs.
    pci_bus_id = absl::AsciiStrToLower(pci_bus_id);
    builder.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = TryToReadNumaNode(pci_bus_id, device_ordinal);
    builder.set_numa_node(numa_node);
  }

  hipDeviceProp_t prop;
  if (GpuDriver::GetDeviceProperties(&prop, device_ordinal)) {
    builder.set_threads_per_block_limit(prop.maxThreadsPerBlock);

    ThreadDim thread_dim_limit;
    thread_dim_limit.x = prop.maxThreadsDim[0];
    thread_dim_limit.y = prop.maxThreadsDim[1];
    thread_dim_limit.z = prop.maxThreadsDim[2];
    builder.set_thread_dim_limit(thread_dim_limit);

    float clock_rate_ghz = static_cast<float>(prop.clockRate) / 1e6;
    builder.set_clock_rate_ghz(clock_rate_ghz);

    // mem_bandwidth = 2 * mem_bus_width_in_bytes * mem_clock_rate_in_hz
    int64_t memory_bandwidth = 2 * (int64_t(prop.memoryBusWidth) / 8) *
                               (int64_t(prop.memoryClockRate) * 1000);
    builder.set_memory_bandwidth(memory_bandwidth);
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

  {
    BlockDim block_dim_limit;
    FillBlockDimLimit(device, &block_dim_limit);
    builder.set_block_dim_limit(block_dim_limit);
  }

  {
    string device_name;
    TF_RETURN_IF_ERROR(GpuDriver::GetDeviceName(device, &device_name));
    builder.set_name(device_name);
  }

  builder.set_platform_version(
      absl::StrCat("AMDGPU ISA version: ", gcn_arch_name));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  builder.set_device_address_bits(64);

  builder.set_device_vendor("Advanced Micro Devices, Inc");
  builder.set_rocm_compute_capability(gcn_arch_name);

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
  builder.set_registers_per_core_limit(64 * 1024);

  return builder.Build();
}

}  // namespace gpu

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(rocm_gpu_executor, {});
