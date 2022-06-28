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
class MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc() {
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

// Implements the StreamExecutor interface by passing through to its
// implementation_ value (in pointer-to-implementation style), which
// implements StreamExecutorInterface.

#include "tensorflow/stream_executor/stream_executor_pimpl.h"

#include <atomic>
#include <memory>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/fft.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/stacktrace.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace {
bool FLAGS_check_device_leaks = false;
}  // namespace

namespace stream_executor {
namespace {

std::string StackTraceIfVLOG10() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_0(mht_0_v, 220, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StackTraceIfVLOG10");

  if (VLOG_IS_ON(10)) {
    return absl::StrCat(" ", port::CurrentStackTrace(), "\n");
  } else {
    return "";
  }
}

// Make sure the executor is done with its work; we know (because this isn't
// publicly visible) that all enqueued work is quick.
void BlockOnThreadExecutor(port::ThreadPool* executor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_1(mht_1_v, 233, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "BlockOnThreadExecutor");

  absl::Notification n;
  executor->Schedule([&n]() { n.Notify(); });
  n.WaitForNotification();
}

std::atomic_int_fast64_t correlation_id_generator(0);

}  // namespace

template <typename BeginCallT, typename CompleteCallT, typename ReturnT,
          typename... BeginArgsT>
class ScopedTracer {
 public:
  ScopedTracer(StreamExecutor* stream_exec, BeginCallT begin_call,
               CompleteCallT complete_call, const ReturnT* result,
               BeginArgsT... begin_args)
      : stream_exec_(stream_exec),
        complete_call_(complete_call),
        result_(result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_2(mht_2_v, 255, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "ScopedTracer");

    if (stream_exec_->tracing_enabled_) {
      correlation_id_ =
          correlation_id_generator.fetch_add(1, std::memory_order_relaxed) - 1;
      Trace(begin_call, begin_args...);
    }
  }

  ~ScopedTracer() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_3(mht_3_v, 266, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "~ScopedTracer");

    if (stream_exec_->tracing_enabled_) {
      Trace(complete_call_, result_);
    }
  }

 private:
  template <typename CallbackT, typename... TraceArgsT>
  void Trace(CallbackT callback, TraceArgsT... args) {
    {
      // Instance tracers held in a block to limit the lock lifetime.
      absl::ReaderMutexLock lock{&stream_exec_->mu_};
      for (TraceListener* listener : stream_exec_->listeners_) {
        (listener->*callback)(correlation_id_,
                              std::forward<TraceArgsT>(args)...);
      }
    }
  }

  StreamExecutor* stream_exec_;
  CompleteCallT complete_call_;
  const ReturnT* result_;
  int64_t correlation_id_;
};

template <typename BeginCallT, typename CompleteCallT, typename ReturnT,
          typename... BeginArgsT>
ScopedTracer<BeginCallT, CompleteCallT, ReturnT, BeginArgsT...>
MakeScopedTracer(StreamExecutor* stream_exec, BeginCallT begin_call,
                 CompleteCallT complete_call, ReturnT* result,
                 BeginArgsT... begin_args) {
  return ScopedTracer<BeginCallT, CompleteCallT, ReturnT, BeginArgsT...>(
      stream_exec, begin_call, complete_call, result,
      std::forward<BeginArgsT>(begin_args)...);
}

#define SCOPED_TRACE(LOC, ...) \
  auto tracer =                \
      MakeScopedTracer(this, &LOC##Begin, &LOC##Complete, ##__VA_ARGS__);

/* static */ absl::Mutex StreamExecutor::static_mu_{absl::kConstInit};

// Get per-device memory limit in bytes. Returns 0 if
// TF_PER_DEVICE_MEMORY_LIMIT_MB environment variable is not set.
static int64_t GetMemoryLimitBytes() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_4(mht_4_v, 313, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "GetMemoryLimitBytes");

  int64_t value;
  SE_CHECK_OK(tensorflow::ReadInt64FromEnvVar("TF_PER_DEVICE_MEMORY_LIMIT_MB",
                                              0, &value));
  return value * (1ll << 20);
}

StreamExecutor::StreamExecutor(
    const Platform* platform,
    std::unique_ptr<internal::StreamExecutorInterface> implementation,
    int device_ordinal)
    : platform_(platform),
      implementation_(std::move(implementation)),
      device_ordinal_(device_ordinal),
      background_threads_(new port::ThreadPool(
          port::Env::Default(), "stream_executor", kNumBackgroundThreads)),
      live_stream_count_(0),
      tracing_enabled_(false),
      mem_alloc_bytes_(0),
      memory_limit_bytes_(GetMemoryLimitBytes()),
      allocator_(this) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_5(mht_5_v, 336, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::StreamExecutor");

  std::string name = absl::AsciiStrToLower(platform_->Name());
  if (name == "cuda") {
    platform_kind_ = PlatformKind::kCuda;
  } else if (name == "rocm") {
    platform_kind_ = PlatformKind::kROCm;
  } else if (name == "opencl") {
    platform_kind_ = PlatformKind::kOpenCL;
  } else if (name == "host") {
    platform_kind_ = PlatformKind::kHost;
  } else {
    platform_kind_ = PlatformKind::kInvalid;
  }
}

StreamExecutor::~StreamExecutor() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_6(mht_6_v, 354, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::~StreamExecutor");

  BlockOnThreadExecutor(background_threads_.get());

  if (live_stream_count_.load() != 0) {
    LOG(WARNING) << "Not all streams were deallocated at executor destruction "
                 << "time. This may lead to unexpected/bad behavior - "
                 << "especially if any stream is still active!";
  }

  if (FLAGS_check_device_leaks) {
    for (const auto& it : mem_allocs_) {
      LOG(INFO) << "Memory alloced at executor exit: addr: "
                << absl::StrFormat("%p", it.first)
                << ", bytes: " << it.second.bytes << ", trace: \n"
                << it.second.stack_trace;
    }
  }
}

port::Status StreamExecutor::Init(DeviceOptions device_options) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_7(mht_7_v, 376, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::Init");

  return implementation_->Init(device_ordinal_, std::move(device_options));
}

port::Status StreamExecutor::Init() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_8(mht_8_v, 383, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::Init");
 return Init(DeviceOptions::Default()); }

port::Status StreamExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                                       KernelBase* kernel) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_9(mht_9_v, 389, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetKernel");

  return implementation_->GetKernel(spec, kernel);
}

void StreamExecutor::UnloadKernel(const KernelBase* kernel) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_10(mht_10_v, 396, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::UnloadKernel");

  implementation_->UnloadKernel(kernel);
}

port::Status StreamExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                                        ModuleHandle* module_handle) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_11(mht_11_v, 404, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::LoadModule");

  return implementation_->LoadModule(spec, module_handle);
}

bool StreamExecutor::UnloadModule(ModuleHandle module_handle) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_12(mht_12_v, 411, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::UnloadModule");

  return implementation_->UnloadModule(module_handle);
}

port::StatusOr<std::shared_ptr<DeviceMemoryBase>>
StreamExecutor::CreateOrShareConstant(Stream* stream,
                                      const std::vector<uint8_t>& content) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_13(mht_13_v, 420, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::CreateOrShareConstant");

  return implementation_->CreateOrShareConstant(stream, std::move(content));
}

void StreamExecutor::Deallocate(DeviceMemoryBase* mem) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_14(mht_14_v, 427, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::Deallocate");

  VLOG(1) << "Called StreamExecutor::Deallocate(mem=" << mem->opaque()
          << ") mem->size()=" << mem->size() << StackTraceIfVLOG10();

  if (mem->opaque() != nullptr) {
    EraseAllocRecord(mem->opaque());
  }
  implementation_->Deallocate(mem);
  mem->Reset(nullptr, 0);
}

void StreamExecutor::GetMemAllocs(std::map<void*, AllocRecord>* records_out) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_15(mht_15_v, 441, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetMemAllocs");

  absl::ReaderMutexLock lock(&mu_);
  *records_out = mem_allocs_;
}

bool StreamExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_16(mht_16_v, 449, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::CanEnablePeerAccessTo");

  return implementation_->CanEnablePeerAccessTo(other->implementation_.get());
}

port::Status StreamExecutor::EnablePeerAccessTo(StreamExecutor* other) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_17(mht_17_v, 456, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::EnablePeerAccessTo");

  return implementation_->EnablePeerAccessTo(other->implementation_.get());
}

const DeviceDescription& StreamExecutor::GetDeviceDescription() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_18(mht_18_v, 463, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetDeviceDescription");

  absl::MutexLock lock(&mu_);
  if (device_description_ != nullptr) {
    return *device_description_;
  }

  device_description_ = CreateDeviceDescription();
  return *device_description_;
}

int64_t StreamExecutor::GetDeviceLoad() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_19(mht_19_v, 476, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetDeviceLoad");

  return implementation_->GetDeviceLoad();
}

int StreamExecutor::PlatformDeviceCount() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_20(mht_20_v, 483, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::PlatformDeviceCount");

  return implementation_->PlatformDeviceCount();
}

bool StreamExecutor::SupportsBlas() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_21(mht_21_v, 490, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SupportsBlas");

  return implementation_->SupportsBlas();
}

bool StreamExecutor::SupportsRng() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_22(mht_22_v, 497, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SupportsRng");

  return implementation_->SupportsRng();
}

bool StreamExecutor::SupportsDnn() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_23(mht_23_v, 504, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SupportsDnn");

  return implementation_->SupportsDnn();
}

bool StreamExecutor::GetConvolveAlgorithms(
    dnn::ConvolutionKind kind,
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_24(mht_24_v, 513, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetConvolveAlgorithms");

  dnn::DnnSupport* dnn_support = AsDnn();
  if (!dnn_support) {
    return false;
  }
  switch (kind) {
    default:
      return false;
    case dnn::ConvolutionKind::FORWARD:
    case dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION:
      return dnn_support->GetConvolveAlgorithms(
          GetDeviceDescription().cuda_compute_capability(), out_algorithms);
    case dnn::ConvolutionKind::BACKWARD_DATA:
      return dnn_support->GetConvolveBackwardDataAlgorithms(
          GetDeviceDescription().cuda_compute_capability(), out_algorithms);
    case dnn::ConvolutionKind::BACKWARD_FILTER:
      return dnn_support->GetConvolveBackwardFilterAlgorithms(
          GetDeviceDescription().cuda_compute_capability(), out_algorithms);
  }
}

port::Status StreamExecutor::GetConvolveRunners(
    bool use_cudnn_frontend, dnn::ConvolutionKind kind,
    dnn::DataType input_type, dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    ScratchAllocator* scratch_allocator,
    std::vector<std::unique_ptr<const dnn::ConvRunner>>* out_exec_plans) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_25(mht_25_v, 546, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetConvolveRunners");

  dnn::DnnSupport* dnn_support = AsDnn();
  if (!dnn_support) {
    return port::UnimplementedError("DNN library is not found.");
  }
  return dnn_support->GetConvolveRunners(
      use_cudnn_frontend, kind, input_type, output_type, stream,
      input_descriptor, input_data, filter_descriptor, filter_data,
      output_descriptor, output_data, convolution_descriptor, use_fallback,
      scratch_allocator, out_exec_plans);
}

port::Status StreamExecutor::GetFusedConvolveRunners(
    bool use_cudnn_frontend, dnn::ConvolutionKind kind,
    dnn::DataType input_type, dnn::DataType bias_type,
    dnn::DataType output_type, double conv_input_scale, double side_input_scale,
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    dnn::ActivationMode activation_mode,
    std::vector<std::unique_ptr<const dnn::FusedConvRunner>>* out_exec_plans) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_26(mht_26_v, 571, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetFusedConvolveRunners");

  dnn::DnnSupport* dnn_support = AsDnn();
  if (!dnn_support) {
    return port::UnimplementedError("DNN library is not found.");
  }
  return dnn_support->GetFusedConvolveRunners(
      use_cudnn_frontend, kind, input_type, bias_type, output_type,
      conv_input_scale, side_input_scale, stream, input_descriptor,
      filter_descriptor, bias_descriptor, output_descriptor,
      convolution_descriptor, use_fallback, activation_mode, out_exec_plans);
}

bool StreamExecutor::GetMIOpenConvolveAlgorithms(
    dnn::ConvolutionKind kind, dnn::DataType element_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    ScratchAllocator* scratch_allocator,
    std::vector<dnn::ProfileResult>* out_algorithms) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_27(mht_27_v, 594, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetMIOpenConvolveAlgorithms");

  dnn::DnnSupport* dnn_support = AsDnn();
  if (!dnn_support) {
    return false;
  }
  return dnn_support->GetMIOpenConvolveAlgorithms(
      kind, element_type, stream, input_descriptor, input_data,
      filter_descriptor, filter_data, output_descriptor, output_data,
      convolution_descriptor, scratch_allocator, out_algorithms);
}

bool StreamExecutor::GetRnnAlgorithms(
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_28(mht_28_v, 609, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetRnnAlgorithms");

  dnn::DnnSupport* dnn_support = AsDnn();
  if (!dnn_support) {
    return false;
  }
  return dnn_support->GetRnnAlgorithms(out_algorithms);
}

bool StreamExecutor::GetBlasGemmAlgorithms(
    std::vector<blas::AlgorithmType>* out_algorithms) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_29(mht_29_v, 621, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetBlasGemmAlgorithms");

  blas::BlasSupport* blas_support = AsBlas();
  if (!blas_support) {
    return false;
  }
  return blas_support->GetBlasGemmAlgorithms(out_algorithms);
}

port::StatusOr<std::unique_ptr<blas::IBlasLtMatmulPlan>>
StreamExecutor::CreateBlasLtMatmulPlan(
    const blas::BlasLtMatmulPlanParams& params) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_30(mht_30_v, 634, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::CreateBlasLtMatmulPlan");

  blas::BlasSupport* blas_support = AsBlas();
  if (!blas_support) {
    return port::Status(port::error::UNKNOWN,
                        "Fail to find the blas implementation.");
  }
  return blas_support->CreateBlasLtMatmulPlan(params);
}

port::StatusOr<std::vector<std::unique_ptr<blas::IBlasLtMatmulAlgorithm>>>
StreamExecutor::GetBlasLtMatmulAlgorithms(const blas::IBlasLtMatmulPlan* plan,
                                          size_t max_workspace_size,
                                          int max_algorithm_count) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_31(mht_31_v, 649, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetBlasLtMatmulAlgorithms");

  blas::BlasSupport* blas_support = AsBlas();
  if (!blas_support) {
    return port::Status(port::error::UNKNOWN,
                        "Fail to find the blas implementation.");
  }
  return blas_support->GetBlasLtMatmulAlgorithms(plan, max_workspace_size,
                                                 max_algorithm_count);
}

port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
StreamExecutor::createRnnDescriptor(
    int num_layers, int hidden_size, int input_size, int cell_size,
    int batch_size, dnn::RnnInputMode input_mode,
    dnn::RnnDirectionMode direction_mode, dnn::RnnMode rnn_mode,
    dnn::DataType data_type, const dnn::AlgorithmConfig& algorithm_config,
    float dropout, uint64_t seed, ScratchAllocator* state_allocator,
    bool use_padded_io) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_32(mht_32_v, 669, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::createRnnDescriptor");

  dnn::DnnSupport* dnn_support = AsDnn();
  if (!dnn_support) {
    return port::Status(port::error::UNKNOWN,
                        "Fail to find the dnn implementation.");
  }
  return dnn_support->createRnnDescriptor(
      num_layers, hidden_size, input_size, cell_size, batch_size, input_mode,
      direction_mode, rnn_mode, data_type, algorithm_config, dropout, seed,
      state_allocator, use_padded_io);
}

port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
StreamExecutor::createRnnSequenceTensorDescriptor(int max_seq_length,
                                                  int batch_size, int data_size,
                                                  dnn::DataType data_type) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_33(mht_33_v, 687, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::createRnnSequenceTensorDescriptor");

  dnn::DnnSupport* dnn_support = AsDnn();
  if (!dnn_support) {
    return port::Status(port::error::UNKNOWN,
                        "Fail to find the dnn implementation.");
  }
  return dnn_support->createRnnSequenceTensorDescriptor(
      max_seq_length, batch_size, data_size, data_type);
}

port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
StreamExecutor::createRnnSequenceTensorDescriptor(
    int max_seq_length, int batch_size, int data_size,
    const absl::Span<const int>& seq_lengths, bool time_major,
    dnn::DataType data_type) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_34(mht_34_v, 704, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::createRnnSequenceTensorDescriptor");

  dnn::DnnSupport* dnn_support = AsDnn();
  if (!dnn_support) {
    return port::Status(port::error::UNKNOWN,
                        "Fail to find the dnn implementation.");
  }
  return dnn_support->createRnnSequenceTensorDescriptor(
      max_seq_length, batch_size, data_size, seq_lengths, time_major,
      data_type);
}

port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
StreamExecutor::createRnnStateTensorDescriptor(int num_layer, int batch_size,
                                               int data_size,
                                               dnn::DataType data_type) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_35(mht_35_v, 721, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::createRnnStateTensorDescriptor");

  dnn::DnnSupport* dnn_support = AsDnn();
  if (!dnn_support) {
    return port::Status(port::error::UNKNOWN,
                        "Fail to find the dnn implementation.");
  }
  return dnn_support->createRnnStateTensorDescriptor(num_layer, batch_size,
                                                     data_size, data_type);
}

dnn::DnnSupport* StreamExecutor::AsDnn() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_36(mht_36_v, 734, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::AsDnn");

  absl::MutexLock lock(&mu_);
  if (dnn_ != nullptr) {
    return dnn_.get();
  }

  dnn_.reset(implementation_->CreateDnn());
  return dnn_.get();
}

blas::BlasSupport* StreamExecutor::AsBlas() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_37(mht_37_v, 747, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::AsBlas");

  absl::MutexLock lock(&mu_);
  if (blas_ != nullptr) {
    return blas_.get();
  }

  blas_.reset(implementation_->CreateBlas());
  return blas_.get();
}

fft::FftSupport* StreamExecutor::AsFft() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_38(mht_38_v, 760, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::AsFft");

  absl::MutexLock lock(&mu_);
  if (fft_ != nullptr) {
    return fft_.get();
  }

  fft_.reset(implementation_->CreateFft());
  return fft_.get();
}

rng::RngSupport* StreamExecutor::AsRng() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_39(mht_39_v, 773, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::AsRng");

  absl::MutexLock lock(&mu_);
  if (rng_ != nullptr) {
    return rng_.get();
  }

  rng_.reset(implementation_->CreateRng());
  return rng_.get();
}

port::Status StreamExecutor::Launch(Stream* stream,
                                    const ThreadDim& thread_dims,
                                    const BlockDim& block_dims,
                                    const KernelBase& kernel,
                                    const KernelArgsArrayBase& args) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_40(mht_40_v, 790, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::Launch");

  SubmitTrace(&TraceListener::LaunchSubmit, stream, thread_dims, block_dims,
              kernel, args);

  return implementation_->Launch(stream, thread_dims, block_dims, kernel, args);
}

port::Status StreamExecutor::BlockHostUntilDone(Stream* stream) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_41(mht_41_v, 800, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::BlockHostUntilDone");

  port::Status result;
  SCOPED_TRACE(TraceListener::BlockHostUntilDone, &result, stream);

  result = implementation_->BlockHostUntilDone(stream);
  return result;
}

port::Status StreamExecutor::GetStatus(Stream* stream) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_42(mht_42_v, 811, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetStatus");

  return implementation_->GetStatus(stream);
}

DeviceMemoryBase StreamExecutor::Allocate(uint64_t size, int64_t memory_space) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_43(mht_43_v, 818, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::Allocate");

  if (memory_limit_bytes_ > 0 &&
      static_cast<int64_t>(mem_alloc_bytes_ + size) > memory_limit_bytes_) {
    LOG(WARNING) << "Not enough memory to allocate " << size << " on device "
                 << device_ordinal_
                 << " within provided limit. [used=" << mem_alloc_bytes_
                 << ", limit=" << memory_limit_bytes_ << "]";
    return DeviceMemoryBase();
  }
  DeviceMemoryBase buf = implementation_->Allocate(size, memory_space);
  VLOG(1) << "Called StreamExecutor::Allocate(size=" << size
          << ", memory_space=" << memory_space << ") returns " << buf.opaque()
          << StackTraceIfVLOG10();
  CreateAllocRecord(buf.opaque(), size);

  return buf;
}

port::StatusOr<DeviceMemoryBase> StreamExecutor::GetUntypedSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("symbol_name: \"" + symbol_name + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_44(mht_44_v, 841, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetUntypedSymbol");

  // If failed to get the symbol, opaque/bytes are unchanged. Initialize them to
  // be nullptr/0 for consistency with DeviceMemory semantics.
  void* opaque = nullptr;
  size_t bytes = 0;
  if (GetSymbol(symbol_name, module_handle, &opaque, &bytes)) {
    return DeviceMemoryBase(opaque, bytes);
  }

  return port::Status(
      port::error::NOT_FOUND,
      absl::StrCat("Check if module containing symbol ", symbol_name,
                   " is loaded (module_handle = ",
                   reinterpret_cast<uintptr_t>(module_handle.id()), ")"));
}

bool StreamExecutor::GetSymbol(const std::string& symbol_name,
                               ModuleHandle module_handle, void** mem,
                               size_t* bytes) {
   std::vector<std::string> mht_45_v;
   mht_45_v.push_back("symbol_name: \"" + symbol_name + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_45(mht_45_v, 863, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetSymbol");

  return implementation_->GetSymbol(symbol_name, module_handle, mem, bytes);
}

void* StreamExecutor::UnifiedMemoryAllocate(uint64_t bytes) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_46(mht_46_v, 870, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::UnifiedMemoryAllocate");

  void* buffer = implementation_->UnifiedMemoryAllocate(bytes);
  VLOG(1) << "Called StreamExecutor::UnifiedMemoryAllocate(size=" << bytes
          << ") returns " << buffer << StackTraceIfVLOG10();
  return buffer;
}

void StreamExecutor::UnifiedMemoryDeallocate(void* location) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_47(mht_47_v, 880, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::UnifiedMemoryDeallocate");

  VLOG(1) << "Called StreamExecutor::UnifiedMemoryDeallocate(location="
          << location << ")" << StackTraceIfVLOG10();

  return implementation_->UnifiedMemoryDeallocate(location);
}

void* StreamExecutor::HostMemoryAllocate(uint64_t size) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_48(mht_48_v, 890, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::HostMemoryAllocate");

  void* buffer = implementation_->HostMemoryAllocate(size);
  VLOG(1) << "Called StreamExecutor::HostMemoryAllocate(size=" << size
          << ") returns " << buffer << StackTraceIfVLOG10();
  return buffer;
}

void StreamExecutor::HostMemoryDeallocate(void* location) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_49(mht_49_v, 900, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::HostMemoryDeallocate");

  VLOG(1) << "Called StreamExecutor::HostMemoryDeallocate(location=" << location
          << ")" << StackTraceIfVLOG10();

  return implementation_->HostMemoryDeallocate(location);
}

bool StreamExecutor::HostMemoryRegister(void* location, uint64_t size) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_50(mht_50_v, 910, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::HostMemoryRegister");

  VLOG(1) << "Called StreamExecutor::HostMemoryRegister(location=" << location
          << ", size=" << size << ")" << StackTraceIfVLOG10();
  if (location == nullptr || size == 0) {
    LOG(WARNING) << "attempting to register null or zero-sized memory: "
                 << location << "; size " << size;
  }
  return implementation_->HostMemoryRegister(location, size);
}

bool StreamExecutor::HostMemoryUnregister(void* location) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_51(mht_51_v, 923, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::HostMemoryUnregister");

  VLOG(1) << "Called StreamExecutor::HostMemoryUnregister(location=" << location
          << ")" << StackTraceIfVLOG10();
  return implementation_->HostMemoryUnregister(location);
}

bool StreamExecutor::SynchronizeAllActivity() {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_52(mht_52_v, 932, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SynchronizeAllActivity");

  VLOG(1) << "Called StreamExecutor::SynchronizeAllActivity()"
          << StackTraceIfVLOG10();
  bool ok = implementation_->SynchronizeAllActivity();

  // This should all be quick and infallible work, so we can perform the
  // synchronization even in the case of failure.
  BlockOnThreadExecutor(background_threads_.get());

  return ok;
}

port::Status StreamExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                                uint64_t size) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_53(mht_53_v, 948, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SynchronousMemZero");

  VLOG(1) << "Called StreamExecutor::SynchronousMemZero(location=" << location
          << ", size=" << size << ")" << StackTraceIfVLOG10();

  return implementation_->SynchronousMemZero(location, size);
}

port::Status StreamExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                               int value, uint64_t size) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_54(mht_54_v, 959, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SynchronousMemSet");

  VLOG(1) << "Called StreamExecutor::SynchronousMemSet(location=" << location
          << ", value=" << value << ", size=" << size << ")"
          << StackTraceIfVLOG10();

  return implementation_->SynchronousMemSet(location, value, size);
}

bool StreamExecutor::SynchronousMemcpy(DeviceMemoryBase* device_dst,
                                       const void* host_src, uint64_t size) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_55(mht_55_v, 971, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SynchronousMemcpy");

  VLOG(1) << "Called StreamExecutor::SynchronousMemcpy(device_dst="
          << device_dst->opaque() << ", host_src=" << host_src
          << ", size=" << size << ") H2D" << StackTraceIfVLOG10();

  // Tracing overloaded methods is very difficult due to issues with type
  // inference on template args. Since use of these overloaded methods is
  // discouraged anyway, this isn't a huge deal.
  port::Status status =
      implementation_->SynchronousMemcpy(device_dst, host_src, size);
  if (!status.ok()) {
    LOG(ERROR) << "synchronous memcpy: " << status;
  }
  return status.ok();
}

bool StreamExecutor::SynchronousMemcpy(void* host_dst,
                                       const DeviceMemoryBase& device_src,
                                       uint64_t size) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_56(mht_56_v, 992, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SynchronousMemcpy");

  VLOG(1) << "Called StreamExecutor::SynchronousMemcpy(host_dst=" << host_dst
          << ", device_src=" << device_src.opaque() << ", size=" << size
          << ") D2H" << StackTraceIfVLOG10();

  port::Status status =
      implementation_->SynchronousMemcpy(host_dst, device_src, size);
  if (!status.ok()) {
    LOG(ERROR) << "synchronous memcpy: " << status;
  }
  return status.ok();
}

bool StreamExecutor::SynchronousMemcpy(DeviceMemoryBase* device_dst,
                                       const DeviceMemoryBase& device_src,
                                       uint64_t size) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_57(mht_57_v, 1010, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SynchronousMemcpy");

  VLOG(1) << "Called StreamExecutor::SynchronousMemcpy(device_dst="
          << device_dst->opaque() << ", device_src=" << device_src.opaque()
          << ", size=" << size << ") D2D" << StackTraceIfVLOG10();

  port::Status status = implementation_->SynchronousMemcpyDeviceToDevice(
      device_dst, device_src, size);
  if (!status.ok()) {
    LOG(ERROR) << "synchronous memcpy: " << status;
  }
  return status.ok();
}

port::Status StreamExecutor::SynchronousMemcpyD2H(
    const DeviceMemoryBase& device_src, int64_t size, void* host_dst) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_58(mht_58_v, 1027, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SynchronousMemcpyD2H");

  VLOG(1) << "Called StreamExecutor::SynchronousMemcpyD2H(device_src="
          << device_src.opaque() << ", size=" << size
          << ", host_dst=" << host_dst << ")" << StackTraceIfVLOG10();

  port::Status result;
  SCOPED_TRACE(TraceListener::SynchronousMemcpyD2H, &result, device_src, size,
               host_dst);

  result = implementation_->SynchronousMemcpy(host_dst, device_src, size);
  if (!result.ok()) {
    result = port::Status(
        port::error::INTERNAL,
        absl::StrFormat("failed to synchronously memcpy device-to-host: device "
                        "%p to host %p size %d: %s",
                        device_src.opaque(), host_dst, size,
                        result.ToString()));
  }

  return result;
}

port::Status StreamExecutor::SynchronousMemcpyH2D(
    const void* host_src, int64_t size, DeviceMemoryBase* device_dst) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_59(mht_59_v, 1053, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::SynchronousMemcpyH2D");

  VLOG(1) << "Called StreamExecutor::SynchronousMemcpyH2D(host_src=" << host_src
          << ", size=" << size << ", device_dst=" << device_dst->opaque() << ")"
          << StackTraceIfVLOG10();

  port::Status result;
  SCOPED_TRACE(TraceListener::SynchronousMemcpyH2D, &result, host_src, size,
               device_dst);

  result = implementation_->SynchronousMemcpy(device_dst, host_src, size);
  if (!result.ok()) {
    result = port::Status(
        port::error::INTERNAL,
        absl::StrFormat("failed to synchronously memcpy host-to-device: host "
                        "%p to device %p size %d: %s",
                        host_src, device_dst->opaque(), size,
                        result.ToString()));
  }

  return result;
}

bool StreamExecutor::Memcpy(Stream* stream, void* host_dst,
                            const DeviceMemoryBase& device_src, uint64_t size) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_60(mht_60_v, 1079, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::Memcpy");

  return implementation_->Memcpy(stream, host_dst, device_src, size);
}

bool StreamExecutor::Memcpy(Stream* stream, DeviceMemoryBase* device_dst,
                            const void* host_src, uint64_t size) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_61(mht_61_v, 1087, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::Memcpy");

  return implementation_->Memcpy(stream, device_dst, host_src, size);
}

bool StreamExecutor::MemcpyDeviceToDevice(Stream* stream,
                                          DeviceMemoryBase* device_dst,
                                          const DeviceMemoryBase& device_src,
                                          uint64_t size) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_62(mht_62_v, 1097, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::MemcpyDeviceToDevice");

  return implementation_->MemcpyDeviceToDevice(stream, device_dst, device_src,
                                               size);
}

port::Status StreamExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                     uint64_t size) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_63(mht_63_v, 1106, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::MemZero");

  return implementation_->MemZero(stream, location, size);
}

port::Status StreamExecutor::Memset32(Stream* stream,
                                      DeviceMemoryBase* location,
                                      uint32 pattern, uint64_t size) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_64(mht_64_v, 1115, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::Memset32");

  CHECK_EQ(0, size % 4)
      << "need 32-bit multiple size to fill with 32-bit pattern";
  return implementation_->Memset32(stream, location, pattern, size);
}

bool StreamExecutor::HostCallback(Stream* stream,
                                  std::function<void()> callback) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_65(mht_65_v, 1125, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::HostCallback");

  return implementation_->HostCallback(stream, std::move(callback));
}

bool StreamExecutor::HostCallback(Stream* stream,
                                  std::function<port::Status()> callback) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_66(mht_66_v, 1133, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::HostCallback");

  return implementation_->HostCallback(stream, std::move(callback));
}

port::Status StreamExecutor::AllocateEvent(Event* event) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_67(mht_67_v, 1140, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::AllocateEvent");

  return implementation_->AllocateEvent(event);
}

port::Status StreamExecutor::DeallocateEvent(Event* event) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_68(mht_68_v, 1147, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::DeallocateEvent");

  return implementation_->DeallocateEvent(event);
}

port::Status StreamExecutor::RecordEvent(Stream* stream, Event* event) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_69(mht_69_v, 1154, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::RecordEvent");

  return implementation_->RecordEvent(stream, event);
}

port::Status StreamExecutor::WaitForEvent(Stream* stream, Event* event) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_70(mht_70_v, 1161, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::WaitForEvent");

  return implementation_->WaitForEvent(stream, event);
}

Event::Status StreamExecutor::PollForEventStatus(Event* event) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_71(mht_71_v, 1168, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::PollForEventStatus");

  return implementation_->PollForEventStatus(event);
}

bool StreamExecutor::AllocateStream(Stream* stream) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_72(mht_72_v, 1175, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::AllocateStream");

  live_stream_count_.fetch_add(1, std::memory_order_relaxed);
  if (!implementation_->AllocateStream(stream)) {
    auto count = live_stream_count_.fetch_sub(1);
    CHECK_GE(count, 0) << "live stream count should not dip below zero";
    LOG(INFO) << "failed to allocate stream; live stream count: " << count;
    return false;
  }

  return true;
}

void StreamExecutor::DeallocateStream(Stream* stream) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_73(mht_73_v, 1190, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::DeallocateStream");

  dnn::DnnSupport* dnn;
  {
    absl::MutexLock lock(&mu_);
    dnn = dnn_.get();
  }
  if (dnn) {
    dnn->NotifyStreamDestroyed(stream);
  }
  implementation_->DeallocateStream(stream);
  CHECK_GE(live_stream_count_.fetch_sub(1), 0)
      << "live stream count should not dip below zero";
}

bool StreamExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_74(mht_74_v, 1207, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::CreateStreamDependency");

  return implementation_->CreateStreamDependency(dependent, other);
}

bool StreamExecutor::AllocateTimer(Timer* timer) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_75(mht_75_v, 1214, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::AllocateTimer");

  return implementation_->AllocateTimer(timer);
}

void StreamExecutor::DeallocateTimer(Timer* timer) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_76(mht_76_v, 1221, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::DeallocateTimer");

  return implementation_->DeallocateTimer(timer);
}

bool StreamExecutor::StartTimer(Stream* stream, Timer* timer) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_77(mht_77_v, 1228, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::StartTimer");

  return implementation_->StartTimer(stream, timer);
}

bool StreamExecutor::StopTimer(Stream* stream, Timer* timer) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_78(mht_78_v, 1235, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::StopTimer");

  return implementation_->StopTimer(stream, timer);
}

std::unique_ptr<DeviceDescription> StreamExecutor::CreateDeviceDescription()
    const {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_79(mht_79_v, 1243, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::CreateDeviceDescription");

  auto desc_status = implementation_->CreateDeviceDescription();
  return desc_status.ConsumeValueOrDie();
}

bool StreamExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_80(mht_80_v, 1251, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::DeviceMemoryUsage");

  return implementation_->DeviceMemoryUsage(free, total);
}

void StreamExecutor::EnqueueOnBackgroundThread(std::function<void()> task) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_81(mht_81_v, 1258, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::EnqueueOnBackgroundThread");

  background_threads_->Schedule(std::move(task));
}

void StreamExecutor::CreateAllocRecord(void* opaque, uint64_t bytes) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_82(mht_82_v, 1265, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::CreateAllocRecord");

  if (FLAGS_check_device_leaks && opaque != nullptr && bytes != 0) {
    absl::MutexLock lock(&mu_);
    mem_allocs_[opaque] = AllocRecord{bytes, ""};
    mem_alloc_bytes_ += bytes;
  }
}

void StreamExecutor::EraseAllocRecord(void* opaque) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_83(mht_83_v, 1276, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::EraseAllocRecord");

  if (FLAGS_check_device_leaks && opaque != nullptr) {
    absl::MutexLock lock(&mu_);
    if (mem_allocs_.find(opaque) == mem_allocs_.end()) {
      LOG(ERROR) << "Deallocating unknown pointer: " << opaque;
    } else {
      mem_alloc_bytes_ -= mem_allocs_[opaque].bytes;
      mem_allocs_.erase(opaque);
    }
  }
}

void StreamExecutor::EnableTracing(bool enabled) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_84(mht_84_v, 1291, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::EnableTracing");
 tracing_enabled_ = enabled; }

void StreamExecutor::RegisterTraceListener(TraceListener* listener) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_85(mht_85_v, 1296, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::RegisterTraceListener");

  {
    absl::MutexLock lock(&mu_);
    if (listeners_.find(listener) != listeners_.end()) {
      LOG(INFO) << "Attempt to register already-registered listener, "
                << listener;
    } else {
      listeners_.insert(listener);
    }
  }

  implementation_->RegisterTraceListener(listener);
}

bool StreamExecutor::UnregisterTraceListener(TraceListener* listener) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_86(mht_86_v, 1313, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::UnregisterTraceListener");

  {
    absl::MutexLock lock(&mu_);
    if (listeners_.find(listener) == listeners_.end()) {
      LOG(INFO) << "Attempt to unregister unknown listener, " << listener;
      return false;
    }
    listeners_.erase(listener);
  }

  implementation_->UnregisterTraceListener(listener);
  return true;
}

absl::optional<AllocatorStats> StreamExecutor::GetAllocatorStats() {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_87(mht_87_v, 1330, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::GetAllocatorStats");

  return implementation_->GetAllocatorStats();
}

bool StreamExecutor::ClearAllocatorStats() {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_88(mht_88_v, 1337, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::ClearAllocatorStats");

  return implementation_->ClearAllocatorStats();
}

template <typename TraceCallT, typename... ArgsT>
void StreamExecutor::SubmitTrace(TraceCallT trace_call, ArgsT&&... args) {
  if (tracing_enabled_) {
    {
      // instance tracers held in a block to limit the lock lifetime.
      absl::ReaderMutexLock lock(&mu_);
      for (TraceListener* listener : listeners_) {
        (listener->*trace_call)(std::forward<ArgsT>(args)...);
      }
    }
  }
}

internal::StreamExecutorInterface* StreamExecutor::implementation() {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_89(mht_89_v, 1357, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutor::implementation");

  return implementation_->GetUnderlyingExecutor();
}

StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator(
    StreamExecutor* executor)
    : DeviceMemoryAllocator(executor->platform()) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_90(mht_90_v, 1366, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator");

  stream_executors_ = {executor};
}

StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator(
    const Platform* platform,
    absl::Span<StreamExecutor* const> stream_executors)
    : DeviceMemoryAllocator(platform),
      stream_executors_(stream_executors.begin(), stream_executors.end()) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_91(mht_91_v, 1377, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator");
}

port::StatusOr<OwningDeviceMemory> StreamExecutorMemoryAllocator::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_92(mht_92_v, 1384, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutorMemoryAllocator::Allocate");

  TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                      GetStreamExecutor(device_ordinal));
  DeviceMemoryBase result = executor->AllocateArray<uint8>(size, memory_space);
  if (size > 0 && result == nullptr) {
    return tensorflow::errors::ResourceExhausted(absl::StrFormat(
        "Failed to allocate request for %s (%uB) on device ordinal %d",
        tensorflow::strings::HumanReadableNumBytes(size), size,
        device_ordinal));
  }
  VLOG(3) << absl::StreamFormat(
      "Allocated %s (%uB) on device ordinal %d: %p",
      tensorflow::strings::HumanReadableNumBytes(size), size, device_ordinal,
      result.opaque());
  return OwningDeviceMemory(result, device_ordinal, this);
}

port::Status StreamExecutorMemoryAllocator::Deallocate(int device_ordinal,
                                                       DeviceMemoryBase mem) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_93(mht_93_v, 1405, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutorMemoryAllocator::Deallocate");

  if (!mem.is_null()) {
    TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                        GetStreamExecutor(device_ordinal));
    VLOG(3) << absl::StreamFormat("Freeing %p on device ordinal %d",
                                  mem.opaque(), device_ordinal);
    executor->Deallocate(&mem);
  }
  return port::Status::OK();
}

port::StatusOr<StreamExecutor*>
StreamExecutorMemoryAllocator::GetStreamExecutor(int device_ordinal) const {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_94(mht_94_v, 1420, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutorMemoryAllocator::GetStreamExecutor");

  if (device_ordinal < 0) {
    return tensorflow::errors::InvalidArgument(absl::StrFormat(
        "device ordinal value (%d) must be non-negative", device_ordinal));
  }
  for (StreamExecutor* se : stream_executors_) {
    if (se->device_ordinal() == device_ordinal) {
      return se;
    }
  }
  return tensorflow::errors::NotFound(
      absl::StrFormat("Device %s:%d present but not supported",
                      platform()->Name(), device_ordinal));
}

bool StreamExecutorMemoryAllocator::AllowsAsynchronousDeallocation() const {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_95(mht_95_v, 1438, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutorMemoryAllocator::AllowsAsynchronousDeallocation");

  return false;
}

port::StatusOr<Stream*> StreamExecutorMemoryAllocator::GetStream(
    int device_ordinal) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_pimplDTcc mht_96(mht_96_v, 1446, "", "./tensorflow/stream_executor/stream_executor_pimpl.cc", "StreamExecutorMemoryAllocator::GetStream");

  CHECK(!AllowsAsynchronousDeallocation())
      << "The logic below only works for synchronous allocators";
  TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                      GetStreamExecutor(device_ordinal));
  Stream* out = [&] {
    absl::MutexLock lock(&mutex_);
    if (!streams_.count(device_ordinal)) {
      auto p = streams_.emplace(std::piecewise_construct,
                                std::forward_as_tuple(device_ordinal),
                                std::forward_as_tuple(executor));
      p.first->second.Init();
      return &p.first->second;
    }
    return &streams_.at(device_ordinal);
  }();
  return out;
}

}  // namespace stream_executor
