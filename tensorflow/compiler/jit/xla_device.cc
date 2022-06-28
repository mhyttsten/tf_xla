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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc() {
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

#include "tensorflow/compiler/jit/xla_device.h"

#include <stdlib.h>

#include <unordered_set>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/compiler/jit/xla_device_context.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace tensorflow {

// Default PaddedShapeFn implementation that simply returns the unpadded
// on-device shape. This is accurate for CPU and GPU devices that neither
// transpose nor pad tensors.
Status DefaultPaddedShapeFn(const Tensor& tensor, xla::Shape* shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_0(mht_0_v, 236, "", "./tensorflow/compiler/jit/xla_device.cc", "DefaultPaddedShapeFn");

  const tensorflow::XlaTensor* xla_tensor =
      tensorflow::XlaTensor::FromTensor(&tensor);
  if (xla_tensor == nullptr) {
    return TensorShapeToXLAShape(tensor.dtype(), tensor.shape(), shape);
  }

  const xla::ShapedBuffer& shaped_buffer = xla_tensor->shaped_buffer();
  *shape = shaped_buffer.on_device_shape();
  return Status::OK();
}

// Caches a XlaDeviceAllocator per <backend, device ordinal> pair. A
// XlaDeviceAllocator is created on demand and is associated with a
// XlaDevice. It outlives the device itself (for instance, the buffer
// backing a tensor holds a pointer to the allocator for book-keeping,
// and this buffer can outlast the device).
class XlaDeviceAllocatorState {
 public:
  // Creates or returns a cached XlaDeviceAllocator for a given
  // backend and device_ordinal.
  static XlaDeviceAllocator* GetOrCreateXlaDeviceAllocator(
      const xla::Backend* backend, int device_ordinal);

 private:
  // Returns the singleton instance of XlaDeviceAllocatorState.
  static XlaDeviceAllocatorState& Singleton();
  XlaDeviceAllocatorState();
  ~XlaDeviceAllocatorState();

  mutex allocator_mutex_;  // Guards the singleton allocator state.
  std::unordered_map<std::pair<const xla::Backend*, int>,
                     std::unique_ptr<XlaDeviceAllocator>,
                     hash<std::pair<const xla::Backend*, int>>>
      allocators_ TF_GUARDED_BY(allocator_mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(XlaDeviceAllocatorState);
};

/* static */ XlaDeviceAllocatorState& XlaDeviceAllocatorState::Singleton() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_1(mht_1_v, 278, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDeviceAllocatorState::Singleton");

  static auto a = new XlaDeviceAllocatorState;
  return *a;
}

XlaDeviceAllocatorState::XlaDeviceAllocatorState() = default;
XlaDeviceAllocatorState::~XlaDeviceAllocatorState() = default;

XlaDeviceAllocator* XlaDeviceAllocatorState::GetOrCreateXlaDeviceAllocator(
    const xla::Backend* backend, int device_ordinal) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_2(mht_2_v, 290, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDeviceAllocatorState::GetOrCreateXlaDeviceAllocator");

  XlaDeviceAllocatorState& state = Singleton();
  mutex_lock lock(state.allocator_mutex_);

  auto it = state.allocators_.find({backend, device_ordinal});
  if (it != state.allocators_.end()) {
    return it->second.get();
  }

  std::unique_ptr<XlaDeviceAllocator> alloc =
      absl::make_unique<XlaDeviceAllocator>(
          backend->stream_executors()[device_ordinal]);
  XlaDeviceAllocator* alloc_ptr = alloc.get();
  state.allocators_[{backend, device_ordinal}] = std::move(alloc);
  return alloc_ptr;
}

namespace {

static DeviceAttributes BuildXlaDeviceAttributes(const string& name_prefix,
                                                 const string& device_name,
                                                 int device_ordinal) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name_prefix: \"" + name_prefix + "\"");
   mht_3_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_3(mht_3_v, 316, "", "./tensorflow/compiler/jit/xla_device.cc", "BuildXlaDeviceAttributes");

  return Device::BuildDeviceAttributes(
      absl::StrCat(name_prefix, "/device:", device_name, ":", device_ordinal),
      DeviceType(device_name), Bytes(16ULL << 30), DeviceLocality(),
      absl::StrCat("device: ", device_name, " device"));
}

}  // namespace

XlaDevice::Metadata::Metadata(
    int device_ordinal, se::Platform* platform, const DeviceType& device_type,
    std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
        shape_determination_fns,
    PaddedShapeFn padded_shape_fn, bool use_multiple_streams)
    : device_ordinal_(device_ordinal),
      device_type_(device_type),
      platform_(platform),
      shape_determination_fns_(std::move(shape_determination_fns)),
      padded_shape_fn_(std::move(padded_shape_fn)),
      use_multiple_streams_(use_multiple_streams) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_4(mht_4_v, 338, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::Metadata::Metadata");
}

int XlaDevice::Metadata::device_ordinal() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_5(mht_5_v, 343, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::Metadata::device_ordinal");
 return device_ordinal_; }

se::Platform* XlaDevice::Metadata::platform() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_6(mht_6_v, 348, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::Metadata::platform");
 return platform_; }

xla::LocalClient* XlaDevice::Metadata::client() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_7(mht_7_v, 353, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::Metadata::client");

  auto client = xla::ClientLibrary::GetOrCreateLocalClient(platform_);
  return client.ValueOrDie();
}

const DeviceType& XlaDevice::Metadata::jit_device_type() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_8(mht_8_v, 361, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::Metadata::jit_device_type");

  return device_type_;
}

/*static*/ Status XlaDevice::GetMetadataFromDevice(
    DeviceBase* device, const XlaDevice::Metadata** metadata) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_9(mht_9_v, 369, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::GetMetadataFromDevice");

  *metadata = nullptr;
  XlaDevice* xla_device = dynamic_cast<XlaDevice*>(device->UnderlyingDevice());
  if (xla_device == nullptr) {
    return errors::Internal(
        "Cannot get XLA metadata from non-XLA device \"", device->name(),
        "\". GetMetadata must only be called on an XLA device. Either an "
        "internal bug has been triggered, or an XLA-specific op has been "
        "placed on the wrong device.");
  }
  *metadata = &(xla_device->xla_metadata_);
  return Status::OK();
}

/* static */ Status XlaDevice::GetMetadata(OpKernelContext* ctx,
                                           const Metadata** metadata) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_10(mht_10_v, 387, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::GetMetadata");

  return GetMetadataFromDevice(ctx->device(), metadata);
}

/* static */ Status XlaDevice::GetMetadata(OpKernelConstruction* ctx,
                                           const Metadata** metadata) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_11(mht_11_v, 395, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::GetMetadata");

  return GetMetadataFromDevice(ctx->device(), metadata);
}

/* static */ mutex XlaDevice::global_mu_(LINKER_INITIALIZED);
/* static */ std::vector<std::shared_ptr<se::Stream>>*
    XlaDevice::global_compute_streams_ =
        new std::vector<std::shared_ptr<se::Stream>>;

XlaDevice::XlaDevice(const SessionOptions& session_options,
                     const Options& options)
    : LocalDevice(session_options,
                  BuildXlaDeviceAttributes(options.device_name_prefix,
                                           options.device_name,
                                           options.device_ordinal)),
      xla_metadata_(options.device_ordinal, options.platform,
                    DeviceType(options.compilation_device_name),
                    options.shape_determination_fns,
                    options.padded_shape_fn ? options.padded_shape_fn
                                            : DefaultPaddedShapeFn,
                    options.use_multiple_streams),
      device_ordinal_(options.device_ordinal),
      jit_device_name_(options.compilation_device_name),
      platform_(options.platform),
      intra_op_parallelism_threads_(
          session_options.config.intra_op_parallelism_threads()),
      use_multiple_streams_(options.use_multiple_streams),
      shape_determination_fns_(options.shape_determination_fns),
      allowed_devices_(options.allowed_devices),
      use_global_compute_stream_(options.use_global_compute_stream) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_12(mht_12_v, 427, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::XlaDevice");

  if (options.shape_determination_fns.empty()) {
    LOG(ERROR) << "shape_representation_fns must be non-empty.";
  }
  VLOG(1) << "Created XLA device " << options.compilation_device_name << " "
          << options.device_ordinal << " " << this;
  VLOG(1) << "XlaDevice options: use_multiple_streams: "
          << options.use_multiple_streams << " use_global_compute_stream: "
          << options.use_global_compute_stream;
  thread_pool_.reset(new thread::ThreadPool(session_options.env, "xla_device",
                                            /*num_threads=*/1));

  // We have multiple device to device streams to allow for some concurrency
  // between transfers. The particular value of '4' is chosen fairly
  // arbitrarily. It may be necessary to make this tunable via
  // XlaDevice::Options.
  static constexpr int kNumDeviceToDeviceStreams = 4;
  device_to_device_streams_.resize(kNumDeviceToDeviceStreams);
}

XlaDevice::~XlaDevice() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_13(mht_13_v, 450, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::~XlaDevice");

  VLOG(1) << "Destroying XLA device " << jit_device_name_ << " " << this;
  mutex_lock lock(mu_);
  for (const auto& iter : device_contexts_) {
    iter->Unref();
  }
}

StatusOr<xla::LocalClient*> XlaDevice::GetOrCreateClient() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_14(mht_14_v, 461, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::GetOrCreateClient");

  // We lazily create the client because the platform commits to the
  // details of the host hardware when the client is created, so we
  // don't want to do it until we get a chance to hook the platform up
  // to a simulator.

  xla::LocalClientOptions options;
  options.set_platform(platform_)
      .set_allowed_devices(allowed_devices_)
      .set_intra_op_parallelism_threads(intra_op_parallelism_threads_);
  return xla::ClientLibrary::GetOrCreateLocalClient(options);
}

Allocator* XlaDevice::GetAllocator(AllocatorAttributes attr) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_15(mht_15_v, 477, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::GetAllocator");

  mutex_lock lock(mu_);
  return GetAllocatorLocked(attr);
}

Allocator* XlaDevice::GetAllocatorLocked(AllocatorAttributes attr) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_16(mht_16_v, 485, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::GetAllocatorLocked");

  if (attr.on_host()) {
    return cpu_allocator();
  }

  if (xla_allocator_ == nullptr) {
    // TODO(b/78468222): This can fail, at least when the backend is GPU and
    // there is no GPU on the host.
    xla::Backend* backend = GetOrCreateClient().ValueOrDie()->mutable_backend();
    xla_allocator_ = XlaDeviceAllocatorState::GetOrCreateXlaDeviceAllocator(
        backend, device_ordinal_);
  }
  return xla_allocator_;
}

Status XlaDevice::EnsureDeviceContextOk() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_17(mht_17_v, 503, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::EnsureDeviceContextOk");

  mutex_lock lock(mu_);
  return GetDeviceContextLocked().status();
}

Status XlaDevice::EnsureStreamOkLocked(xla::Backend* backend,
                                       const string& name,
                                       std::shared_ptr<se::Stream>* stream,
                                       bool* stream_was_changed) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_18(mht_18_v, 515, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::EnsureStreamOkLocked");

  if (!(*stream) || !(*stream)->ok()) {
    xla::StreamPool::Ptr ptr;
    TF_ASSIGN_OR_RETURN(ptr, backend->BorrowStream(device_ordinal_));
    *stream = std::shared_ptr<se::Stream>(std::move(ptr));
    VLOG(1) << "XlaDevice " << this << " new " << name << " "
            << (*stream)->DebugStreamPointers();
    *stream_was_changed = true;
  }
  return Status::OK();
}

StatusOr<std::vector<XlaDeviceContext*>> XlaDevice::GetDeviceContextLocked() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_19(mht_19_v, 530, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::GetDeviceContextLocked");

  TF_ASSIGN_OR_RETURN(xla::LocalClient * client, GetOrCreateClient());
  xla::Backend* backend = client->mutable_backend();

  // Ensure all our streams are valid, borrowing new streams if necessary.
  bool need_new_device_context = device_contexts_.empty();
  if (use_global_compute_stream_) {
    mutex_lock lock(global_mu_);
    if (global_compute_streams_->size() <= device_ordinal_) {
      global_compute_streams_->resize(device_ordinal_ + 1, nullptr);
    }

    auto& global_stream = global_compute_streams_->at(device_ordinal_);
    if (global_stream != nullptr && global_stream->ok()) {
      stream_ = global_stream;
    } else {
      // Directly create the stream here instead of borrowing from the stream
      // pool to avoid potential lifetime issues.
      stream_ = absl::make_unique<se::Stream>(
          backend->stream_executors()[device_ordinal_]);
      stream_->Init();
      TF_RETURN_IF_ERROR(EnsureStreamOkLocked(backend, "stream", &stream_,
                                              &need_new_device_context));
      (*global_compute_streams_)[device_ordinal_] = stream_;
    }
  } else {
    TF_RETURN_IF_ERROR(EnsureStreamOkLocked(backend, "stream", &stream_,
                                            &need_new_device_context));
  }

  std::shared_ptr<se::Stream> host_to_device_stream;
  std::shared_ptr<se::Stream> device_to_host_stream;
  std::vector<std::shared_ptr<se::Stream>> device_to_device_streams;
  if (use_multiple_streams_) {
    TF_RETURN_IF_ERROR(EnsureStreamOkLocked(backend, "host_to_device_stream",
                                            &host_to_device_stream_,
                                            &need_new_device_context));
    for (std::shared_ptr<se::Stream>& stream : device_to_device_streams_) {
      TF_RETURN_IF_ERROR(
          EnsureStreamOkLocked(backend, "device_to_device_stream", &stream,
                               &need_new_device_context));
    }
    host_to_device_stream = host_to_device_stream_;
    device_to_device_streams = device_to_device_streams_;
    // The data transfer requests from device to host could arrive out of order,
    // so a single stream would cause deadlock. For this case,
    // xla_device_context would borrow a stream for each transfer request.
    device_to_host_stream = nullptr;
  } else {
    host_to_device_stream = stream_;
    device_to_host_stream = stream_;
    device_to_device_streams = {stream_};
  }

  if (!need_new_device_context) {
    return device_contexts_;
  }

  // At this point we know we need a new device context.
  // Call GetAllocator for the side-effect of ensuring the allocator is created.
  GetAllocatorLocked({});
  for (const auto& iter : device_contexts_) {
    iter->Unref();
  }
  // The XlaDeviceContext keeps a reference count to the streams, and the
  // XlaDeviceContext remains live for the duration of a Executor run. This
  // ensures that the streams remain live for the duration of a run, even if
  // an error is encountered and the streams are replaced with new ones.
  for (const auto& iter : shape_determination_fns_) {
    auto device_context = new XlaDeviceContext(
        stream_, host_to_device_stream, device_to_host_stream,
        device_to_device_streams, client, iter, thread_pool_.get());
    VLOG(1) << "XlaDevice " << this << " new XlaDeviceContext "
            << device_context;
    device_contexts_.emplace_back(device_context);
  }

  // Create and set a new GpuDeviceInfo, if necessary.
  //
  // TODO(b/78232898): This isn't thread-safe; there is a race between the call
  // to set_tensorflow_gpu_device_info() with ops that call the getter
  // tensorflow_gpu_device_info(). This isn't trivially fixed by adding locking
  // to those methods; see the bug for details. Our only saving grace at the
  // moment is that this race doesn't seem to occur in practice.
  if (use_gpu_device_info_) {
    auto gpu_device_info =
        absl::make_unique<DeviceBase::AcceleratorDeviceInfo>();
    gpu_device_info->stream = stream_.get();
    gpu_device_info->default_context = device_contexts_.at(0);
    set_tensorflow_accelerator_device_info(gpu_device_info.get());
    gpu_device_info_ = std::move(gpu_device_info);
    VLOG(1) << "XlaDevice " << this << " new GpuDeviceInfo "
            << gpu_device_info_.get();
  }

  return device_contexts_;
}

StatusOr<XlaDeviceContext*> XlaDevice::GetDeviceContextWithIndex(int index) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_20(mht_20_v, 631, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::GetDeviceContextWithIndex");

  mutex_lock lock(mu_);
  TF_ASSIGN_OR_RETURN(auto device_contexts, GetDeviceContextLocked());
  return device_contexts.at(index);
}

StatusOr<XlaDeviceContext*> XlaDevice::GetDeviceContextDefault() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_21(mht_21_v, 640, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::GetDeviceContextDefault");

  return GetDeviceContextWithIndex(0);
}

Status XlaDevice::UseGpuDeviceInfo() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_22(mht_22_v, 647, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::UseGpuDeviceInfo");

  mutex_lock lock(mu_);
  use_gpu_device_info_ = true;
  return GetDeviceContextLocked().status();
}

Status XlaDevice::TryGetDeviceContext(DeviceContext** out_context) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_23(mht_23_v, 656, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::TryGetDeviceContext");

  TF_ASSIGN_OR_RETURN(auto device_context, GetDeviceContextDefault());
  device_context->Ref();
  *out_context = device_context;
  return Status::OK();
}

// Warn about XLA_CPU/XLA_GPU exactly once.
static void ShowXlaDeviceDeprecationWarning(
    absl::string_view compilation_device_name) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("compilation_device_name: \"" + std::string(compilation_device_name.data(), compilation_device_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_24(mht_24_v, 669, "", "./tensorflow/compiler/jit/xla_device.cc", "ShowXlaDeviceDeprecationWarning");

  static absl::once_flag once;
  if (absl::StrContains(compilation_device_name, "CPU") ||
      absl::StrContains(compilation_device_name, "GPU")) {
    absl::call_once(once, [] {
      LOG(INFO) << "XLA_GPU and XLA_CPU devices are deprecated and will be "
                   "removed in subsequent releases. Instead, use either "
                   "@tf.function(jit_compile=True) for must-compile "
                   "semantics, or run with TF_XLA_FLAGS=--tf_xla_auto_jit=2 "
                   "for auto-clustering best-effort compilation.";
    });
  }
}

void XlaDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_25(mht_25_v, 686, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::Compute");

  VLOG(2) << "XlaDevice::Compute " << op_kernel->name() << ":"
          << op_kernel->type_string();
  ShowXlaDeviceDeprecationWarning(jit_device_name_.type_string());
  op_kernel->Compute(context);
}

void XlaDevice::ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                             AsyncOpKernel::DoneCallback done) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_26(mht_26_v, 697, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::ComputeAsync");

  ShowXlaDeviceDeprecationWarning(jit_device_name_.type_string());
  VLOG(2) << "XlaDevice::ComputeAsync " << op_kernel->name() << ":"
          << op_kernel->type_string();
  op_kernel->ComputeAsync(context, done);
}

Status XlaDevice::Sync() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_27(mht_27_v, 707, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::Sync");

  VLOG(1) << "XlaDevice::Sync";
  profiler::TraceMe activity("XlaDevice::Sync", profiler::TraceMeLevel::kInfo);
  std::shared_ptr<se::Stream> stream;
  {
    mutex_lock lock(mu_);
    stream = stream_;
  }
  if (!stream) return Status::OK();

  Status status = stream->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(status);
  if (!stream->ok()) {
    return errors::Internal("XlaDevice::Sync() failed.");
  }
  VLOG(1) << "XlaDevice::Sync completed";
  return Status::OK();
}

// TODO(b/112409994): This is no longer necessary. Consolidate it with the
// synchronous version.
void XlaDevice::Sync(const DoneCallback& done) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_28(mht_28_v, 731, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::Sync");

  VLOG(1) << "XlaDevice::Sync (asynchronous)";
  std::shared_ptr<se::Stream> stream;
  {
    mutex_lock lock(mu_);
    stream = stream_;
  }
  if (!stream) {
    done(Status::OK());
    return;
  }

  // The call to ThenEnqueueOnBackgroundThread below enqueues a host callback at
  // the end of the stream, after everything that has already been enqueued
  // there at this moment. When the host callback is called, everything before
  // it must have already finished, and the host callback will then place the
  // task below onto a background thread. (See the implementation of
  // ThenEnqueueOnBackgroundThread for details.) Therefore, when the done
  // callback is finally called from that background thread, we know for sure
  // that everything enqueued onto the stream (i.e., the device) at this very
  // moment--when ThenEnqueueOnBackgroundThread is called--will have finished.
  // This achieves a device-wide sync.
  stream->ThenEnqueueOnBackgroundThread([stream, done](se::StreamExecutor*) {
    profiler::TraceMe activity("XlaDevice::Sync::Callback",
                               profiler::TraceMeLevel::kInfo);
    done(stream->ok() ? Status::OK()
                      : errors::Internal("XlaDevice::Sync() failed."));
  });
}

Status XlaDevice::MakeTensorFromProto(XlaDeviceContext* device_context,
                                      const TensorProto& tensor_proto,
                                      const AllocatorAttributes alloc_attrs,
                                      Tensor* tensor) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_29(mht_29_v, 767, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::MakeTensorFromProto");

  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }

  Status status;
  if (alloc_attrs.on_host()) {
    *tensor = parsed;
  } else {
    Allocator* allocator;
    {
      mutex_lock lock(mu_);
      allocator = GetAllocatorLocked(alloc_attrs);
    }
    Tensor copy(allocator, parsed.dtype(), parsed.shape());
    TF_RETURN_IF_ERROR(
        device_context->CopyCPUTensorToDeviceSync(&parsed, this, &copy));
    *tensor = copy;
  }
  VLOG(2) << "Allocated tensor at " << DMAHelper::base(tensor);
  return status;
}

Status XlaDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                      const AllocatorAttributes alloc_attrs,
                                      Tensor* tensor) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_30(mht_30_v, 797, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::MakeTensorFromProto");

  VLOG(1) << "XlaDevice::MakeTensorFromProto";
  XlaDeviceContext* device_context;
  TF_ASSIGN_OR_RETURN(device_context, GetDeviceContextDefault());
  return MakeTensorFromProto(device_context, tensor_proto, alloc_attrs, tensor);
}

void XlaDevice::SetAllowsSyncOnCompletion(bool sync_on_completion) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_31(mht_31_v, 807, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::SetAllowsSyncOnCompletion");

  mutex_lock lock(mu_);
  sync_on_completion_ = sync_on_completion;
}

bool XlaDevice::AllowsSyncOnCompletion() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_32(mht_32_v, 815, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::AllowsSyncOnCompletion");

  mutex_lock lock(mu_);
  return sync_on_completion_;
}

void XlaDevice::SetHandleDeviceErrorCallback(std::function<Status()> callback) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_33(mht_33_v, 823, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::SetHandleDeviceErrorCallback");

  mutex_lock lock(mu_);
  device_error_callback_ = callback;
}

Status XlaDevice::HandleDeviceError() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_34(mht_34_v, 831, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::HandleDeviceError");

  std::function<Status()> local_device_error_callback;
  {
    mutex_lock lock(mu_);
    local_device_error_callback = device_error_callback_;
  }
  if (local_device_error_callback != nullptr) {
    return local_device_error_callback();
  }
  return Status::OK();
}

Status XlaDevice::RefreshStatus() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_35(mht_35_v, 846, "", "./tensorflow/compiler/jit/xla_device.cc", "XlaDevice::RefreshStatus");

  std::shared_ptr<se::Stream> stream;
  {
    mutex_lock lock(mu_);
    stream = stream_;
  }
  if (!stream) {
    return Status::OK();
  }
  Status status = stream->RefreshStatus();
  if (!status.ok()) {
    // Ignore errors from HandleDeviceError, since by definition the status is
    // already non-ok, so there's nothing extra to report if HandleDeviceError
    // itself returns an error.
    HandleDeviceError().IgnoreError();
  }
  return status;
}

XlaDeviceOpRegistrations* RegisterXlaDeviceKernels(const char* device,
                                                   const char* jit_device) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("device: \"" + (device == nullptr ? std::string("nullptr") : std::string((char*)device)) + "\"");
   mht_36_v.push_back("jit_device: \"" + (jit_device == nullptr ? std::string("nullptr") : std::string((char*)jit_device)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTcc mht_36(mht_36_v, 871, "", "./tensorflow/compiler/jit/xla_device.cc", "RegisterXlaDeviceKernels");

  // Any op assigned to the device that isn't rewritten by the graph rewriter
  // gets executed by an XlaCompileOnDemandOp, which compiles it and executes
  // it just-in-time.
  auto factory = [](OpKernelConstruction* context) -> OpKernel* {
    return new XlaCompileOnDemandOp(context);
  };
  XlaOpRegistry::RegisterCompilationKernels();
  XlaDeviceOpRegistrations* registrations = new XlaDeviceOpRegistrations;
  for (const KernelDef* jit_def : XlaOpRegistry::DeviceKernels(
           jit_device,
           /*include_compilation_only_kernels=*/false)) {
    KernelDef* def = new KernelDef(*jit_def);
    const std::unordered_set<std::string>* constant_inputs =
        XlaOpRegistry::CompileTimeConstantInputArgNames(def->op());

    for (const std::string& arg_name : *constant_inputs) {
      def->add_host_memory_arg(arg_name);
    }

    def->set_device_type(device);
    registrations->op_kernel_registrars.emplace_back(
        new kernel_factory::OpKernelRegistrar(def, "XlaCompileOnDemandOp",
                                              factory));
  }
  return registrations;
}

}  // namespace tensorflow
