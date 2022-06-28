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
class MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_compile_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_compile_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_compile_opsDTcc() {
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

// Classes for compiling XLA computations and managing handles that refer to
// them.

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_compilation_cache.h"
#include "tensorflow/compiler/xrt/xrt_device.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"
#include "tensorflow/compiler/xrt/xrt_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/timed.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

Status GenerateXlaDeviceAssignment(
    const xrt::DeviceAssignment& xrt_device_assignment, int num_replicas,
    int num_cores_per_replica, xla::DeviceAssignment* device_assignment) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_compile_opsDTcc mht_0(mht_0_v, 225, "", "./tensorflow/compiler/xrt/kernels/xrt_compile_ops.cc", "GenerateXlaDeviceAssignment");

  if (num_cores_per_replica !=
      xrt_device_assignment.computation_devices_size()) {
    return errors::InvalidArgument(
        "Device assignment does not have the correct number of "
        "computation_devices: num_cores_per_replica=",
        num_cores_per_replica, " computation_devices=",
        xrt_device_assignment.computation_devices_size());
  }
  for (int64_t c = 0; c < xrt_device_assignment.computation_devices_size();
       ++c) {
    const auto& computation_devices =
        xrt_device_assignment.computation_devices(c);
    if (num_replicas != computation_devices.replica_devices_size()) {
      return errors::InvalidArgument(
          "Device assignment does not have the correct number of "
          "replica_device_ids: num_replicas=",
          num_replicas,
          " replica_devices=", computation_devices.replica_devices_size());
    }
    for (int64_t r = 0; r < computation_devices.replica_devices_size(); ++r) {
      const auto& coords = computation_devices.replica_devices(r);
      if (coords.value_size() != 4) {
        return errors::InvalidArgument(
            "Device assignment mesh coordinates must have 4 entries, got ",
            coords.value_size());
      }
      for (int n = 0; n < 3; ++n) {
        if (coords.value(n) != 0) {
          return errors::InvalidArgument("Mesh coordinate at index ", n,
                                         " must be 0, got ", coords.value(n));
        }
      }
      (*device_assignment)(r, c) = coords.value(3);
    }
  }
  return Status::OK();
}

class XRTCompileOp : public OpKernel {
 public:
  explicit XRTCompileOp(OpKernelConstruction* ctx);
  ~XRTCompileOp() override;
  XRTCompileOp(const XRTCompileOp&) = delete;
  XRTCompileOp& operator=(const XRTCompileOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 private:
  Status Compile(OpKernelContext* ctx,
                 const xrt::XLAComputation& computation_proto,
                 std::unique_ptr<xla::LocalExecutable>* program);
};

Status CompilationCacheKey(const xrt::XLAComputation& computation,
                           string* key) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_compile_opsDTcc mht_1(mht_1_v, 283, "", "./tensorflow/compiler/xrt/kernels/xrt_compile_ops.cc", "CompilationCacheKey");

  const size_t size = computation.ByteSizeLong();
  auto serialized = absl::make_unique<char[]>(size);
  TF_RET_CHECK(
      SerializeToBufferDeterministic(computation, serialized.get(), size));
  uint64 fingerprint = Fingerprint64(absl::string_view(serialized.get(), size));
  *key = absl::StrCat(fingerprint);
  return Status::OK();
}

XRTCompileOp::XRTCompileOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_compile_opsDTcc mht_2(mht_2_v, 296, "", "./tensorflow/compiler/xrt/kernels/xrt_compile_ops.cc", "XRTCompileOp::XRTCompileOp");
}

Status XRTCompileOp::Compile(OpKernelContext* ctx,
                             const xrt::XLAComputation& computation_proto,
                             std::unique_ptr<xla::LocalExecutable>* program) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_compile_opsDTcc mht_3(mht_3_v, 303, "", "./tensorflow/compiler/xrt/kernels/xrt_compile_ops.cc", "XRTCompileOp::Compile");

  const xrt::XLAComputationConfig& config = computation_proto.config();
  // Sanity checks for options not yet supported.
  int num_cores_per_replica = std::max<int>(config.num_cores_per_replica(), 1);
  TF_RET_CHECK(num_cores_per_replica == 1);
  TF_RET_CHECK(config.per_core_program_shape_size() == 0);

  // The default config value is 0; treat it as 1 for convenience.
  int num_replicas = config.num_replicas() ? config.num_replicas() : 1;

  // We are guaranteed that the underlying device object won't be deleted out
  // from under us, while the ScopedRef is live.
  class XRTGenericDeviceAccessor::ScopedRef device_ref;
  TF_RETURN_IF_ERROR(XRTGenericDeviceAccessor::InitScopedRef(ctx, &device_ref));

  xla::LocalClient* client = device_ref.client();

  // There is officially no way to use XLA in a client/server architecture where
  // client and server are built from different revisions, because the XLA team
  // does not want to give any guarantees about the stability of the Hlo
  // proto. For cloud TPU this is fine because server and client versions can be
  // assumed to be synced to the same version. For general use the mechanism
  // here (using a snapshot from XlaComputation) works as well as the "official"
  // XLA client/server design, which serializes the same proto between client
  // and server, so in reality is probably fine.
  TF_ASSIGN_OR_RETURN(xla::XlaComputation computation,
                      client->LoadSnapshot(computation_proto.hlo_snapshot()));

  std::vector<xla::Shape> argument_layouts(
      config.program_shape().parameters_size());
  std::vector<const xla::Shape*> argument_layout_ptrs(
      config.program_shape().parameters_size());
  for (int i = 0; i < config.program_shape().parameters_size(); ++i) {
    argument_layouts[i] = xla::Shape(config.program_shape().parameters(i));
    argument_layout_ptrs[i] = &argument_layouts[i];
  }
  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(device_ref.device_ordinal());
  build_options.set_num_replicas(num_replicas);
  build_options.set_result_layout(xla::Shape(config.program_shape().result()));
  build_options.set_device_allocator(device_ref.allocator());
  if (config.has_debug_options()) {
    *build_options.mutable_debug_options() =
        BuildXlaDebugOptions(config.debug_options());
  }
  if (config.has_device_assignment()) {
    xla::DeviceAssignment device_assignment(num_replicas,
                                            num_cores_per_replica);
    TF_RETURN_IF_ERROR(
        GenerateXlaDeviceAssignment(config.device_assignment(), num_replicas,
                                    num_cores_per_replica, &device_assignment));
    build_options.set_device_assignment(device_assignment);
  }

  VLOG(1) << "Building executable";
  TF_ASSIGN_OR_RETURN(
      auto executables,
      client->Compile(computation, argument_layout_ptrs, build_options));
  TF_RET_CHECK(executables.size() == 1);
  *program = std::move(executables[0]);
  return Status::OK();
}

void XRTCompileOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_compile_opsDTcc mht_4(mht_4_v, 369, "", "./tensorflow/compiler/xrt/kernels/xrt_compile_ops.cc", "XRTCompileOp::Compute");

  VLOG(1) << "XRTCompileOp::Compute";
  auto timed = monitoring::MakeTimed(xrt_metrics::GetCompileCell());

  ResourceMgr* rm;
  OP_REQUIRES_OK(ctx, XRTGenericDeviceAccessor::GetResourceManager(ctx, &rm));

  const Tensor& computation_input = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(computation_input.shape()),
              errors::Internal("computation input should be a string scalar"));

  xrt::XLAComputation computation_proto;
  OP_REQUIRES(ctx,
              ParseFromTString(computation_input.scalar<tstring>()(),
                               &computation_proto),
              errors::InvalidArgument(
                  "Unable to parse computation input to XLAComputation"));

  string key;
  OP_REQUIRES_OK(ctx, CompilationCacheKey(computation_proto, &key));

  // Process-wide cache of XLA executables.
  auto cache_or = XRTGenericDeviceAccessor::GetOrCreateCompilationCache(
      ctx, /*max_number_of_entries=*/0);
  OP_REQUIRES_OK(ctx, cache_or.status());
  auto cache = cache_or.ConsumeValueOrDie();

  int64_t uid;
  OP_REQUIRES_OK(
      ctx, cache->CompileIfKeyAbsent(
               key, &uid, [&](std::unique_ptr<xla::LocalExecutable>* program) {
                 VLOG(1) << "Compiling XLA executable";
                 return Compile(ctx, computation_proto, program);
               }));
  std::unique_ptr<XRTCompilationCacheEntryRef> entry;
  OP_REQUIRES_OK(ctx, cache->Lookup(uid, &entry));

  Tensor handle_output(DT_INT64, TensorShape({}));
  handle_output.scalar<int64_t>()() = uid;
  ctx->set_output(0, handle_output);

  xla::LocalExecutable* executable = entry->get().get_executable();
  xla::ProgramShapeProto program_shape = executable->executable()
                                             ->module()
                                             .config()
                                             .entry_computation_layout()
                                             .ComputeProgramShape()
                                             .ToProto();
  Tensor program_shape_output(DT_STRING, TensorShape({1}));
  program_shape_output.vec<tstring>()(0) = program_shape.SerializeAsString();
  ctx->set_output(1, program_shape_output);
}

XRTCompileOp::~XRTCompileOp() = default;

class XRTReleaseCompilationRefOp : public OpKernel {
 public:
  explicit XRTReleaseCompilationRefOp(OpKernelConstruction* ctx);
  ~XRTReleaseCompilationRefOp() override;
  XRTReleaseCompilationRefOp(const XRTReleaseCompilationRefOp&) = delete;
  XRTReleaseCompilationRefOp& operator=(const XRTReleaseCompilationRefOp&) =
      delete;

  void Compute(OpKernelContext* ctx) override;
};

XRTReleaseCompilationRefOp::XRTReleaseCompilationRefOp(
    OpKernelConstruction* ctx)
    : OpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_compile_opsDTcc mht_5(mht_5_v, 440, "", "./tensorflow/compiler/xrt/kernels/xrt_compile_ops.cc", "XRTReleaseCompilationRefOp::XRTReleaseCompilationRefOp");
}

XRTReleaseCompilationRefOp::~XRTReleaseCompilationRefOp() = default;

void XRTReleaseCompilationRefOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPSxrt_compile_opsDTcc mht_6(mht_6_v, 447, "", "./tensorflow/compiler/xrt/kernels/xrt_compile_ops.cc", "XRTReleaseCompilationRefOp::Compute");

  VLOG(1) << "XRTReleaseCompilationRefOp::Compute";
  auto timed = monitoring::MakeTimed(xrt_metrics::GetReleaseCompilationCell());

  // Process-wide cache of XLA executables.
  auto cache_or = XRTGenericDeviceAccessor::GetOrCreateCompilationCache(
      ctx, /*max_number_of_entries=*/0);
  OP_REQUIRES_OK(ctx, cache_or.status());
  auto cache = cache_or.ConsumeValueOrDie();

  const Tensor& keys_tensor = ctx->input(0);
  auto flat_keys = keys_tensor.flat<int64_t>();
  for (int64_t i = 0; i < flat_keys.size(); ++i) {
    int64_t key = flat_keys(i);
    OP_REQUIRES_OK(ctx, cache->Release(key));
    VLOG(2) << "Released computation handle " << key;
  }
}

}  // namespace

REGISTER_KERNEL_BUILDER(Name("XRTCompile")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("computation")
                            .HostMemory("handle"),
                        XRTCompileOp);
REGISTER_KERNEL_BUILDER(Name("XRTCompile")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("computation")
                            .HostMemory("handle"),
                        XRTCompileOp);

REGISTER_KERNEL_BUILDER(Name("XRTReleaseCompilationHandle")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle"),
                        XRTReleaseCompilationRefOp);
REGISTER_KERNEL_BUILDER(Name("XRTReleaseCompilationHandle")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle"),
                        XRTReleaseCompilationRefOp);

}  // namespace tensorflow
