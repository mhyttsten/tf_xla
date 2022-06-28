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
class MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_compile_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_compile_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_compile_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"
#include "tensorflow/compiler/xrt/xrt_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/timed.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_key.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_op_util.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

class XRTCompileOp : public OpKernel {
 public:
  explicit XRTCompileOp(OpKernelConstruction* ctx);
  ~XRTCompileOp() override;
  XRTCompileOp(const XRTCompileOp&) = delete;
  XRTCompileOp& operator=(const XRTCompileOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 private:
  Status Compile(const XLA_TpuMeshState* xla_mesh_state,
                 const xrt::XLAComputation& computation_proto,
                 tensorflow::tpu::TpuProgramGroupInterface* tpu_program_group);
};

XRTCompileOp::XRTCompileOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_compile_opsDTcc mht_0(mht_0_v, 250, "", "./tensorflow/compiler/xrt/kernels/tpu_compile_ops.cc", "XRTCompileOp::XRTCompileOp");
}

Status XRTCompileOp::Compile(
    const XLA_TpuMeshState* xla_mesh_state,
    const xrt::XLAComputation& computation_proto,
    tensorflow::tpu::TpuProgramGroupInterface* tpu_program_group) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_compile_opsDTcc mht_1(mht_1_v, 258, "", "./tensorflow/compiler/xrt/kernels/tpu_compile_ops.cc", "XRTCompileOp::Compile");

  return tensorflow::tpu::TpuProgramGroup::CompileAndBuild(
      computation_proto, xla_mesh_state, tpu_program_group);
}

tpu::TpuCompilationCacheKey CompilationCacheKey(
    const xrt::XLAComputation& computation,
    tensorflow::tpu::TpuMeshStateInterface* mesh_state, int num_replicas,
    int num_cores_per_replica) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_compile_opsDTcc mht_2(mht_2_v, 269, "", "./tensorflow/compiler/xrt/kernels/tpu_compile_ops.cc", "CompilationCacheKey");

  string computation_serialized;
  CHECK(SerializeToStringDeterministic(computation, &computation_serialized));
  tpu::TPUCompileMetadataProto metadata;
  metadata.set_num_replicas(num_replicas);
  metadata.set_num_cores_per_replica(num_cores_per_replica);
  const tpu::TpuCompilationCacheKey key = CreateCompilationCacheKey(
      "compile", 0, tensorflow::Fingerprint64(computation_serialized), {}, {},
      metadata, *mesh_state);
  return key;
}

void ExitCountdown(Env* env, std::shared_ptr<std::atomic<bool>> done) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_compile_opsDTcc mht_3(mht_3_v, 284, "", "./tensorflow/compiler/xrt/kernels/tpu_compile_ops.cc", "ExitCountdown");

  const int kSleepSeconds = 300;
  LOG(INFO) << "TpuCompileOp was cancelled. Sleeping for " << kSleepSeconds
            << " seconds to give time for TPUCompileOp to finished.";
  env->SleepForMicroseconds(kSleepSeconds * 1000000);
  if (done->load()) {
    // If the TpuCompileOp has finished, then terminate peacefully.
    return;
  }

  LOG(ERROR) << "Aborting process due to cancelled TpuCompileOp. This "
             << "termination is to ensure a consistent state.";
  std::exit(42);
}

void XRTCompileOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_compile_opsDTcc mht_4(mht_4_v, 302, "", "./tensorflow/compiler/xrt/kernels/tpu_compile_ops.cc", "XRTCompileOp::Compute");

  VLOG(1) << "XRTCompileOp::Compute";
  auto timed = monitoring::MakeTimed(xrt_metrics::GetCompileCell());

  std::shared_ptr<std::atomic<bool>> done(new std::atomic<bool>(false));
  CancellationToken token =
      ctx->cancellation_manager()->get_cancellation_token();
  const bool already_cancelled =
      !ctx->cancellation_manager()->RegisterCallback(token, [ctx, done]() {
        if (tpu::OpsApiFn()
                ->TpuCompile_ShouldTpuCompileOpIgnoreCancellationFn()) {
          return;
        }

        // Sleep and exit in another thread so the cancellation manager can
        // continue running callbacks.
        Env* env = ctx->env();
        env->SchedClosure([env, done]() { ExitCountdown(env, done); });
      });

  // If the RPC was cancelled before we registered the cancellation callback,
  // don't compile the TPU program.
  OP_REQUIRES(ctx, !already_cancelled,
              errors::Cancelled("RPC cancelled, not compiling TPU program"));

  // We only want to abort the process if a cancellation actually occurs during
  // compilation; we must deregister the callback in the success case. It
  // doesn't hurt to also deregister the callback in the failure case; the
  // CancellationManager ensures that already-registered callbacks will be run
  // once cancellation has started.
  auto cancellation_cleanup = absl::MakeCleanup([ctx, token, done] {
    ctx->cancellation_manager()->DeregisterCallback(token);
    done->store(true);
  });

  VLOG(1) << "Retrieving pod state";
  // Retrieve the topology from the resource manager
  ResourceMgr* rm = GetTPUConfigResourceMgr();
  tensorflow::tpu::TpuMeshStateInterface* mesh_state;
  OP_REQUIRES_OK(ctx,
                 rm->Lookup(rm->default_container(),
                            tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                            &mesh_state));
  core::ScopedUnref mesh_state_unref(mesh_state);

  const Tensor& computation_input = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(computation_input.shape()),
              errors::Internal("computation input should be a string scalar"));

  xrt::XLAComputation computation_proto;
  OP_REQUIRES(
      ctx,
      computation_proto.ParseFromString(computation_input.scalar<tstring>()()),
      errors::InvalidArgument(
          "Unable to parse computation input to XLAComputation"));

  const xrt::XLAComputationConfig& config = computation_proto.config();
  int num_replicas = config.num_replicas() ? config.num_replicas() : 1;
  CHECK_GT(num_replicas, 0);
  int num_cores_per_replica =
      config.num_cores_per_replica() ? config.num_cores_per_replica() : 1;

  const tpu::TpuCompilationCacheKey key = CompilationCacheKey(
      computation_proto, mesh_state, num_replicas, num_cores_per_replica);

  // Process-wide cache of Tpu executables.
  tpu::TpuCompilationCacheInterface* cache;
  OP_REQUIRES_OK(ctx, rm->Lookup<tpu::TpuCompilationCacheInterface>(
                          rm->default_container(),
                          tpu::kCompilationCacheResourceName, &cache));
  core::ScopedUnref cache_unref(cache);

  int64_t uid;
  std::vector<string> proto_key;
  std::vector<string> shard_key;
  std::vector<bool> may_modify_variables;
  absl::Span<const xla::HloProto* const> hlo_metadata;
  OP_REQUIRES_OK(
      ctx, cache->CompileIfKeyAbsent(
               key, /*session_metadata=*/nullptr,
               /*per_step_ref_holder=*/nullptr, &uid, &proto_key, &shard_key,
               &may_modify_variables, &hlo_metadata,
               [&](tpu::TpuProgramGroupInterface* tpu_program_group) {
                 VLOG(1) << "Compiling TPU executable";
                 return Compile(mesh_state->data(), computation_proto,
                                tpu_program_group);
               }));

  Tensor output(DT_INT64, TensorShape({}));
  output.scalar<int64_t>()() = uid;
  ctx->set_output(0, output);

  Tensor program_shape_output(DT_STRING, TensorShape({num_cores_per_replica}));
  for (int64_t i = 0; i < num_cores_per_replica; ++i) {
    xla::ProgramShapeProto program_shape =
        hlo_metadata[i]->hlo_module().host_program_shape();
    program_shape_output.vec<tstring>()(i) = program_shape.SerializeAsString();
  }
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_compile_opsDTcc mht_5(mht_5_v, 421, "", "./tensorflow/compiler/xrt/kernels/tpu_compile_ops.cc", "XRTReleaseCompilationRefOp::XRTReleaseCompilationRefOp");
}

XRTReleaseCompilationRefOp::~XRTReleaseCompilationRefOp() = default;

void XRTReleaseCompilationRefOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_compile_opsDTcc mht_6(mht_6_v, 428, "", "./tensorflow/compiler/xrt/kernels/tpu_compile_ops.cc", "XRTReleaseCompilationRefOp::Compute");

  VLOG(1) << "XRTReleaseCompilationRefOp::Compute";
  auto timed = monitoring::MakeTimed(xrt_metrics::GetReleaseCompilationCell());
  ResourceMgr* rm = GetTPUConfigResourceMgr();
  OP_REQUIRES(ctx, rm != nullptr, errors::Internal("No resource manager."));

  // Process-wide cache of Tpu executables.
  tpu::TpuCompilationCacheInterface* cache;
  OP_REQUIRES_OK(ctx, rm->Lookup<tpu::TpuCompilationCacheInterface>(
                          rm->default_container(),
                          tpu::kCompilationCacheResourceName, &cache));
  core::ScopedUnref cache_unref(cache);

  const Tensor& keys_tensor = ctx->input(0);
  auto flat_keys = keys_tensor.flat<int64_t>();
  for (int64_t i = 0; i < flat_keys.size(); ++i) {
    int64_t key = flat_keys(i);
    OP_REQUIRES_OK(ctx, cache->Release(key));
    VLOG(2) << "Released computation handle " << key;
  }
}

REGISTER_KERNEL_BUILDER(Name("XRTCompile")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("computation")
                            .HostMemory("handle"),
                        XRTCompileOp);

REGISTER_KERNEL_BUILDER(Name("XRTReleaseCompilationHandle")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("handle"),
                        XRTReleaseCompilationRefOp);

}  // namespace tensorflow
