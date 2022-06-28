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
class MHTracer_DTPStensorflowPScorePSncclPScollective_communicatorDTcc {
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
   MHTracer_DTPStensorflowPScorePSncclPScollective_communicatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSncclPScollective_communicatorDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/nccl/collective_communicator.h"

#include "tensorflow/core/framework/cancellation.h"

#if TENSORFLOW_USE_NCCL && (GOOGLE_CUDA || TENSORFLOW_USE_ROCM)

#include "absl/memory/memory.h"
#include "tensorflow/core/nccl/nccl_manager.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

class NcclCommunicator : public NcclCommunicatorInterface {
 public:
  string GenerateCommunicatorKey() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSncclPScollective_communicatorDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/nccl/collective_communicator.cc", "GenerateCommunicatorKey");

    return nccl_manager_.GenerateCommunicatorKey();
  }

  void Enqueue(std::shared_ptr<CollectiveContext> col_ctx,
               StatusCallback done) override;

  void StartAbort(const Status& s) override;

 private:
  NcclManager nccl_manager_;
};

namespace {
Status ReductionOp(const string& merge_op, ncclRedOp_t* reduction_op) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("merge_op: \"" + merge_op + "\"");
   MHTracer_DTPStensorflowPScorePSncclPScollective_communicatorDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/nccl/collective_communicator.cc", "ReductionOp");

  if (merge_op == "Add") {
    *reduction_op = ncclSum;
    return Status::OK();
  } else if (merge_op == "Mul") {
    *reduction_op = ncclProd;
    return Status::OK();
  } else if (merge_op == "Maximum") {
    *reduction_op = ncclMax;
    return Status::OK();
  } else if (merge_op == "Minimum") {
    *reduction_op = ncclMin;
    return Status::OK();
  } else {
    return errors::Internal(
        "Expected merge_op to be in [Add, Mul, Maximum, Minimum], found ",
        merge_op);
  }
}

string NcclCollectiveKey(const string& exec_key, int step_id) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("exec_key: \"" + exec_key + "\"");
   MHTracer_DTPStensorflowPScorePSncclPScollective_communicatorDTcc mht_2(mht_2_v, 242, "", "./tensorflow/core/nccl/collective_communicator.cc", "NcclCollectiveKey");

  return strings::StrCat(exec_key, ":", step_id);
}
}  // namespace

std::unique_ptr<NcclCommunicatorInterface> MaybeCreateNcclCommunicator(
    const ConfigProto& config) {
  // Skip creating a NcclCommunicator if there are 0 GPUs configured.
  const auto& device_count = config.device_count();
  auto item = device_count.find("GPU");
  if (item != device_count.end() && item->second == 0) {
    return nullptr;
  }
  return absl::make_unique<NcclCommunicator>();
}

void NcclCommunicator::Enqueue(std::shared_ptr<CollectiveContext> col_ctx,
                               StatusCallback done) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSncclPScollective_communicatorDTcc mht_3(mht_3_v, 262, "", "./tensorflow/core/nccl/collective_communicator.cc", "NcclCommunicator::Enqueue");

  const CollectiveParams* col_params = col_ctx->col_params.get();
  const int num_global_devices = col_params->group.group_size;
  const int num_local_devices = col_params->group.num_devices_per_task.at(
      col_params->group.members[col_params->default_rank].task);
  const string nccl_collective_key =
      NcclCollectiveKey(col_ctx->exec_key, col_ctx->step_id);
  auto* compute_stream = col_ctx->op_ctx->op_device_context()->stream();
  auto* gpu_info =
      col_ctx->op_ctx->device()->tensorflow_accelerator_device_info();
  auto participant = absl::make_unique<NcclManager::Participant>(
      compute_stream->parent(), compute_stream, gpu_info, col_ctx->input,
      col_ctx->output, col_ctx->col_params->default_rank,
      /*done_callback=*/nullptr);
  CancellationManager* cancel_mgr = col_ctx->op_ctx->cancellation_manager();
  if (cancel_mgr == nullptr) {
    participant->done_callback = std::move(done);
  } else {
    CancellationToken cancel_token = cancel_mgr->get_cancellation_token();
    bool already_cancelled =
        !cancel_mgr->RegisterCallback(cancel_token, [this]() {
          nccl_manager_.StartAbort(errors::Cancelled("op cancelled"));
          nccl_manager_.Reset();
        });
    if (already_cancelled) {
      done(errors::Cancelled("op cancelled"));
      return;
    }
    participant->done_callback = [cancel_mgr, cancel_token,
                                  done = std::move(done)](const Status& s) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSncclPScollective_communicatorDTcc mht_4(mht_4_v, 294, "", "./tensorflow/core/nccl/collective_communicator.cc", "lambda");

      // Do not block on deregistration since this can be invoked by
      // NcclManager::StartAbort() in the cancellation callback.
      cancel_mgr->TryDeregisterCallback(cancel_token);
      done(s);
    };
  }
  NcclManager::Context context(
      nccl_collective_key, num_local_devices, num_global_devices,
      col_params->group.runtime_details.communicator_key,
      col_params->source_rank);
  VLOG(1) << "NcclCommunicator::Enqueue type " << col_params->instance.type
          << " num_tasks " << col_params->group.num_tasks << " current task "
          << col_params->group.members[col_params->default_rank].task
          << " num local devices " << num_local_devices
          << " num global devices " << num_global_devices << " device "
          << col_ctx->device_name << " instance "
          << col_params->instance.instance_key;
  // `AddTo*` performs consistency checks for the NCCL call and enqueues the
  // `Participant` struct locally.  When all local participants with this
  // `nccl_collective_key` have called `AddToAllReduce` and
  // `SignalMultiNodeReady`, all devices at this worker are ready to process
  // this NCCL op.
  //
  // The `NcclManager` uses a dedicated CUDA stream for NCCL kernels.  At this
  // point, it synchronizes the NCCL stream with the compute stream, and then
  // enqueues the NCCL kernel on the NCCL stream.
  switch (col_params->instance.type) {
    case REDUCTION_COLLECTIVE: {
      ncclRedOp_t reduction_op;
      Status s =
          ReductionOp(col_params->merge_op->type_string(), &reduction_op);
      if (!s.ok()) {
        participant->done_callback(s);
        return;
      }
      nccl_manager_.AddToAllReduce(std::move(participant), context,
                                   reduction_op);
      break;
    }
    case GATHER_COLLECTIVE: {
      nccl_manager_.AddToAllGather(std::move(participant), context);
      break;
    }
    case BROADCAST_COLLECTIVE: {
      if (col_params->is_source) {
        nccl_manager_.AddBroadcastSend(std::move(participant), context);
      } else {
        nccl_manager_.AddBroadcastRecv(std::move(participant), context);
      }
      break;
    }
    default: {
      participant->done_callback(errors::Internal("Unexpected CollectiveType ",
                                                  col_params->instance.type));
      return;
    }
  }
  // NOTE(ayushd): We need to synchronize NCCL launches across nodes to prevent
  // deadlocks.  In the current implementation, we define a deterministic
  // sequential launch order between potentially concurrent collective instances
  // by introducing control information during static graph analysis in
  // graph/collective_order.cc.  This can be either in the form of explicit
  // control edges or via `wait_for` attribute on the collective op.
  //
  // The other end of the design spectrum would have a distinguished node
  // dynamically signal the next collective to launch to all other participants.
  // This has higher degree of runtime coordination, but it may be able to
  // achieve better performance if the (arbitrary) static execution order
  // assigned in the first approach turns out to not be good from a scheduling
  // perspective.  e.g. consider a graph in which c1, c2, and c3 are three
  // concurrent collective instances, and the static ordering assigns c1 -> c2
  // -> c3.  In practice, it could turn out that c3 is always ready to execute
  // before c1 or c2.
  {
    // `WaitForDependencies` may block if the collective instances on which this
    // op depends have not yet launched.  When this function returns, this op is
    // ready to go.
    profiler::TraceMe activity("WaitForDependencies",
                               profiler::TraceMeLevel::kInfo);
    col_ctx->col_exec->WaitForDependencies(*col_params);
    nccl_manager_.SignalMultiNodeReady(nccl_collective_key);
  }
  {
    // When all devices at this worker have called `SignalMultiNodeReady`, the
    // `NcclManager` will enqueue the NCCL kernel on the NCCL stream.  Thus the
    // implementation of `UnblockDependencies` keeps track of the number of
    // devices that have launched.
    profiler::TraceMe activity("Schedule", profiler::TraceMeLevel::kInfo);
    col_ctx->col_exec->UnblockDependencies(*col_params);
  }
}

void NcclCommunicator::StartAbort(const Status& s) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSncclPScollective_communicatorDTcc mht_5(mht_5_v, 390, "", "./tensorflow/core/nccl/collective_communicator.cc", "NcclCommunicator::StartAbort");

  nccl_manager_.StartAbort(s);
}

}  // namespace tensorflow

#else
namespace tensorflow {
std::unique_ptr<NcclCommunicatorInterface> MaybeCreateNcclCommunicator(
    const ConfigProto& config) {
  return nullptr;
}
}  // namespace tensorflow
#endif  // TENSORFLOW_USE_NCCL && (GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
