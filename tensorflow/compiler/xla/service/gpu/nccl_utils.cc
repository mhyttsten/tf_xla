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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace gpu {

bool IsGlobalNcclConfig() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/gpu/nccl_utils.cc", "IsGlobalNcclConfig");

  static const bool global_nccl_config = std::getenv("NCCL_COMM_ID") != nullptr;
  return global_nccl_config;
}

bool IsNcclLaunchModeParallel() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/xla/service/gpu/nccl_utils.cc", "IsNcclLaunchModeParallel");

  static const bool is_launch_mode_parallel =
      absl::string_view(std::getenv("NCCL_LAUNCH_MODE")) == "PARALLEL";
  return is_launch_mode_parallel;
}

Status ToStatus(ncclResult_t s, const char* file, int64_t line,
                const char* expr) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   mht_2_v.push_back("expr: \"" + (expr == nullptr ? std::string("nullptr") : std::string((char*)expr)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc mht_2(mht_2_v, 224, "", "./tensorflow/compiler/xla/service/gpu/nccl_utils.cc", "ToStatus");

  if (s == ncclSuccess) {
    return Status::OK();
  }
  return tensorflow::errors::Internal(
      absl::StrFormat("%s:%d: NCCL operation %s failed: %s", file, line, expr,
                      ncclGetErrorString(s)));
}

ncclRedOp_t ToNcclReduction(ReductionKind kind) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/xla/service/gpu/nccl_utils.cc", "ToNcclReduction");

  switch (kind) {
    case ReductionKind::SUM:
      return ncclSum;
    case ReductionKind::PRODUCT:
      return ncclProd;
    case ReductionKind::MIN:
      return ncclMin;
    case ReductionKind::MAX:
      return ncclMax;
  }
}

namespace {

StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType element_type) {
  switch (element_type) {
    case S8:
      return ncclInt8;
    case PRED:
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
    case C64:
      return ncclFloat32;
    case F64:
    case C128:
      return ncclFloat64;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case BF16:
      return ncclBfloat16;
#endif
    default:
      return tensorflow::errors::InvalidArgument(absl::StrFormat(
          "Unsupported data type: %s", PrimitiveType_Name(element_type)));
  }
}

StatusOr<ncclUniqueId> ToNcclUniqueId(const std::string& id_str) {
  static_assert(sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES,
                "NCCL_UNIQUE_ID_BYTES");

  TF_RET_CHECK(id_str.size() == NCCL_UNIQUE_ID_BYTES);
  ncclUniqueId id;
  absl::c_copy(id_str, id.internal);
  return id;
}

template <typename K, typename V>
class ThreadSafeMap {
 public:
  V& operator[](const K& key) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc mht_4(mht_4_v, 300, "", "./tensorflow/compiler/xla/service/gpu/nccl_utils.cc", "lambda");

    absl::MutexLock lock(&mutex_);
    std::unique_ptr<V>& value = map_[key];
    if (value == nullptr) value = std::make_unique<V>();
    return *value;
  }

  void ForEachValue(const std::function<void(V&)>& fn) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc mht_5(mht_5_v, 310, "", "./tensorflow/compiler/xla/service/gpu/nccl_utils.cc", "ForEachValue");

    absl::MutexLock lock(&mutex_);
    for (const auto& it : map_) fn(*it.second);
  }

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<K, std::unique_ptr<V>> map_ ABSL_GUARDED_BY(mutex_);
};

StatusOr<std::string> LocalNcclUniqueIdCallback(const NcclCliqueKey&) {
  ncclUniqueId id;
  XLA_CUDA_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  return std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
}

void WaitAndLogIfStuck(absl::Mutex& mutex, const absl::Condition& condition) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc mht_6(mht_6_v, 329, "", "./tensorflow/compiler/xla/service/gpu/nccl_utils.cc", "WaitAndLogIfStuck");

  constexpr absl::Duration kTimeout = absl::Seconds(10);
  if (mutex.AwaitWithTimeout(condition, kTimeout)) {
    return;
  }

  LOG(ERROR) << "This thread has been waiting for "
             << absl::ToInt64Seconds(kTimeout) << "s and may be stuck:";

  int64_t termination_timeout = xla::GetDebugOptionsFromFlags()
                                    .xla_gpu_nccl_termination_timeout_seconds();
  // infinite timeout is equivalent to await call without timeout.
  absl::Duration kTerminationTimeout = termination_timeout >= 0
                                           ? absl::Seconds(termination_timeout)
                                           : absl::InfiniteDuration();

  if (mutex.AwaitWithTimeout(condition, kTerminationTimeout)) {
    LOG(ERROR) << "Thread is unstuck! Warning above was a false-positive. "
                  "Perhaps the timeout is too short.";
    return;
  }
  LOG(ERROR)
      << "Termination timeout of " << termination_timeout
      << " seconds exceeded. Exiting to ensure a consistent program state.";
  std::exit(42);
}

// A rendezvous for a group of threads.
//
// The group of threads identifies itself with a key that must be unique to the
// the group. When all threads have arrived at the rendezvous, one thread
// executes the given function and all threads received the result.
// TODO(cjfj): Replace XLA rendezvous code with this simpler implementation.
template <typename R, typename K>
std::shared_ptr<R> Rendezvous(const K& key, size_t num_threads,
                              const std::function<R()>& fn) {
  // Fast-path (DO NOT REMOVE: the logic below doesn't work for single thread).
  if (num_threads == 1) return std::make_shared<R>(fn());

  struct State {
    absl::Mutex mutex;
    size_t num_threads_arrived ABSL_GUARDED_BY(mutex) = 0;
    std::shared_ptr<R> result ABSL_GUARDED_BY(mutex);
  };

  static auto& states = *new ThreadSafeMap<K, State>;
  State& state = states[key];

  absl::MutexLock lock(&state.mutex);
  ++state.num_threads_arrived;

  std::shared_ptr<R> result;
  if (state.num_threads_arrived == num_threads) {
    // Last thread to arrive executes the function.
    CHECK(state.result == nullptr);
    result = std::make_shared<R>(fn());
    state.result = result;
    state.num_threads_arrived = 0;
  } else {
    absl::Condition result_ready(
        +[](std::shared_ptr<R>* ptr) { return ptr->get() != nullptr; },
        &state.result);
    WaitAndLogIfStuck(state.mutex, result_ready);

    // There is one use of the result in the shared state, plus one use for each
    // thread that has already retrieved the result.
    if (state.result.use_count() < num_threads) {
      result = state.result;
    } else {
      // Last thread to retrieve the result takes the result from the state,
      // allowing the other threads to exit the function.
      return std::move(state.result);
    }
  }

  // Wait for all threads to have retrieved the result. Without this, a thread
  // could duplicate or delete its copy of the result, invalidating the use
  // count logic above.
  absl::Condition result_taken(
      +[](std::shared_ptr<R>* ptr) { return ptr->get() == nullptr; },
      &state.result);
  WaitAndLogIfStuck(state.mutex, result_taken);
  return result;
}

struct NcclCliqueState {
  ncclUniqueId unique_id;
  int64_t run_id = -1;
};

using NcclClique = Lockable<NcclCliqueState>;

std::shared_ptr<StatusOr<NcclClique::Lock>> AcquireNcclClique(
    RunId run_id, OpId op_id, NcclCliqueKey clique_key,
    const NcclUniqueIdCallback& unique_id_callback,
    size_t num_local_participants) {
  static auto& cliques = *new ThreadSafeMap<NcclCliqueKey, NcclClique>;

  auto rendezvous_key = std::make_tuple(run_id, op_id, std::move(clique_key));

  return Rendezvous<StatusOr<NcclClique::Lock>>(
      rendezvous_key, num_local_participants,
      [&]() -> StatusOr<NcclClique::Lock> {
        const NcclCliqueKey& clique_key = std::get<2>(rendezvous_key);
        NcclClique::Lock clique = cliques[clique_key].Acquire();
        if (clique->run_id < 0) {
          TF_ASSIGN_OR_RETURN(std::string id, unique_id_callback(clique_key));
          TF_ASSIGN_OR_RETURN(clique->unique_id, ToNcclUniqueId(id));
        }
        // If multiple executable are running simultaneously while using
        // multiple hosts, it is possible that different executables could
        // acquire the same clique on different hosts. We protect against this
        // by checking that the run ID increases monotonically.
        bool is_local = clique_key.devices().size() == num_local_participants;
        TF_RET_CHECK(is_local || (run_id.ToInt() >= clique->run_id));
        clique->run_id = run_id.ToInt();
        return clique;
      });
}

void CheckNcclAsyncError(NcclComm& lockable_comm) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc mht_7(mht_7_v, 452, "", "./tensorflow/compiler/xla/service/gpu/nccl_utils.cc", "CheckNcclAsyncError");

  ncclComm_t comm = *lockable_comm.Acquire();
  if (comm == nullptr) return;

  Status status = [comm] {
    ncclResult_t async_err;
    XLA_CUDA_RETURN_IF_ERROR(ncclCommGetAsyncError(comm, &async_err));
    if (async_err != ncclSuccess) {
      LOG(ERROR) << "Aborting communicator: " << comm
                 << " due to async NCCL error: "
                 << ncclGetErrorString(async_err);
      XLA_CUDA_RETURN_IF_ERROR(ncclCommAbort(comm));
    }
    return XLA_CUDA_STATUS(async_err);
  }();

  if (!status.ok()) LOG(ERROR) << status.ToString();
}

}  // namespace

StatusOr<std::pair<ncclDataType_t, int>> ToNcclDataTypeAndCountMultiplier(
    PrimitiveType element_type) {
  TF_ASSIGN_OR_RETURN(ncclDataType_t dtype, ToNcclDataType(element_type));
  bool is_complex = primitive_util::IsComplexType(element_type);
  return std::make_pair(dtype, is_complex ? 2 : 1);
}

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_utilsDTcc mht_8(mht_8_v, 485, "", "./tensorflow/compiler/xla/service/gpu/nccl_utils.cc", "GetNumLocalParticipants");

  if (local_devices == nullptr) return participants.size();

  return absl::c_count_if(participants, [&](const GlobalDeviceId& device_id) {
    return absl::c_linear_search(*local_devices, device_id);
  });
}

StatusOr<const NcclUniqueIdCallback*> GetNcclUniqueIdCallback(
    const NcclUniqueIdCallback* unique_id_callback, bool is_local) {
  if (unique_id_callback != nullptr) return unique_id_callback;

  TF_RET_CHECK(is_local || IsGlobalNcclConfig())
      << "If non-local devices are taking part of a collective API on "
         "GPU, the nccl_unique_id_callback must be provided by the client.";

  static NcclUniqueIdCallback local_callback(LocalNcclUniqueIdCallback);
  return &local_callback;
}

StatusOr<NcclComm::Lock> AcquireNcclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    size_t num_local_participants,
    const NcclUniqueIdCallback& unique_id_callback, int rank) {
  // Ensure that this group of threads have exclusive access to the clique to
  // prevent threads from different groups locking communicators in the clique.
  NcclCliqueKey clique_key(std::move(participants));
  std::shared_ptr<StatusOr<NcclClique::Lock>> clique = AcquireNcclClique(
      run_id, op_id, clique_key, unique_id_callback, num_local_participants);

  if (!clique->ok()) return clique->status();

  auto comm_key = std::make_pair(std::move(clique_key), rank);
  static auto& comms = *new ThreadSafeMap<decltype(comm_key), NcclComm>;

  // Launch a thread that periodically checks all NCCL communicators for
  // asynchronous errors. If an asynchronous error is observed, the communicator
  // is aborted and an error message logged.
  static auto check_async_error_thread =
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions(), "nccl_async_error_thread", [&] {
            while (true) {
              absl::SleepFor(absl::Seconds(30));
              comms.ForEachValue(CheckNcclAsyncError);
            }
          });
  (void)check_async_error_thread;  // Silence unused variable warning.

  NcclComm::Lock comm = comms[comm_key].Acquire();
  if (*comm == nullptr) {
    int nranks = comm_key.first.devices().size();
    const ncclUniqueId& id = (**clique)->unique_id;
    XLA_CUDA_RETURN_IF_ERROR(ncclCommInitRank(comm.get(), nranks, id, rank));
  }
  return comm;
}

}  // namespace gpu
}  // namespace xla
