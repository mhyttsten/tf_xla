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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/local_device_state.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/stream_executor/stream.h"

namespace xla {

LocalDeviceState::LocalDeviceState(se::StreamExecutor* executor,
                                   LocalClient* client,
                                   AllocationModel allocation_model,
                                   int max_inflight_computations,
                                   bool allow_event_reuse,
                                   bool use_callback_stream)
    : allocation_model_(allocation_model),
      event_pool_(allow_event_reuse),
      compute_semaphore_(
          /*capacity=*/max_inflight_computations),
      executor_(executor),
      client_(client),
      prng_seed_generator_(prng_seed_device_()),
      prng_seed_distribution_(std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/pjrt/local_device_state.cc", "LocalDeviceState::LocalDeviceState");

  compute_stream_ = std::make_unique<se::Stream>(executor);
  host_to_device_stream_ = std::make_unique<se::Stream>(executor);
  compute_stream_->Init();
  host_to_device_stream_->Init();
  if (use_callback_stream) {
    callback_stream_map_ =
        absl::flat_hash_map<se::Stream*, std::unique_ptr<se::Stream>>();
  }
  device_to_host_streams_.reserve(kNumDeviceToHostStreams);
  for (int i = 0; i < kNumDeviceToHostStreams; ++i) {
    auto stream = std::make_unique<se::Stream>(executor);
    stream->Init();
    device_to_host_streams_.push_back(std::move(stream));
  }
  device_to_device_streams_.reserve(kNumDeviceToDeviceStreams);
  for (int i = 0; i < kNumDeviceToDeviceStreams; ++i) {
    auto stream = std::make_unique<se::Stream>(executor);
    stream->Init();
    device_to_device_streams_.push_back(std::move(stream));
  }
  execute_thread_ = std::make_unique<WorkerThread>(tensorflow::Env::Default(),
                                                   "py_xla_execute");
  callback_thread_ = std::make_unique<WorkerThread>(tensorflow::Env::Default(),
                                                    "py_xla_callback");
}

LocalDeviceState::~LocalDeviceState() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc mht_1(mht_1_v, 243, "", "./tensorflow/compiler/xla/pjrt/local_device_state.cc", "LocalDeviceState::~LocalDeviceState");

  Status status = SynchronizeAllActivity();
  if (!status.ok()) {
    LOG(ERROR) << "Error when closing device: " << status;
  }
}

Status LocalDeviceState::SynchronizeAllActivity() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/xla/pjrt/local_device_state.cc", "LocalDeviceState::SynchronizeAllActivity");

  Status status;
  // TODO(phawkins): in theory the call to SynchronizeAllActivity below should
  // suffice. However on the Host platform SynchronizeAllActivity is a dummy
  // implementation that doesn't actually block. To make sure activity has
  // stopped, also block on the compute stream. If SynchronizeAllActivity is
  // fixed, we could remove the BlockHostUntilDone call.
  status.Update(compute_stream_->BlockHostUntilDone());
  if (callback_stream_map_.has_value()) {
    for (auto& callback_stream : callback_stream_map_.value()) {
      status.Update(callback_stream.second->BlockHostUntilDone());
    }
  }
  bool ok = compute_stream_->parent()->SynchronizeAllActivity();
  if (!ok) {
    status.Update(Unknown("SynchronizeAllActivity failed."));
  }
  return status;
}

Status LocalDeviceState::ThenMemcpyDeviceToDevice(
    se::Stream* transfer_stream, se::Stream* dst_stream,
    se::DeviceMemoryBase src_buffer, se::DeviceMemoryBase dst_buffer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc mht_3(mht_3_v, 278, "", "./tensorflow/compiler/xla/pjrt/local_device_state.cc", "LocalDeviceState::ThenMemcpyDeviceToDevice");

  // The default implementation simply calls ThenMemcpyD2D, and assumes that
  // the buffer addresses identify the devices. This does not work
  // on all platforms; this method is virtual so it can be overridden.
  transfer_stream->ThenMemcpyD2D(&dst_buffer, src_buffer, dst_buffer.size());
  return Status::OK();
}

void LocalDeviceState::ThenExecuteCallback(se::Stream* stream,
                                           std::function<void()> callback) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc mht_4(mht_4_v, 290, "", "./tensorflow/compiler/xla/pjrt/local_device_state.cc", "LocalDeviceState::ThenExecuteCallback");

  tensorflow::profiler::TraceMe traceme("ThenExecuteCallback");
  if (callback_stream_map_.has_value()) {
    // Prevent concurrent updates to the callback stream map.
    absl::MutexLock lock(&mu_);
    auto callback_stream = callback_stream_map_->find(stream);
    if (callback_stream == callback_stream_map_->end()) {
      auto new_stream = std::make_unique<se::Stream>(executor_);
      new_stream->Init();
      callback_stream =
          callback_stream_map_->insert({stream, std::move(new_stream)}).first;
    }
    callback_stream->second->ThenWaitFor(stream);
    stream = callback_stream->second.get();
  }
  stream->ThenDoHostCallback([this, callback{std::move(callback)}]() mutable {
    callback_thread_->Schedule(std::move(callback));
  });
}

se::Stream* LocalDeviceState::GetDeviceToHostStream() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc mht_5(mht_5_v, 313, "", "./tensorflow/compiler/xla/pjrt/local_device_state.cc", "LocalDeviceState::GetDeviceToHostStream");

  absl::MutexLock lock(&mu_);
  int i = next_device_to_host_stream_;
  next_device_to_host_stream_ =
      (next_device_to_host_stream_ + 1) % device_to_host_streams_.size();
  return device_to_host_streams_.at(i).get();
}

se::Stream* LocalDeviceState::GetDeviceToDeviceStream() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc mht_6(mht_6_v, 324, "", "./tensorflow/compiler/xla/pjrt/local_device_state.cc", "LocalDeviceState::GetDeviceToDeviceStream");

  absl::MutexLock lock(&mu_);
  int i = next_device_to_device_stream_;
  next_device_to_device_stream_ =
      (next_device_to_device_stream_ + 1) % device_to_device_streams_.size();
  return device_to_device_streams_.at(i).get();
}

std::unique_ptr<se::Stream> LocalDeviceState::BorrowStreamFromPool() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc mht_7(mht_7_v, 335, "", "./tensorflow/compiler/xla/pjrt/local_device_state.cc", "LocalDeviceState::BorrowStreamFromPool");

  absl::MutexLock lock(&mu_);
  if (usage_stream_pool_.empty()) {
    auto stream = std::make_unique<se::Stream>(compute_stream_->parent());
    stream->Init();
    return stream;
  } else {
    std::unique_ptr<se::Stream> stream = std::move(usage_stream_pool_.top());
    usage_stream_pool_.pop();
    auto status = stream->RefreshStatus();  // Can return error::Unimplemented
    // Stream may fail with "ABORTED: Bad connection".
    if (status.code() != tensorflow::error::ABORTED) {
      CHECK(stream->ok()) << status;
    }
    return stream;
  }
}

void LocalDeviceState::ReturnStreamToPool(std::unique_ptr<se::Stream> stream) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc mht_8(mht_8_v, 356, "", "./tensorflow/compiler/xla/pjrt/local_device_state.cc", "LocalDeviceState::ReturnStreamToPool");

  auto status = stream->RefreshStatus();  // Can return error::Unimplemented
  // Stream may fail with "ABORTED: Bad connection".
  if (status.code() != tensorflow::error::ABORTED) {
    CHECK(stream->ok()) << status;
  }
  absl::MutexLock lock(&mu_);
  usage_stream_pool_.push(std::move(stream));
}

int LocalDeviceState::GetNewPrngSeed() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTcc mht_9(mht_9_v, 369, "", "./tensorflow/compiler/xla/pjrt/local_device_state.cc", "LocalDeviceState::GetNewPrngSeed");

  absl::MutexLock lock(&mu_);
  int x = 0;
  do {
    x = prng_seed_distribution_(prng_seed_generator_);
  } while (x == 0);
  return x;
}

}  // namespace xla
