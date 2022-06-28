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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_BASE_COLLECTIVE_EXECUTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_BASE_COLLECTIVE_EXECUTOR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTh() {
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


#include <memory>
#include <string>

#include "tensorflow/core/common_runtime/buf_rendezvous.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {
class CollectiveImplementation;
class DeviceMgr;
class Device;

// Helper interface that aliases regular subfields of a Tensor as separate
// Tensors for in-place update.
class CollectiveAdapter {
 public:
  virtual ~CollectiveAdapter() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTh mht_0(mht_0_v, 204, "", "./tensorflow/core/common_runtime/base_collective_executor.h", "~CollectiveAdapter");
}

  // Move the backing tensor to 'output' with its original storage and
  // shape. After this call this CollectiveAdapter object should be
  // deleted immediately without calling any of its other methods.
  virtual void ConsumeFinalValue(Tensor* output) = 0;

  // const access to entire intermediate value for debugging
  virtual const Tensor& Value() const = 0;

  // Returns tensor for chunk i which aliases the backing buffer.
  virtual Tensor ChunkAlias(int i) = 0;

  // Returns tensor allocated on the same device but with its own
  // separate backing buffer.  Will have same type and size as
  // chunk i.
  virtual Tensor TempChunk(int i) const = 0;

  // Bytes in chunk i
  virtual int64_t ChunkBytes(int i) const = 0;

  // Generate a CPU RAM scalar tensor of the same DataType as the
  // backing tensor with the given integer value.
  virtual Tensor Scalar(int v) const = 0;

  // Generate a scalar tensor of same DataType and on the same device
  // as the backing tensor.
  virtual Tensor Scalar(Allocator* a,
                        const AllocationAttributes& attr) const = 0;

  // Debugging string describing buffer location
  virtual string TBounds(const Tensor& t) const = 0;

  virtual string DebugString() const = 0;

  // Computes the number of elements per alias chunk tensor.
  //
  // A CHECK in tensor.cc expects that the memory buffer backing a
  // Tensor will be aligned according to EIGEN_MAX_ALIGN_BYTES.  To
  // ensure that all chunk aliasing Tensors maintain this alignment we
  // need to pick a chunk size that preserves it.  Note than in extreme
  // cases (impractical, but possible with very small tensors) one or
  // more tail chunks can end up emptby.
  static int64_t AlignedChunkElts(int64_t elt_bytes, int64_t total_elts,
                                  int64_t num_chunks);
};

// Create a CollectiveAdaptor wrapping 'output', specialized to its
// data-type and shape.  If align_chunks == true then chunk size may
// be larger than output->NumElements() / num_chunks and one or more
// of the suffix chunks may be empty.  Chunks will be arranged to start
// and end on alignment boundaries.  If align_chunks == false then
// output->NumElements() % num_chunks must be 0 and all chunks will
// have exactly the same size, ignoring alignment issues.
CollectiveAdapter* MakeCollectiveAdapter(Tensor* output, int num_chunks,
                                         Allocator* allocator,
                                         bool align_chunks = true);

// Default implementation of CollectiveExecutor.  Delegates the actual
// work of moving data to a class specialized for the operation type,
// arguments and device+interconnect topology.
class BaseCollectiveExecutor : public CollectiveExecutor {
 public:
  BaseCollectiveExecutor(CollectiveExecutorMgrInterface* cem,
                         CollectiveRemoteAccess* remote_access, int64_t step_id,
                         const DeviceMgr* dev_mgr,
                         std::shared_ptr<UnboundedWorkQueue> work_queue)
      : CollectiveExecutor(cem),
        step_id_(step_id),
        dev_mgr_(dev_mgr),
        remote_access_(remote_access),
        work_queue_(std::move(work_queue)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTh mht_1(mht_1_v, 278, "", "./tensorflow/core/common_runtime/base_collective_executor.h", "BaseCollectiveExecutor");
}

  ~BaseCollectiveExecutor() override;

  void StartAbort(const Status& s) override TF_LOCKS_EXCLUDED(status_mu_);

  void ExecuteAsync(OpKernelContext* ctx, const CollectiveParams* col_params,
                    const string& exec_key, StatusCallback done) override;

  void CompleteParamsAsync(const DeviceAttributes& device, CollectiveParams* cp,
                           CancellationManager* cancel_mgr,
                           StatusCallback done) override;

  CollectiveRemoteAccess* remote_access() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTh mht_2(mht_2_v, 294, "", "./tensorflow/core/common_runtime/base_collective_executor.h", "remote_access");

    return remote_access_.get();
  }

  void RunClosure(std::function<void()> closure) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTh mht_3(mht_3_v, 301, "", "./tensorflow/core/common_runtime/base_collective_executor.h", "RunClosure");

    work_queue_->Schedule(std::move(closure));
  }

  // If we need to enforce an ordering on any portion of collective
  // implementation, and the ordering is encoded via attribute on the collective
  // op, this function will block until all dependencies for this collective
  // have completed.
  void WaitForDependencies(const CollectiveParams& col_params) override;
  // Record that this collective has completed the portion of the implementation
  // that needs to be ordered wrt other collectives, to unblock any of its
  // dependent ops.
  void UnblockDependencies(const CollectiveParams& col_params) override;

 protected:
  const int64_t step_id_;
  const DeviceMgr* dev_mgr_;  // Not owned.
  std::unique_ptr<CollectiveRemoteAccess> remote_access_;
  // Ownership of `work_queue_` is shared between `this` and
  // `CollectiveExecutorMgr`.
  std::shared_ptr<UnboundedWorkQueue> work_queue_;
  mutex launch_mu_;
  condition_variable launch_cv_;
  // collective instance key -> number of local devices for which NCCL ops have
  // been launched.
  std::unordered_map<int32, int32> launched_ TF_GUARDED_BY(launch_mu_);
  mutex status_mu_;
  Status status_ TF_GUARDED_BY(status_mu_);

 private:
  Status CreateCollective(const CollectiveParams& col_params,
                          CollectiveImplementationInterface** col_impl);
  // Check if all ops on which this collective depends on have launched.
  bool CheckDependencies(const CollectiveParams& col_params)
      TF_EXCLUSIVE_LOCKS_REQUIRED(launch_mu_);
  // Tries to return the status that is the original error. It returns the
  // aborted status if the collective executor is aborted.
  Status GetStatus(const Status& s) TF_LOCKS_EXCLUDED(status_mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_BASE_COLLECTIVE_EXECUTOR_H_
