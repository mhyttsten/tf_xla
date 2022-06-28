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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_TEST_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_TEST_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_test_utilDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_test_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_test_utilDTh() {
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


#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/test_collective_executor_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {

// Wraps CollectiveRemoteAccessLocal with the ability to return an
// error status to the N'th action.
class FailTestRMA : public CollectiveRemoteAccessLocal {
 public:
  FailTestRMA(const DeviceMgr* dev_mgr, DeviceResolverInterface* dev_resolver,
              int64_t step_id);

  // Sets when it should fail. Setting to zero disables the failure.
  void set_fail_after(int fail_after) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_test_utilDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/common_runtime/collective_test_util.h", "set_fail_after");

    mutex_lock l(mu_);
    fail_after_ = fail_after;
  }

  void RecvFromPeer(const string& peer_device, const string& peer_task,
                    bool peer_is_local, const string& key, Device* to_device,
                    DeviceContext* to_device_ctx,
                    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
                    const DeviceLocality& client_locality,
                    int dev_to_dev_stream_index,
                    CancellationManager* cancellation_manager,
                    const StatusCallback& done) override;

  void PostToPeer(const string& peer_device, const string& peer_task,
                  const string& key, Device* from_device,
                  DeviceContext* from_device_ctx,
                  const AllocatorAttributes& from_alloc_attr,
                  const Tensor* from_tensor,
                  const DeviceLocality& client_locality,
                  CancellationManager* cancellation_manager,
                  const StatusCallback& done) override;

 private:
  bool MaybeFail(const StatusCallback& done);

  mutex mu_;
  int fail_after_ TF_GUARDED_BY(mu_);
};

struct CollectiveTestEnv {
  int num_workers;
  int num_devices_per_worker;
  DeviceType device_type;
  std::unique_ptr<ParamResolverInterface> param_resolver;
  std::unique_ptr<TestCollectiveExecutorMgr> col_exec_mgr;
  std::shared_ptr<UnboundedWorkQueue> work_queue;
  std::unique_ptr<tensorflow::DeviceMgr> device_mgr;
  std::unique_ptr<DeviceResolverInterface> device_resolver;
  core::RefCountPtr<CollectiveExecutor> col_exec;
  FailTestRMA* remote_access;

  CollectiveTestEnv() : device_type(DEVICE_DEFAULT) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_test_utilDTh mht_1(mht_1_v, 258, "", "./tensorflow/core/common_runtime/collective_test_util.h", "CollectiveTestEnv");
}
};

std::unique_ptr<CollectiveTestEnv> CreateCollectiveTestEnv(
    int num_workers, int num_devices_per_worker, DeviceType device_type);

core::RefCountPtr<CollectiveParams> CreateCollectiveParams(
    const CollectiveTestEnv& test_env, int rank, const string& collective_name,
    CollectiveType collective_type, DataType dtype, const TensorShape& shape,
    const std::vector<std::vector<int>> user_specified_rank_per_worker = {{}});

std::vector<int> GenerateEvenSubdivOffsets(int num_devices_per_worker,
                                           int num_subdivs);

// Runs a collective. input and output should be on the host.
Status RunCollective(CollectiveTestEnv* test_env, CollectiveParams* col_params,
                     Device* device, Tensor* input, Tensor* output);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_TEST_UTIL_H_
