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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTh() {
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


#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"

namespace tensorflow {

class WorkerSession;

namespace eager {

// EagerClusterFunctionLibraryRuntime contains methods to Instantiate and Run
// functions across processes by making RPCs through eager service.
class EagerClusterFunctionLibraryRuntime
    : public DistributedFunctionLibraryRuntime {
 public:
  EagerClusterFunctionLibraryRuntime(const uint64 context_id, EagerContext* ctx,
                                     DeviceMgr* remote_device_mgr)
      : context_id_(context_id),
        ctx_(ctx),
        remote_device_mgr_(remote_device_mgr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTh mht_0(mht_0_v, 211, "", "./tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.h", "EagerClusterFunctionLibraryRuntime");
}

  ~EagerClusterFunctionLibraryRuntime() override{
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTh mht_1(mht_1_v, 216, "", "./tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.h", "~EagerClusterFunctionLibraryRuntime");
};

  // Register a partition (i.e., component function) of a multi-device function
  // on the remote target specified in `options.target`. This should be
  // triggered as part of instantiating a multi-device function in
  // ProcessFunctionLibraryRuntime.
  void Instantiate(const string& function_name,
                   const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
                   const FunctionLibraryRuntime::InstantiateOptions& options,
                   FunctionLibraryRuntime::LocalHandle* handle,
                   FunctionLibraryRuntime::DoneCallback done) override;

  // Execute the component function specified by `handle` on its instantiated
  // remote target. This should be triggered as part of driving a multi-device
  // function execution in ProcessFunctionLibraryRuntime. Running the component
  // function remotely is purely asynchronous, and multiple component functions
  // with the same remote target are not executed in any particular ordering.
  // The main function side must wait for all component functions to finish
  // (i.e., the done callbacks triggered) before finishing its execution.
  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::LocalHandle handle,
           gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done) override;

  // The component function inputs `args` and outputs `rets` may refer to remote
  // tensors on a remote device, which will be lazily resolved remotely where
  // the inputs/outputs are actually consumed.
  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::LocalHandle handle,
           gtl::ArraySlice<FunctionArg> args, std::vector<FunctionRet>* rets,
           FunctionLibraryRuntime::DoneCallback done) override;

  void CleanUp(uint64 step_id, FunctionLibraryRuntime::LocalHandle handle,
               FunctionLibraryRuntime::DoneCallback done) override;

  DeviceMgr* remote_device_mgr() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTh mht_2(mht_2_v, 254, "", "./tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.h", "remote_device_mgr");
 return remote_device_mgr_; }

 private:
  const uint64 context_id_;
  EagerContext* ctx_;
  DeviceMgr* remote_device_mgr_;  // not owned.

  struct FunctionData {
    const string target;
    const absl::optional<std::vector<int>> ret_indices;
    core::RefCountPtr<EagerClient> eager_client;
    std::unique_ptr<EagerOperation> op;

    FunctionData(const string& target,
                 const absl::optional<std::vector<int>>& ret_indices,
                 EagerClient* eager_client, std::unique_ptr<EagerOperation> op)
        : target(target),
          ret_indices(ret_indices),
          eager_client(core::RefCountPtr<EagerClient>(eager_client)),
          op(std::move(op)) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTh mht_3(mht_3_v, 277, "", "./tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.h", "FunctionData");

      eager_client->Ref();
    }
  };

  mutable mutex mu_;
  std::vector<FunctionData> function_data_ TF_GUARDED_BY(mu_);
};

DistributedFunctionLibraryRuntime* CreateClusterFLR(
    const uint64 context_id, EagerContext* ctx, WorkerSession* worker_session);

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_
