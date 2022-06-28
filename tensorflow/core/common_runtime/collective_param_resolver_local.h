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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_PARAM_RESOLVER_LOCAL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_PARAM_RESOLVER_LOCAL_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTh() {
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


#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
class CompleteGroupRequest;
class CompleteGroupResponse;
class CompleteInstanceRequest;
class CompleteInstanceResponse;
class ConfigProto;
class DeviceMgr;

// Implements ParamResolverInterface for a single-task context.
// It also implements the functionality necessary to serve as the
// group leader for param resolution in a multi-task context.
class CollectiveParamResolverLocal : public ParamResolverInterface {
 public:
  CollectiveParamResolverLocal(const ConfigProto& config,
                               const DeviceMgr* dev_mgr,
                               DeviceResolverInterface* dev_resolver,
                               NcclCommunicatorInterface* nccl_communicator,
                               const string& task_name);

  ~CollectiveParamResolverLocal() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTh mht_0(mht_0_v, 218, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.h", "~CollectiveParamResolverLocal");
}

  void CompleteParamsAsync(const DeviceAttributes& device, CollectiveParams* cp,
                           CancellationManager* cancel_mgr,
                           const StatusCallback& done) override;

  void CompleteGroupAsync(const DeviceAttributes& device,
                          CollGroupParams* group_params,
                          CancellationManager* cancel_mgr,
                          const StatusCallback& done) override;

  void CompleteInstanceAsync(const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             CancellationManager* cancel_mgr,
                             const StatusCallback& done) override;

  Status LookupGroup(int32_t group_key, CollGroupParams* group) override;

  void StartAbort(const Status& s) override;

 protected:
  // For access to InstanceRec and CompleteDefaultRanking.
  friend class CollectiveParamResolverLocalTest;

  // Used to complete/verify CollGroup.
  struct GroupRec {
    mutable mutex mu;
    CollGroupParams group TF_GUARDED_BY(mu);
    Status status TF_GUARDED_BY(mu);
    std::unordered_map<string, int64_t> incarnations_by_device_name
        TF_GUARDED_BY(mu);
    std::vector<CollGroupParams*> pending_params TF_GUARDED_BY(mu);
    std::vector<StatusCallback> pending_done TF_GUARDED_BY(mu);
  };

  // Finds the GroupRec that corresponds to group_params->group_key.
  // Also populates group_params from that group_rec.
  // Will wait until GroupRec is fully populated or an error arises before
  // calling done.  Callback GroupRec* arg is only valid if status is ok.
  // Ownership of GroupRec stays with this object and does not pass to the
  // callback.
  void CompleteGroupLocal(const DeviceAttributes& device,
                          CollGroupParams* group_params,
                          CancellationManager* cancel_mgr, StatusCallback done)
      TF_LOCKS_EXCLUDED(group_mu_);

  // Finishes the group parameters once all members of the group are there.
  void FinishGroup(GroupRec* gr) TF_EXCLUSIVE_LOCKS_REQUIRED(gr->mu);

  // Cancels the group if it's still pending.
  void CancelGroup(int32 group_key) TF_LOCKS_EXCLUDED(group_mu_);

  // Lookup and populate parameters from an already initialized group.
  Status LookupAndPopulateGroupParams(CollGroupParams* group_params);

  // Used to complete/verify CollInstance.
  struct InstanceRec;

  typedef std::function<void(InstanceRec*)> IRConsumer;
  struct InstanceRec {
    mutex mu;
    // Values to be shared by all instances, constant after initialization.
    CollectiveParams* shared;
    // If an error occurs during initialization this structure stays in the
    // table with a non-OK status. Purging the table and restarting needs to be
    // done at a higher level.
    Status status TF_GUARDED_BY(mu);

    // These fields are used to count the instances that have called
    // in and become known while resolving broadcast source identity and
    // communicator key.
    int source_rank TF_GUARDED_BY(mu);
    string communicator_key TF_GUARDED_BY(mu);
    int known_count TF_GUARDED_BY(mu);
    std::vector<bool> known TF_GUARDED_BY(mu);
    std::vector<IRConsumer> known_waiters TF_GUARDED_BY(mu);

    InstanceRec()
        : shared(new CollectiveParams()), source_rank(-1), known_count(0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTh mht_1(mht_1_v, 299, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.h", "InstanceRec");
}
    ~InstanceRec() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTh mht_2(mht_2_v, 303, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.h", "~InstanceRec");
 shared->Unref(); }
  };

  // Find the InstanceRec with the same instance_key as cp.  If it doesn't
  // already exist, create and initialize from gr and cp.
  // created is set to true if a new IRec is created, false otherwise.
  //
  // Precondition: *gr must be a complete GroupRec, i.e. the value set
  // by CompleteGroupLocal. *cp must be populated with all the fields
  // required by InitInstanceSharedParams.  Ownership of InstanceRec stays
  // with this object and does not pass to the callback.
  InstanceRec* GetOrCreateInstanceRec(CollectiveParams* cp, bool* created)
      TF_LOCKS_EXCLUDED(instance_mu_, group_mu_);

  // Populate *ir with device membership from gr, then initialize to be specific
  // to cp->instance_key, i.e. order the devices and tasks.
  //
  // Preconditions:
  //  cp is populated with all DeviceLocalities
  void InitInstanceSharedParams(const CollectiveParams* cp, InstanceRec* ir);

  // Establishes the final order of gp->device_names and gp->task_names by
  // considering localities of all devices.
  void CompleteDefaultRanking(CollGroupParams* gp);

  // Finish populating *cp.
  // Precondition: *gr has been fully populated by CompleteGroupLocal.
  void CompleteInstanceLocal(const string& device, CollectiveParams* cp,
                             const StatusCallback& done)
      TF_LOCKS_EXCLUDED(instance_mu_, group_mu_);

  // Finish populating *cp from fully initialized *ir.
  // Precondition: *gr and *ir are fully populated.
  void CompleteInstanceFromInitializedIRec(const string& device,
                                           CollectiveParams* cp,
                                           InstanceRec* ir,
                                           const StatusCallback& done)
      TF_LOCKS_EXCLUDED(ir->mu);

  // Complete instance params after waiting for group.
  // Precondition: *cp has complete group data and default_rank.
  void WaitForGroup(InstanceRec* ir, CollectiveParams* cp, const IRConsumer& f)
      TF_LOCKS_EXCLUDED(ir->mu);

  // If cp.device_names contains only devices local to this process
  // populates *localities, else returns an error.
  Status GetLocalDeviceLocalities(const CollectiveParams& cp,
                                  std::vector<DeviceLocality>* localities);

  // Sets cp->instance_default_rank according to location of device in
  // current ordering of cp->instance.device_names.
  void SetDefaultRank(const string& device, CollectiveParams* cp);

  // Sets cp->instance.type based on collective op type, and attempts to assign
  // best implementation.
  void AssignCollectiveType(CollectiveParams* cp);

  void StartAbortLocal(const Status& s)
      TF_LOCKS_EXCLUDED(status_mu_, group_mu_, instance_mu_);

  const bool nccl_;
  const DeviceMgr* dev_mgr_;
  DeviceResolverInterface* dev_resolver_;  // Not owned.
  NcclCommunicatorInterface* nccl_communicator_;  // Not owned.
  string task_name_;
  string gpu_ring_order_;
  mutex group_mu_;
  gtl::FlatMap<int32, std::unique_ptr<GroupRec>> group_table_
      TF_GUARDED_BY(group_mu_);
  mutex instance_mu_;
  gtl::FlatMap<int32, gtl::FlatMap<int32, std::unique_ptr<InstanceRec>>>
      instance_table_ TF_GUARDED_BY(instance_mu_);
  mutex status_mu_;
  Status status_ TF_GUARDED_BY(status_mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_PARAM_RESOLVER_LOCAL_H_
