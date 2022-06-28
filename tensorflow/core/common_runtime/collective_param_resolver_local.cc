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
class MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc() {
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
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"

#include <stddef.h>

#include <algorithm>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

CollectiveParamResolverLocal::CollectiveParamResolverLocal(
    const ConfigProto& config, const DeviceMgr* dev_mgr,
    DeviceResolverInterface* dev_resolver,
    NcclCommunicatorInterface* nccl_communicator, const string& task_name)
    : nccl_(config.experimental().collective_nccl()),
      dev_mgr_(dev_mgr),
      dev_resolver_(dev_resolver),
      nccl_communicator_(nccl_communicator),
      task_name_(task_name),
      gpu_ring_order_(
          config.gpu_options().experimental().collective_ring_order()) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("task_name: \"" + task_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_0(mht_0_v, 225, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::CollectiveParamResolverLocal");
}

void CollectiveParamResolverLocal::CompleteGroupAsync(
    const DeviceAttributes& device, CollGroupParams* group_params,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::CompleteGroupAsync");

  CompleteGroupLocal(device, group_params, cancel_mgr, done);
}

namespace {
const char* GetCollectiveName(const CollectiveParams* cp, bool nccl) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_2(mht_2_v, 240, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "GetCollectiveName");

  switch (cp->instance.type) {
    case BROADCAST_COLLECTIVE:
      return nccl ? "NcclBroadcast" : "HierarchicalTreeBroadcast";

    case REDUCTION_COLLECTIVE:
      return nccl ? "NcclReduce" : "RingReduce";

    case GATHER_COLLECTIVE:
      return nccl ? "NcclGather" : "RingGather";

    case PERMUTE_COLLECTIVE:
      return "Permute";

    case ALL_TO_ALL_COLLECTIVE:
      return "AllToAll";

    default:
      return "undef";
  }
}

string TaskNameFromDeviceName(const string& device_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "TaskNameFromDeviceName");

  DeviceNameUtils::ParsedName parsed_device;
  CHECK(DeviceNameUtils::ParseFullName(device_name, &parsed_device));
  string task_name;
  CHECK(DeviceNameUtils::GetTaskName(parsed_device, &task_name));
  return task_name;
}

struct RankFormatter {
  void operator()(std::string* out, CollGroupMember m) const {
    out->append(std::to_string(m.rank));
  }
};

Status CheckUserSpecifiedRanks(const std::vector<CollGroupMember> members) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_4(mht_4_v, 283, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CheckUserSpecifiedRanks");

  absl::flat_hash_set<int> user_ranks = {};
  bool at_least_one_member_with_no_rank = false;
  bool at_least_one_member_with_user_rank = false;
  for (const auto& m : members) {
    if (m.rank == -1) {
      at_least_one_member_with_no_rank = true;
    } else {
      at_least_one_member_with_user_rank = true;
      user_ranks.insert(m.rank);
    }
  }

  auto received_ranks = absl::StrJoin(members, ",", RankFormatter());
  if (at_least_one_member_with_no_rank && at_least_one_member_with_user_rank) {
    return errors::InvalidArgument(
        "Only part of the group members have user given rank specified.",
        "Received ranks: ", received_ranks);
  }

  if (at_least_one_member_with_user_rank &&
      user_ranks.size() < members.size()) {
    return errors::InvalidArgument(
        "Duplicate ranks specified for group members. Received ranks: ",
        received_ranks);
  }
  return Status::OK();
}
}  // namespace

void CollectiveParamResolverLocal::CompleteGroupLocal(
    const DeviceAttributes& device, CollGroupParams* group_params,
    CancellationManager* cancel_mgr, StatusCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_5(mht_5_v, 318, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::CompleteGroupLocal");

  VLOG(1) << "CompleteGroup device=" << device.name() << ": "
          << group_params->ToString();
  std::vector<StatusCallback> to_be_called;

  GroupRec* gr = nullptr;
  Status status;
  {
    mutex_lock l(group_mu_);
    auto it = group_table_.find(group_params->group_key);
    if (it == group_table_.end()) {
      gr = new GroupRec;
      mutex_lock grl(gr->mu);
      gr->group.group_key = group_params->group_key;
      gr->group.group_size = group_params->group_size;
      gr->group.device_type = group_params->device_type;
      if (nccl_communicator_ != nullptr) {
        gr->group.runtime_details.communicator_key =
            nccl_communicator_->GenerateCommunicatorKey();
      }
      // Store GroupRec in group_table_ which is shared between all devices on
      // this worker.
      group_table_[gr->group.group_key].reset(gr);
      VLOG(2) << "New group_key=" << gr->group.group_key
              << " group_size=" << gr->group.group_size
              << " runtime_details=" << gr->group.runtime_details.ToString();
    } else {
      gr = it->second.get();
    }
  }
  {
    mutex_lock l(status_mu_);
    status = status_;
  }
  if (!status.ok()) {
    done(status);
    return;
  }

  if (cancel_mgr != nullptr) {
    CancellationToken token = cancel_mgr->get_cancellation_token();
    bool is_cancelled = !cancel_mgr->RegisterCallback(
        token, std::bind(&CollectiveParamResolverLocal::CancelGroup, this,
                         group_params->group_key));
    if (is_cancelled) {
      done(errors::Cancelled("CompleteGroup is cancelled before it starts"));
      return;
    }
    done = [cancel_mgr, token,
            original_done = std::move(done)](const Status& status) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_6(mht_6_v, 370, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "lambda");

      cancel_mgr->TryDeregisterCallback(token);
      original_done(status);
    };
  }

  {
    mutex_lock gr_lock(gr->mu);
    // If there is ever an error associated with a group key, we store the error
    // status and invoke all waiting and future callbacks with this error
    // status.
    VLOG(2) << "gr device_type=" << gr->group.device_type
            << " cp device_type=" << group_params->device_type
            << " current device=" << device.name();
    if (gr->status.ok()) {
      // Check for consistency with existing GroupRec.
      if (group_params->device_type != gr->group.device_type) {
        gr->status = errors::Internal(
            "Device ", device.name(),
            " is joining a group with incompatible device type",
            gr->group.device_type.type_string(),
            " (group_key=", gr->group.group_key, ")");
      } else if (group_params->group_size != gr->group.group_size) {
        gr->status = errors::Internal(
            "Device ", device.name(), " is joining a group with size",
            group_params->group_size, ", but that group has size ",
            gr->group.group_size, " (group_key=", gr->group.group_key, ")");
      }
    }
    bool new_device = false;
    if (gr->status.ok()) {
      // Insert device if not already present.
      auto it = gr->incarnations_by_device_name.find(device.name());
      if (it == gr->incarnations_by_device_name.end()) {
        if (gr->group.members.size() == gr->group.group_size) {
          // The group is already full.
          gr->status =
              errors::Internal("Device ", device.name(),
                               " is joining a group that is already full",
                               " (group_key=", gr->group.group_key, ")");
        } else {
          // This is a new device that has not yet joined the group.
          gr->incarnations_by_device_name[device.name()] = device.incarnation();
          CollGroupMember member;
          member.device = device;
          if (group_params->user_specified_rank == -1 ||
              (group_params->user_specified_rank >= 0 &&
               group_params->user_specified_rank < gr->group.group_size)) {
            member.rank = group_params->user_specified_rank;
          } else {
            gr->status = errors::InvalidArgument(
                "User Provided rank is invalid. It should be between [0, "
                "group_size)");
          }
          gr->group.members.push_back(std::move(member));
          new_device = true;
          if (VLOG_IS_ON(1)) {
            string dev_buf;
            for (const auto& m : gr->group.members) {
              strings::StrAppend(&dev_buf, ",", m.device.name());
            }
            VLOG(1) << "CompleteGroupLocal group_key=" << gr->group.group_key
                    << " group_size=" << gr->group.group_size << " (current"
                    << " devices)=(" << dev_buf << ") (number of"
                    << " devices pending)="
                    << (gr->group.group_size - gr->group.members.size());
          }
        }
      } else {
        // If the device already exists, check if the incarnation matches.
        if (it->second != device.incarnation()) {
          gr->status = errors::FailedPrecondition(
              "Device ", device.name(),
              " current incarnation doesn't match with one in the group. This "
              "usually means this worker has restarted but the collective "
              "leader hasn't, or this worker connects to a wrong cluster.");
        }
      }
    }

    if (gr->status.ok()) {
      // If the group is not yet complete, queue to wait for it.
      VLOG(2) << "group_size " << gr->group.group_size << " set size "
              << gr->group.members.size() << " gr " << gr;

      if (gr->group.members.size() < gr->group.group_size) {
        gr->pending_done.push_back(std::move(done));
        gr->pending_params.push_back(group_params);
        return;
      }
      CHECK_EQ(gr->group.members.size(), gr->group.group_size);
      // We get a full group. Fill in remaining fields in gr->group.
      auto st = CheckUserSpecifiedRanks(gr->group.members);
      if (!st.ok()) {
        gr->status = st;
      }
      if (new_device) {
        FinishGroup(gr);
      }
      // Copy to all pending CollGroupParams;
      *group_params = gr->group;
      for (auto* params : gr->pending_params) {
        *params = gr->group;
      }
    }
    // At this point, we either have a full group, or an error status.  Ensure
    // that all callbacks are invoked with the appropriate status.
    to_be_called.swap(gr->pending_done);
    gr->pending_params.clear();
    status = gr->status;
  }
  done(status);
  for (int i = 0; i < to_be_called.size(); ++i) {
    to_be_called[i](status);
  }
}

namespace {
struct DevRec {
  string task;
  string device;
  int original_rank;
  int local_rank;
  int global_rank;
  const DeviceLocality* locality;
};
typedef std::unordered_map<string, DevRec> TaskDeviceMap;
typedef std::unordered_map<string, TaskDeviceMap> GlobalDeviceMap;

// Create a populated GlobalDeviceMap from CollInstanceParams and localities.
GlobalDeviceMap BuildDevRecs(const CollGroupParams& gp) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_7(mht_7_v, 503, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "BuildDevRecs");

  GlobalDeviceMap gdm;
  CHECK_EQ(gp.members.size(), gp.members.size());
  for (int i = 0; i < gp.members.size(); ++i) {
    TaskDeviceMap& tdm = gdm[gp.members[i].task];
    DevRec* dr = &tdm[gp.members[i].device.name()];
    dr->task = gp.members[i].task;
    dr->device = gp.members[i].device.name();
    dr->original_rank = i;
    dr->local_rank = 0;   // Will be populated later by OrderTaskDeviceMap.
    dr->global_rank = 0;  // Will be populated later by EstablishGlobalRank.
    dr->locality = &gp.members[i].device.locality();
  }
  return gdm;
}

bool ParseRingOrder(const string& gpu_ring_order_str, TaskDeviceMap* tdm) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("gpu_ring_order_str: \"" + gpu_ring_order_str + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_8(mht_8_v, 523, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "ParseRingOrder");

  std::vector<string> split_gpu_ring_order_str =
      str_util::Split(gpu_ring_order_str, ',');
  if (split_gpu_ring_order_str.size() != tdm->size()) return false;

  // gpu id -> local rank
  gtl::FlatMap<int32, int32> gpu_ranks;
  for (int32_t rank = 0;
       rank < static_cast<int32>(split_gpu_ring_order_str.size()); ++rank) {
    int32_t tmp;
    if (strings::safe_strto32(split_gpu_ring_order_str[rank], &tmp)) {
      gpu_ranks[tmp] = rank;
    } else {
      return false;
    }
  }

  for (auto& tdm_it : *tdm) {
    DeviceNameUtils::ParsedName parsed_name;
    DevRec* dr = &tdm_it.second;
    if (!DeviceNameUtils::ParseFullName(dr->device, &parsed_name)) {
      return false;
    }
    auto rank_it = gpu_ranks.find(parsed_name.id);
    if (rank_it == gpu_ranks.end()) return false;
    dr->local_rank = rank_it->second;
  }
  VLOG(2) << "Assigned local ranks based on ring order " << gpu_ring_order_str;
  return true;
}

void OrderTaskDeviceMap(const string& gpu_ring_order, TaskDeviceMap* tdm) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("gpu_ring_order: \"" + gpu_ring_order + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_9(mht_9_v, 558, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "OrderTaskDeviceMap");

  CHECK_GT(tdm->size(), 0);  // Should never be called with 0 devices

  // If a valid ring order has been passed in via ConfigProto, use that.
  if (ParseRingOrder(gpu_ring_order, tdm)) return;

  // Either no ring order was passed in, or the format was unexpected.
  // We now assign a ring order based on link strengths.  Note that this
  // algorithm is not optimal and may not always find the best ring order.
  int least_rank = -1;
  string next_device;
  std::set<string> selected;
  // Starting device is one with the least initial rank.
  for (const auto& it : *tdm) {
    if (least_rank < 0 || it.second.original_rank < least_rank) {
      least_rank = it.second.original_rank;
      next_device = it.second.device;
    }
  }
  CHECK_GE(least_rank, 0);
  DeviceNameUtils::ParsedName parsed_name;
  CHECK(DeviceNameUtils::ParseFullName(next_device, &parsed_name));
  // NOTE: InterconnectLink has only a device_id, nothing more, so for
  // the time being if there's more than one device at a task we
  // assume they're all GPUs.

  int next_rank = 0;
  while (true) {
    selected.insert(next_device);
    auto next_dev_it = tdm->find(next_device);
    CHECK(next_dev_it != tdm->end());
    DevRec* dr = &next_dev_it->second;
    dr->local_rank = next_rank;
    ++next_rank;
    if (selected.size() == tdm->size()) {
      break;
    }
    // For the present time we assume Locality links only cover GPUs.
    // For multiple CPUs, just take them in order.
    const InterconnectLink* best_link = nullptr;
    if (parsed_name.type == "GPU") {
      for (const InterconnectLink& il : dr->locality->links().link()) {
        parsed_name.id = il.device_id();
        string endpoint_device =
            DeviceNameUtils::ParsedNameToString(parsed_name);
        // Skip the device if we've already seen it.
        if (selected.find(endpoint_device) != selected.end()) {
          continue;
        }
        // Skip the device if it is not participating in this collective
        // instance.
        if (tdm->find(endpoint_device) == tdm->end()) {
          continue;
        }
        if (best_link == nullptr || il.strength() > best_link->strength()) {
          best_link = &il;
        }
      }
    }
    if (best_link != nullptr) {
      // Follow the best edge
      parsed_name.id = best_link->device_id();
      next_device = DeviceNameUtils::ParsedNameToString(parsed_name);
    } else {
      // No good edges, alas. Pick the lowest initial rank among remaining
      // devices.
      least_rank = -1;
      for (const auto& it : *tdm) {
        if (selected.find(it.second.device) != selected.end()) {
          continue;
        }
        if (least_rank < 0 || it.second.original_rank < least_rank) {
          least_rank = it.second.original_rank;
          next_device = it.second.device;
        }
      }
      CHECK_GE(least_rank, 0);
    }
  }
}

// The first time a CollGroupParams is established for a group we compute a good
// rank order for all the devices in the group, that is appropriate for a ring
// algorithm.
GlobalDeviceMap EstablishGlobalRank(const CollGroupParams& gp,
                                    const string& gpu_ring_order) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("gpu_ring_order: \"" + gpu_ring_order + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_10(mht_10_v, 647, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "EstablishGlobalRank");

  VLOG(1) << "EstablishGlobalRank";
  GlobalDeviceMap gdm = BuildDevRecs(gp);
  for (auto& iter : gdm) {
    TaskDeviceMap& tdm = iter.second;
    OrderTaskDeviceMap(gpu_ring_order, &tdm);
  }
  // Connect the global rank order by the lexicographical order of the tasks.
  std::set<string> tasks;
  for (const CollGroupMember& member : gp.members) {
    tasks.insert(member.task);
  }
  int next_rank = 0;
  for (const string& task : tasks) {
    TaskDeviceMap* tdm = &gdm[task];
    for (auto& it : *tdm) {
      it.second.global_rank = it.second.local_rank + next_rank;
    }
    next_rank += tdm->size();
  }
  return gdm;
}

// Count the devices associated with each task and set
// gp->same_num_devices_per_task.  Requires gp->task_names
// be sorted.
void SetDevPerTask(CollGroupParams* gp) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_11(mht_11_v, 676, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "SetDevPerTask");

  gp->num_devices_per_task.clear();
  for (const CollGroupMember& member : gp->members) {
    gp->num_devices_per_task[member.task]++;
  }
  gp->same_num_devices_per_task = false;
  int dev_per_task = -1;
  for (const auto& task_dev : gp->num_devices_per_task) {
    if (dev_per_task == -1) {
      dev_per_task = task_dev.second;
    } else if (dev_per_task != task_dev.second) {
      return;
    }
  }
  gp->same_num_devices_per_task = true;
}

}  // namespace

void CollectiveParamResolverLocal::FinishGroup(GroupRec* gr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_12(mht_12_v, 698, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::FinishGroup");

  // Populate group member task and is_local.
  for (CollGroupMember& member : gr->group.members) {
    member.task = TaskNameFromDeviceName(member.device.name());
    member.is_local = member.task == task_name_;
  }
  // Establish the order of the members by considering localities of all
  // devices.
  CompleteDefaultRanking(&gr->group);
  SetDevPerTask(&gr->group);
  gr->group.num_tasks =
      static_cast<int32>(gr->group.num_devices_per_task.size());
}

void CollectiveParamResolverLocal::CancelGroup(int32 group_key) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_13(mht_13_v, 715, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::CancelGroup");

  std::vector<StatusCallback> pending_done;
  GroupRec* gr = nullptr;
  {
    mutex_lock l(group_mu_);
    auto it = group_table_.find(group_key);
    if (it == group_table_.end()) {
      return;
    }
    gr = it->second.get();
  }
  {
    mutex_lock l(gr->mu);
    if (gr->group.members.size() == gr->group.group_size) {
      // The group is already complete. There's no need to cancel.
      return;
    }
    gr->status = errors::Cancelled("group is cancelled");
    pending_done.swap(gr->pending_done);
    gr->pending_params.clear();
  }
  for (const StatusCallback& done : pending_done) {
    done(errors::Cancelled("group is cancelled"));
  }
}

void CollectiveParamResolverLocal::SetDefaultRank(const string& device,
                                                  CollectiveParams* cp) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_14(mht_14_v, 746, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::SetDefaultRank");

  CHECK_EQ(cp->group.group_size, cp->group.members.size()) << cp->ToString();
  for (int i = 0; i < cp->group.group_size; ++i) {
    if (cp->group.members[i].device.name() == device) {
      cp->default_rank = i;
    }
    // Set member rank to default rank if not user specified.
    if (cp->group.members[i].rank == -1) {
      cp->group.members[i].rank = i;
    }
  }
}

void CollectiveParamResolverLocal::InitInstanceSharedParams(
    const CollectiveParams* cp, InstanceRec* ir) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_15(mht_15_v, 763, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::InitInstanceSharedParams");

  ir->shared->instance = cp->instance;
  ir->shared->default_rank = -1;
}

// NOTE(ayushd): The DeviceLocality objects in attributes will have LocalLinks
// to all devices that they are physically connected to and visible to the
// TensorFlow runtime.  This set of devices may be a superset of the devices
// participating in this instance of collectives.
void CollectiveParamResolverLocal::CompleteDefaultRanking(CollGroupParams* gp) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_16(mht_16_v, 775, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::CompleteDefaultRanking");

  // Sort gp->member to avoid indeterminism.
  std::sort(gp->members.begin(), gp->members.end(),
            [](const CollGroupMember& lhs, const CollGroupMember& rhs) {
              return lhs.device.name() < rhs.device.name();
            });
  // Establish an instance-specific default rank order for devices
  // based on localities.  This rank order should be a good ring
  // order, if possible.
  GlobalDeviceMap gdm = EstablishGlobalRank(*gp, gpu_ring_order_);
  // Reflect the new global ranking on shared
  std::vector<CollGroupMember> new_members(gp->group_size);
  for (const auto& git : gdm) {
    const TaskDeviceMap& tdm = git.second;
    for (const auto& tit : tdm) {
      const DevRec& dr = tit.second;
      new_members[dr.global_rank] = std::move(gp->members[dr.original_rank]);
    }
  }

  if (VLOG_IS_ON(2)) {
    string buf;
    for (const auto& m : new_members)
      strings::StrAppend(&buf, "\n", m.device.name());
    VLOG(2) << "Optimized device order for group " << gp->group_key << ": "
            << buf;
  }
  gp->members = std::move(new_members);
}

CollectiveParamResolverLocal::InstanceRec*
CollectiveParamResolverLocal::GetOrCreateInstanceRec(CollectiveParams* cp,
                                                     bool* created) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_17(mht_17_v, 810, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::GetOrCreateInstanceRec");

  *created = false;
  InstanceRec* irec = nullptr;
  {
    mutex_lock l(instance_mu_);
    auto group_it = instance_table_.find(cp->group.group_key);
    if (group_it != instance_table_.end()) {
      auto instance_it = group_it->second.find(cp->instance.instance_key);
      if (instance_it != group_it->second.end()) {
        irec = instance_it->second.get();
      }
    }
    if (irec == nullptr) {
      // Create new InstanceRec.
      irec = new InstanceRec;
      *created = true;
      {
        mutex_lock il(irec->mu);
        irec->known.resize(cp->group.group_size, false);
      }
      InitInstanceSharedParams(cp, irec);
      instance_table_[cp->group.group_key][cp->instance.instance_key].reset(
          irec);
    }
  }
  Status status;
  {
    mutex_lock l(status_mu_);
    status = status_;
  }
  if (!status.ok()) {
    mutex_lock l(irec->mu);
    irec->status = status;
  }
  return irec;
}

Status CollectiveParamResolverLocal::LookupGroup(int32_t group_key,
                                                 CollGroupParams* group) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_18(mht_18_v, 851, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::LookupGroup");

  mutex_lock l(group_mu_);
  auto group_rec = group_table_.find(group_key);
  if (group_rec == group_table_.end()) {
    return errors::InvalidArgument("Group ", group_key,
                                   " is not "
                                   "initialized. Please call group "
                                   "initialization op first before invoking "
                                   "collective op.");
  }
  mutex_lock lock(group_rec->second->mu);
  if (!group_rec->second->status.ok()) {
    return errors::FailedPrecondition(
        "Failed to run collective due to "
        "unsuccessful group initialization. "
        "Group initialization failed with error ",
        group_rec->second->status.ToString());
  }
  *group = group_rec->second->group;
  return Status::OK();
}

void CollectiveParamResolverLocal::CompleteParamsAsync(
    const DeviceAttributes& device, CollectiveParams* cp,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_19(mht_19_v, 878, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::CompleteParamsAsync");

  VLOG(1) << "CompleteParams local " << device.name() << " for " << cp << ": "
          << cp->ToString();
  if (cp->run_group_initialization) {
    CompleteGroupLocal(device, &cp->group, cancel_mgr,
                       [this, device, cp, done](const Status& s) {
                         if (s.ok()) {
                           CompleteInstanceLocal(device.name(), cp, done);
                         } else {
                           done(s);
                         }
                       });
  } else {
    // For Collective V3 ops, group is already initialized. Fetch attributes
    // for the already initialized group to pass to Insitance initialization.
    const auto s = LookupGroup(cp->group.group_key, &cp->group);
    if (s.ok()) {
      CompleteInstanceLocal(device.name(), cp, done);
    } else {
      done(s);
    }
  }
}

void CollectiveParamResolverLocal::CompleteInstanceAsync(
    const CompleteInstanceRequest* request, CompleteInstanceResponse* response,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_20(mht_20_v, 907, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::CompleteInstanceAsync");

  done(
      errors::Internal("CompleteInstance is not implemented by "
                       "CollectiveParamResolverLocal which is "
                       "intended only for non-distributed deployment."));
}

// TODO(b/111897089): we need a better way to pick the collective
// implementation.  The ideal way would depend upon the topology and link
// strength before picking a particular implementation.
void CollectiveParamResolverLocal::AssignCollectiveType(CollectiveParams* cp) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_21(mht_21_v, 920, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::AssignCollectiveType");

  // We use the NCCL implementation if this is an environment which supports
  // NCCL, i.e. `LookupParamResolverInstance` for `NcclReduce` returns OK, and
  // also if indicated either in `ConfigProto` or `communication_hint`.
  //
  // After enough testing, we may simplify this logic to use NCCL whenever
  // available.
  CollectiveImplementationInterface* col_impl;
  bool use_nccl =
      (nccl_ || cp->instance.impl_details.communication_hint == "nccl") &&
      cp->group.device_type == DEVICE_GPU &&
      CollectiveRegistry::LookupParamResolverInstance("NcclReduce", &col_impl)
          .ok();
  cp->instance.impl_details.collective_name = GetCollectiveName(cp, use_nccl);
  VLOG(1) << "AssignCollectiveType "
          << cp->instance.impl_details.collective_name;
}

void CollectiveParamResolverLocal::CompleteInstanceLocal(
    const string& device, CollectiveParams* cp, const StatusCallback& done) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_22(mht_22_v, 943, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::CompleteInstanceLocal");

  VLOG(1) << "CompleteInstanceLocal " << device
          << " instance_key: " << cp->instance.instance_key << " group_key "
          << cp->group.group_key;

  bool created_irec;
  InstanceRec* ir = GetOrCreateInstanceRec(cp, &created_irec);
  if (!created_irec) {
    // Check that the preexisting IRec is consistent with the params passed into
    // this invocation.
    if (ir->shared->instance.type != cp->instance.type ||
        ir->shared->instance.data_type != cp->instance.data_type) {
      done(errors::Internal("Collective instance ", cp->instance.instance_key,
                            " expected type ", ir->shared->instance.type,
                            " and data_type ", ir->shared->instance.data_type,
                            " but got type ", cp->instance.type,
                            " and data_type ", cp->instance.data_type));
      return;
    }
  }
  CompleteInstanceFromInitializedIRec(device, cp, ir, done);
}

void CollectiveParamResolverLocal::CompleteInstanceFromInitializedIRec(
    const string& device, CollectiveParams* cp, InstanceRec* ir,
    const StatusCallback& done) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_23(mht_23_v, 972, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::CompleteInstanceFromInitializedIRec");

  auto expected_shape = cp->instance.shape;
  Status status;
  // Populate the fields common across instance.
  {
    mutex_lock l(ir->mu);
    status = ir->status;
    if (status.ok()) {
      // custom operator= does a deep copy.
      cp->instance = ir->shared->instance;
    }
  }
  if (!status.ok()) {
    done(status);
    return;
  }
  if (expected_shape != cp->instance.shape) {
    done(errors::InvalidArgument(
        "Shape mismatch in the collective instance ", cp->instance.instance_key,
        ". Op at device ", device, " expected shape ",
        expected_shape.DebugString(), " but another member in the group ",
        "expected shape ", cp->instance.shape.DebugString(), ". This is likely",
        " due to different input shapes at different members of the collective",
        " op."));
    return;
  }
  // Populate the fields common across task.
  AssignCollectiveType(cp);
  SetDefaultRank(device, cp);

  CollectiveImplementationInterface* col_impl;
  status = CollectiveRegistry::LookupParamResolverInstance(
      cp->instance.impl_details.collective_name, &col_impl);
  if (!status.ok()) {
    done(status);
    return;
  }

  //  We may need to wait for the group, if this is a broadcast, for source
  //  discovery.
  if (cp->instance.type == BROADCAST_COLLECTIVE) {
    WaitForGroup(ir, cp, [col_impl, ir, device, cp, done](InstanceRec* irec) {
      Status s;
      if (ir != irec) {
        s = errors::Internal("Expected ir ", ir, " and irec ", irec,
                             " to be equal");
      } else {
        mutex_lock l(irec->mu);
        s = irec->status;
        cp->source_rank = irec->source_rank;
      }
      if (s.ok()) {
        s = col_impl->InitializeCollectiveParams(cp);
      }
      done(s);
    });
  } else {
    done(col_impl->InitializeCollectiveParams(cp));
  }
}

void CollectiveParamResolverLocal::WaitForGroup(InstanceRec* ir,
                                                CollectiveParams* cp,
                                                const IRConsumer& f) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_24(mht_24_v, 1038, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::WaitForGroup");

  std::vector<IRConsumer> ready_waiters;
  do {
    mutex_lock l(ir->mu);
    if (!ir->status.ok()) {
      break;
    }
    CHECK_EQ(cp->group.group_size, ir->known.size());
    CHECK_GE(cp->default_rank, 0);
    if (!ir->known[cp->default_rank]) {
      ir->known[cp->default_rank] = true;
      ++ir->known_count;
      if (cp->is_source) {
        // Initialize source rank.
        if (ir->source_rank >= 0) {
          ir->status = errors::Internal("Instance ", cp->instance.instance_key,
                                        " already has source ", ir->source_rank,
                                        ", received second claim from ",
                                        cp->default_rank);
        } else {
          ir->source_rank = cp->default_rank;
        }
      }
    }
    if (ir->known_count < cp->group.group_size) {
      ir->known_waiters.push_back(f);
      return;
    }
    CHECK_EQ(ir->known_count, cp->group.group_size);
    if (ir->source_rank < 0) {
      // NOTE(ayushd): changing the error message below would also require
      // updating CompleteParamsBroadcastForgotSend test in
      // CollectiveParamResolverLocalTest.
      ir->status =
          errors::Internal("Instance ", cp->instance.instance_key,
                           " found no source for broadcast.  This "
                           "could mean that there were group_size=",
                           ir->known_count, " BcastRecvs but no BcastSend.");
    }
    if (!ir->known_waiters.empty()) {
      ready_waiters = std::move(ir->known_waiters);
    }
  } while (false);
  f(ir);
  for (auto& f : ready_waiters) {
    f(ir);
  }
}

void CollectiveParamResolverLocal::StartAbort(const Status& s) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_25(mht_25_v, 1090, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::StartAbort");

  {
    mutex_lock l(status_mu_);
    if (!status_.ok()) {
      VLOG(2) << "CollectiveParamResolverLocal already aborted. Ignoring "
                 "subsequent abortion with status: "
              << s;
      return;
    }
    status_ = s;
  }
  StartAbortLocal(s);
}

void CollectiveParamResolverLocal::StartAbortLocal(const Status& s) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_localDTcc mht_26(mht_26_v, 1107, "", "./tensorflow/core/common_runtime/collective_param_resolver_local.cc", "CollectiveParamResolverLocal::StartAbortLocal");

  std::vector<StatusCallback> pending_done;
  {
    mutex_lock l(group_mu_);
    for (const auto& item : group_table_) {
      GroupRec* gr = item.second.get();
      {
        mutex_lock gl(gr->mu);
        gr->status = s;
        for (auto& done : gr->pending_done) {
          pending_done.push_back(std::move(done));
        }
        gr->pending_done.clear();
        gr->pending_params.clear();
      }
    }
  }
  for (const StatusCallback& done : pending_done) {
    done(s);
  }
  std::vector<InstanceRec*> instances;
  {
    mutex_lock l(instance_mu_);
    for (const auto& group_entry : instance_table_) {
      for (const auto& item : group_entry.second) {
        instances.push_back(item.second.get());
      }
    }
  }
  for (InstanceRec* ir : instances) {
    std::vector<IRConsumer> known_waiters;
    {
      mutex_lock il(ir->mu);
      ir->status = s;
      known_waiters.swap(ir->known_waiters);
    }
    for (const IRConsumer& done : known_waiters) {
      done(ir);
    }
  }
}

}  // namespace tensorflow
