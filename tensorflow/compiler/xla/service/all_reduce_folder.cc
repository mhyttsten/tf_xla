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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_folderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_folderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_folderDTcc() {
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

#include "tensorflow/compiler/xla/service/all_reduce_folder.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/all_reduce_key.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"

namespace xla {
namespace {
// Folds the given two sets of non-empty replica groups into a single set if
// possible.
absl::optional<std::vector<ReplicaGroup>> FoldReplicaGroups(
    absl::Span<const ReplicaGroup> replica_groups0,
    absl::Span<const ReplicaGroup> replica_groups1) {
  // For a valid all-reduce with non-empty replica groups, the groups should
  // list each replica exactly once.
  int64_t num_replicas = 0;
  for (const ReplicaGroup &rg : replica_groups0) {
    for (int64_t id : rg.replica_ids()) {
      num_replicas = std::max(num_replicas, id);
    }
  }
  num_replicas++;

  // We will build, for each replica, the effective set of replicas which
  // contribute to the output of that replica by essentially tracing through
  // the 2 sets of replica groups.

  // For each replica, remember its replica group # from replica_group0
  std::vector<int> replica_group_no(num_replicas, -1);
  for (int group_no = 0; group_no < replica_groups0.size(); ++group_no) {
    for (int64_t id : replica_groups0[group_no].replica_ids()) {
      replica_group_no[id] = group_no;
    }
  }

  // For each replica, trace through the 2 replica groups to build the set of
  // contributing replicas for each replica's output. In an all-reduce, each
  // contributor can contribute only once, so if see a contributing replica more
  // than once, such replica groups cannot be folded.

  // Note: Using std::vector<bool> instead of flat_hash_set for contributor sets
  // since flat_hash_set cannot be used as a flat_hash_map key.
  // Map to a contributor set to its unique id.
  absl::flat_hash_map<std::vector<bool>, int64_t> contributor_set_id;

  // Map each replica to the unique id for the set of its contributors.
  std::vector<int64_t> contributing_replicas_set_id(num_replicas, 0);

  int64_t next_id = 1;
  for (const ReplicaGroup &rg : replica_groups1) {
    std::vector<bool> contributors(num_replicas, false);
    for (int64_t id : rg.replica_ids()) {
      int64_t group_no = replica_group_no[id];
      for (int64_t contrib : replica_groups0[group_no].replica_ids()) {
        // If the contributor already preset in the set, fail. As an example
        // rg0 = {0, 1}
        // rg1 = {0, 1}
        // In such a case, when processing id = 1 from rg0, replica #0 will
        // already be present, so the groups cannot be merged.
        if (contributors[contrib]) {
          return absl::nullopt;
        }
        contributors[contrib] = true;
      }
    }

    // Uniquefy the contributor sets by assigning a unique id to each unique
    // set.
    int64_t set_id;
    auto it = contributor_set_id.find(contributors);
    if (it != contributor_set_id.end()) {
      set_id = it->second;
    } else {
      set_id = next_id++;
      contributor_set_id[contributors] = set_id;
    }

    // All replica id in the group have the same set of contributors.
    for (int64_t id : rg.replica_ids()) {
      contributing_replicas_set_id[id] = set_id;
    }
  }

  // Now verify, for each unique set of contributors, whether for all of the
  // associated replica's have the same contributors. These unique sets now
  // become the folded replica groups.
  std::vector<ReplicaGroup> new_replica_groups;
  new_replica_groups.reserve(contributor_set_id.size());

  for (const auto &it : contributor_set_id) {
    const std::vector<bool> &contributors = it.first;
    const int64_t set_id = it.second;
    new_replica_groups.emplace_back();
    ReplicaGroup &group = new_replica_groups.back();
    for (int64_t replica = 0; replica < num_replicas; ++replica) {
      if (contributors[replica]) {
        if (contributing_replicas_set_id[replica] != set_id) {
          return absl::nullopt;
        }
        group.add_replica_ids(replica);
      }
    }
  }

  // Sort the replica groups by the first id for stable behavior. Otherwise,
  // groups are formed according to the order in the contributer_set_id map,
  // which is not stable.
  absl::c_sort(new_replica_groups,
               [](const ReplicaGroup &a, const ReplicaGroup &b) {
                 return a.replica_ids(0) < b.replica_ids(0);
               });
  return new_replica_groups;
}

}  // namespace

StatusOr<bool> AllReduceFolder::Run(HloModule *module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_folderDTcc mht_0(mht_0_v, 308, "", "./tensorflow/compiler/xla/service/all_reduce_folder.cc", "AllReduceFolder::Run");

  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1) << "Skip AllReduceFolder because the module contains all-reduce "
               "with constrained layouts";
    return false;
  }

  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (auto computation : module->computations()) {
    for (HloInstruction *inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kAllReduce ||
          inst->operand(0)->opcode() != HloOpcode::kAllReduce) {
        continue;
      }

      auto *ar0 = Cast<HloAllReduceInstruction>(inst->mutable_operand(0));
      auto *ar1 = Cast<HloAllReduceInstruction>(inst);

      if (ar0->user_count() != 1) {
        continue;
      }

      // Check if the 2 all-reduce instructions are compatible with the
      // exception of the replica groups.
      absl::optional<AllReduceKey> key0 = GetAllReduceKey(
          ar0, /*domain_map=*/nullptr, /*ignore_replica_groups=*/true);
      absl::optional<AllReduceKey> key1 = GetAllReduceKey(
          ar1, /*domain_map=*/nullptr, /*ignore_replica_groups=*/true);
      if (!key0 || !key1 || *key0 != *key1 || ar0->replica_groups().empty() ||
          ar1->replica_groups().empty()) {
        continue;
      }

      // Since both all-reduces have non-empty replica groups, they list all the
      // participants. We essentially build, for each participant, which replica
      // contributes to the result of second all-reduce for that participant.
      // For example, for the below sequence:
      //   ar0 = all-reduce(x)   replica_groups={{0,1},{2,3},{4,5},{6,7}}
      //   ar1 = all-reduce(ar0) replica_groups={{0,2},{1,3},{4,6},{5,7}}

      // ar1 output for replica 0 contains { x0, x1, x2, x3}, where x_i is the
      // value of x in replica i.
      // r1 = { x0, x1, x2, x3} as well.
      // After we have these sets, we check if these sets are compatible for
      // forming a new all-reduce.

      absl::optional<std::vector<ReplicaGroup>> new_replica_groups =
          FoldReplicaGroups(ar0->replica_groups(), ar1->replica_groups());
      if (!new_replica_groups) {
        continue;
      }
      absl::optional<int64_t> channel_id;
      if (ar0->channel_id()) {
        channel_id = next_channel_id++;
      }

      // Create new all-reduce and delete the 2 existing ones.
      HloInstruction *new_ar =
          computation->AddInstruction(HloInstruction::CreateAllReduce(
              ar0->shape(), ar0->operands(), ar0->to_apply(),
              *new_replica_groups, /*constrain_layout=*/false, channel_id,
              ar0->use_global_device_ids()));
      TF_RETURN_IF_ERROR(ar1->ReplaceAllUsesWith(new_ar));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar1));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar0));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
