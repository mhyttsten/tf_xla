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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_metadataDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_metadataDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_metadataDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_module_metadata.h"

#include <algorithm>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

StatusOr<HloPassMetadata*> HloModuleMetadata::GetCurrentHloPassMetadata() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_metadataDTcc mht_0(mht_0_v, 194, "", "./tensorflow/compiler/xla/service/hlo_module_metadata.cc", "HloModuleMetadata::GetCurrentHloPassMetadata");

  if (running_passes_.empty()) {
    return NotFound(
        "HloPassMetadata for currently running pass not found, either because "
        "the pass did not call RecordPassStart or because a pass is "
        "creating/switching modules without using "
        "HloModuleGroup::ReplaceModule.");
  }
  return running_passes_.back();
}

Status HloModuleMetadata::MutateCurrentHloPassMetadata(
    const std::function<void(HloPassMetadata*)>& mutator) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_metadataDTcc mht_1(mht_1_v, 209, "", "./tensorflow/compiler/xla/service/hlo_module_metadata.cc", "HloModuleMetadata::MutateCurrentHloPassMetadata");

  TF_ASSIGN_OR_RETURN(HloPassMetadata * pass_metadata,
                      GetCurrentHloPassMetadata());
  mutator(pass_metadata);
  return Status::OK();
}

void HloModuleMetadata::RecordPassStart() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_metadataDTcc mht_2(mht_2_v, 219, "", "./tensorflow/compiler/xla/service/hlo_module_metadata.cc", "HloModuleMetadata::RecordPassStart");

  HloPassMetadata* pass_metadata = module_metadata_.add_pass_metadata();
  pass_metadata->set_pass_id(next_pass_id_++);
  pass_metadata->set_start_timestamp_usec(env_->NowMicros());
  running_passes_.push_back(pass_metadata);
}

Status HloModuleMetadata::RecordPassEnd() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_metadataDTcc mht_3(mht_3_v, 229, "", "./tensorflow/compiler/xla/service/hlo_module_metadata.cc", "HloModuleMetadata::RecordPassEnd");

  TF_ASSIGN_OR_RETURN(HloPassMetadata * pass_metadata,
                      GetCurrentHloPassMetadata());
  pass_metadata->set_end_timestamp_usec(env_->NowMicros());
  running_passes_.pop_back();
  return Status::OK();
}

void HloModuleMetadata::set_prepartitioning_metadata(
    const HloModuleMetadata& prepartitioning_metadata) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_metadataDTcc mht_4(mht_4_v, 241, "", "./tensorflow/compiler/xla/service/hlo_module_metadata.cc", "HloModuleMetadata::set_prepartitioning_metadata");

  module_metadata_.set_original_module_id(
      prepartitioning_metadata.proto().canonical_module_id());
  prepartitioning_metadata_ = prepartitioning_metadata.proto();
  prepartitioning_metadata_->clear_pass_metadata();

  // Because HloPassMetadata represents the completion of a pass, metadata for
  // all currently running passes need to be moved over to the new module.
  absl::flat_hash_set<HloPassMetadata*> running_passes(
      prepartitioning_metadata.running_passes_.begin(),
      prepartitioning_metadata.running_passes_.end());
  for (const HloPassMetadata& pass_metadata :
       prepartitioning_metadata.proto().pass_metadata()) {
    if (running_passes.contains(&pass_metadata)) {
      HloPassMetadata* added_pass_metadata =
          module_metadata_.add_pass_metadata();
      *added_pass_metadata = pass_metadata;
      running_passes_.push_back(added_pass_metadata);
      next_pass_id_ =
          std::max(next_pass_id_,
                   static_cast<int64_t>(added_pass_metadata->pass_id()) + 1);
    } else {
      *prepartitioning_metadata_->add_pass_metadata() = pass_metadata;
    }
  }
}

}  // namespace xla
