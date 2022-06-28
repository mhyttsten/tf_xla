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
class MHTracer_DTPStensorflowPScorePSdataPSservicePScommonDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePScommonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePScommonDTcc() {
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
#include "tensorflow/core/data/service/common.h"

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

namespace {
constexpr const char kAuto[] = "AUTO";
constexpr const char kAny[] = "ANY";
constexpr const char kLocal[] = "LOCAL";

constexpr const char kColocated[] = "COLOCATED";
constexpr const char kRemote[] = "REMOTE";
constexpr const char kHybrid[] = "HYBRID";
}  // namespace

bool IsNoShard(const ProcessingModeDef& processing_mode) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScommonDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/data/service/common.cc", "IsNoShard");

  return processing_mode.sharding_policy() == ProcessingModeDef::OFF;
}

bool IsDynamicShard(const ProcessingModeDef& processing_mode) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScommonDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/data/service/common.cc", "IsDynamicShard");

  return processing_mode.sharding_policy() == ProcessingModeDef::DYNAMIC;
}

bool IsStaticShard(const ProcessingModeDef& processing_mode) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScommonDTcc mht_2(mht_2_v, 223, "", "./tensorflow/core/data/service/common.cc", "IsStaticShard");

  return processing_mode.sharding_policy() == ProcessingModeDef::FILE ||
         processing_mode.sharding_policy() == ProcessingModeDef::DATA ||
         processing_mode.sharding_policy() == ProcessingModeDef::FILE_OR_DATA ||
         processing_mode.sharding_policy() == ProcessingModeDef::HINT;
}

Status ValidateProcessingMode(const ProcessingModeDef& processing_mode) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScommonDTcc mht_3(mht_3_v, 233, "", "./tensorflow/core/data/service/common.cc", "ValidateProcessingMode");

  if (!IsNoShard(processing_mode) && !IsDynamicShard(processing_mode) &&
      !IsStaticShard(processing_mode)) {
    return errors::Internal(
        "ProcessingMode ", processing_mode.ShortDebugString(),
        " does not "
        "specify a valid sharding policy. Please add the policy to either "
        "`IsDynamicShard` or `IsStaticShard` (i.e., auto-shard).");
  }
  return Status::OK();
}

StatusOr<AutoShardPolicy> ToAutoShardPolicy(
    const ProcessingModeDef::ShardingPolicy sharding_policy) {
  switch (sharding_policy) {
    case ProcessingModeDef::FILE:
      return AutoShardPolicy::FILE;
    case ProcessingModeDef::DATA:
      return AutoShardPolicy::DATA;
    case ProcessingModeDef::FILE_OR_DATA:
      return AutoShardPolicy::AUTO;
    case ProcessingModeDef::HINT:
      return AutoShardPolicy::HINT;
    case ProcessingModeDef::DYNAMIC:
    case ProcessingModeDef::OFF:
      return AutoShardPolicy::OFF;
    default:
      return errors::Internal(
          "tf.data service sharding policy ",
          ProcessingModeDef::ShardingPolicy_Name(sharding_policy),
          " is not convertible to a valid auto-shard policy. If you're "
          "defining a new sharding policy, please update the policy mapping.");
  }
}

StatusOr<TargetWorkers> ParseTargetWorkers(absl::string_view s) {
  std::string str_upper = absl::AsciiStrToUpper(s);
  if (str_upper.empty() || str_upper == kAuto) {
    return TARGET_WORKERS_AUTO;
  }
  if (str_upper == kAny) {
    return TARGET_WORKERS_ANY;
  }
  if (str_upper == kLocal) {
    return TARGET_WORKERS_LOCAL;
  }
  return errors::InvalidArgument("Unrecognized target workers: ", s);
}

std::string TargetWorkersToString(TargetWorkers target_workers) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScommonDTcc mht_4(mht_4_v, 285, "", "./tensorflow/core/data/service/common.cc", "TargetWorkersToString");

  switch (target_workers) {
    case TARGET_WORKERS_AUTO:
      return kAuto;
    case TARGET_WORKERS_ANY:
      return kAny;
    case TARGET_WORKERS_LOCAL:
      return kLocal;
    default:
      DCHECK(false);
      return "UNKNOWN";
  }
}

StatusOr<DeploymentMode> ParseDeploymentMode(absl::string_view s) {
  std::string str_upper = absl::AsciiStrToUpper(s);
  if (str_upper == kColocated) {
    return DEPLOYMENT_MODE_COLOCATED;
  }
  if (str_upper == kRemote) {
    return DEPLOYMENT_MODE_REMOTE;
  }
  if (str_upper == kHybrid) {
    return DEPLOYMENT_MODE_HYBRID;
  }
  return errors::InvalidArgument("Invalid tf.data service deployment mode: ", s,
                                 ". Supported modes are "
                                 "COLOCATED, REMOTE, and HYBRID.");
}

bool IsPreemptedError(const Status& status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScommonDTcc mht_5(mht_5_v, 318, "", "./tensorflow/core/data/service/common.cc", "IsPreemptedError");

  return errors::IsAborted(status) || errors::IsCancelled(status) ||
         errors::IsUnavailable(status);
}
}  // namespace data
}  // namespace tensorflow
