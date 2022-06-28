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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_utilDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/tf2xla/sharding_util.h"

#include "absl/strings/match.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {
const char kDeviceSuffixReplicatedCore[] = "REPLICATED_CORE";
const char kShardingAttribute[] = "_XlaSharding";
const char kShardingOpAttribute[] = "sharding";
}  // namespace

namespace {
xla::OpMetadata CreateOpMetadata(const std::string& op_type,
                                 const std::string& op_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_type: \"" + op_type + "\"");
   mht_0_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_utilDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/tf2xla/sharding_util.cc", "CreateOpMetadata");

  xla::OpMetadata metadata;
  metadata.set_op_type(op_type);
  metadata.set_op_name(op_name);
  return metadata;
}

void AssignOpMetadataToSharding(xla::OpSharding& sharding,
                                const string& op_type, const string& op_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op_type: \"" + op_type + "\"");
   mht_1_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_utilDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/tf2xla/sharding_util.cc", "AssignOpMetadataToSharding");

  auto metadata = CreateOpMetadata(op_type, op_name);
  if (sharding.type() == xla::OpSharding::TUPLE) {
    for (auto& sharding_element : *sharding.mutable_tuple_shardings()) {
      *sharding_element.add_metadata() = metadata;
    }
  } else {
    *sharding.add_metadata() = metadata;
  }
}

Status CoreOutOfRangeError(int core, int num_cores_per_replica) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_utilDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/tf2xla/sharding_util.cc", "CoreOutOfRangeError");

  return errors::InvalidArgument(
      "Invalid replicated core id: ", core,
      "; num_cores_per_replica=", num_cores_per_replica);
}
}  // namespace

StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromDevice(
    const string& device_name, int num_cores_per_replica,
    absl::optional<xla::OpSharding> explicit_sharding,
    absl::optional<xla::OpMetadata> metadata) {
  if (device_name.empty()) {
    return explicit_sharding;
  }
  DeviceNameUtils::ParsedName parsed_device;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed_device)) {
    return errors::InvalidArgument("Malformed assigned device '", device_name,
                                   "'");
  }

  if (explicit_sharding.has_value()) {
    return explicit_sharding;
  } else if (!parsed_device.has_type || !parsed_device.has_id ||
             !absl::StrContains(parsed_device.type,
                                kDeviceSuffixReplicatedCore)) {
    return absl::optional<xla::OpSharding>();
  } else {
    const int core = parsed_device.id;
    if (core < 0 || core >= num_cores_per_replica) {
      return CoreOutOfRangeError(core, num_cores_per_replica);
    }
    auto sharding = xla::sharding_builder::AssignDevice(core);
    if (metadata.has_value()) {
      *sharding.add_metadata() = metadata.value();
    }
    return absl::optional<xla::OpSharding>(sharding);
  }
}

StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromDevice(
    const NodeDef& node_def, int num_cores_per_replica, bool add_metadata) {
  const string& device_name = node_def.device();
  TF_ASSIGN_OR_RETURN(absl::optional<xla::OpSharding> sharding,
                      GetShardingFromNodeDef(node_def, add_metadata));
  return ParseShardingFromDevice(
      device_name, num_cores_per_replica, sharding,
      add_metadata ? absl::optional<xla::OpMetadata>(
                         CreateOpMetadata(node_def.op(), node_def.name()))
                   : absl::nullopt);
}

StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromDevice(
    const Node& node, int num_cores_per_replica, bool add_metadata) {
  string device_name = node.assigned_device_name();
  if (device_name.empty()) {
    device_name = node.requested_device();
  }
  TF_ASSIGN_OR_RETURN(absl::optional<xla::OpSharding> sharding,
                      GetShardingFromNodeDef(node.def(), add_metadata));
  return ParseShardingFromDevice(
      device_name, num_cores_per_replica, sharding,
      add_metadata ? absl::optional<xla::OpMetadata>(
                         CreateOpMetadata(node.type_string(), node.name()))
                   : absl::nullopt);
}

StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromEdgeSource(
    const Edge& edge, int num_cores_per_replica, bool add_metadata) {
  if (edge.src() == nullptr) {
    return tensorflow::errors::InvalidArgument(
        "Null src for ParseShardingFromEdgeSource edge=", edge.DebugString());
  }
  TF_ASSIGN_OR_RETURN(absl::optional<xla::OpSharding> sharding,
                      ParseShardingFromDevice(
                          *edge.src(), num_cores_per_replica, add_metadata));
  if (sharding.has_value() &&
      sharding.value().type() == xla::OpSharding::TUPLE) {
    if (edge.src_output() < 0 ||
        edge.src_output() >= sharding.value().tuple_shardings_size()) {
      return tensorflow::errors::InvalidArgument(
          "Tuple index out of bound: edge=", edge.DebugString(),
          " sharding=", sharding->DebugString());
    }
    absl::optional<xla::OpSharding> subsharding =
        sharding.value().tuple_shardings(edge.src_output());
    return subsharding;
  }
  return sharding;
}

void SetShardingDeviceAssignmentFromNode(const Node& src, Node* dst) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSsharding_utilDTcc mht_3(mht_3_v, 322, "", "./tensorflow/compiler/tf2xla/sharding_util.cc", "SetShardingDeviceAssignmentFromNode");

  string device_name = src.assigned_device_name();
  if (device_name.empty()) {
    device_name = src.requested_device();
  }
  dst->set_assigned_device_name(device_name);
  if (const AttrValue* attr = src.attrs().Find(kShardingAttribute)) {
    dst->AddAttr(kShardingAttribute, *attr);
  }
}

namespace {

StatusOr<absl::optional<xla::OpSharding>> GetShardingFromNodeDefInternal(
    const NodeDef& node_def, bool add_metadata, const char* attribute) {
  if (!HasNodeAttr(node_def, attribute)) {
    return absl::optional<xla::OpSharding>();
  }
  string value;
  xla::OpSharding sharding;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, attribute, &value));
  if (!sharding.ParseFromString(value)) {
    return xla::InvalidArgument(
        "Experimental %s attribute was not a valid encoded xla::OpSharding "
        "proto.",
        attribute);
  }
  if (add_metadata) {
    AssignOpMetadataToSharding(sharding, node_def.op(), node_def.name());
  }
  return absl::optional<xla::OpSharding>(sharding);
}

}  // namespace

xla::StatusOr<absl::optional<xla::OpSharding>> GetShardingFromNodeDef(
    const NodeDef& node_def, bool add_metadata) {
  if (node_def.op() == "XlaSharding") {
    TF_ASSIGN_OR_RETURN(auto sharding,
                        GetShardingFromNodeDefInternal(node_def, add_metadata,
                                                       kShardingOpAttribute));
    if (sharding.has_value()) {
      return sharding;
    }
  }
  return GetShardingFromNodeDefInternal(node_def, add_metadata,
                                        kShardingAttribute);
}

}  // namespace tensorflow
