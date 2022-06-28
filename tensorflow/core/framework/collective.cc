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
class MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc() {
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
#include "tensorflow/core/framework/collective.h"

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

namespace {
// A RegistrationInfo object stores a collective implementation registration
// details.  `factory` is used to create instances of the collective
// implementation.
struct RegistrationInfo {
  // This constructor also creates, and stores in `param_resolver_instance`,
  // what is effectively a static instance of the collective implementation.
  // During param resolution of collective ops we return this static instance.
  // The actual op execution gets a fresh instance using `factory`.
  RegistrationInfo(const string& n, CollectiveRegistry::Factory f)
      : name(n),
        factory(std::move(f)),
        param_resolver_instance(this->factory()) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("n: \"" + n + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/framework/collective.cc", "RegistrationInfo");
}
  string name;
  CollectiveRegistry::Factory factory;
  CollectiveImplementationInterface* param_resolver_instance;
};

std::vector<RegistrationInfo>* MutableCollectiveRegistry() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/framework/collective.cc", "MutableCollectiveRegistry");

  static std::vector<RegistrationInfo>* registry =
      new std::vector<RegistrationInfo>;
  return registry;
}
}  // namespace

string CollGroupRuntimeDetails::ToString() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/framework/collective.cc", "CollGroupRuntimeDetails::ToString");

  return strings::StrCat("CollGroupRuntimeDetails {communicator_key=",
                         absl::CEscape(communicator_key), "}");
}

string CollGroupParams::ToString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_3(mht_3_v, 235, "", "./tensorflow/core/framework/collective.cc", "CollGroupParams::ToString");

  string v = strings::StrCat(
      "CollGroupParams {group_key=", group_key, " group_size=", group_size,
      " device_type=", device_type.type_string(), " num_tasks=", num_tasks,
      " runtime_details=", runtime_details.ToString(), " devices {");
  for (const auto& m : members) {
    strings::StrAppend(&v, m.device.name(), ",");
  }
  strings::StrAppend(&v, "} num_devices_per_task={");
  for (const auto& dpt : num_devices_per_task) {
    strings::StrAppend(&v, dpt.first, ": ", dpt.second, ", ");
  }
  strings::StrAppend(&v, "}");
  return v;
}

CollInstanceParams& CollInstanceParams::operator=(
    const CollInstanceParams& other) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_4(mht_4_v, 255, "", "./tensorflow/core/framework/collective.cc", "=");

  if (this != &other) {
    instance_key = other.instance_key;
    type = other.type;
    data_type = other.data_type;
    shape = other.shape;
    impl_details.subdiv_offsets.assign(
        other.impl_details.subdiv_offsets.begin(),
        other.impl_details.subdiv_offsets.end());
    impl_details.subdiv_permutations.clear();
    for (auto p : other.impl_details.subdiv_permutations) {
      impl_details.subdiv_permutations.push_back(
          std::vector<int>(p.begin(), p.end()));
    }
    impl_details.subdiv_source_rank.assign(
        other.impl_details.subdiv_source_rank.begin(),
        other.impl_details.subdiv_source_rank.end());
    impl_details.dependencies = other.impl_details.dependencies;
    devices.assign(other.devices.begin(), other.devices.end());
    permutation.assign(other.permutation.begin(), other.permutation.end());
  }
  return *this;
}

string CollInstanceParams::ToString() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_5(mht_5_v, 282, "", "./tensorflow/core/framework/collective.cc", "CollInstanceParams::ToString");

  string v =
      strings::StrCat("CollInstanceParams { instance_key=", instance_key,
                      " type=", type, " data_type=", DataTypeString(data_type),
                      " shape=", shape.DebugString(), " devices {");
  strings::StrAppend(&v, "}, collective_name=", impl_details.collective_name,
                     ", subdiv_offsets={");
  strings::StrAppend(&v, "}, subdiv_offsets={");
  for (const auto& d : impl_details.subdiv_offsets) {
    strings::StrAppend(&v, d, ",");
  }
  strings::StrAppend(&v, "}, subdiv_perms={");
  for (const auto& p : impl_details.subdiv_permutations) {
    strings::StrAppend(&v, "{");
    for (const auto& i : p) {
      strings::StrAppend(&v, i, ",");
    }
    strings::StrAppend(&v, "}");  // one subdiv
  }
  if (!impl_details.subdiv_source_rank.empty()) {
    strings::StrAppend(&v, " subdiv_source_rank={");
    for (const auto& r : impl_details.subdiv_source_rank) {
      strings::StrAppend(&v, r, ",");
    }
    strings::StrAppend(&v, "}");
  }  // all subdivs
  if (type == PERMUTE_COLLECTIVE) {
    strings::StrAppend(&v, "}, permute_devices {");
    for (const auto& d : devices) {
      strings::StrAppend(&v, d, ",");
    }
    strings::StrAppend(&v, "}, permute_permutation {");
    for (const auto& p : permutation) {
      strings::StrAppend(&v, p, ",");
    }
    strings::StrAppend(&v, "}");
  }
  return v;
}

string CollectiveParams::ToString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_6(mht_6_v, 325, "", "./tensorflow/core/framework/collective.cc", "CollectiveParams::ToString");

  string v = strings::StrCat("CollectiveParams ", name, " {", group.ToString());
  strings::StrAppend(&v, " ", instance.ToString());
  strings::StrAppend(&v, " default_rank=", default_rank,
                     " is_source=", is_source, " source_rank=", source_rank,
                     " subdiv_rank={");
  for (const auto& r : subdiv_rank) {
    strings::StrAppend(&v, r, ",");
  }
  strings::StrAppend(&v, "}}");
  return v;
}

/*static*/ OpKernelContext::Params* CollectiveExecutor::CtxParams(
    OpKernelContext* ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_7(mht_7_v, 342, "", "./tensorflow/core/framework/collective.cc", "CollectiveExecutor::CtxParams");

  return ctx->params_;
}

CollectiveContext::CollectiveContext(
    CollectiveExecutor* col_exec, NcclCommunicatorInterface* nccl_communicator,
    const DeviceMgr* dev_mgr, OpKernelContext* ctx,
    OpKernelContext::Params* op_params, const CollectiveParams* col_params,
    const string& exec_key, int64_t step_id, const Tensor* input,
    Tensor* output)
    : col_exec(col_exec),
      nccl_communicator(nccl_communicator),
      dev_mgr(dev_mgr),
      op_ctx(ctx),
      op_params(op_params),
      col_params(col_params, /*add_ref=*/true),
      exec_key(exec_key),
      step_id(step_id),
      input(input),
      output(output),
      device(nullptr),
      device_name(
          col_params->group.members[col_params->default_rank].device.name()) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("exec_key: \"" + exec_key + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_8(mht_8_v, 368, "", "./tensorflow/core/framework/collective.cc", "CollectiveContext::CollectiveContext");
}

/*static*/
int64_t CollectiveExecutor::kInvalidId = -1;

/*static*/
Status CollectiveRegistry::Lookup(
    const string& collective_name,
    CollectiveImplementationInterface** implementation) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("collective_name: \"" + collective_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_9(mht_9_v, 380, "", "./tensorflow/core/framework/collective.cc", "CollectiveRegistry::Lookup");

  return LookupHelper(collective_name, implementation, false);
}

/*static*/
Status CollectiveRegistry::LookupParamResolverInstance(
    const string& collective_name,
    CollectiveImplementationInterface** implementation) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("collective_name: \"" + collective_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_10(mht_10_v, 391, "", "./tensorflow/core/framework/collective.cc", "CollectiveRegistry::LookupParamResolverInstance");

  return LookupHelper(collective_name, implementation, true);
}

/*static*/
void CollectiveRegistry::GetAll(
    std::vector<CollectiveImplementationInterface*>* implementations) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_11(mht_11_v, 400, "", "./tensorflow/core/framework/collective.cc", "CollectiveRegistry::GetAll");

  std::vector<RegistrationInfo>* registry = MutableCollectiveRegistry();
  for (const RegistrationInfo& reg_info : *registry)
    implementations->emplace_back(reg_info.factory());
}

/*static*/
Status CollectiveRegistry::Register(const string& collective_name,
                                    Factory factory) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("collective_name: \"" + collective_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_12(mht_12_v, 412, "", "./tensorflow/core/framework/collective.cc", "CollectiveRegistry::Register");

  std::vector<RegistrationInfo>* registry = MutableCollectiveRegistry();
  for (const RegistrationInfo& reg_info : *registry) {
    if (reg_info.name == collective_name)
      return errors::Internal("Already registered collective ",
                              collective_name);
  }
  registry->emplace_back(collective_name, std::move(factory));
  return Status::OK();
}

/*static*/
Status CollectiveRegistry::LookupHelper(
    const string& collective_name,
    CollectiveImplementationInterface** implementation, bool param_resolver) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("collective_name: \"" + collective_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScollectiveDTcc mht_13(mht_13_v, 430, "", "./tensorflow/core/framework/collective.cc", "CollectiveRegistry::LookupHelper");

  std::vector<RegistrationInfo>* registry = MutableCollectiveRegistry();
  for (const RegistrationInfo& reg_info : *registry) {
    if (reg_info.name == collective_name) {
      if (param_resolver) {
        *implementation = reg_info.param_resolver_instance;
      } else {
        *implementation = reg_info.factory();
      }
      return Status::OK();
    }
  }
  return errors::Internal(
      "CollectiveRegistry::Lookup did not find collective implementation ",
      collective_name);
}

}  // namespace tensorflow
