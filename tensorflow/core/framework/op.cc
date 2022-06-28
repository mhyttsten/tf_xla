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
class MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

Status DefaultValidator(const OpRegistryInterface& op_registry) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/framework/op.cc", "DefaultValidator");

  LOG(WARNING) << "No kernel validator registered with OpRegistry.";
  return Status::OK();
}

// OpRegistry -----------------------------------------------------------------

OpRegistryInterface::~OpRegistryInterface() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/framework/op.cc", "OpRegistryInterface::~OpRegistryInterface");
}

Status OpRegistryInterface::LookUpOpDef(const string& op_type_name,
                                        const OpDef** op_def) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/framework/op.cc", "OpRegistryInterface::LookUpOpDef");

  *op_def = nullptr;
  const OpRegistrationData* op_reg_data = nullptr;
  TF_RETURN_IF_ERROR(LookUp(op_type_name, &op_reg_data));
  *op_def = &op_reg_data->op_def;
  return Status::OK();
}

OpRegistry::OpRegistry()
    : initialized_(false), op_registry_validator_(DefaultValidator) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_3(mht_3_v, 233, "", "./tensorflow/core/framework/op.cc", "OpRegistry::OpRegistry");
}

OpRegistry::~OpRegistry() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/framework/op.cc", "OpRegistry::~OpRegistry");

  for (const auto& e : registry_) delete e.second;
}

void OpRegistry::Register(const OpRegistrationDataFactory& op_data_factory) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_5(mht_5_v, 245, "", "./tensorflow/core/framework/op.cc", "OpRegistry::Register");

  mutex_lock lock(mu_);
  if (initialized_) {
    TF_QCHECK_OK(RegisterAlreadyLocked(op_data_factory));
  } else {
    deferred_.push_back(op_data_factory);
  }
}

namespace {
// Helper function that returns Status message for failed LookUp.
Status OpNotFound(const string& op_type_name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_6(mht_6_v, 260, "", "./tensorflow/core/framework/op.cc", "OpNotFound");

  Status status = errors::NotFound(
      "Op type not registered '", op_type_name, "' in binary running on ",
      port::Hostname(), ". ",
      "Make sure the Op and Kernel are registered in the binary running in "
      "this process. Note that if you are loading a saved graph which used ops "
      "from tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done "
      "before importing the graph, as contrib ops are lazily registered when "
      "the module is first accessed.");
  VLOG(1) << status.ToString();
  return status;
}
}  // namespace

Status OpRegistry::LookUp(const string& op_type_name,
                          const OpRegistrationData** op_reg_data) const {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_7(mht_7_v, 279, "", "./tensorflow/core/framework/op.cc", "OpRegistry::LookUp");

  if ((*op_reg_data = LookUp(op_type_name))) return Status::OK();
  return OpNotFound(op_type_name);
}

const OpRegistrationData* OpRegistry::LookUp(const string& op_type_name) const {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_8(mht_8_v, 288, "", "./tensorflow/core/framework/op.cc", "OpRegistry::LookUp");

  {
    tf_shared_lock l(mu_);
    if (initialized_) {
      if (const OpRegistrationData* res =
              gtl::FindWithDefault(registry_, op_type_name, nullptr)) {
        return res;
      }
    }
  }
  return LookUpSlow(op_type_name);
}

const OpRegistrationData* OpRegistry::LookUpSlow(
    const string& op_type_name) const {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_9(mht_9_v, 306, "", "./tensorflow/core/framework/op.cc", "OpRegistry::LookUpSlow");

  const OpRegistrationData* res = nullptr;

  bool first_call = false;
  bool first_unregistered = false;
  {  // Scope for lock.
    mutex_lock lock(mu_);
    first_call = MustCallDeferred();
    res = gtl::FindWithDefault(registry_, op_type_name, nullptr);

    static bool unregistered_before = false;
    first_unregistered = !unregistered_before && (res == nullptr);
    if (first_unregistered) {
      unregistered_before = true;
    }
    // Note: Can't hold mu_ while calling Export() below.
  }
  if (first_call) {
    TF_QCHECK_OK(op_registry_validator_(*this));
  }
  if (res == nullptr) {
    if (first_unregistered) {
      OpList op_list;
      Export(true, &op_list);
      if (VLOG_IS_ON(3)) {
        LOG(INFO) << "All registered Ops:";
        for (const auto& op : op_list.op()) {
          LOG(INFO) << SummarizeOpDef(op);
        }
      }
    }
  }
  return res;
}

void OpRegistry::GetRegisteredOps(std::vector<OpDef>* op_defs) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_10(mht_10_v, 344, "", "./tensorflow/core/framework/op.cc", "OpRegistry::GetRegisteredOps");

  mutex_lock lock(mu_);
  MustCallDeferred();
  for (const auto& p : registry_) {
    op_defs->push_back(p.second->op_def);
  }
}

void OpRegistry::GetOpRegistrationData(
    std::vector<OpRegistrationData>* op_data) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_11(mht_11_v, 356, "", "./tensorflow/core/framework/op.cc", "OpRegistry::GetOpRegistrationData");

  mutex_lock lock(mu_);
  MustCallDeferred();
  for (const auto& p : registry_) {
    op_data->push_back(*p.second);
  }
}

Status OpRegistry::SetWatcher(const Watcher& watcher) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_12(mht_12_v, 367, "", "./tensorflow/core/framework/op.cc", "OpRegistry::SetWatcher");

  mutex_lock lock(mu_);
  if (watcher_ && watcher) {
    return errors::AlreadyExists(
        "Cannot over-write a valid watcher with another.");
  }
  watcher_ = watcher;
  return Status::OK();
}

void OpRegistry::Export(bool include_internal, OpList* ops) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_13(mht_13_v, 380, "", "./tensorflow/core/framework/op.cc", "OpRegistry::Export");

  mutex_lock lock(mu_);
  MustCallDeferred();

  std::vector<std::pair<string, const OpRegistrationData*>> sorted(
      registry_.begin(), registry_.end());
  std::sort(sorted.begin(), sorted.end());

  auto out = ops->mutable_op();
  out->Clear();
  out->Reserve(sorted.size());

  for (const auto& item : sorted) {
    if (include_internal || !absl::StartsWith(item.first, "_")) {
      *out->Add() = item.second->op_def;
    }
  }
}

void OpRegistry::DeferRegistrations() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_14(mht_14_v, 402, "", "./tensorflow/core/framework/op.cc", "OpRegistry::DeferRegistrations");

  mutex_lock lock(mu_);
  initialized_ = false;
}

void OpRegistry::ClearDeferredRegistrations() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_15(mht_15_v, 410, "", "./tensorflow/core/framework/op.cc", "OpRegistry::ClearDeferredRegistrations");

  mutex_lock lock(mu_);
  deferred_.clear();
}

Status OpRegistry::ProcessRegistrations() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_16(mht_16_v, 418, "", "./tensorflow/core/framework/op.cc", "OpRegistry::ProcessRegistrations");

  mutex_lock lock(mu_);
  return CallDeferred();
}

string OpRegistry::DebugString(bool include_internal) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_17(mht_17_v, 426, "", "./tensorflow/core/framework/op.cc", "OpRegistry::DebugString");

  OpList op_list;
  Export(include_internal, &op_list);
  string ret;
  for (const auto& op : op_list.op()) {
    strings::StrAppend(&ret, SummarizeOpDef(op), "\n");
  }
  return ret;
}

bool OpRegistry::MustCallDeferred() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_18(mht_18_v, 439, "", "./tensorflow/core/framework/op.cc", "OpRegistry::MustCallDeferred");

  if (initialized_) return false;
  initialized_ = true;
  for (size_t i = 0; i < deferred_.size(); ++i) {
    TF_QCHECK_OK(RegisterAlreadyLocked(deferred_[i]));
  }
  deferred_.clear();
  return true;
}

Status OpRegistry::CallDeferred() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_19(mht_19_v, 452, "", "./tensorflow/core/framework/op.cc", "OpRegistry::CallDeferred");

  if (initialized_) return Status::OK();
  initialized_ = true;
  for (size_t i = 0; i < deferred_.size(); ++i) {
    Status s = RegisterAlreadyLocked(deferred_[i]);
    if (!s.ok()) {
      return s;
    }
  }
  deferred_.clear();
  return Status::OK();
}

Status OpRegistry::RegisterAlreadyLocked(
    const OpRegistrationDataFactory& op_data_factory) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_20(mht_20_v, 469, "", "./tensorflow/core/framework/op.cc", "OpRegistry::RegisterAlreadyLocked");

  std::unique_ptr<OpRegistrationData> op_reg_data(new OpRegistrationData);
  Status s = op_data_factory(op_reg_data.get());
  if (s.ok()) {
    s = ValidateOpDef(op_reg_data->op_def);
    if (s.ok() &&
        !gtl::InsertIfNotPresent(&registry_, op_reg_data->op_def.name(),
                                 op_reg_data.get())) {
      s = errors::AlreadyExists("Op with name ", op_reg_data->op_def.name());
    }
  }
  Status watcher_status = s;
  if (watcher_) {
    watcher_status = watcher_(s, op_reg_data->op_def);
  }
  if (s.ok()) {
    op_reg_data.release();
  } else {
    op_reg_data.reset();
  }
  return watcher_status;
}

// static
OpRegistry* OpRegistry::Global() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_21(mht_21_v, 496, "", "./tensorflow/core/framework/op.cc", "OpRegistry::Global");

  static OpRegistry* global_op_registry = new OpRegistry;
  return global_op_registry;
}

// OpListOpRegistry -----------------------------------------------------------

OpListOpRegistry::OpListOpRegistry(const OpList* op_list) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_22(mht_22_v, 506, "", "./tensorflow/core/framework/op.cc", "OpListOpRegistry::OpListOpRegistry");

  for (const OpDef& op_def : op_list->op()) {
    auto* op_reg_data = new OpRegistrationData();
    op_reg_data->op_def = op_def;
    index_[op_def.name()] = op_reg_data;
  }
}

OpListOpRegistry::~OpListOpRegistry() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_23(mht_23_v, 517, "", "./tensorflow/core/framework/op.cc", "OpListOpRegistry::~OpListOpRegistry");

  for (const auto& e : index_) delete e.second;
}

const OpRegistrationData* OpListOpRegistry::LookUp(
    const string& op_type_name) const {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_24(mht_24_v, 526, "", "./tensorflow/core/framework/op.cc", "OpListOpRegistry::LookUp");

  auto iter = index_.find(op_type_name);
  if (iter == index_.end()) {
    return nullptr;
  }
  return iter->second;
}

Status OpListOpRegistry::LookUp(const string& op_type_name,
                                const OpRegistrationData** op_reg_data) const {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTcc mht_25(mht_25_v, 539, "", "./tensorflow/core/framework/op.cc", "OpListOpRegistry::LookUp");

  if ((*op_reg_data = LookUp(op_type_name))) return Status::OK();
  return OpNotFound(op_type_name);
}

namespace register_op {

InitOnStartupMarker OpDefBuilderWrapper::operator()() {
  OpRegistry::Global()->Register(
      [builder =
           std::move(builder_)](OpRegistrationData* op_reg_data) -> Status {
        return builder.Finalize(op_reg_data);
      });
  return {};
}

}  //  namespace register_op

}  // namespace tensorflow
