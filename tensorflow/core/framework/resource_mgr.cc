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
class MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc() {
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

#include "tensorflow/core/framework/resource_mgr.h"

#include <atomic>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/demangle.h"
#include "tensorflow/core/platform/stacktrace.h"

namespace tensorflow {

ResourceHandle MakeResourceHandle(
    const string& container, const string& name, const DeviceBase& device,
    const TypeIndex& type_index,
    const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes,
    const absl::optional<ManagedStackTrace>& definition_stack_trace) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("container: \"" + container + "\"");
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/framework/resource_mgr.cc", "MakeResourceHandle");

  ResourceHandle result;
  result.set_device(device.name());
  result.set_container(container);
  result.set_definition_stack_trace(definition_stack_trace);
  if (name == ResourceHandle::ANONYMOUS_NAME) {
    result.set_name(
        strings::StrCat("_AnonymousVar", ResourceHandle::GenerateUniqueId()));
  } else {
    result.set_name(name);
  }
  result.set_hash_code(type_index.hash_code());
  result.set_maybe_type_name(type_index.name());
  result.set_dtypes_and_shapes(dtypes_and_shapes);
  return result;
}

Status MakeResourceHandleToOutput(OpKernelContext* context, int output_index,
                                  const string& container, const string& name,
                                  const TypeIndex& type_index) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("container: \"" + container + "\"");
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/framework/resource_mgr.cc", "MakeResourceHandleToOutput");

  Tensor* handle;
  TF_RETURN_IF_ERROR(
      context->allocate_output(output_index, TensorShape({}), &handle));
  handle->scalar<ResourceHandle>()() =
      MakeResourceHandle(container, name, *context->device(), type_index);
  return Status::OK();
}

namespace internal {

Status ValidateDevice(OpKernelContext* ctx, const ResourceHandle& p) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/framework/resource_mgr.cc", "ValidateDevice");

  if (ctx->device()->attributes().name() != p.device()) {
    return errors::InvalidArgument(
        "Trying to access resource ", p.name(), " located in device ",
        p.device(), " from device ", ctx->device()->attributes().name());
  }
  return Status::OK();
}

}  // end namespace internal

Status ResourceMgr::InsertDebugTypeName(uint64 hash_code,
                                        const string& type_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("type_name: \"" + type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_3(mht_3_v, 262, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::InsertDebugTypeName");

  auto iter = debug_type_names_.emplace(hash_code, type_name);
  if (iter.first->second != type_name) {
    return errors::AlreadyExists("Duplicate hash code found for type ",
                                 type_name);
  }
  return Status::OK();
}

const char* ResourceMgr::DebugTypeName(uint64 hash_code) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_4(mht_4_v, 274, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::DebugTypeName");

  auto type_name_iter = debug_type_names_.find(hash_code);
  if (type_name_iter == debug_type_names_.end()) {
    return "<unknown>";
  } else {
    return type_name_iter->second.c_str();
  }
}

ResourceMgr::ResourceAndName::ResourceAndName() : name(nullptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_5(mht_5_v, 286, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::ResourceAndName::ResourceAndName");
}

ResourceMgr::ResourceAndName::ResourceAndName(const string& name)
    : name(absl::make_unique<string>(name)) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_6(mht_6_v, 293, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::ResourceAndName::ResourceAndName");
}

core::RefCountPtr<ResourceBase> ResourceMgr::ResourceAndName::GetResource()
    const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_7(mht_7_v, 299, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::ResourceAndName::GetResource");

  if (absl::holds_alternative<core::RefCountPtr<ResourceBase>>(resource)) {
    ResourceBase* ptr =
        absl::get<core::RefCountPtr<ResourceBase>>(resource).get();
    ptr->Ref();
    return core::RefCountPtr<ResourceBase>(ptr);
  } else if (absl::holds_alternative<core::WeakPtr<ResourceBase>>(resource)) {
    return absl::get<core::WeakPtr<ResourceBase>>(resource).GetNewRef();
  } else {
    return nullptr;
  }
}

ResourceMgr::ResourceAndName::ResourceAndName(
    ResourceAndName&& other) noexcept {
  name = std::move(other.name);
  resource = std::move(other.resource);
}

ResourceMgr::ResourceAndName::~ResourceAndName() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_8(mht_8_v, 321, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::ResourceAndName::~ResourceAndName");
}

ResourceMgr::ResourceAndName& ResourceMgr::ResourceAndName::operator=(
    ResourceAndName&& other) noexcept {
  name = std::move(other.name);
  resource = std::move(other.resource);
  return *this;
}

ResourceMgr::ResourceMgr() : default_container_("localhost") {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_9(mht_9_v, 333, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::ResourceMgr");
}

ResourceMgr::ResourceMgr(const string& default_container)
    : default_container_(default_container) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("default_container: \"" + default_container + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_10(mht_10_v, 340, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::ResourceMgr");
}

ResourceMgr::~ResourceMgr() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_11(mht_11_v, 345, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::~ResourceMgr");
 Clear(); }

void ResourceMgr::Clear() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_12(mht_12_v, 350, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::Clear");

  // We do the deallocation outside of the lock to avoid a potential deadlock
  // in case any of the destructors access the resource manager.
  absl::flat_hash_map<string, Container*> tmp_containers;
  {
    mutex_lock l(mu_);
    tmp_containers = std::move(containers_);
  }
  for (const auto& p : tmp_containers) {
    delete p.second;
  }
  tmp_containers.clear();
}

string ResourceMgr::DebugString() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_13(mht_13_v, 367, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::DebugString");

  mutex_lock l(mu_);
  struct Line {
    const string* container;
    const string type;
    const string* resource;
    const string detail;
  };
  std::vector<Line> lines;
  for (const auto& p : containers_) {
    const string& container = p.first;
    for (const auto& q : *p.second) {
      const Key& key = q.first;
      const char* type = DebugTypeName(key.first);
      const core::RefCountPtr<ResourceBase> resource = q.second.GetResource();
      Line l{&container, port::Demangle(type), q.second.name.get(),
             resource ? resource->DebugString() : "<nullptr>"};
      lines.push_back(l);
    }
  }
  std::vector<string> text;
  text.reserve(lines.size());
  for (const Line& line : lines) {
    text.push_back(strings::Printf(
        "%-20s | %-40s | %-40s | %-s", line.container->c_str(),
        line.type.c_str(), line.resource->c_str(), line.detail.c_str()));
  }
  std::sort(text.begin(), text.end());
  return absl::StrJoin(text, "\n");
}

Status ResourceMgr::DoCreate(const string& container_name, TypeIndex type,
                             const string& name, ResourceBase* resource,
                             bool owns_resource) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("container_name: \"" + container_name + "\"");
   mht_14_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_14(mht_14_v, 405, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::DoCreate");

  Container* container = [&]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_15(mht_15_v, 409, "", "./tensorflow/core/framework/resource_mgr.cc", "lambda");

    Container** ptr = &containers_[container_name];
    if (*ptr == nullptr) {
      *ptr = new Container;
    }
    return *ptr;
  }();

  // NOTE: Separating out the construction of the map key and value so that the
  // key can contain a StringPiece that borrows from the string in the value.
  ResourceAndName resource_and_name(name);

  StringPiece borrowed_name(*resource_and_name.name);

  if (owns_resource) {
    resource_and_name.resource = core::RefCountPtr<ResourceBase>(resource);
  } else {
    auto cleanup_fn = [this, container, type, borrowed_name]() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_16(mht_16_v, 429, "", "./tensorflow/core/framework/resource_mgr.cc", "lambda");

      mutex_lock l(mu_);
      auto iter = container->find({type.hash_code(), borrowed_name});
      if (iter != container->end()) {
        container->erase(iter);
      }
    };
    resource_and_name.resource =
        core::WeakPtr<ResourceBase>(resource, cleanup_fn);
  }

  Container::value_type key_and_value(Key(type.hash_code(), borrowed_name),
                                      std::move(resource_and_name));

  auto st = container->insert(std::move(key_and_value));
  if (st.second) {
    TF_RETURN_IF_ERROR(InsertDebugTypeName(type.hash_code(), type.name()));
    return Status::OK();
  }
  return errors::AlreadyExists("Resource ", container_name, "/", name, "/",
                               type.name());
}

Status ResourceMgr::Lookup(const ResourceHandle& handle,
                           ResourceBase** resource) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_17(mht_17_v, 456, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::Lookup");

  tf_shared_lock l(mu_);
  return DoLookup(handle.container(), handle.hash_code(),
                  /*type_name=*/"ResourceBase", handle.name(), resource);
}

Status ResourceMgr::DoLookup(const string& container, TypeIndex type,
                             const string& name,
                             ResourceBase** resource) const {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("container: \"" + container + "\"");
   mht_18_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_18(mht_18_v, 469, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::DoLookup");

  return DoLookup(container, type.hash_code(), type.name(), name, resource);
}

Status ResourceMgr::DoLookup(const string& container, uint64 type_hash_code,
                             const string& type_name,
                             const string& resource_name,
                             ResourceBase** resource) const {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("container: \"" + container + "\"");
   mht_19_v.push_back("type_name: \"" + type_name + "\"");
   mht_19_v.push_back("resource_name: \"" + resource_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_19(mht_19_v, 482, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::DoLookup");

  const Container* b = gtl::FindPtrOrNull(containers_, container);
  if (b == nullptr) {
    return errors::NotFound("Container ", container,
                            " does not exist. (Could not find resource: ",
                            container, "/", resource_name, ")");
  }
  auto iter = b->find({type_hash_code, resource_name});
  if (iter == b->end()) {
    return errors::NotFound("Resource ", container, "/", resource_name, "/",
                            type_name, " does not exist.");
  }
  ResourceBase* ptr = iter->second.GetResource().release();
  if (ptr == nullptr) {
    return errors::NotFound("Resource ", container, "/", resource_name, "/",
                            type_name, " has been destroyed.");
  }
  *resource = ptr;
  return Status::OK();
}

Status ResourceMgr::PopResourceAndName(const string& container,
                                       uint64 type_hash_code,
                                       const string& resource_name,
                                       const string& type_name,
                                       ResourceAndName& resource_and_name) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("container: \"" + container + "\"");
   mht_20_v.push_back("resource_name: \"" + resource_name + "\"");
   mht_20_v.push_back("type_name: \"" + type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_20(mht_20_v, 513, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::PopResourceAndName");

  mutex_lock l(mu_);
  Container* b = gtl::FindPtrOrNull(containers_, container);
  if (b == nullptr) {
    return errors::NotFound("Container ", container, " does not exist.");
  }
  auto iter = b->find({type_hash_code, resource_name});
  if (iter == b->end()) {
    return errors::NotFound("Resource ", container, "/", resource_name, "/",
                            type_name, " does not exist.");
  }
  std::swap(resource_and_name, iter->second);
  b->erase(iter);
  return Status::OK();
}

Status ResourceMgr::DoDelete(const string& container, uint64 type_hash_code,
                             const string& resource_name,
                             const string& type_name) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("container: \"" + container + "\"");
   mht_21_v.push_back("resource_name: \"" + resource_name + "\"");
   mht_21_v.push_back("type_name: \"" + type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_21(mht_21_v, 537, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::DoDelete");

  ResourceAndName resource_and_name;
  TF_RETURN_IF_ERROR(PopResourceAndName(
      container, type_hash_code, resource_name, type_name, resource_and_name));

  if (absl::holds_alternative<core::WeakPtr<ResourceBase>>(
          resource_and_name.resource)) {
    return errors::Internal(
        "Cannot delete an unowned Resource ", container, "/", resource_name,
        "/", type_name, " from ResourceMgr. ",
        "This indicates ref-counting ResourceHandle is exposed to weak "
        "ResourceHandle code paths.");
  }
  return Status::OK();
}

Status ResourceMgr::DoDelete(const string& container, TypeIndex type,
                             const string& resource_name) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("container: \"" + container + "\"");
   mht_22_v.push_back("resource_name: \"" + resource_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_22(mht_22_v, 559, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::DoDelete");

  return DoDelete(container, type.hash_code(), resource_name, type.name());
}

Status ResourceMgr::Delete(const ResourceHandle& handle) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_23(mht_23_v, 566, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::Delete");

  return DoDelete(handle.container(), handle.hash_code(), handle.name(),
                  "<unknown>");
}

Status ResourceMgr::Cleanup(const string& container) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("container: \"" + container + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_24(mht_24_v, 575, "", "./tensorflow/core/framework/resource_mgr.cc", "ResourceMgr::Cleanup");

  {
    tf_shared_lock l(mu_);
    if (!gtl::FindOrNull(containers_, container)) {
      // Nothing to cleanup.
      return Status::OK();
    }
  }
  Container* b = nullptr;
  {
    mutex_lock l(mu_);
    auto iter = containers_.find(container);
    if (iter == containers_.end()) {
      // Nothing to cleanup, it's OK (concurrent cleanup).
      return Status::OK();
    }
    b = iter->second;
    containers_.erase(iter);
  }
  CHECK(b != nullptr);
  delete b;
  return Status::OK();
}

static bool IsValidContainerName(StringPiece s) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_25(mht_25_v, 602, "", "./tensorflow/core/framework/resource_mgr.cc", "IsValidContainerName");

  using ::tensorflow::strings::Scanner;
  return Scanner(s)
      .One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH)
      .Eos()
      .GetResult();
}

Status ContainerInfo::Init(ResourceMgr* rmgr, const NodeDef& ndef,
                           bool use_node_name_as_default) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_26(mht_26_v, 615, "", "./tensorflow/core/framework/resource_mgr.cc", "ContainerInfo::Init");

  CHECK(rmgr);
  rmgr_ = rmgr;
  string attr_container;
  TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "container", &attr_container));
  if (!attr_container.empty() && !IsValidContainerName(attr_container)) {
    return errors::InvalidArgument("container contains invalid characters: ",
                                   attr_container);
  }
  string attr_shared_name;
  TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "shared_name", &attr_shared_name));
  if (!attr_shared_name.empty() && (attr_shared_name[0] == '_')) {
    return errors::InvalidArgument("shared_name cannot start with '_':",
                                   attr_shared_name);
  }
  if (!attr_container.empty()) {
    container_ = attr_container;
  } else {
    container_ = rmgr_->default_container();
  }
  if (!attr_shared_name.empty()) {
    name_ = attr_shared_name;
  } else if (use_node_name_as_default) {
    name_ = ndef.name();
  } else {
    resource_is_private_to_kernel_ = true;
    static std::atomic<int64_t> counter(0);
    name_ = strings::StrCat("_", counter.fetch_add(1), "_", ndef.name());
  }
  return Status::OK();
}

string ContainerInfo::DebugString() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_27(mht_27_v, 650, "", "./tensorflow/core/framework/resource_mgr.cc", "ContainerInfo::DebugString");

  return strings::StrCat("[", container(), ",", name(), ",",
                         resource_is_private_to_kernel() ? "private" : "public",
                         "]");
}

const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_28(mht_28_v, 659, "", "./tensorflow/core/framework/resource_mgr.cc", "HandleFromInput");

  return ctx->input(input).flat<ResourceHandle>()(0);
}

Status HandleFromInput(OpKernelContext* ctx, StringPiece input,
                       ResourceHandle* handle) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_29(mht_29_v, 667, "", "./tensorflow/core/framework/resource_mgr.cc", "HandleFromInput");

  const Tensor* tensor;
  TF_RETURN_IF_ERROR(ctx->input(input, &tensor));
  *handle = tensor->flat<ResourceHandle>()(0);
  return Status::OK();
}

Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      ResourceBase** value) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_30(mht_30_v, 678, "", "./tensorflow/core/framework/resource_mgr.cc", "LookupResource");

  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  if (p.IsRefCounting()) {
    TF_ASSIGN_OR_RETURN(*value, p.GetResource<ResourceBase>());
    (*value)->Ref();
    return Status::OK();
  }
  return ctx->resource_manager()->Lookup(p, value);
}

Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgrDTcc mht_31(mht_31_v, 691, "", "./tensorflow/core/framework/resource_mgr.cc", "DeleteResource");

  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  if (p.IsRefCounting()) {
    return Status::OK();
  }
  return ctx->resource_manager()->Delete(p);
}

}  //  end namespace tensorflow
