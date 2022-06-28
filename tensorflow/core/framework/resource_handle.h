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

#ifndef TENSORFLOW_FRAMEWORK_RESOURCE_HANDLE_H_
#define TENSORFLOW_FRAMEWORK_RESOURCE_HANDLE_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh() {
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


#include <string>

#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/intrusive_ptr.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace tensorflow {

class ResourceHandleProto;

// Class representing a handle to a tensorflow resource. Handles are
// not valid across executions, but can be serialized back and forth from within
// a single run (except for those created from MakeRefCountingHandle i.e. whose
// resource_ field is not empty).
//
// This is the native C++ class equivalent of ResourceHandleProto.  They are
// separate so that kernels do not need to depend on protos.
class ResourceHandle {
 public:
  ResourceHandle();
  ResourceHandle(const ResourceHandleProto& proto);
  ~ResourceHandle();

  // Use this factory method if the `proto` comes from user controlled input, to
  // prevent a denial of service.
  static Status BuildResourceHandle(const ResourceHandleProto& proto,
                                    ResourceHandle* out);

  // Unique name for the device containing the resource.
  const std::string& device() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_0(mht_0_v, 224, "", "./tensorflow/core/framework/resource_handle.h", "device");
 return device_; }

  void set_device(const std::string& device) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_1(mht_1_v, 230, "", "./tensorflow/core/framework/resource_handle.h", "set_device");
 device_ = device; }

  // Container in which this resource is placed.
  const std::string& container() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_2(mht_2_v, 236, "", "./tensorflow/core/framework/resource_handle.h", "container");
 return container_; }
  void set_container(const std::string& container) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("container: \"" + container + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_3(mht_3_v, 241, "", "./tensorflow/core/framework/resource_handle.h", "set_container");
 container_ = container; }

  // Unique name of this resource.
  const std::string& name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_4(mht_4_v, 247, "", "./tensorflow/core/framework/resource_handle.h", "name");
 return name_; }
  void set_name(const std::string& name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_5(mht_5_v, 252, "", "./tensorflow/core/framework/resource_handle.h", "set_name");
 name_ = name; }

  // Hash code for the type of the resource. Is only valid in the same device
  // and in the same execution.
  uint64 hash_code() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_6(mht_6_v, 259, "", "./tensorflow/core/framework/resource_handle.h", "hash_code");
 return hash_code_; }
  void set_hash_code(uint64 hash_code) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_7(mht_7_v, 263, "", "./tensorflow/core/framework/resource_handle.h", "set_hash_code");
 hash_code_ = hash_code; }

  // For debug-only, the name of the type pointed to by this handle, if
  // available.
  const std::string& maybe_type_name() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_8(mht_8_v, 270, "", "./tensorflow/core/framework/resource_handle.h", "maybe_type_name");
 return maybe_type_name_; }
  void set_maybe_type_name(const std::string& value) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_9(mht_9_v, 275, "", "./tensorflow/core/framework/resource_handle.h", "set_maybe_type_name");

    maybe_type_name_ = value;
  }

  // Data types and shapes for the underlying resource.
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes() const {
    return dtypes_and_shapes_;
  }
  void set_dtypes_and_shapes(
      const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_10(mht_10_v, 287, "", "./tensorflow/core/framework/resource_handle.h", "set_dtypes_and_shapes");

    dtypes_and_shapes_ = dtypes_and_shapes;
  }

  void set_definition_stack_trace(
      const absl::optional<ManagedStackTrace>& definition_stack_trace) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_11(mht_11_v, 295, "", "./tensorflow/core/framework/resource_handle.h", "set_definition_stack_trace");

    definition_stack_trace_ = definition_stack_trace;
  }

  const absl::optional<ManagedStackTrace>& definition_stack_trace() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_12(mht_12_v, 302, "", "./tensorflow/core/framework/resource_handle.h", "definition_stack_trace");

    return definition_stack_trace_;
  }

  // Conversion to and from ResourceHandleProto
  void AsProto(ResourceHandleProto* proto) const;
  Status FromProto(const ResourceHandleProto& proto);

  // Serialization via ResourceHandleProto
  std::string SerializeAsString() const;
  bool ParseFromString(const std::string& s);

  std::string DebugString() const;

  std::string SummarizeValue() const;

  // GUID for anonymous resources. Resources with this shared_name will have
  // their shared_name replaced with a GUID at creation time
  static constexpr const char* ANONYMOUS_NAME =
      "cd2c89b7-88b7-44c8-ad83-06c2a9158347";

  // Creates a `ResourceHandle` that holds a pointer to a resource and takes
  // ownership of it. Normally a `ResourceHandle` only contains the name (and
  // some other metadata) of the resource. When created via this function,
  // the handle will own the resource, in the sense that it will destroy the
  // resource automatically when the resource is no longer needed. It does this
  // via automatic ref-counting on the resource: when the handle is copied, it
  // will call `Ref` on the resource (remember that all resources inherit from
  // `ResourceBase` which inherits from `RefCounted`), and when the handle is
  // destroyed, it will call `Unref` on the resource. When the last handle goes
  // out of scope, the resource's ref-count will go down to zero and the
  // resource will be destroyed. When calling this function, the `resource`
  // argument should have a ref-count of one (which is the case when the
  // resource is newly created).
  //
  // For those familiar with `ResourceMgr`, when you create a handle by the
  // `MakeResourceHandle` function in resource_mgr.h, the handle doesn't hold a
  // strong reference to the resource, and the resource is owned by the
  // resource manager whose strong reference must be manually deleted by
  // calling `ResourceMgr::Delete`. In contrast, a handle created by this
  // function holds a strong reference to the resource. The resource manager
  // does not hold a strong reference to the resource.
  template <typename T>
  static ResourceHandle MakeRefCountingHandle(
      T* resource, const string& device_name,
      const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes = {},
      const absl::optional<ManagedStackTrace>& definition_stack_trace = {}) {
    return MakeRefCountingHandle(resource, device_name, TypeIndex::Make<T>(),
                                 dtypes_and_shapes, definition_stack_trace);
  }

  static ResourceHandle MakeRefCountingHandle(
      ResourceBase* resource, const string& device_name,
      const TypeIndex& type_index,
      const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes = {},
      const absl::optional<ManagedStackTrace>& definition_stack_trace = {});

  // Pointer to the resource.
  const core::IntrusivePtr<ResourceBase>& resource() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_13(mht_13_v, 363, "", "./tensorflow/core/framework/resource_handle.h", "resource");
 return resource_; }

  // Gets the resource pointer in `handle` as `T*`, or an error if the actual
  // resource type is not `T`.
  template <typename T>
  StatusOr<T*> GetResource() const {
    TF_RETURN_IF_ERROR(ValidateType<T>());
    return down_cast<T*>(resource_.get());
  }

  // Returns True if the resource handle is ref-counting.
  // See MakeRefCountingHandle.
  bool IsRefCounting() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_14(mht_14_v, 378, "", "./tensorflow/core/framework/resource_handle.h", "IsRefCounting");
 return resource_.get() != nullptr; }

  // Validates that the resource type in `handle` is `T`.
  template <typename T>
  Status ValidateType() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTh mht_15(mht_15_v, 385, "", "./tensorflow/core/framework/resource_handle.h", "ValidateType");

    return ValidateType(TypeIndex::Make<T>());
  }

  Status ValidateType(const TypeIndex& type_index) const;

  // Generates unique IDs (e.g. for names of anonymous variables)
  static int64_t GenerateUniqueId();

 private:
  std::string device_;
  std::string container_;
  std::string name_;
  uint64 hash_code_ = 0;
  std::string maybe_type_name_;
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes_;
  absl::optional<ManagedStackTrace> definition_stack_trace_;
  // A smart pointer to the actual resource. When this field is not empty, the
  // handle is in a "ref-counting" mode, owning the resource; otherwise it's in
  // a "weak-ref" mode, only containing the name of the resource (conceptually a
  // weak reference).
  core::IntrusivePtr<ResourceBase> resource_;
  static std::atomic<int64_t> current_id_;
};

// For backwards compatibility for when this was a proto
std::string ProtoDebugString(const ResourceHandle& handle);

// Encodes a list of ResourceHandle protos in the given StringListEncoder.
void EncodeResourceHandleList(const ResourceHandle* p, int64_t n,
                              std::unique_ptr<port::StringListEncoder> e);

// Decodes a list of ResourceHandle protos from the given StringListDecoder.
bool DecodeResourceHandleList(std::unique_ptr<port::StringListDecoder> d,
                              ResourceHandle* ps, int64_t n);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_RESOURCE_HANDLE_H_
