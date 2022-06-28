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
class MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc() {
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

#include "tensorflow/core/framework/resource_handle.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/demangle.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

namespace {
std::string DtypeAndShapesToString(
    const std::vector<DtypeAndPartialTensorShape>& dtype_and_shapes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/framework/resource_handle.cc", "DtypeAndShapesToString");

  std::vector<std::string> dtype_and_shape_strings;
  dtype_and_shape_strings.reserve(dtype_and_shapes.size());
  for (const DtypeAndPartialTensorShape& dtype_and_shape : dtype_and_shapes) {
    // Note that it is a bit unfortunate to return int/enum as dtype, given we
    // can't directly use DataTypeString due to circular dependency.
    dtype_and_shape_strings.push_back(
        absl::StrFormat("DType enum: %d, Shape: %s", dtype_and_shape.dtype,
                        dtype_and_shape.shape.DebugString()));
  }
  return absl::StrFormat("[ %s ]", absl::StrJoin(dtype_and_shape_strings, ","));
}
}  // namespace

// Must be declared here for pre-C++17 compatibility.
/* static */ constexpr const char* ResourceHandle::ANONYMOUS_NAME;

ResourceHandle::ResourceHandle() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::ResourceHandle");
}

ResourceHandle::ResourceHandle(const ResourceHandleProto& proto) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::ResourceHandle");

  TF_CHECK_OK(FromProto(proto));
}

Status ResourceHandle::BuildResourceHandle(const ResourceHandleProto& proto,
                                           ResourceHandle* out) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::BuildResourceHandle");

  if (out == nullptr)
    return errors::Internal(
        "BuildResourceHandle() was called with nullptr for the output");
  return out->FromProto(proto);
}

ResourceHandle::~ResourceHandle() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_4(mht_4_v, 247, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::~ResourceHandle");
}

void ResourceHandle::AsProto(ResourceHandleProto* proto) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_5(mht_5_v, 252, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::AsProto");

  proto->set_device(device());
  proto->set_container(container());
  proto->set_name(name());
  proto->set_hash_code(hash_code());
  proto->set_maybe_type_name(maybe_type_name());
  for (const auto& dtype_and_shape_pair : dtypes_and_shapes_) {
    auto dtype_and_shape = proto->add_dtypes_and_shapes();
    dtype_and_shape->set_dtype(dtype_and_shape_pair.dtype);
    dtype_and_shape_pair.shape.AsProto(dtype_and_shape->mutable_shape());
  }
}

Status ResourceHandle::FromProto(const ResourceHandleProto& proto) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_6(mht_6_v, 268, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::FromProto");

  set_device(proto.device());
  set_container(proto.container());
  set_name(proto.name());
  set_hash_code(proto.hash_code());
  set_maybe_type_name(proto.maybe_type_name());
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes;
  for (const auto& dtype_and_shape : proto.dtypes_and_shapes()) {
    DataType dtype = dtype_and_shape.dtype();
    PartialTensorShape shape;
    Status s = PartialTensorShape::BuildPartialTensorShape(
        dtype_and_shape.shape(), &shape);
    if (!s.ok()) {
      return s;
    }
    dtypes_and_shapes.push_back(DtypeAndPartialTensorShape{dtype, shape});
  }
  dtypes_and_shapes_ = std::move(dtypes_and_shapes);
  return Status::OK();
}

string ResourceHandle::SerializeAsString() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_7(mht_7_v, 292, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::SerializeAsString");

  ResourceHandleProto proto;
  AsProto(&proto);
  return proto.SerializeAsString();
}

bool ResourceHandle::ParseFromString(const string& s) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_8(mht_8_v, 302, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::ParseFromString");

  ResourceHandleProto proto;
  return proto.ParseFromString(s) && FromProto(proto).ok();
}

string ResourceHandle::DebugString() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_9(mht_9_v, 310, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::DebugString");

  return absl::StrFormat(
      "device: %s container: %s name: %s hash_code: 0x%X maybe_type_name %s, "
      "dtype and shapes : %s",
      device(), container(), name(), hash_code(),
      port::Demangle(maybe_type_name()),
      DtypeAndShapesToString(dtypes_and_shapes()));
}
string ResourceHandle::SummarizeValue() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_10(mht_10_v, 321, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::SummarizeValue");

  return absl::StrFormat(
      "ResourceHandle(name=\"%s\", device=\"%s\", container=\"%s\", "
      "type=\"%s\", dtype and shapes : \"%s\")",
      name(), device(), container(), port::Demangle(maybe_type_name()),
      DtypeAndShapesToString(dtypes_and_shapes()));
}

ResourceHandle ResourceHandle::MakeRefCountingHandle(
    ResourceBase* resource, const string& device_name,
    const TypeIndex& type_index,
    const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes,
    const absl::optional<ManagedStackTrace>& definition_stack_trace) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_11(mht_11_v, 337, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::MakeRefCountingHandle");

  ResourceHandle result;
  result.resource_.reset(resource, /*add_ref=*/false);
  result.set_device(device_name);
  // All resources owned by anonymous handles are put into the same container,
  // and they get process-unique handle names.
  result.set_container("Anonymous");
  result.set_definition_stack_trace(definition_stack_trace);
  result.set_name(
      absl::StrFormat("Resource-%d-at-%p", GenerateUniqueId(), resource));
  result.set_hash_code(type_index.hash_code());
  result.set_maybe_type_name(type_index.name());
  result.set_dtypes_and_shapes(dtypes_and_shapes);
  return result;
}

Status ResourceHandle::ValidateType(const TypeIndex& type_index) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_12(mht_12_v, 356, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::ValidateType");

  if (type_index.hash_code() != hash_code()) {
    return errors::InvalidArgument(
        "Trying to access a handle's resource using the wrong type. ",
        "The handle points to a resource (name '", name(), "') of type '",
        port::Demangle(maybe_type_name()), "' (hash code ", hash_code(),
        ") but you are trying to access the resource as type '",
        port::Demangle(type_index.name()), "' (hash code ",
        type_index.hash_code(), ")");
  }
  return Status::OK();
}

std::atomic<int64_t> ResourceHandle::current_id_;

int64_t ResourceHandle::GenerateUniqueId() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_13(mht_13_v, 374, "", "./tensorflow/core/framework/resource_handle.cc", "ResourceHandle::GenerateUniqueId");
 return current_id_.fetch_add(1); }

string ProtoDebugString(const ResourceHandle& handle) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_14(mht_14_v, 379, "", "./tensorflow/core/framework/resource_handle.cc", "ProtoDebugString");

  return handle.DebugString();
}

void EncodeResourceHandleList(const ResourceHandle* p, int64_t n,
                              std::unique_ptr<port::StringListEncoder> e) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_15(mht_15_v, 387, "", "./tensorflow/core/framework/resource_handle.cc", "EncodeResourceHandleList");

  ResourceHandleProto proto;
  for (int i = 0; i < n; ++i) {
    p[i].AsProto(&proto);
    e->Append(proto);
  }
  e->Finalize();
}

bool DecodeResourceHandleList(std::unique_ptr<port::StringListDecoder> d,
                              ResourceHandle* ps, int64_t n) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_handleDTcc mht_16(mht_16_v, 400, "", "./tensorflow/core/framework/resource_handle.cc", "DecodeResourceHandleList");

  std::vector<uint32> sizes(n);
  if (!d->ReadSizes(&sizes)) return false;

  ResourceHandleProto proto;
  for (int i = 0; i < n; ++i) {
    if (!proto.ParseFromArray(d->Data(sizes[i]), sizes[i])) {
      return false;
    }
    if (!ps[i].FromProto(proto).ok()) {
      return false;
    }
  }
  return true;
}

}  // namespace tensorflow
