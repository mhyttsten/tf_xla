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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc() {
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

#include "tensorflow/core/common_runtime/eager/attr_builder.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace {

mutex g_op_name_to_attr_type_map_lock(LINKER_INITIALIZED);

tensorflow::gtl::FlatMap<string, const AttrTypeMap*>* OpNameToAttrTypeMap() {
  static auto* const m =
      new tensorflow::gtl::FlatMap<string, const AttrTypeMap*>;
  return m;
}

const uint32 kIsList = 1U << 31;

AttrTypeMap* DefaultFunctionAttrTypeMap() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "DefaultFunctionAttrTypeMap");

  AttrTypeMap* map = new AttrTypeMap();
  (*map)["executor_type"] = TF_ATTR_STRING;
  (*map)["config_proto"] = TF_ATTR_STRING;
  return map;
}

const AttrTypeMap* GetDefaultFunctionAttrTypeMap() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "GetDefaultFunctionAttrTypeMap");

  static const AttrTypeMap* map = DefaultFunctionAttrTypeMap();
  return map;
}

}  // namespace

Status OpDefForOp(const string& op_name, const OpDef** op_def) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_2(mht_2_v, 233, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "OpDefForOp");

  const OpRegistrationData* op_reg_data = nullptr;
  Status s = OpRegistry::Global()->LookUp(op_name, &op_reg_data);
  if (s.ok()) {
    *op_def = &op_reg_data->op_def;
  }
  return s;
}

Status AttrTypeMapForOp(const char* op_name, const AttrTypeMap** out,
                        bool* is_function) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_3(mht_3_v, 247, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrTypeMapForOp");

  {
    tf_shared_lock l(g_op_name_to_attr_type_map_lock);
    *is_function = false;
    *out = gtl::FindPtrOrNull(*OpNameToAttrTypeMap(), op_name);
    if (*out != nullptr) return Status::OK();
  }

  mutex_lock l(g_op_name_to_attr_type_map_lock);

  // Check the existence of AttrTypeMap for op_name again because another thread
  // may insert this map after the tf_shared_lock is released but before the
  // mutex_lock is acquired.
  *out = gtl::FindPtrOrNull(*OpNameToAttrTypeMap(), op_name);
  if (*out != nullptr) return Status::OK();

  const OpDef* op_def = nullptr;
  Status s = OpDefForOp(op_name, &op_def);
  if (errors::IsNotFound(s)) {
    // If we did not find the op def, we assume `op_name` is a function.
    // If it is actually a misspelled op, user will get another error when
    // trying to run it.
    // TODO(iga): If we ever have a use case for different attribute specs
    // in different functions, we will need to look at the OpDef in the
    // function def to retrieve their types.
    *out = GetDefaultFunctionAttrTypeMap();
    *is_function = true;
    return Status::OK();
  } else if (!s.ok()) {
    return s;
  }
  std::unique_ptr<AttrTypeMap> m(new AttrTypeMap);
  // TODO(agarwal): Avoid having to create this "registry" at runtime,
  // perhaps can be done at op registration time?
  for (const auto& attr : op_def->attr()) {
    string type = attr.type();
    const bool is_list = (type.length() > 6 && type.compare(0, 4, "list") == 0);
    if (is_list) {
      type = type.substr(5, type.length() - 6);
    }
    uint32 t = is_list ? kIsList : 0;
    if (type == "string") {
      t |= TF_ATTR_STRING;
    } else if (type == "int") {
      t |= TF_ATTR_INT;
    } else if (type == "float") {
      t |= TF_ATTR_FLOAT;
    } else if (type == "bool") {
      t |= TF_ATTR_BOOL;
    } else if (type == "type") {
      t |= TF_ATTR_TYPE;
    } else if (type == "shape") {
      t |= TF_ATTR_SHAPE;
    } else if (type == "tensor") {
      t |= TF_ATTR_TENSOR;
    } else if (type == "func") {
      t |= TF_ATTR_FUNC;
    } else {
      return errors::Unimplemented(
          "TODO(agarwal): Enable support for ops with attributes of type '",
          type, "'");
    }
    gtl::InsertIfNotPresent(m.get(), attr.name(), t);
  }
  *out = m.get();
  auto r = OpNameToAttrTypeMap()->emplace(op_name, m.release());
  DCHECK(r.second) << "AttrTypeMap already exists for " << op_name;

  return Status::OK();
}

#define DEFINE_GET_ATTR(TYPE, FIELD, ATTR_TYPE)                         \
  template <>                                                           \
  Status AttrBuilder::Get(StringPiece attr_name, TYPE* value) const {   \
    auto it = encoded_attrs_.find(string(attr_name));                   \
    if (it == encoded_attrs_.end()) {                                   \
      return errors::NotFound("No attr named '", attr_name,             \
                              "' found in AttrBuilder for ", op_name_); \
    }                                                                   \
    attr_tmp_.ParseFromString(it->second);                              \
    TF_RETURN_IF_ERROR(AttrValueHasType(attr_tmp_, ATTR_TYPE));         \
    *value = attr_tmp_.FIELD();                                         \
    return Status::OK();                                                \
  }

DEFINE_GET_ATTR(float, f, "float");
DEFINE_GET_ATTR(int, i, "int");
DEFINE_GET_ATTR(int64_t, i, "int");
DEFINE_GET_ATTR(bool, b, "bool");
DEFINE_GET_ATTR(tensorflow::DataType, type, "type");

#undef DEFINE_GET_ATTR

template <>
Status AttrBuilder::Get(StringPiece attr_name,
                        absl::InlinedVector<DataType, 4>* value) const {
  auto it = encoded_attrs_.find(string(attr_name));
  if (it == encoded_attrs_.end()) {
    return errors::NotFound("No attr named '", attr_name,
                            "' found in AttrBuilder for ", op_name_);
  }
  attr_tmp_.ParseFromString(it->second);
  TF_RETURN_IF_ERROR(AttrValueHasType(attr_tmp_, "list(type)"));
  for (size_t i = 0; i < attr_tmp_.list().type_size(); i++) {
    value->push_back(attr_tmp_.list().type(i));
  }
  return Status::OK();
}

AttrBuilder& AttrBuilder::NumInputs(int n) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_4(mht_4_v, 359, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::NumInputs");

  DCHECK(!node_def_finalized_) << "Calling NumInputs after BuildNodeDef.";
  num_inputs_ = n;
  return *this;
}

void AttrBuilder::FillAttrValueMap(AttrValueMap* m) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_5(mht_5_v, 368, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::FillAttrValueMap");

  for (auto& entry : encoded_attrs_) {
    attr_tmp_.ParseFromString(entry.second);
    m->insert(AttrValueMap::value_type(entry.first, attr_tmp_));
  }
  // For any attr-value pairs that exist in the op def (from op registry) but
  // not `m`, fill them into `m`, so that we can run a TFE_Op without having to
  // specify all the default attr values (e.g. for matmul, the `transpose_a`
  // attr defaults to false).
  const OpDef* op_def = nullptr;
  Status s = OpDefForOp(op_name().c_str(), &op_def);
  // This is expected, if this op is a custom function, and is therefore not
  // present in the op registry.
  if (!s.ok()) return;

  DCHECK(op_def);
  for (const auto& attr_def : op_def->attr()) {
    if (attr_def.has_default_value() && !m->count(attr_def.name())) {
      SetInAttrValueMap(m, attr_def.name(), attr_def.default_value());
    }
  }
}

namespace {

bool ValueMatchesDefault(const OpDef* op_def, const string& attr_name,
                         const AttrValue& attr_value) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_6(mht_6_v, 398, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "ValueMatchesDefault");

  // TODO(iga): It might make sense to augment OpRegistrationData with a
  // {attr_name -> default_attr_value} FlatMap to avoid the loop here.
  for (const OpDef::AttrDef& attr_def : op_def->attr()) {
    if (attr_def.name() == attr_name && attr_def.has_default_value() &&
        AreAttrValuesEqual(attr_def.default_value(), attr_value)) {
      return true;
    }
  }
  return false;
}

}  // namespace

void AttrBuilder::FillAttrValueMapWithoutDefaults(AttrValueMap* m) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_7(mht_7_v, 415, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::FillAttrValueMapWithoutDefaults");

  const OpDef* op_def = nullptr;
  Status s = OpDefForOp(op_name().c_str(), &op_def);

  for (auto& entry : encoded_attrs_) {
    attr_tmp_.ParseFromString(entry.second);
    // Insert the attr-value pair if we did not find the OpDef or if the value
    // is different from default.
    if (!s.ok() || !ValueMatchesDefault(op_def, entry.first, attr_tmp_)) {
      m->insert(AttrValueMap::value_type(entry.first, attr_tmp_));
    }
  }
}

void AttrBuilder::AddAttrIfNotPresent(StringPiece attr_name,
                                      const AttrValue& value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_8(mht_8_v, 433, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::AddAttrIfNotPresent");

  encoded_attrs_.emplace(string(attr_name), value.SerializeAsString());
}

const NodeDef& AttrBuilder::BuildNodeDef() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_9(mht_9_v, 440, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::BuildNodeDef");

  if (node_def_finalized_) return node_def_;
  if (!node_def_initialized_) {
    InitializeNodeDef();
  }
  for (int i = 0; i < num_inputs_; ++i) {
    node_def_.add_input("dummy_input");
  }
  FillAttrValueMap(node_def_.mutable_attr());
  node_def_finalized_ = true;
  return node_def_;
}

void AttrBuilder::CopyAttributes(const AttrBuilder& other) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_10(mht_10_v, 456, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::CopyAttributes");

  encoded_attrs_.insert(other.encoded_attrs_.begin(),
                        other.encoded_attrs_.end());
}

Status AttrTypeByName(const AttrTypeMap& m, const string& attr_name,
                      TF_AttrType* out, unsigned char* is_list) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_name: \"" + attr_name + "\"");
   mht_11_v.push_back("is_list: \"" + (is_list == nullptr ? std::string("nullptr") : std::string((char*)is_list)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_11(mht_11_v, 467, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrTypeByName");

  auto* t = gtl::FindOrNull(m, attr_name);
  if (t == nullptr) {
    return errors::InvalidArgument("Attribute '", attr_name,
                                   "' does not exist for this operation");
  }
  *out = static_cast<TF_AttrType>(*t & ~kIsList);
  if (*t & kIsList) {
    *is_list = 1;
  } else {
    *is_list = 0;
  }
  return Status::OK();
}

namespace {
inline tensorflow::Fprint128 FingerprintCat128(const tensorflow::Fprint128& a,
                                               const tensorflow::Fprint128& b) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_12(mht_12_v, 487, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "FingerprintCat128");

  return {tensorflow::FingerprintCat64(a.low64, b.low64),
          tensorflow::FingerprintCat64(a.high64, b.high64)};
}

void CombineUnordered(const tensorflow::Fprint128& a,
                      tensorflow::Fprint128* b) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_13(mht_13_v, 496, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "CombineUnordered");

  b->low64 += a.low64;
  b->high64 += a.high64;
}

inline tensorflow::Fprint128 CacheKeyHelper(StringPiece s,
                                            const tensorflow::Fprint128& b) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_14(mht_14_v, 505, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "CacheKeyHelper");

  tensorflow::Fprint128 a = tensorflow::Fingerprint128(s);
  return FingerprintCat128(a, b);
}

inline tensorflow::Fprint128 CacheKeyHelper(StringPiece s, uint64 b) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_15(mht_15_v, 513, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "CacheKeyHelper");

  return CacheKeyHelper(s, {b, b});
}

}  // namespace

tensorflow::Fprint128 AttrBuilder::CacheKey(const StringPiece device) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_16(mht_16_v, 522, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::CacheKey");

  if (!cached_cache_key_ || device != device_for_cached_cache_key_) {
    cached_cache_key_ = BuildCacheKeyForDevice(device);
    device_for_cached_cache_key_ = string(device);
  }

  return *cached_cache_key_;
}

tensorflow::Fprint128 AttrBuilder::BuildCacheKeyForDevice(
    const StringPiece device) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_17(mht_17_v, 535, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::BuildCacheKeyForDevice");

  tensorflow::Fprint128 f = tensorflow::Fingerprint128(op_name());
  f = tensorflow::FingerprintCat128(f, tensorflow::Fingerprint128(device));
  for (const auto& p : encoded_attrs_) {
    CombineUnordered(
        CacheKeyHelper(p.first, tensorflow::Fingerprint128(p.second)), &f);
  }
  return f;
}

void AttrBuilder::InitializeNodeDef() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_18(mht_18_v, 548, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::InitializeNodeDef");

  DCHECK(!node_def_initialized_);
  node_def_.Clear();
  node_def_.set_name(op_name_);
  node_def_.set_op(op_name_);
  node_def_initialized_ = true;
}

void AttrBuilder::GetNameAttrList(
    tensorflow::NameAttrList* name_and_attrs) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_19(mht_19_v, 560, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::GetNameAttrList");

  FillAttrValueMap(name_and_attrs->mutable_attr());
  name_and_attrs->set_name(op_name());
}

Status AttrBuilder::GetTypeList(
    absl::string_view attr_name,
    absl::InlinedVector<DataType, 4>* type_list) const {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_20(mht_20_v, 571, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::GetTypeList");

  return Get(attr_name, type_list);
}

bool AttrBuilder::GetInt(absl::string_view attr_name, int64_t* result) const {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_21(mht_21_v, 579, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::GetInt");

  Status s = Get(attr_name, result);
  return s.ok();
}
bool AttrBuilder::GetFloat(absl::string_view attr_name, float* result) const {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_22(mht_22_v, 587, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::GetFloat");

  Status s = Get(attr_name, result);
  return s.ok();
}
bool AttrBuilder::GetBool(absl::string_view attr_name, bool* result) const {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_23(mht_23_v, 595, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::GetBool");

  Status s = Get(attr_name, result);
  return s.ok();
}

bool AttrBuilder::GetType(absl::string_view attr_name,
                          tensorflow::DataType* result) const {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTcc mht_24(mht_24_v, 605, "", "./tensorflow/core/common_runtime/eager/attr_builder.cc", "AttrBuilder::GetType");

  Status s = Get(attr_name, result);
  return s.ok();
}

}  // namespace tensorflow
