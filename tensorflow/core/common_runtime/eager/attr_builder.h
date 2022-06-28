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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_ATTR_BUILDER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_ATTR_BUILDER_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh() {
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


// Support for eager execution of TensorFlow kernels.

#include <memory>
#include <unordered_map>

#include "tensorflow/c/eager/abstract_op_attrs.h"
#include "tensorflow/c/tf_attrtype.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {

// Maps attribute name to an encoding of the type of the attribute value.
// If the type is not a list type, the value is the same as the TF_AttrType type
// of the value. Else, the highest order bit is on, and the rest of the bits
// represent the TF_AttrType type of the values in the list.
typedef std::unordered_map<string, uint32> AttrTypeMap;

// Look up OpDef for `op_name`.
Status OpDefForOp(const string& op_name, const OpDef** op_def);

// Returns the AttrTypeMap for the TensorFlow operation named op_name.
// If op_name is not registered in global op registry, AttrTypeMapForOp assumes
// the op to be a function and returns the default attributes for a function.
// `is_function` is set to true in this case.
Status AttrTypeMapForOp(const char* op_name, const AttrTypeMap** out,
                        bool* is_function);

// Looks for 'attr_name' in 'm' and sets 'out' and 'is_list'.
Status AttrTypeByName(const AttrTypeMap& m, const string& attr_name,
                      TF_AttrType* out, unsigned char* is_list);

// KernelAndDevice::Init needs a NodeDef only to pass the attribute map through.
// An AttrBuilder is a convenience class to help with that - providing a smaller
// interface than NodeDefBuilder and avoiding expensive (unnecessary?) sanity
// checks (like number of inputs matching the OpDef - we only care about
// attributes here).
//
// TODO(ashankar): Take a closer look at checks in NodeDefBuilder and see which
// ones make sense to replicate.

// This is a helper class for creating a NodeDef. Additionally, this class
// allows computing a cache key based on fingerprinting the attributes of this
// NodeDef.
//
// Example usage:
// AttrBuilder a;
// a.NumInputs(2);
// a.Set("T", TF_FLOAT);
// tensorflow::Fprint128 cache_key = a.CacheKey("cpu:0");
// const NodeDef& n = a.BuildNodeDef();
//
// Note that all calls to Set and NumInputs should happen before calling
// BuildNodeDef. Also, calls to NumInputs or Set between multiple invocations
// to CacheKey may cause different values to be returned by CacheKey.
//
// For performance reasons, the class internally delays the actual construction
// of the NodeDef till BuildNodeDef is called, or Set is called with certain
// uncommon types (see template specializations of Set to see which types
// trigger a NodeDef creation).
//
// Setting attributes via `Set` may cause arena-allocated protocol buffer
// messages to be destructed, which is not thread safe. This means that it is
// currently not safe to set attributes on *different* AttrBuilder objects from
// multiple threads. This does not apply to `CopyAttributes`.
class AttrBuilder : public AbstractOpAttrs {
 public:
  AttrBuilder()
      : AbstractOpAttrs(AbstractOpAttrs::AbstractOpAttrsKind::kEager) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh mht_0(mht_0_v, 264, "", "./tensorflow/core/common_runtime/eager/attr_builder.h", "AttrBuilder");
}

  ~AttrBuilder() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh mht_1(mht_1_v, 269, "", "./tensorflow/core/common_runtime/eager/attr_builder.h", "~AttrBuilder");
}
  explicit AttrBuilder(const char* op)
      : AbstractOpAttrs(AbstractOpAttrs::AbstractOpAttrsKind::kEager) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh mht_2(mht_2_v, 275, "", "./tensorflow/core/common_runtime/eager/attr_builder.h", "AttrBuilder");

    Reset(op);
  }

  void Reset(const char* op) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh mht_3(mht_3_v, 283, "", "./tensorflow/core/common_runtime/eager/attr_builder.h", "Reset");

    op_name_ = op;
    num_inputs_ = 0;
    encoded_attrs_.clear();
    node_def_initialized_ = false;
    node_def_finalized_ = false;
    cached_cache_key_ = absl::nullopt;
    device_for_cached_cache_key_.clear();
  }

  const string& op_name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh mht_4(mht_4_v, 296, "", "./tensorflow/core/common_runtime/eager/attr_builder.h", "op_name");
 return op_name_; }

  // Needed to work around call to ValidateNodeDef in CreateOpKernel.
  AttrBuilder& NumInputs(int n);

  template <class T>
  AttrBuilder& Set(StringPiece attr_name, T&& value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh mht_5(mht_5_v, 305, "", "./tensorflow/core/common_runtime/eager/attr_builder.h", "Set");

    SetAttrValue(value, &attr_tmp_);
    AddAttrIfNotPresent(attr_name, attr_tmp_);
    cached_cache_key_ = absl::nullopt;
    return *this;
  }

  size_t NumAttributes() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh mht_6(mht_6_v, 315, "", "./tensorflow/core/common_runtime/eager/attr_builder.h", "NumAttributes");
 return encoded_attrs_.size(); }

  AttrBuilder& Set(StringPiece attr_name, const AttrValue& value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh mht_7(mht_7_v, 320, "", "./tensorflow/core/common_runtime/eager/attr_builder.h", "Set");

    AddAttrIfNotPresent(attr_name, value);
    cached_cache_key_ = absl::nullopt;
    return *this;
  }

  // Retrieves the attribute value.
  // Note that Get() can involve a linear scan of all attributes with the same
  // value type in this Node. This is not an issue, because Get is used rarely
  // and nodes have a small number of attributes.
  template <class T>
  Status Get(StringPiece attr_name, T* value) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh mht_8(mht_8_v, 334, "", "./tensorflow/core/common_runtime/eager/attr_builder.h", "Get");

    // Common attributes are stored in AttrVecs. This Get() template
    // is specialized for them below. If we end up here, the type must be
    // among those that we store in the node_def_.
    if (!node_def_initialized_) {
      return errors::NotFound("No attr named'", attr_name,
                              "' found in AttrBuilder for ", op_name_);
    }
    return GetNodeAttr(AttrSlice(node_def_), attr_name, value);
  }

  tensorflow::Fprint128 CacheKey(const StringPiece device);

  // Fill `m` with the attr-value pairs set via AttrBuilder::Set() so far, as
  // well as any default attr-value pairs from the associated op_def, if there
  // is one.
  void FillAttrValueMap(AttrValueMap* m) const;

  // Fill `m` with the attr-value pairs set via AttrBuilder::Set() so far except
  // when the value matches the default for this attr.
  // More precisely, if the global op registry contains an OpDef for this op
  // and if an attribute value is the same as the default (according to the
  // OpDef), this attr-value pair is not added to `m`.
  void FillAttrValueMapWithoutDefaults(AttrValueMap* m) const;
  const NodeDef& BuildNodeDef();

  // Transfers the attributes from `other` to this AttrBuilder. Does not
  // overwrite existing attributes. Since it does not require deserializing and
  // re-serializing attributes, it is much more efficient than going through an
  // AttrValueMap.
  void CopyAttributes(const AttrBuilder& other);

  void GetNameAttrList(tensorflow::NameAttrList* name_and_attrs) const override;

  bool GetInt(absl::string_view attr_name, int64_t* result) const override;
  bool GetFloat(absl::string_view attr_name, float* result) const override;
  bool GetBool(absl::string_view attr_name, bool* result) const override;
  bool GetType(absl::string_view attr_name,
               tensorflow::DataType* result) const override;
  Status GetTypeList(
      absl::string_view attr_name,
      absl::InlinedVector<DataType, 4>* type_list) const override;

 private:
  tensorflow::Fprint128 BuildCacheKeyForDevice(const StringPiece device) const;

  // Initialize the node_def_ object.
  // REQUIRES: node_def_initialized_ = false
  void InitializeNodeDef();

  template <class T>
  void SetInAttrValueMap(AttrValueMap* m, const string& attr_name,
                         T&& value) const {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSattr_builderDTh mht_9(mht_9_v, 390, "", "./tensorflow/core/common_runtime/eager/attr_builder.h", "SetInAttrValueMap");

    DCHECK(!node_def_finalized_)
        << "Calling SetInAttrValueMap after BuildNodeDef.";
    // If attribute is set more than once, its first value prevails
    m->insert({attr_name, value});
  }

  void AddAttrIfNotPresent(StringPiece attr_name, const AttrValue& value);

  gtl::FlatMap<string, string> encoded_attrs_;
  mutable AttrValue attr_tmp_;  // For encoding

  string op_name_;  // Conceptually const, but can't be because of Reset(...)
  int num_inputs_;
  NodeDef node_def_;
  bool node_def_initialized_;
  bool node_def_finalized_;

  absl::optional<tensorflow::Fprint128> cached_cache_key_;
  string device_for_cached_cache_key_;
};

template <>
Status AttrBuilder::Get(StringPiece attr_name, int* value) const;
template <>
Status AttrBuilder::Get(StringPiece attr_name, float* value) const;
template <>
Status AttrBuilder::Get(StringPiece attr_name, bool* value) const;
template <>
Status AttrBuilder::Get(StringPiece attr_name,
                        tensorflow::DataType* value) const;
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_ATTR_BUILDER_H_
