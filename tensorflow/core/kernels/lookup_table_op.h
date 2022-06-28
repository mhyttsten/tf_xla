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

#ifndef TENSORFLOW_CORE_KERNELS_LOOKUP_TABLE_OP_H_
#define TENSORFLOW_CORE_KERNELS_LOOKUP_TABLE_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh() {
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


#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Lookup table op that supports different table implementations specified by
// the 'Container' template. Container must be derived from LookupInterface. The
// key and value are of the templated type "key_dtype" and "value_dtype"
// respectively.
template <class Container, class key_dtype, class value_dtype>
class LookupTableOp : public OpKernel {
 public:
  // ctx is not owned by this class.
  explicit LookupTableOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_set_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/lookup_table_op.h", "LookupTableOp");

    if (ctx->output_type(0) == DT_RESOURCE) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(tensorflow::DT_RESOURCE,
                                        tensorflow::TensorShape({}), &table_));
    } else {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(tensorflow::DT_STRING,
                                        tensorflow::TensorShape({2}), &table_));
    }
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
  }

  // ctx is not owned by this function.
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_1(mht_1_v, 233, "", "./tensorflow/core/kernels/lookup_table_op.h", "Compute");

    mutex_lock l(mu_);

    if (!table_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator =
        [ctx, this](lookup::LookupInterface** ret)
            TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_2(mht_2_v, 246, "", "./tensorflow/core/kernels/lookup_table_op.h", "lambda");

              lookup::LookupInterface* container = new Container(ctx, this);
              if (!ctx->status().ok()) {
                container->Unref();
                return ctx->status();
              }
              if (ctx->track_allocations()) {
                ctx->record_persistent_memory_allocation(
                    container->MemoryUsed() + table_.AllocatedBytes());
              }
              *ret = container;
              return Status::OK();
            };

    lookup::LookupInterface* table = nullptr;
    OP_REQUIRES_OK(ctx,
                   cinfo_.resource_manager()
                       ->template LookupOrCreate<lookup::LookupInterface>(
                           cinfo_.container(), cinfo_.name(), &table, creator));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, lookup::CheckTableDataTypes(
                            *table, DataTypeToEnum<key_dtype>::v(),
                            DataTypeToEnum<value_dtype>::v(), cinfo_.name()));

    if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
      if (!table_set_) {
        auto h = table_.template scalar<ResourceHandle>();
        h() = MakeResourceHandle<lookup::LookupInterface>(
            ctx, cinfo_.container(), cinfo_.name());
      }
      ctx->set_output(0, table_);
    } else {
      if (!table_set_) {
        auto h = table_.template flat<tstring>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
      ctx->set_output_ref(0, &mu_, &table_);
    }
    table_set_ = true;
  }

  ~LookupTableOp() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_3(mht_3_v, 292, "", "./tensorflow/core/kernels/lookup_table_op.h", "~LookupTableOp");

    // If the table object was not shared, delete it.
    if (table_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<lookup::LookupInterface>(cinfo_.container(),
                                                          cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

 private:
  mutex mu_;
  Tensor table_ TF_GUARDED_BY(mu_);
  bool table_set_ TF_GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(LookupTableOp);
};

// An anonymous version of LookupTableOp, which creates a new table resource
// everytime `Compute` is called. The resource can only be accessed by the
// returned resource handle (e.g. it can't be looked up by a name in a resource
// manager). The resource will be automatically deleted when all resource
// handles pointing to it are gone.
template <class Container, class key_dtype, class value_dtype>
class AnonymousLookupTableOp : public OpKernel {
 public:
  explicit AnonymousLookupTableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_4(mht_4_v, 325, "", "./tensorflow/core/kernels/lookup_table_op.h", "AnonymousLookupTableOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_5(mht_5_v, 330, "", "./tensorflow/core/kernels/lookup_table_op.h", "Compute");

    lookup::LookupInterface* table = new Container(ctx, this);
    if (!ctx->status().ok()) {
      table->Unref();
      return;
    }
    Tensor table_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(tensorflow::DT_RESOURCE,
                                tensorflow::TensorShape({}), &table_tensor));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() +
                                               table_tensor.AllocatedBytes());
    }
    table_tensor.scalar<ResourceHandle>()() =
        ResourceHandle::MakeRefCountingHandle<lookup::LookupInterface>(
            table, ctx->device()->name());
    ctx->set_output(0, table_tensor);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AnonymousLookupTableOp);
};

namespace lookup {

// Ensure that the compiler cannot elide a copy into a local, for
// bounds checking on source tensors that might be updated asynchronously for
// integral types. However non-integer variables are not allowed and therefore
// the local copy is unnecessary.
template <typename T>
T SubtleMustCopyIfIntegral(const T& value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_6(mht_6_v, 364, "", "./tensorflow/core/kernels/lookup_table_op.h", "SubtleMustCopyIfIntegral");

  return internal::SubtleMustCopy(value);
}

inline const tstring& SubtleMustCopyIfIntegral(const tstring& value) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("value: \"" + (std::string)value + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_7(mht_7_v, 372, "", "./tensorflow/core/kernels/lookup_table_op.h", "SubtleMustCopyIfIntegral");

  return value;
}

inline const float SubtleMustCopyIfIntegral(const float value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_8(mht_8_v, 379, "", "./tensorflow/core/kernels/lookup_table_op.h", "SubtleMustCopyIfIntegral");
 return value; }

inline const double SubtleMustCopyIfIntegral(const double value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_9(mht_9_v, 384, "", "./tensorflow/core/kernels/lookup_table_op.h", "SubtleMustCopyIfIntegral");

  return value;
}

inline const Variant& SubtleMustCopyIfIntegral(const Variant& value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_10(mht_10_v, 391, "", "./tensorflow/core/kernels/lookup_table_op.h", "SubtleMustCopyIfIntegral");

  return value;
}

inline const ResourceHandle& SubtleMustCopyIfIntegral(
    const ResourceHandle& value) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_11(mht_11_v, 399, "", "./tensorflow/core/kernels/lookup_table_op.h", "SubtleMustCopyIfIntegral");

  return value;
}

// Returns a unique node name starting with "base".
std::string UniqueNodeName(const std::string& base);

// Lookup table that wraps an flat_hash_map, where the key and value data type
// is specified.
//
// This table is recommended for any variations to key values.
//
// For look up, the table is required to be initialized (allocated
// and populated). Once the table is marked as initialized it becomes read-only.
//
// Sample use case:
//
// HashTable<int64, int64> table;  // int64 -> int64.
// table.Initialize(...);
// table.Find(in_t, &out_t, default_t)
//
template <class K, class V>
class HashTable : public InitializableLookupTable {
 public:
  HashTable(OpKernelContext* ctx, OpKernel* kernel) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_12(mht_12_v, 426, "", "./tensorflow/core/kernels/lookup_table_op.h", "HashTable");
}

  Status AsGraphDef(GraphDefBuilder* builder, Node** out) const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_13(mht_13_v, 431, "", "./tensorflow/core/kernels/lookup_table_op.h", "AsGraphDef");

    // We set use_node_name_sharing with a unique node name so that the resource
    // can outlive the HashTableV2 kernel. This means that the lifetime of the
    // HashTable resource will be tied to the lifetime of the resource manager
    // it is created in.
    // TODO(b/181695913): Provide a mechanism for deleting this resource
    // earlier when appropriate.
    Node* hash_table_node = ops::SourceOp(
        "HashTableV2", builder->opts()
                           .WithName(UniqueNodeName("HashTableFromGraphDef"))
                           .WithAttr("key_dtype", key_dtype())
                           .WithAttr("value_dtype", value_dtype())
                           .WithAttr("use_node_name_sharing", true));
    if (table_.empty()) {
      *out = hash_table_node;
      return Status::OK();
    }

    if (initializer_serializer_ == nullptr) {
      std::string message =
          "Failed to serialize lookup table: no initialization function was "
          "specified. Falling back to serializing a handle to the table.";
      LOG(WARNING) << message;
      return errors::Unimplemented(message);
    }
    Node* initializer;
    TF_RETURN_IF_ERROR(initializer_serializer_->AsGraphDef(
        builder, hash_table_node, &initializer));
    *out = ops::UnaryOp("Identity", hash_table_node,
                        builder->opts().WithControlInput(initializer));
    return Status::OK();
  }

  size_t size() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_14(mht_14_v, 467, "", "./tensorflow/core/kernels/lookup_table_op.h", "size");

    if (!is_initialized())
      return 0;
    else
      return table_.size();
  }

  Status ExportValues(OpKernelContext* context) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_15(mht_15_v, 477, "", "./tensorflow/core/kernels/lookup_table_op.h", "ExportValues");

    if (!is_initialized()) {
      return errors::Aborted("HashTable is not initialized.");
    }

    const int64_t size = table_.size();

    Tensor* keys;
    Tensor* values;
    TF_RETURN_IF_ERROR(
        context->allocate_output("keys", TensorShape({size}), &keys));
    TF_RETURN_IF_ERROR(
        context->allocate_output("values", TensorShape({size}), &values));

    auto keys_data = keys->flat<K>();
    auto values_data = values->flat<V>();
    int64_t i = 0;
    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      keys_data(i) = it->first;
      values_data(i) = it->second;
    }
    return Status::OK();
  }

  DataType key_dtype() const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_16(mht_16_v, 504, "", "./tensorflow/core/kernels/lookup_table_op.h", "key_dtype");
 return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_17(mht_17_v, 509, "", "./tensorflow/core/kernels/lookup_table_op.h", "value_dtype");
 return DataTypeToEnum<V>::v(); }

 protected:
  Status DoPrepare(size_t size) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_18(mht_18_v, 515, "", "./tensorflow/core/kernels/lookup_table_op.h", "DoPrepare");

    if (is_initialized()) {
      return errors::Aborted("HashTable already initialized.");
    }
    if (size > 0) {
      table_.reserve(size);
    }
    return Status::OK();
  };

  Status DoLazyPrepare(std::function<int64(void)> size_fn) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_19(mht_19_v, 528, "", "./tensorflow/core/kernels/lookup_table_op.h", "DoLazyPrepare");

    return DoPrepare(size_fn());
  }

  Status DoInsert(const Tensor& keys, const Tensor& values) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_20(mht_20_v, 535, "", "./tensorflow/core/kernels/lookup_table_op.h", "DoInsert");

    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();
    for (int64_t i = 0; i < key_values.size(); ++i) {
      auto&& key = SubtleMustCopyIfIntegral(key_values(i));
      auto&& value = SubtleMustCopyIfIntegral(value_values(i));
      auto result = table_.try_emplace(key, value);
      if (!result.second && result.first->second != value) {
        return errors::FailedPrecondition(
            "HashTable has different value for same key. Key ", key, " has ",
            result.first->second, " and trying to add value ", value);
      }
    }
    return Status::OK();
  }

  Status DoFind(const Tensor& key, Tensor* value,
                const Tensor& default_value) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_21(mht_21_v, 555, "", "./tensorflow/core/kernels/lookup_table_op.h", "DoFind");

    const V default_val = default_value.flat<V>()(0);
    const auto key_values = key.flat<K>();
    auto value_values = value->flat<V>();

    for (int64_t i = 0; i < key_values.size(); ++i) {
      value_values(i) = gtl::FindWithDefault(
          table_, SubtleMustCopyIfIntegral(key_values(i)), default_val);
    }
    return Status::OK();
  }

  int64_t MemoryUsed() const override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_opDTh mht_22(mht_22_v, 570, "", "./tensorflow/core/kernels/lookup_table_op.h", "MemoryUsed");

    if (!is_initialized()) {
      return 0;
    }
    const int64_t num_elements = table_.size();
    return num_elements * (sizeof(K) + sizeof(V));
  }

 private:
  absl::flat_hash_map<K, V> table_;
};

}  // namespace lookup

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOOKUP_TABLE_OP_H_
