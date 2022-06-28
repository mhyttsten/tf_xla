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
class MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc {
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
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc() {
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

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/strcat.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

// Implement a simple hash table as a Resource using ref-counting.
// This demonstrates a Stateful Op for a general Create/Read/Update/Delete
// (CRUD) style use case.  To instead make an op for a specific lookup table
// case, it is preferable to follow the implementation style of
// the kernels for TensorFlow's tf.lookup internal ops which use
// LookupInterface.
template <class K, class V>
class SimpleHashTableResource : public ::tensorflow::ResourceBase {
 public:
  Status Insert(const Tensor& key, const Tensor& value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_0(mht_0_v, 206, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Insert");

    const K key_val = key.flat<K>()(0);
    const V value_val = value.flat<V>()(0);

    mutex_lock l(mu_);
    table_[key_val] = value_val;
    return Status::OK();
  }

  Status Find(const Tensor& key, Tensor* value, const Tensor& default_value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_1(mht_1_v, 218, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Find");

    // Note that tf_shared_lock could be used instead of mutex_lock
    // in ops that do not not modify data protected by a mutex, but
    // go/totw/197 recommends using exclusive lock instead of a shared
    // lock when the lock is not going to be held for a significant amount
    // of time.
    mutex_lock l(mu_);

    const V default_val = default_value.flat<V>()(0);
    const K key_val = key.flat<K>()(0);
    auto value_val = value->flat<V>();
    value_val(0) = gtl::FindWithDefault(table_, key_val, default_val);
    return Status::OK();
  }

  Status Remove(const Tensor& key) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_2(mht_2_v, 236, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Remove");

    mutex_lock l(mu_);

    const K key_val = key.flat<K>()(0);
    if (table_.erase(key_val) != 1) {
      return errors::NotFound("Key for remove not found: ", key_val);
    }
    return Status::OK();
  }

  // Save all key, value pairs to tensor outputs to support SavedModel
  Status Export(OpKernelContext* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_3(mht_3_v, 250, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Export");

    mutex_lock l(mu_);
    int64_t size = table_.size();
    Tensor* keys;
    Tensor* values;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({size}), &keys));
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("values", TensorShape({size}), &values));
    auto keys_data = keys->flat<K>();
    auto values_data = values->flat<V>();
    int64_t i = 0;
    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      keys_data(i) = it->first;
      values_data(i) = it->second;
    }
    return Status::OK();
  }

  // Load all key, value pairs from tensor inputs to support SavedModel
  Status Import(const Tensor& keys, const Tensor& values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_4(mht_4_v, 273, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Import");

    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();

    mutex_lock l(mu_);
    table_.clear();
    for (int64_t i = 0; i < key_values.size(); ++i) {
      gtl::InsertOrUpdate(&table_, key_values(i), value_values(i));
    }
    return Status::OK();
  }

  // Create a debug string with the content of the map if this is small,
  // or some example data if this is large, handling both the cases where the
  // hash table has many entries and where the entries are long strings.
  std::string DebugString() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_5(mht_5_v, 291, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "DebugString");
 return DebugString(3); }
  std::string DebugString(int num_pairs) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_6(mht_6_v, 295, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "DebugString");

    std::string rval = "SimpleHashTable {";
    size_t count = 0;
    const size_t max_kv_str_len = 100;
    mutex_lock l(mu_);
    for (const auto& pair : table_) {
      if (count >= num_pairs) {
        strings::StrAppend(&rval, "...");
        break;
      }
      std::string kv_str = strings::StrCat(pair.first, ": ", pair.second);
      strings::StrAppend(&rval, kv_str.substr(0, max_kv_str_len));
      if (kv_str.length() > max_kv_str_len) strings::StrAppend(&rval, " ...");
      strings::StrAppend(&rval, ", ");
      count += 1;
    }
    strings::StrAppend(&rval, "}");
    return rval;
  }

 private:
  mutable mutex mu_;
  absl::flat_hash_map<K, V> table_ TF_GUARDED_BY(mu_);
};

template <class K, class V>
class SimpleHashTableCreateOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableCreateOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_7(mht_7_v, 327, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "SimpleHashTableCreateOpKernel");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_8(mht_8_v, 332, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Compute");

    Tensor handle_tensor;
    AllocatorAttributes attr;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                           &handle_tensor, attr));
    handle_tensor.scalar<ResourceHandle>()() =
        ResourceHandle::MakeRefCountingHandle(
            new SimpleHashTableResource<K, V>(), ctx->device()->name(),
            /*dtypes_and_shapes=*/{}, ctx->stack_trace());
    ctx->set_output(0, handle_tensor);
  }

 private:
  // Just to be safe, avoid accidentally copying the kernel.
  TF_DISALLOW_COPY_AND_ASSIGN(SimpleHashTableCreateOpKernel);
};

// GetResource retrieves a Resource using a handle from the first
// input in "ctx" and saves it in "resource" without increasing
// the reference count for that resource.
template <class K, class V>
Status GetResource(OpKernelContext* ctx,
                   SimpleHashTableResource<K, V>** resource) {
  const Tensor& handle_tensor = ctx->input(0);
  const ResourceHandle& handle = handle_tensor.scalar<ResourceHandle>()();
  typedef SimpleHashTableResource<K, V> resource_type;
  TF_ASSIGN_OR_RETURN(*resource, handle.GetResource<resource_type>());
  return Status::OK();
}

template <class K, class V>
class SimpleHashTableFindOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableFindOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_9(mht_9_v, 369, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "SimpleHashTableFindOpKernel");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_10(mht_10_v, 374, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Compute");

    DataTypeVector expected_inputs = {DT_RESOURCE, DataTypeToEnum<K>::v(),
                                      DataTypeToEnum<V>::v()};
    DataTypeVector expected_outputs = {DataTypeToEnum<V>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));
    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    // Note that ctx->input(0) is the Resource handle
    const Tensor& key = ctx->input(1);
    const Tensor& default_value = ctx->input(2);
    TensorShape output_shape = default_value.shape();
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("value", output_shape, &out));
    OP_REQUIRES_OK(ctx, resource->Find(key, out, default_value));
  }
};

template <class K, class V>
class SimpleHashTableInsertOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableInsertOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_11(mht_11_v, 398, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "SimpleHashTableInsertOpKernel");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_12(mht_12_v, 403, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Compute");

    DataTypeVector expected_inputs = {DT_RESOURCE, DataTypeToEnum<K>::v(),
                                      DataTypeToEnum<V>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));
    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    // Note that ctx->input(0) is the Resource handle
    const Tensor& key = ctx->input(1);
    const Tensor& value = ctx->input(2);
    OP_REQUIRES_OK(ctx, resource->Insert(key, value));
  }
};

template <class K, class V>
class SimpleHashTableRemoveOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableRemoveOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_13(mht_13_v, 423, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "SimpleHashTableRemoveOpKernel");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_14(mht_14_v, 428, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Compute");

    DataTypeVector expected_inputs = {DT_RESOURCE, DataTypeToEnum<K>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));
    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    // Note that ctx->input(0) is the Resource handle
    const Tensor& key = ctx->input(1);
    OP_REQUIRES_OK(ctx, resource->Remove(key));
  }
};

template <class K, class V>
class SimpleHashTableExportOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableExportOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_15(mht_15_v, 446, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "SimpleHashTableExportOpKernel");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_16(mht_16_v, 451, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Compute");

    DataTypeVector expected_inputs = {DT_RESOURCE};
    DataTypeVector expected_outputs = {DataTypeToEnum<K>::v(),
                                       DataTypeToEnum<V>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));
    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    OP_REQUIRES_OK(ctx, resource->Export(ctx));
  }
};

template <class K, class V>
class SimpleHashTableImportOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableImportOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_17(mht_17_v, 469, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "SimpleHashTableImportOpKernel");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_kernelDTcc mht_18(mht_18_v, 474, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_kernel.cc", "Compute");

    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    DataTypeVector expected_inputs = {DT_RESOURCE, DataTypeToEnum<K>::v(),
                                      DataTypeToEnum<V>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));
    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES(
        ctx, keys.shape() == values.shape(),
        errors::InvalidArgument("Shapes of keys and values are not the same: ",
                                keys.shape().DebugString(), " for keys, ",
                                values.shape().DebugString(), "for values."));

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = resource->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, resource->Import(keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(resource->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

// The "Name" used by REGISTER_KERNEL_BUILDER is defined by REGISTER_OP,
// see simple_hash_table_op.cc.
#define REGISTER_KERNEL(key_dtype, value_dtype)               \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableCreate")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableCreateOpKernel<key_dtype, value_dtype>); \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableFind")                    \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableFindOpKernel<key_dtype, value_dtype>);   \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableInsert")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableInsertOpKernel<key_dtype, value_dtype>)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableRemove")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableRemoveOpKernel<key_dtype, value_dtype>)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableExport")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableExportOpKernel<key_dtype, value_dtype>)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableImport")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableImportOpKernel<key_dtype, value_dtype>);

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int32, tstring);
REGISTER_KERNEL(int64_t, double);
REGISTER_KERNEL(int64_t, float);
REGISTER_KERNEL(int64_t, int32);
REGISTER_KERNEL(int64_t, int64_t);
REGISTER_KERNEL(int64_t, tstring);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64_t);
REGISTER_KERNEL(tstring, tstring);
#undef REGISTER_KERNEL

}  // namespace custom_op_examples
}  // namespace tensorflow
