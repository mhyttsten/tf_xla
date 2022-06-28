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

#ifndef TENSORFLOW_CORE_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
#define TENSORFLOW_CORE_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh() {
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


#include <atomic>

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace lookup {

// Base class for lookup tables that require initialization.
class InitializableLookupTable : public LookupInterface {
 public:
  class InitTableIterator;
  class InitializerSerializer;

  // Performs batch lookups, for every element in the key tensor, Find returns
  // the corresponding value into the values tensor.
  // If an element is not present in the table, the given default value is used.
  //
  // For tables that require initialization, `Find` is available once the table
  // is marked as initialized.
  //
  // Returns the following statuses:
  // - OK: when the find finishes successfully.
  // - FailedPrecondition: if the table is not initialized.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - In addition, other implementations may provide another non-OK status
  //   specific to their failure modes.
  Status Find(OpKernelContext* ctx, const Tensor& keys, Tensor* values,
              const Tensor& default_value) final;

  // Returns errors::Unimplemented.
  Status Insert(OpKernelContext* ctx, const Tensor& keys,
                const Tensor& values) final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "Insert");

    return errors::Unimplemented(
        "Insert not supported by InitializableLookupTable implementations");
  }

  // Returns errors::Unimplemented.
  Status Remove(OpKernelContext* ctx, const Tensor& keys) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_1(mht_1_v, 230, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "Remove");

    return errors::Unimplemented(
        "Remove not supported by InitializableLookupTable implementations");
  }

  Status ExportValues(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_2(mht_2_v, 238, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "ExportValues");

    return errors::Unimplemented(
        "ExportValues not supported by InitializableLookupTable "
        "implementations");
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) final;

  TensorShape key_shape() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_3(mht_3_v, 250, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "key_shape");
 return TensorShape(); }

  TensorShape value_shape() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_4(mht_4_v, 255, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "value_shape");
 return TensorShape(); }

  // Returns whether the table was initialized and is ready to serve lookups.
  bool is_initialized() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_5(mht_5_v, 261, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "is_initialized");

    return is_initialized_.load(std::memory_order_acquire);
  }

  // Initializes the table from the given init table iterator.
  //
  // Atomically, this operation prepares the table, populates it with the given
  // iterator, and marks the table as initialized.
  //
  // Returns the following statuses:
  // - OK: when the initialization was successful.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - FailedPrecondition: if the table is already initialized and
  //   fail_if_initialized is set to true.
  // - In addition, other implementations may provide another non-OK status
  //   specific to their failure modes.
  Status Initialize(InitTableIterator& iter);

  // Initializes the table from the given init table iterator. `serializer` may
  // specify how to serialize the table initializer, so that the table can be
  // serialized using its metadata (as opposed to serializing a handle to the
  // table).
  Status Initialize(InitTableIterator& iter,
                    std::unique_ptr<InitializerSerializer> serializer);

  // Basic iterator to initialize lookup tables.
  // It yields a sequence of pairs of `keys()` and `values()` Tensors, so that
  // the consumer may insert key-value pairs in batches.
  //
  // Then the iterator is exhausted, valid returns false and status returns
  // Status::OutOfRange.
  //
  // This class is Thread-unsafe.
  class InitTableIterator {
   public:
    InitTableIterator() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_6(mht_6_v, 300, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "InitTableIterator");
}

    virtual ~InitTableIterator() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_7(mht_7_v, 305, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "~InitTableIterator");
}

    // Prepares the next batch of key and value tensors.
    virtual void Next() = 0;

    // Returns true if keys and values point to valid tensors.
    virtual bool Valid() const = 0;

    // Returns a tensor that contains the current batch of 'key' values.
    virtual const Tensor& keys() const = 0;

    // Returns a tensor that contains the current batch of 'value' values.
    virtual const Tensor& values() const = 0;

    // Returns an error if one has occurred, otherwise returns Status::OK.
    virtual Status status() const = 0;

    // Returns the total number of elements that the iterator will produce.
    // It might return -1 in case of error.
    virtual int64_t total_size() const = 0;

   private:
    TF_DISALLOW_COPY_AND_ASSIGN(InitTableIterator);
  };

  InitializableLookupTable* GetInitializableLookupTable() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_8(mht_8_v, 333, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "GetInitializableLookupTable");

    return this;
  }

  // Logic specifying how to represent an initializer as a GraphDef, so that a
  // lookup table can be serialized using its metadata (as opposed to
  // serializing the content of the table, or a handle to the table).
  class InitializerSerializer {
   public:
    // A function which builds a graph so that executing `*out` will initialize
    // `table`.
    using SerializeFn = std::function<Status(GraphDefBuilder* builder,
                                             Node* table, Node** out)>;
    // A function which performs any necessary cleanup for the serializer.
    using CleanupFn = std::function<void()>;

    // Wraps serialization logic that requires no cleanup.
    explicit InitializerSerializer(SerializeFn serialize)
        : serialize_(std::move(serialize)), cleanup_([] {}) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_9(mht_9_v, 354, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "InitializerSerializer");
}

    // Wraps serialization logic along with a cleanup function. `cleanup` will
    // be run when the serializer is destroyed.
    explicit InitializerSerializer(SerializeFn serialize, CleanupFn cleanup)
        : serialize_(std::move(serialize)), cleanup_(std::move(cleanup)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_10(mht_10_v, 362, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "InitializerSerializer");
}

    ~InitializerSerializer() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_11(mht_11_v, 367, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "~InitializerSerializer");
 cleanup_(); }

    // Builds a graph so that executing `*out` will initialize `table`.
    Status AsGraphDef(GraphDefBuilder* builder, Node* table, Node** out) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_12(mht_12_v, 373, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "AsGraphDef");

      return serialize_(builder, table, out);
    }

   private:
    SerializeFn serialize_;
    CleanupFn cleanup_;
  };

 protected:
  // Prepares and allocates the underlying data structure to store the given
  // number of expected elements.
  virtual Status DoPrepare(size_t expected_num_elements) = 0;

  // Same as DoPrepare() but derived implementations might choose to skip
  // calling get_expected_num_elements if size is not needed for DoPrepare.
  virtual Status DoLazyPrepare(
      std::function<int64_t(void)> get_expected_num_elements) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_13(mht_13_v, 393, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "DoLazyPrepare");

    int64_t expected_num_elements = get_expected_num_elements();
    if (expected_num_elements < 0) {
      return errors::FailedPrecondition("Got negative expected_num_elements.");
    }
    return DoPrepare(expected_num_elements);
  }

  // Populates the table in batches given keys and values as tensors into the
  // underlying data structure.
  virtual Status DoInsert(const Tensor& keys, const Tensor& values) = 0;

  // Performs the batch find operation on the underlying data structure.
  virtual Status DoFind(const Tensor& keys, Tensor* values,
                        const Tensor& default_value) = 0;

  virtual Status AreEntriesSame(const InitTableIterator& iter, bool* result);

  mutex mu_;

 protected:
  // When set, provides a mechanism for serializing the table initializer as
  // GraphDef.
  std::unique_ptr<InitializerSerializer> initializer_serializer_;

 private:
  std::atomic<bool> is_initialized_{false};
};

// Iterator to initialize tables given 'keys' and 'values' tensors.
//
// The two tensors are returned in the first iteration. It doesn't loop
// over each element of the tensor since insertions in the lookup table can
// process batches.
class KeyValueTensorIterator
    : public InitializableLookupTable::InitTableIterator {
 public:
  // keys and values are not owned by the iterator.
  explicit KeyValueTensorIterator(const Tensor* keys, const Tensor* values)
      : keys_(keys), values_(values), valid_(true), status_(Status::OK()) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_14(mht_14_v, 435, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "KeyValueTensorIterator");

    TensorShape key_shape = keys_->shape();
    if (!key_shape.IsSameSize(values_->shape())) {
      valid_ = false;
      status_ = errors::InvalidArgument(
          "keys and values should have the same dimension.",
          key_shape.DebugString(), " vs ", values_->shape().DebugString());
    }
    if (key_shape.num_elements() == 0) {
      valid_ = false;
      status_ =
          errors::InvalidArgument("keys and values cannot be empty tensors.");
    }
  }

  bool Valid() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_15(mht_15_v, 453, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "Valid");
 return valid_; }

  void Next() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_16(mht_16_v, 458, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "Next");

    valid_ = false;
    status_ = errors::OutOfRange("No more data.");
  }

  const Tensor& keys() const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_17(mht_17_v, 466, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "keys");
 return *keys_; }

  const Tensor& values() const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_18(mht_18_v, 471, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "values");
 return *values_; }

  Status status() const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_19(mht_19_v, 476, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "status");
 return status_; }

  int64_t total_size() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinitializable_lookup_tableDTh mht_20(mht_20_v, 481, "", "./tensorflow/core/kernels/initializable_lookup_table.h", "total_size");

    return keys_ == nullptr ? -1 : keys_->NumElements();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(KeyValueTensorIterator);

  const Tensor* keys_;    // Doesn't own it.
  const Tensor* values_;  // Doesn't own it.
  bool valid_;            // true if the iterator points to an existing range.
  Status status_;
};

}  // namespace lookup
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
