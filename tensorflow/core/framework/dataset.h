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
#ifndef TENSORFLOW_CORE_FRAMEWORK_DATASET_H_
#define TENSORFLOW_CORE_FRAMEWORK_DATASET_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh() {
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


#include <deque>
#include <memory>
#include <unordered_map>

#include "absl/memory/memory.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/dataset_metadata.pb.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/dataset_stateful_op_allowlist.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/thread_factory.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/threadpool_interface.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/tracing.h"

// Polymorphic datasets should support all primitive TensorFlow
// types. Use this macro to expand `m(T)` once for each primitive type
// `T`, e.g. to build a `switch` statement.
#define TF_CALL_DATASET_TYPES(m) TF_CALL_ALL_TYPES(m) TF_CALL_QUANTIZED_TYPES(m)

namespace tensorflow {

// Forward declarations to avoid introducing a dependency on headers in
// "tensorflow/core/graph/...".
class GraphDefBuilder;
class Node;

namespace data {

namespace internal {
// Merges Options from source to destination. If there is a conflict on a field,
// the field value from the source takes precedence.
void MergeOptions(const protobuf::Message& source,
                  protobuf::Message* destination);
void MergeOptions(const protobuf::MessageLite& source,
                  protobuf::MessageLite* destination);
}  // namespace internal

using TraceMeMetadata = std::vector<std::pair<StringPiece, string>>;

constexpr char kTFDataFunction[] = "_tf_data_function";

constexpr int kInfiniteCardinality = -1;
constexpr int kUnknownCardinality = -2;

// This constant is a magic number that is used (as a prefix) to identify keys
// used for serialization of iterator state.
constexpr char kFullNameRandomHex[] = "60d899aa0d8ce4351e7c3b419e92d25b";
constexpr char kPipe[] = "|";
constexpr char kColon[] = ":";

constexpr char kTFDataResourceTag[] = "tfdata";
constexpr char kTraceInfoUnavailable[] = "unavailable";
constexpr char kMetadata[] = "metadata";

constexpr char kCardinalityAttrForRewrite[] = "_cardinality";

class DatasetBase;
class SerializationContext;

inline bool IsTFDataFunction(const FunctionDef& func) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_0(mht_0_v, 265, "", "./tensorflow/core/framework/dataset.h", "IsTFDataFunction");

  auto iter = func.attr().find(data::kTFDataFunction);
  return (iter != func.attr().end() && iter->second.b());
}

// Interface for reading values from a key-value store.
// Used for restoring iterator state. This class is thread safe.
// Please see comment on IteratorStateWriter for guidance around using the
// Read*(key, val) vs Read*(name, key, val).
class IteratorStateReader {
 public:
  // Determines whether the iterator state contains the given key.
  virtual bool Contains(StringPiece key) const = 0;
  virtual bool Contains(StringPiece name, StringPiece key) const = 0;

  // Reads an integer for the given key.
  virtual Status ReadScalar(StringPiece key, int64_t* val) const = 0;
  virtual Status ReadScalar(StringPiece name, StringPiece key,
                            int64_t* val) const = 0;

  // Reads a string for the given key.
  virtual Status ReadScalar(StringPiece key, tstring* val) const = 0;
  virtual Status ReadScalar(StringPiece name, StringPiece key,
                            tstring* val) const = 0;

  // Reads a tensor for the given key.
  // TODO(jsimsa): Remove non-FLR overrides once all callers are updated.
  virtual Status ReadTensor(StringPiece key, Tensor* val) const = 0;
  virtual Status ReadTensor(FunctionLibraryRuntime* flr, StringPiece key,
                            Tensor* val) const = 0;
  virtual Status ReadTensor(StringPiece name, StringPiece key,
                            Tensor* val) const = 0;
  virtual Status ReadTensor(FunctionLibraryRuntime* flr, StringPiece name,
                            StringPiece key, Tensor* val) const = 0;

  virtual ~IteratorStateReader() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_1(mht_1_v, 303, "", "./tensorflow/core/framework/dataset.h", "~IteratorStateReader");
}
};

// Interface for writing values to a key-value store.
// Used for saving iterator state. Not thread safe.
// The IteratorStateWriter creates a tensor for each unique iterator name it
// sees. For the Write*(key, val) API's the key is expected to encode this
// name as keys are required to be produced using the full_name() method.
// Each tensor has an upper limit of 2 GB and so if the state for an iterator
// might exceed the 2 GB limit, you can pass an explicit name in via the
// Write*(name, key, val) APIs allowing you to further split up the state
// into more manageable chunks.
class IteratorStateWriter {
 public:
  // Writes an integer for the given key.
  virtual Status WriteScalar(StringPiece key, const int64_t val) = 0;
  virtual Status WriteScalar(StringPiece name, StringPiece key,
                             const int64_t val) = 0;

  // Writes a string for the given key.
  virtual Status WriteScalar(StringPiece key, const tstring& val) = 0;
  virtual Status WriteScalar(StringPiece name, StringPiece key,
                             const tstring& val) = 0;

  // Writes a tensor for the given key.
  virtual Status WriteTensor(StringPiece key, const Tensor& val) = 0;
  virtual Status WriteTensor(StringPiece name, StringPiece key,
                             const Tensor& val) = 0;

  virtual ~IteratorStateWriter() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_2(mht_2_v, 335, "", "./tensorflow/core/framework/dataset.h", "~IteratorStateWriter");
}
};

// Generates a full name key for iterator checkpointing. All keys generated for
// iterator checkpoints should go through this function.
std::string FullName(const std::string& prefix, const std::string& name);

// Wrapper around GraphDefBuilder. Used to serialize Dataset graph.
class GraphDefBuilderWrapper {
 public:
  explicit GraphDefBuilderWrapper(GraphDefBuilder* b) : b_(b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_3(mht_3_v, 348, "", "./tensorflow/core/framework/dataset.h", "GraphDefBuilderWrapper");
}

  // Adds a Const node with scalar value to the Graph.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  template <typename T>
  Status AddScalar(const T& val, Node** output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_4(mht_4_v, 358, "", "./tensorflow/core/framework/dataset.h", "AddScalar");

    Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
    val_t.scalar<T>()() = val;
    AddTensorInternal(val_t, output);
    if (*output == nullptr) {
      return errors::Internal("AddScalar: Failed to build Const op.");
    }
    return Status::OK();
  }

  // Adds a Const node with vector value to the Graph.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  // TODO(shivaniagrawal): Consider changing to gtl::ArraySlice?
  template <typename T>
  Status AddVector(const std::vector<T>& val, Node** output) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_5(mht_5_v, 377, "", "./tensorflow/core/framework/dataset.h", "AddVector");

    Tensor val_t = Tensor(DataTypeToEnum<T>::v(),
                          TensorShape({static_cast<int64_t>(val.size())}));
    for (size_t i = 0; i < val.size(); i++) {
      val_t.flat<T>()(i) = val[i];
    }
    AddTensorInternal(val_t, output);
    if (*output == nullptr) {
      return errors::Internal("AddVector: Failed to build Const op.");
    }
    return Status::OK();
  }

  Status AddVector(const std::vector<string>& val, Node** output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_6(mht_6_v, 393, "", "./tensorflow/core/framework/dataset.h", "AddVector");

    Tensor val_t = Tensor(DataTypeToEnum<tstring>::v(),
                          TensorShape({static_cast<int64_t>(val.size())}));
    for (size_t i = 0; i < val.size(); i++) {
      val_t.flat<tstring>()(i) = val[i];
    }
    AddTensorInternal(val_t, output);
    if (*output == nullptr) {
      return errors::Internal("AddVector: Failed to build Const op.");
    }
    return Status::OK();
  }

  // Adds a `Const` node for the given tensor value to the graph.
  //
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status. The returned `Node`
  // pointer is owned by the backing graph of `GraphDefBuilder`.
  Status AddTensor(const Tensor& val, Node** output) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_7(mht_7_v, 414, "", "./tensorflow/core/framework/dataset.h", "AddTensor");

    AddTensorInternal(val, output);
    if (*output == nullptr) {
      return errors::Internal("AddTensor: Failed to build Const op.");
    }
    return Status::OK();
  }

  // Adds a `Placeholder` node for the given tensor value to the graph.
  //
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status. The returned `Node`
  // pointer is owned by the backing graph of `GraphDefBuilder`.
  Status AddPlaceholder(const Tensor& val, Node** output) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_8(mht_8_v, 430, "", "./tensorflow/core/framework/dataset.h", "AddPlaceholder");

    AddPlaceholderInternal(val, output);
    if (*output == nullptr) {
      return errors::Internal(
          "AddPlaceholder: Failed to build Placeholder op.");
    }
    return Status::OK();
  }

  // Adds a node for the given dataset to the `Graph`. The value of
  // `DatasetBase::type_string()` is used as the op type for the node. Values
  // for the `output_types` and `output_shapes` node attributes are also written
  // if those attributes are defined in the `OpDef`.
  //
  // If `use_dataset_name` is set, the value of `DatasetBase::node_name()` is
  // used as the op name for the node. This argument should only be set when
  // serializing `DatasetBase` instances which might not have been created
  // through op kernel execution to make sure the dataset op name is preserved
  // across serialization boundaries, which is in turn needed to make sure
  // iterator checkpoints are valid across serialization boundaries. When
  // `use_dataset_name` is set, the caller is responsible for making sure that
  // the op name is unique across the graph.
  //
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status. The returned `Node`
  // pointer is owned by the backing `Graph` of `GraphDefBuilder`.
  Status AddDataset(const DatasetBase* dataset,
                    const std::vector<Node*>& inputs, Node** output);
  Status AddDataset(const DatasetBase* dataset,
                    const std::vector<Node*>& inputs,
                    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
                    Node** output);
  Status AddDataset(
      const DatasetBase* dataset,
      const std::vector<std::pair<size_t, Node*>>& inputs,
      const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
      const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
      Node** output);
  Status AddDataset(
      const DatasetBase* dataset,
      const std::vector<std::pair<size_t, Node*>>& inputs,
      const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
      const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
      bool use_dataset_name, Node** output);

  // Adds a user-defined function with name `function_name` to the graph and
  // recursively adds all functions it references. If a function with a matching
  // name has already been added, returns with OK status. If a user-defined with
  // name `function_name` is not found in the context's function library,
  // returns an InvalidArgumentError. If the function with name `function_name`
  // or any of its dependent functions are stateful, and the context does not
  // explicitly permit stateful functions, returns an InvalidArgument error.
  Status AddFunction(SerializationContext* ctx, const string& function_name,
                     const FunctionLibraryDefinition& lib_def);

  template <typename T>
  void BuildAttrValue(const T& value, AttrValue* attr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_9(mht_9_v, 489, "", "./tensorflow/core/framework/dataset.h", "BuildAttrValue");

    SetAttrValue(value, attr);
  }

  template <typename T>
  AttrValue BuildAttrValue(const T& value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_10(mht_10_v, 497, "", "./tensorflow/core/framework/dataset.h", "BuildAttrValue");

    AttrValue attr;
    SetAttrValue(value, &attr);
    return attr;
  }

 protected:
  GraphDefBuilder* builder() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_11(mht_11_v, 507, "", "./tensorflow/core/framework/dataset.h", "builder");
 return b_; }

 private:
  void AddPlaceholderInternal(const Tensor& val, Node** output);
  void AddTensorInternal(const Tensor& val, Node** output);
  bool HasAttr(const string& op_type_name, const string& attr_name) const;

  bool HasAttr(const OpDef* op_def, const string& attr_name) const {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_12(mht_12_v, 518, "", "./tensorflow/core/framework/dataset.h", "HasAttr");

    for (const auto& attr : op_def->attr()) {
      if (attr.name() == attr_name) {
        return true;
      }
    }
    return false;
  }

  Status AddAttrFunctions(SerializationContext* ctx,
                          const AttrValue& attr_value,
                          const FunctionLibraryDefinition& lib_def) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_13(mht_13_v, 532, "", "./tensorflow/core/framework/dataset.h", "AddAttrFunctions");

    if (attr_value.has_func()) {
      TF_RETURN_IF_ERROR(AddFunction(ctx, attr_value.func().name(), lib_def));
    } else if (attr_value.has_list()) {
      for (const NameAttrList& name_attr_list : attr_value.list().func()) {
        TF_RETURN_IF_ERROR(AddFunction(ctx, name_attr_list.name(), lib_def));
      }
    }
    return Status::OK();
  }

  GraphDefBuilder* b_;
};

class StatsAggregator;

// A utility class for running a function and ensuring that there is always a
// `tensorflow::data` symbol on the stack.
class Runner {
 public:
  virtual ~Runner() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_14(mht_14_v, 555, "", "./tensorflow/core/framework/dataset.h", "~Runner");
}

  // Runs the given function.
  virtual void Run(const std::function<void()>& f) = 0;

  // Returns a global singleton Runner.
  static Runner* get();
};

// A class which provides a sequence of splits. Splits represent subdivisions of
// a dataset, e.g. filenames or ranges within files. We use splitting to
// partition input data into smaller pieces for distributed processing (see
// go/tf-data-splitting-design).
//
// Datasets provide a `MakeSplitProvider` method to expose a listing of their
// splits.
//
// Iterators created with a split provider will only iterate over the splits
// provided by the split provider.
class SplitProvider {
 public:
  virtual ~SplitProvider() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_15(mht_15_v, 579, "", "./tensorflow/core/framework/dataset.h", "~SplitProvider");
}
  // Stores the next split in `*split`, setting `*end_of_splits` to indicate
  // whether there were any splits left.
  virtual Status GetNext(Tensor* split, bool* end_of_splits) = 0;
  // Resets the split provider to its beginning.
  virtual Status Reset() = 0;
  // Saves the state of this split provider.
  virtual Status Save(std::function<std::string(std::string)> full_name,
                      IteratorStateWriter* writer) = 0;
  // Restores the state of this split provider.
  virtual Status Restore(std::function<std::string(std::string)> full_name,
                         IteratorStateReader* reader) = 0;
};

// Returns the runner threadpool size from an OpKernelContext.
int32_t GetRunnerThreadpoolSizeFromOpKernelContext(OpKernelContext* ctx);

// A cut-down version of `OpKernelContext` for running computations in
// iterators. Note that we cannot simply use `OpKernelContext` here because we
// might run computation in an iterator whose lifetime is not nested within the
// lifetime of a single `OpKernelContext` (e.g. asynchronous prefetching).
//
// TODO(mrry): We're making some daring assumptions about the lifetime of the
// runner passed in here. A runner will be deleted when the original step ends,
// but all existing runners only close over session-lifetime (or longer-lived)
// state, so we can make a copy of the function. There's nothing in the
// definition of the API from which we took the runner to guarantee that what we
// are doing is safe. We should formalize the properties here.
class IteratorContext {
 public:
  struct Params {
    explicit Params(IteratorContext* ctx)
        : allocator_getter(ctx->allocator_getter()),
          cancellation_manager(ctx->cancellation_manager()),
          collective_executor(ctx->collective_executor()),
          env(ctx->env()),
          flr(ctx->flr()),
          function_handle_cache(ctx->function_handle_cache()),
          interleave_depth(ctx->interleave_depth()),
          is_restoring(ctx->is_restoring()),
          model(ctx->model()),
          options(ctx->options()),
          resource_mgr(ctx->resource_mgr()),
          runner(*(ctx->runner())),
          runner_threadpool_size(ctx->runner_threadpool_size()),
          split_providers(ctx->split_providers()),
          stats_aggregator(ctx->stats_aggregator()),
          thread_factory(ctx->thread_factory()),
          thread_pool(ctx->thread_pool()) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_16(mht_16_v, 630, "", "./tensorflow/core/framework/dataset.h", "Params");
}

    explicit Params(OpKernelContext* ctx)
        : collective_executor(ctx->collective_executor()),
          env(ctx->env()),
          flr(ctx->function_library()) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_17(mht_17_v, 638, "", "./tensorflow/core/framework/dataset.h", "Params");

      // NOTE: need reinterpret_cast because function.h forward-declares Device.
      DeviceBase* device =
          reinterpret_cast<DeviceBase*>(ctx->function_library()->device());
      allocator_getter = [device](AllocatorAttributes attrs) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_18(mht_18_v, 645, "", "./tensorflow/core/framework/dataset.h", "lambda");

        return device->GetAllocator(attrs);
      };

      runner_threadpool_size = GetRunnerThreadpoolSizeFromOpKernelContext(ctx);

      // NOTE: Wrap every runner invocation in a call to Runner()->Run(), so
      // that a symbol in the tensorflow::data namespace is always on the stack
      // when executing a function inside a Dataset.
      runner = std::bind(
          [](
              // Note: `runner` is a const reference to avoid copying it.
              const std::function<void(std::function<void()>)>& ctx_runner,
              std::function<void()> fn) {
            std::function<void()> wrapped_fn = std::bind(
                [](const std::function<void()>& fn) { Runner::get()->Run(fn); },
                std::move(fn));
            ctx_runner(std::move(wrapped_fn));
          },
          *ctx->runner(), std::placeholders::_1);
    }

    // The Allocator to be used to allocate the output of an iterator.
    std::function<Allocator*(AllocatorAttributes)> allocator_getter = nullptr;

    // The CancellationManager to be used to cancel execution of ops.
    CancellationManager* cancellation_manager;

    // Collective support.
    CollectiveExecutor* collective_executor = nullptr;

    // Interface to operating system functionality.
    Env* env = nullptr;

    // The FunctionLibraryRuntime object to be used to make function calls.
    FunctionLibraryRuntime* flr = nullptr;

    // A FunctionHandleCache that owns all the function handles. Not owned.
    FunctionHandleCache* function_handle_cache = nullptr;

    // Records the number of ParallelInterleave operations in the path from the
    // root node to this node (not including this node) in the input pipeline
    // tree.
    int64 interleave_depth = 0;

    // Marks whether the iterator is restored from a checkpoint.
    bool is_restoring = false;

    // If non-null, identifies the object used for performance modeling.
    std::shared_ptr<model::Model> model = nullptr;

    // The input pipeline options.
    const Options* options = nullptr;

    // A resource manager for storing dataset-related state, e.g. random
    // seeds or cached tensors. Not owned.
    ResourceMgr* resource_mgr = nullptr;

    // Function call support.
    std::function<void(std::function<void()>)> runner = nullptr;

    // Number of threads used for executing user-defined functions.
    int32 runner_threadpool_size = 0;

    // Split providers indicating which splits to process. May be empty,
    // indicating that the iterator should process all splits.
    std::vector<std::shared_ptr<SplitProvider>> split_providers;

    // The `StatsAggregator` object to record statistics about the iterator.
    //
    // TODO(b/147325552): Remove this API and any of its uses after we switch to
    // using C++ based implementation for tf.data options (on 4/12/2021).
    std::shared_ptr<StatsAggregator> stats_aggregator = nullptr;

    // A factory for creating threads to perform blocking work.
    std::shared_ptr<ThreadFactory> thread_factory = nullptr;

    // A shared thread pool to schedule computation into.
    thread::ThreadPoolInterface* thread_pool = nullptr;
  };

  explicit IteratorContext(IteratorContext* ctx) : params_(Params{ctx}) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_19(mht_19_v, 729, "", "./tensorflow/core/framework/dataset.h", "IteratorContext");
}

  explicit IteratorContext(OpKernelContext* ctx) : params_(Params{ctx}) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_20(mht_20_v, 734, "", "./tensorflow/core/framework/dataset.h", "IteratorContext");
}

  explicit IteratorContext(Params params) : params_(std::move(params)) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_21(mht_21_v, 739, "", "./tensorflow/core/framework/dataset.h", "IteratorContext");
}

  Allocator* allocator(AllocatorAttributes attrs) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_22(mht_22_v, 744, "", "./tensorflow/core/framework/dataset.h", "allocator");

    return params_.allocator_getter(attrs);
  }

  std::function<Allocator*(AllocatorAttributes)> allocator_getter() {
    return params_.allocator_getter;
  }

  CancellationManager* cancellation_manager() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_23(mht_23_v, 755, "", "./tensorflow/core/framework/dataset.h", "cancellation_manager");

    return params_.cancellation_manager;
  }

  CollectiveExecutor* collective_executor() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_24(mht_24_v, 762, "", "./tensorflow/core/framework/dataset.h", "collective_executor");

    return params_.collective_executor;
  }

  Env* env() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_25(mht_25_v, 769, "", "./tensorflow/core/framework/dataset.h", "env");
 return params_.env; }

  FunctionLibraryRuntime* flr() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_26(mht_26_v, 774, "", "./tensorflow/core/framework/dataset.h", "flr");
 return params_.flr; }

  FunctionHandleCache* function_handle_cache() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_27(mht_27_v, 779, "", "./tensorflow/core/framework/dataset.h", "function_handle_cache");

    return params_.function_handle_cache;
  }

  int64 interleave_depth() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_28(mht_28_v, 786, "", "./tensorflow/core/framework/dataset.h", "interleave_depth");
 return params_.interleave_depth; }

  bool is_restoring() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_29(mht_29_v, 791, "", "./tensorflow/core/framework/dataset.h", "is_restoring");
 return params_.is_restoring; }

  const std::shared_ptr<model::Model>& model() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_30(mht_30_v, 796, "", "./tensorflow/core/framework/dataset.h", "model");
 return params_.model; }

  const Options* options() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_31(mht_31_v, 801, "", "./tensorflow/core/framework/dataset.h", "options");
 return params_.options; }

  ResourceMgr* resource_mgr() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_32(mht_32_v, 806, "", "./tensorflow/core/framework/dataset.h", "resource_mgr");
 return params_.resource_mgr; }

  std::function<void(std::function<void()>)>* runner() {
    return &params_.runner;
  }

  int32 runner_threadpool_size() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_33(mht_33_v, 815, "", "./tensorflow/core/framework/dataset.h", "runner_threadpool_size");
 return params_.runner_threadpool_size; }

  std::vector<std::shared_ptr<SplitProvider>> split_providers() {
    return params_.split_providers;
  }

  std::shared_ptr<StatsAggregator> stats_aggregator() {
    return params_.stats_aggregator;
  }

  const std::shared_ptr<ThreadFactory>& thread_factory() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_34(mht_34_v, 828, "", "./tensorflow/core/framework/dataset.h", "thread_factory");

    return params_.thread_factory;
  }

  thread::ThreadPoolInterface* thread_pool() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_35(mht_35_v, 835, "", "./tensorflow/core/framework/dataset.h", "thread_pool");
 return params_.thread_pool; }

  std::unique_ptr<thread::ThreadPool> CreateThreadPool(const string& name,
                                                       int num_threads) {
    if (params_.thread_pool) {
      // Create a `ThreadPool` instance by wrapping `params_.thread_pool` (which
      // is an instance of `thread::ThreadPoolInterface`). Notably, the
      // ownership of `params_.thread_pool` is *not* transferred onto the newly
      // created `ThreadPool` instance.
      return absl::make_unique<thread::ThreadPool>(params_.thread_pool);
    } else {
      return absl::make_unique<thread::ThreadPool>(params_.env, ThreadOptions(),
                                                   name, num_threads,
                                                   /*low_latency_hint=*/false);
    }
  }

  std::unique_ptr<Thread> StartThread(const string& name,
                                      std::function<void()> fn) {
    if (params_.thread_factory) {
      return params_.thread_factory->StartThread(name, std::move(fn));
    } else {
      return absl::WrapUnique(
          Env::Default()->StartThread({}, name, std::move(fn)));
    }
  }

 private:
  Params params_;
};

// Aggregates runtime support needed for dataset and iterator serialization.
class SerializationContext {
 public:
  // Enum describing what to do during serialization when external state is
  // encountered.
  enum class ExternalStatePolicy : int64 {
    // Proceed with serialization, but log a warning about what state will be
    // lost.
    kWarn = 0,
    // Proceed with serialization without logging any warning.
    kIgnore = 1,
    // Fail the serialization with an error.
    kFail = 2,
  };

  // Handles the CheckExternalState status according to the external state
  // policy.
  Status HandleCheckExternalStateStatus(Status s) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_36(mht_36_v, 886, "", "./tensorflow/core/framework/dataset.h", "HandleCheckExternalStateStatus");

    if (s.ok()) {
      return s;
    }
    switch (params_.external_state_policy) {
      case ExternalStatePolicy::kWarn:
        LOG(WARNING) << s.ToString();
        return Status::OK();
      case ExternalStatePolicy::kIgnore:
        VLOG(2) << "Ignoring error status: " << s.ToString();
        return Status::OK();
      case ExternalStatePolicy::kFail:
        return s;
    }
    LOG(FATAL) << "Control should never reach here";
  }

  struct Params {
    explicit Params() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_37(mht_37_v, 907, "", "./tensorflow/core/framework/dataset.h", "Params");
}

    explicit Params(OpKernelContext* ctx)
        : resource_mgr(ctx->resource_manager()),
          device_name(ctx->device()->attributes().name()) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_38(mht_38_v, 914, "", "./tensorflow/core/framework/dataset.h", "Params");
}

    std::vector<std::pair<string, Tensor>>* input_list = nullptr;  // Not owned.

    // Indicates what to do if the dataset depends on external state.
    ExternalStatePolicy external_state_policy = ExternalStatePolicy::kWarn;

    // Indicates whether the serialization is for rewrites.
    //
    // If true:
    //   * A dataset that doesn't implement serialization is replaced with a
    //     placeholder returned in `input_list`.
    //   * Data tensors are replaced with a placeholder returned in
    //     `input_list`.
    //   * Datasets that use random seeds should not serialize the random seeds.
    //     This doesn't affect datasets that use fixed seeds; fixed seeds will
    //     always be preserved.
    //   * Cardinality is serialized as an unregistered attribute
    //     `_cardinality`.
    // If false:
    //   * A dataset that doesn't implement serialization should result in an
    //     error.
    //   * Data tensors (potentially large) should be serialized.
    //   * Datasets that use random seeds should serialize the random seeds.
    bool is_graph_rewrite = false;

    // A resource manager for looking up resources during serialization.
    ResourceMgr* resource_mgr;

    // The name of the device doing the serialization.
    std::string device_name;
  };

  explicit SerializationContext(Params params) : params_(params) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_39(mht_39_v, 950, "", "./tensorflow/core/framework/dataset.h", "SerializationContext");
}

  std::vector<std::pair<string, Tensor>>* input_list() {
    return params_.input_list;
  }

  ExternalStatePolicy external_state_policy() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_40(mht_40_v, 959, "", "./tensorflow/core/framework/dataset.h", "external_state_policy");

    return params_.external_state_policy;
  }

  bool is_graph_rewrite() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_41(mht_41_v, 966, "", "./tensorflow/core/framework/dataset.h", "is_graph_rewrite");
 return params_.is_graph_rewrite; }

  const ResourceMgr* resource_mgr() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_42(mht_42_v, 971, "", "./tensorflow/core/framework/dataset.h", "resource_mgr");
 return params_.resource_mgr; }

  const std::string& device_name() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_43(mht_43_v, 976, "", "./tensorflow/core/framework/dataset.h", "device_name");
 return params_.device_name; }

 private:
  Params params_;

  TF_DISALLOW_COPY_AND_ASSIGN(SerializationContext);
};

// Represents the current position in a range of outputs, where the
// range of outputs is typically represented by an `DatasetBase`,
// defined below.
class IteratorBase {
 public:
  virtual ~IteratorBase() {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_44(mht_44_v, 992, "", "./tensorflow/core/framework/dataset.h", "~IteratorBase");

    for (auto rit = cleanup_fns_.rbegin(); rit != cleanup_fns_.rend(); ++rit) {
      (*rit)();
    }
  }

  // Gets the next output from the range that this iterator is traversing.
  //
  // If at least one output remains in this iterator's range, that
  // output will be stored in `*out_tensors` and `false` will be
  // stored in `*end_of_sequence`.
  //
  // If no more outputs remain in this iterator's range, `true` will be stored
  // in `*end_of_sequence`, and `*out_tensors` will be empty.
  //
  // Implementations should never return `OutOfRange` error. If at end of
  // sequence, set `*end_of_sequence = true` and return `Status::OK()`.
  // Internally raised `OutOfRange` errors that do not imply end of sequence
  // should be converted to a different error type before being propagated to
  // the caller.
  //
  // Implementations must explicitly set `*end_of_sequence = false` if an
  // `Status::OK()` status is returned and the iterator is not at the end of the
  // sequence.
  //
  // This method is thread-safe.
  //
  // TODO(mrry): Define `GetNextAsync()` or `GetNextManyAsync()`, and
  // potentially remove this method.
  virtual Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) = 0;

  Status GetNext(IteratorContext&& ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_45(mht_45_v, 1028, "", "./tensorflow/core/framework/dataset.h", "GetNext");

    return GetNext(&ctx, out_tensors, end_of_sequence);
  }

  // Skips the next `num_to_skip` outputs from the range that this iterator
  // is traversing.
  //
  // If there are not enough outputs to skip, it will set
  // `*end_of_sequence = true` and return `Status::OK()`. `*num_skipped` will
  // store the number of outputs that are skipped. When `*end_of_sequence` is
  // `false`, `*num_skipped` should equal to `num_to_skip`.
  virtual Status Skip(IteratorContext* ctx, int num_to_skip,
                      bool* end_of_sequence, int* num_skipped) = 0;

  virtual Status Skip(IteratorContext&& ctx, int num_to_skip,
                      bool* end_of_sequence, int* num_skipped) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_46(mht_46_v, 1046, "", "./tensorflow/core/framework/dataset.h", "Skip");

    return Skip(&ctx, num_to_skip, end_of_sequence, num_skipped);
  }

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this
  // iterator.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this iterator.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;

  // Returns a string that identifies the sequence of iterators leading up to
  // this iterator.
  virtual const string& prefix() const = 0;

  // Performs initialization that needs to happen outside of a constructor to
  // properly propagate errors.
  virtual Status Initialize(IteratorContext* ctx) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_47(mht_47_v, 1069, "", "./tensorflow/core/framework/dataset.h", "Initialize");
 return Status::OK(); }

  // Performs initialization of the base iterator.
  Status InitializeBase(IteratorContext* ctx, const IteratorBase* parent);

  // Saves the state of this iterator.
  virtual Status Save(SerializationContext* ctx, IteratorStateWriter* writer) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_48(mht_48_v, 1078, "", "./tensorflow/core/framework/dataset.h", "Save");

    int64_t start_us = EnvTime::NowMicros();
    TF_RETURN_IF_ERROR(SaveInternal(ctx, writer));
    VLOG(1) << "Saved " << prefix() << " in "
            << (EnvTime::NowMicros() - start_us) << "us";
    return Status::OK();
  }

 protected:
  // Returns a node that models this iterator.
  virtual std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const = 0;

  // Restores the state of this iterator.
  virtual Status Restore(IteratorContext* ctx, IteratorStateReader* reader) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_49(mht_49_v, 1095, "", "./tensorflow/core/framework/dataset.h", "Restore");

    int64_t start_us = EnvTime::NowMicros();
    TF_RETURN_IF_ERROR(RestoreInternal(ctx, reader));
    VLOG(1) << "Restored " << prefix() << " in "
            << (EnvTime::NowMicros() - start_us) << "us";
    return Status::OK();
  }

  // This is needed so that sub-classes of IteratorBase can call
  // `SaveInternal` on their input iterators.
  Status SaveInput(SerializationContext* ctx, IteratorStateWriter* writer,
                   const std::unique_ptr<IteratorBase>& input) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_50(mht_50_v, 1109, "", "./tensorflow/core/framework/dataset.h", "SaveInput");

    return input->Save(ctx, writer);
  }

  // This is needed so that sub-classes of IteratorBase can call
  // `RestoreInternal` on their input iterators.
  Status RestoreInput(IteratorContext* ctx, IteratorStateReader* reader,
                      const std::unique_ptr<IteratorBase>& input) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_51(mht_51_v, 1119, "", "./tensorflow/core/framework/dataset.h", "RestoreInput");

    return input->Restore(ctx, reader);
  }

  Status RestoreInput(IteratorContext&& ctx, IteratorStateReader* reader,
                      const std::unique_ptr<IteratorBase>& input) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_52(mht_52_v, 1127, "", "./tensorflow/core/framework/dataset.h", "RestoreInput");

    return RestoreInput(&ctx, reader, input);
  }

  // Saves the state of this iterator.
  //
  // This method is used to store the state of the iterator in a checkpoint.
  // implementations have an override.
  virtual Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) = 0;

  // Restores the state of this iterator.
  //
  // This method is used to restore the state of the iterator from a checkpoint.
  //
  // Implementations may assume that the iterator is in a clean state. That is,
  // its `Initialize` method has been called, but its `GetNext` method has
  // never been called.
  // implementations have an override.
  virtual Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) = 0;

  // Returns a pointer to the node representing this iterator in the performance
  // model. It may be null, if performance modeling is not enabled for this
  // iterator.
  std::shared_ptr<model::Node> model_node() const { return node_; }

  // Returns the number of elements produced by this iterator.
  int64_t num_elements() const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_53(mht_53_v, 1158, "", "./tensorflow/core/framework/dataset.h", "num_elements");

    if (node_) return node_->num_elements();
    return 0;
  }

 private:
  // For access to `AddCleanupFunction` and `Restore`.
  friend class DatasetBase;
  friend class DatasetBaseIterator;  // for access to `node_`

  std::vector<std::function<void()>> cleanup_fns_;
  std::shared_ptr<model::Node> node_ = nullptr;
  const IteratorBase* parent_ = nullptr;  // Not owned.
  int64_t id_ = 0;
  int64_t parent_id_ = 0;
};

// Represents runtime information needed to construct a dataset.
class DatasetContext {
 public:
  struct Params {
    string type_string;  // op type name of this dataset.
    string node_name;    // graph node name of this dataset op, uniquely
                         // identifying the dataset in the graph.
  };

  explicit DatasetContext(Params params) : params_(std::move(params)) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_54(mht_54_v, 1187, "", "./tensorflow/core/framework/dataset.h", "DatasetContext");
}

  explicit DatasetContext(OpKernelContext* ctx) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_55(mht_55_v, 1192, "", "./tensorflow/core/framework/dataset.h", "DatasetContext");

    params_.type_string = ctx->op_kernel().type_string();
    params_.node_name = ctx->op_kernel().name();
  }

  const string& type_string() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_56(mht_56_v, 1200, "", "./tensorflow/core/framework/dataset.h", "type_string");
 return params_.type_string; }
  const string& node_name() const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_57(mht_57_v, 1204, "", "./tensorflow/core/framework/dataset.h", "node_name");
 return params_.node_name; }

 private:
  Params params_;
};

// Returns the number of bytes allocated for the given tensor.
int64_t GetAllocatedBytes(const std::vector<Tensor>& element);

// Returns the estimated memory usage in bytes of the given tensor.
int64_t GetTotalBytes(const std::vector<Tensor>& element);

// Validates and extracts a `DatasetBase` object from `tensor`.
//
// `tensor` must have been written by a call to SetVariantTensorToDataset().
//
// The retrieved pointer is a borrowed reference to the dataset, which is owned
// by the tensor. The consumer must either acquire its own reference to the
// dataset by calling `(*out_dataset)->Ref()`, or ensure that `tensor` is not
// destroyed or mutated while the retrieved pointer is in use.
Status GetDatasetFromVariantTensor(const Tensor& tensor,
                                   DatasetBase** out_dataset);

// Stores a `DatasetBase` object in `tensor`.
//
// The ownership of `dataset` is transferred to `tensor`.
Status StoreDatasetInVariantTensor(DatasetBase* dataset, Tensor* tensor);

// Represents a (potentially infinite) range of outputs, where each
// output is a tuple of tensors.
class DatasetBase : public core::RefCounted {
 public:
  // Key for storing the Dataset graph in the serialized format.
  TF_EXPORT static const char kDatasetGraphKey[];

  // Key for storing the output node of the Dataset graph in the serialized
  // format.
  TF_EXPORT static const char kDatasetGraphOutputNodeKey[];

  explicit DatasetBase(DatasetContext&& ctx)
      : type_string_(ctx.type_string()), node_name_(ctx.node_name()) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_58(mht_58_v, 1247, "", "./tensorflow/core/framework/dataset.h", "DatasetBase");
}

  // Op type name of this dataset.
  const string& type_string() const {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_59(mht_59_v, 1253, "", "./tensorflow/core/framework/dataset.h", "type_string");
 return type_string_; }

  // Graph node name of this dataset op, uniquely identifying the dataset in
  // the graph.
  const string& node_name() const {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_60(mht_60_v, 1260, "", "./tensorflow/core/framework/dataset.h", "node_name");
 return node_name_; }

  // Initializes the dataset.
  void Initialize(const Metadata& metadata);

  const Metadata& metadata() const {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_61(mht_61_v, 1268, "", "./tensorflow/core/framework/dataset.h", "metadata");
 return metadata_; }

  const Options& options() const {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_62(mht_62_v, 1273, "", "./tensorflow/core/framework/dataset.h", "options");
 return options_; }

  int64_t num_sources() const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_63(mht_63_v, 1278, "", "./tensorflow/core/framework/dataset.h", "num_sources");
 return num_sources_; }

  // Returns a new iterator for iterating over the range of elements in
  // this dataset.
  //
  // This method may be called multiple times on the same instance,
  // and the resulting iterators will have distinct state. Each
  // iterator will traverse all elements in this dataset from the
  // start.
  //
  // The prefix identifies the sequence of iterators leading up to the newly
  // created iterator.
  Status MakeIterator(IteratorContext* ctx, const IteratorBase* parent,
                      const string& output_prefix,
                      std::unique_ptr<IteratorBase>* iterator) const;

  Status MakeIterator(IteratorContext&& ctx, const IteratorBase* parent,
                      const string& output_prefix,
                      std::unique_ptr<IteratorBase>* iterator) const {
   std::vector<std::string> mht_64_v;
   mht_64_v.push_back("output_prefix: \"" + output_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_64(mht_64_v, 1300, "", "./tensorflow/core/framework/dataset.h", "MakeIterator");

    return MakeIterator(&ctx, parent, output_prefix, iterator);
  }

  // Returns a new iterator restored from the checkpoint data in `reader`.
  Status MakeIteratorFromCheckpoint(
      IteratorContext* ctx, const string& output_prefix,
      IteratorStateReader* reader,
      std::unique_ptr<IteratorBase>* iterator) const {
   std::vector<std::string> mht_65_v;
   mht_65_v.push_back("output_prefix: \"" + output_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_65(mht_65_v, 1312, "", "./tensorflow/core/framework/dataset.h", "MakeIteratorFromCheckpoint");

    std::unique_ptr<IteratorBase> it;
    IteratorContext::Params params(ctx);
    params.is_restoring = true;
    IteratorContext restore_ctx(std::move(params));
    TF_RETURN_IF_ERROR(MakeIterator(&restore_ctx,
                                    /*parent=*/nullptr, output_prefix, &it));
    TF_RETURN_IF_ERROR(it->Restore(&restore_ctx, reader));
    *iterator = std::move(it);
    return Status::OK();
  }

  Status MakeIteratorFromCheckpoint(
      IteratorContext&& ctx, const string& output_prefix,
      IteratorStateReader* reader,
      std::unique_ptr<IteratorBase>* iterator) const {
   std::vector<std::string> mht_66_v;
   mht_66_v.push_back("output_prefix: \"" + output_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_66(mht_66_v, 1331, "", "./tensorflow/core/framework/dataset.h", "MakeIteratorFromCheckpoint");

    return MakeIteratorFromCheckpoint(&ctx, output_prefix, reader, iterator);
  }

  // Returns a split provider which partitions the dataset's data into splits
  // and provides them in a sequence. The split provider is stored in
  // `*split_provider`.
  virtual Status MakeSplitProviders(
      std::vector<std::unique_ptr<SplitProvider>>* split_providers) const;

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this
  // dataset.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this dataset.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;

  // Returns the number of bytes allocated for tensors of this dataset.
  virtual int64_t AllocatedBytes() const {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_67(mht_67_v, 1355, "", "./tensorflow/core/framework/dataset.h", "AllocatedBytes");
 return 0; }

  // Returns the estimated number of bytes used for tensors of this dataset.
  virtual int64_t TotalBytes() const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_68(mht_68_v, 1361, "", "./tensorflow/core/framework/dataset.h", "TotalBytes");
 return 0; }

  // Returns the cardinality of this dataset.
  // TODO(shilpakrish): Remove this overload once all callers are migrated
  // to the API which passes in the options parameter.
  ABSL_DEPRECATED("Use the overload that passes in the options parameter.")
  int64_t Cardinality() const;

  // Returns the cardinality of this dataset based on the options.
  int64_t Cardinality(CardinalityOptions options) const;

  // Internal implementation of cardinality for a dataset.
  // TODO(shilpakrish): Remove this overload once all callers are migrated
  // to the API which passes in the options parameter.
  ABSL_DEPRECATED("Use the overload that passes in the options parameter.")
  virtual int64_t CardinalityInternal() const { return kUnknownCardinality; }

  // Internal implementation of cardinality for a dataset based on the options.
  virtual int64_t CardinalityInternal(CardinalityOptions options) const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_69(mht_69_v, 1382, "", "./tensorflow/core/framework/dataset.h", "CardinalityInternal");

    return kUnknownCardinality;
  }

  // A human-readable debug string for this dataset.
  virtual string DebugString() const = 0;

  // Stores the dataset's input datasets in `*inputs`. The pointers stored in
  // `*inputs` are borrowed. The only valid non-ok return status is
  // UNIMPLEMENTED in case `InputDatasets` is not implemented by a dataset
  // subclass. Implementing `InputDatasets` enables `DatasetBase` to provide a
  // default implementation of `MakeSplitProvider` when there is a single input
  // dataset.
  virtual Status InputDatasets(std::vector<const DatasetBase*>* inputs) const;

  // Indicates whether the dataset depends on any external state which would
  // prevent it from being serializable. If so, the method returns
  // `errors::FailedPrecondition` with a message that identifies the external
  // state. Otherwise, the method returns `Status::OK()`.
  virtual Status CheckExternalState() const = 0;

  // Indicates whether the dataset is compatible with random access.
  Status CheckRandomAccessCompatible(const int64 index) const;

  // Return the element at a particular index for a randomly accessible dataset.
  virtual Status Get(OpKernelContext* ctx, int64 index,
                     std::vector<Tensor>* out_tensors) const;

  // Return a finalized version of the dataset.  The returned DatasetBase is
  // unowned and lives for as long as this dataset.
  virtual StatusOr<DatasetBase*> Finalize(
      OpKernelContext* ctx,
      std::function<StatusOr<core::RefCountPtr<DatasetBase>>()>
          make_finalized_dataset) const;

  // Wrapper around a GraphDefBuilder which provides support for serializing
  // Datasets as GraphDefs.
  class DatasetGraphDefBuilder : public GraphDefBuilderWrapper {
   public:
    explicit DatasetGraphDefBuilder(GraphDefBuilder* b)
        : GraphDefBuilderWrapper(b) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_70(mht_70_v, 1425, "", "./tensorflow/core/framework/dataset.h", "DatasetGraphDefBuilder");
}
    Status AddInputDataset(SerializationContext* ctx,
                           const DatasetBase* dataset, Node** output);
    Status AddDatasetOrTensor(SerializationContext* ctx, const Tensor& val,
                              Node** output);
    Status AddIdentity(SerializationContext* ctx,
                       const std::string& name_prefix, Node** input,
                       Node** output);

   private:
    Status AddDatasetOrTensorHelper(SerializationContext* ctx,
                                    const Tensor& val, Node** output);
    Status AddResourceHelper(SerializationContext* ctx, const Tensor& val,
                             Node** output);
  };

 protected:
  friend class CapturedFunction;

  // Serializes the dataset into a `GraphDef`, which has two uses:
  //
  // 1) To perform static input pipeline optimizations, tf.data serializes the
  // dataset graph, applies graph rewrites, and then deserializes the graph.
  // If a subclass of `DatasetBase` does not implement this method, then it will
  // be excluded from static optimizations (and so will any upstream datasets).
  //
  // 2) To save the dataset so that it can restore at a later point (possibly in
  // different environment). If a subclass of `DatasetBase` does not implement
  // this method, then this migration will not be possible.
  virtual Status AsGraphDefInternal(SerializationContext* ctx,
                                    DatasetGraphDefBuilder* b,
                                    Node** node) const = 0;

  virtual std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const = 0;

  void set_options(const Options& options) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_71(mht_71_v, 1464, "", "./tensorflow/core/framework/dataset.h", "set_options");
 options_ = options; }

 private:
  // Computes and stores the cardinality of a given dataset.
  Status ComputeCardinality();

  // Computes the number of source datasets feeding into this dataset. A source
  // dataset is a leaf in the subtree of dataset inputs.
  Status ComputeNumSources();

  // Merges options from inputs to this dataset. If there is a conflict in a
  // field value, the options set on this dataset takes precedence over those in
  // the inputs. The order of precedence on the inputs is in the same order as
  // how they appear for this dataset.
  Status MergeOptionsFromInputs();

  const string type_string_;
  const string node_name_;
  Metadata metadata_;
  Options options_;
  mutable mutex mu_;
  mutable mutex cardinality_mu_;
  mutable core::RefCountPtr<DatasetBase> finalized_dataset_;
  //  The number of source datasets feeding into the dataset. A source dataset
  //  is a leaf in the subtree of dataset inputs.
  int64_t num_sources_ = -1;
  mutable int64_t cardinality_ TF_GUARDED_BY(cardinality_mu_) =
      kUnknownCardinality;
};

// Represents an iterator that is associated with a particular dataset.
class DatasetBaseIterator : public IteratorBase {
 public:
  struct BaseParams {
    // Owns one reference on the shared dataset object.
    const DatasetBase* dataset;

    // Identifies the sequence of iterators leading up to this iterator.
    const string prefix;
  };

  explicit DatasetBaseIterator(const BaseParams& params);

  ~DatasetBaseIterator() override;

  virtual const DatasetBase* dataset() const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_72(mht_72_v, 1512, "", "./tensorflow/core/framework/dataset.h", "dataset");
 return params_.dataset; }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_73(mht_73_v, 1517, "", "./tensorflow/core/framework/dataset.h", "output_dtypes");

    return params_.dataset->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_74(mht_74_v, 1524, "", "./tensorflow/core/framework/dataset.h", "output_shapes");

    return params_.dataset->output_shapes();
  }

  const string& prefix() const override {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_75(mht_75_v, 1531, "", "./tensorflow/core/framework/dataset.h", "prefix");
 return params_.prefix; }

  // Returns a name to be used for the TraceMe event.
  //
  // NOTE: TraceMe supports passing key-value pairs of "arguments" using the
  // following format "name#arg_1=value_,...,arg_n=value_n".
  string BuildTraceMeName();

  Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) final;

  Status GetNext(IteratorContext&& ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_76(mht_76_v, 1546, "", "./tensorflow/core/framework/dataset.h", "GetNext");

    return GetNext(&ctx, out_tensors, end_of_sequence);
  }

  Status Skip(IteratorContext* ctx, int num_to_skip, bool* end_of_sequence,
              int* num_skipped) final;

  Status Save(SerializationContext* ctx, IteratorStateWriter* writer) final {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_77(mht_77_v, 1556, "", "./tensorflow/core/framework/dataset.h", "Save");

    VLOG(2) << "Attempting to save checkpoints on iterator (prefix: "
            << prefix() << ") from " << dataset()->DebugString();
    return IteratorBase::Save(ctx, writer);
  }

 protected:
  Status Restore(IteratorContext* ctx, IteratorStateReader* reader) final {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_78(mht_78_v, 1566, "", "./tensorflow/core/framework/dataset.h", "Restore");

    VLOG(2) << "Attempting to restore checkpoints on iterator (prefix: "
            << prefix() << ") from " << dataset()->DebugString();
    return IteratorBase::Restore(ctx, reader);
  }

  // Internal implementation of GetNext that is wrapped in tracing logic.
  //
  // See the docstring of `GetNext` method regaring the contract for
  // `out_tensors` and `end_of_sequence`. Implementations may assume that
  // `*out_tensors` is empty.
  virtual Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) = 0;

  // Internal implementation of Skip that is wrapped in tracing logic
  virtual Status SkipInternal(IteratorContext* ctx, int num_to_skip,
                              bool* end_of_sequence, int* num_skipped);

  string full_name(const string& name) const {
   std::vector<std::string> mht_79_v;
   mht_79_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_79(mht_79_v, 1589, "", "./tensorflow/core/framework/dataset.h", "full_name");

    return FullName(params_.prefix, name);
  }

  // Returns a map of key-value pairs to included in the TraceMe string.
  virtual TraceMeMetadata GetTraceMeMetadata() const {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_80(mht_80_v, 1597, "", "./tensorflow/core/framework/dataset.h", "GetTraceMeMetadata");
 return {}; }

  // By default we model iterators using an unknown node, which acts as
  // pass-through with respect to performance modeling.
  std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const override {
    return model::MakeUnknownNode(std::move(args));
  }

  // When modeling is enabled, this method disables autotuning for the given
  // iterator (and the transitive closure of its inputs).
  void DisableAutotune(IteratorContext* ctx, IteratorBase* iterator) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_81(mht_81_v, 1611, "", "./tensorflow/core/framework/dataset.h", "DisableAutotune");

    if (iterator->node_) {
      iterator->node_->set_autotune(false);
    }
  }

  // When modeling is enabled, this method enables autotuning for the given
  // iterator (and the transitive closure of its inputs).
  void EnableAutotune(IteratorContext* ctx, IteratorBase* iterator) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_82(mht_82_v, 1622, "", "./tensorflow/core/framework/dataset.h", "EnableAutotune");

    if (iterator->node_) {
      iterator->node_->set_autotune(true);
    }
  }

  // When modeling is enabled, this method records the fact that this iterator
  // has dequeued an element from an internal buffer.
  void RecordBufferDequeue(IteratorContext* ctx,
                           const std::vector<Tensor>& element) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_83(mht_83_v, 1634, "", "./tensorflow/core/framework/dataset.h", "RecordBufferDequeue");

    if (collect_resource_usage(ctx)) {
      node_->record_buffer_event(-GetAllocatedBytes(element), -1);

      DCHECK_GE(node_->buffered_elements(), 0);
    }
  }

  // When modeling is enabled, this method records the fact that this iterator
  // has enqueued an element in an internal buffer.
  void RecordBufferEnqueue(IteratorContext* ctx,
                           const std::vector<Tensor>& element) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_84(mht_84_v, 1648, "", "./tensorflow/core/framework/dataset.h", "RecordBufferEnqueue");

    if (collect_resource_usage(ctx)) {
      node_->record_buffer_event(GetAllocatedBytes(element), 1);
    }
  }

  // When modeling is enabled, this method records the fact that this iterator
  // has produced an element and its size in bytes.
  void RecordElement(IteratorContext* ctx, std::vector<Tensor>* out_tensors) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_85(mht_85_v, 1659, "", "./tensorflow/core/framework/dataset.h", "RecordElement");

    if (collect_resource_usage(ctx)) {
      int64_t num_bytes = GetAllocatedBytes(*out_tensors);
      node_->record_element();
      node_->record_bytes_produced(num_bytes);
      if (node_->output()) {
        node_->output()->record_bytes_consumed(num_bytes);
      }
    }
  }

  // When modeling is enabled, this method records the fact that a thread of
  // this iterator has started work.
  void RecordStart(IteratorContext* ctx) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_86(mht_86_v, 1675, "", "./tensorflow/core/framework/dataset.h", "RecordStart");

    if (collect_resource_usage(ctx)) {
      int64_t now_nanos = EnvTime::NowNanos();
      node_->record_start(now_nanos);
    }
  }

  // When modeling is enabled, this method records the fact that a thread of
  // this iterator has stopped work.
  void RecordStop(IteratorContext* ctx) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_87(mht_87_v, 1687, "", "./tensorflow/core/framework/dataset.h", "RecordStop");

    if (collect_resource_usage(ctx)) {
      int64_t now_nanos = EnvTime::NowNanos();
      node_->record_stop(now_nanos);
    }
  }

  // Returns whether work is currently being recorded, i.e. whether we are
  // currently between a `RecordStart` and a `RecordStop`.
  bool IsRecording(IteratorContext* ctx) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_88(mht_88_v, 1699, "", "./tensorflow/core/framework/dataset.h", "IsRecording");

    return node_ && node_->is_recording();
  }

 private:
  bool collect_resource_usage(IteratorContext* ctx) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_89(mht_89_v, 1707, "", "./tensorflow/core/framework/dataset.h", "collect_resource_usage");

    return ctx->model() && node_;
  }

  string traceme_metadata_;
  BaseParams params_;
};

// Represents an iterator that is associated with a particular dataset
// with a particular type.
template <class DatasetType>
class DatasetIterator : public DatasetBaseIterator {
 public:
  struct Params {
    // Borrowed pointer to the dataset.
    const DatasetType* dataset;

    // Identifies the sequence of iterators leading up to this iterator.
    const string prefix;
  };

  explicit DatasetIterator(const Params& params)
      : DatasetBaseIterator({params.dataset, params.prefix}),
        typed_dataset_(params.dataset) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_90(mht_90_v, 1733, "", "./tensorflow/core/framework/dataset.h", "DatasetIterator");
}

  // The dataset from which this iterator was created.
  const DatasetType* dataset() const final {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_91(mht_91_v, 1739, "", "./tensorflow/core/framework/dataset.h", "dataset");
 return typed_dataset_; }

 private:
  const DatasetType* const typed_dataset_;  // Not owned.
};

template <typename T>
Status ParseScalarArgument(OpKernelContext* ctx,
                           const StringPiece& argument_name, T* output) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_92(mht_92_v, 1750, "", "./tensorflow/core/framework/dataset.h", "ParseScalarArgument");

  const Tensor* argument_t;
  TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
  if (!TensorShapeUtils::IsScalar(argument_t->shape())) {
    return errors::InvalidArgument(argument_name, " must be a scalar");
  }
  *output = argument_t->scalar<T>()();
  return Status::OK();
}

template <typename T>
Status ParseVectorArgument(OpKernelContext* ctx,
                           const StringPiece& argument_name,
                           std::vector<T>* output) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_93(mht_93_v, 1766, "", "./tensorflow/core/framework/dataset.h", "ParseVectorArgument");

  const Tensor* argument_t;
  TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
  if (!TensorShapeUtils::IsVector(argument_t->shape())) {
    return errors::InvalidArgument(argument_name, " must be a vector");
  }
  int size = argument_t->vec<T>().size();
  output->reserve(size);
  for (int i = 0; i < size; ++i) {
    output->push_back(argument_t->vec<T>()(i));
  }
  return Status::OK();
}

// Encapsulates the work required to plug a DatasetBase into the core TensorFlow
// graph execution engine.
class DatasetOpKernel : public OpKernel {
 public:
  explicit DatasetOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_94(mht_94_v, 1787, "", "./tensorflow/core/framework/dataset.h", "DatasetOpKernel");

    if (ctx->HasAttr(kMetadata)) {
      std::string serialized_metadata;
      OP_REQUIRES_OK(ctx, ctx->GetAttr(kMetadata, &serialized_metadata));
      OP_REQUIRES(ctx, metadata_.ParseFromString(serialized_metadata),
                  errors::InvalidArgument(absl::StrCat(
                      "Could not parse the 'metadata' attribute.")));
    }
  }

  void Compute(OpKernelContext* ctx) final;

  // Checks whether the given op is a tf.data operation.
  //
  // NOTE: The check uses a heuristic and can produce both false positives and
  // false negatives. In particular, tf.data operations are expected to use
  // names that end with "Dataset" or "DatasetV[0-9]+".
  static bool IsDatasetOp(const OpDef& op_def);

  string TraceString(const OpKernelContext& ctx, bool verbose) const override;

 protected:
  // Subclasses should implement this method. It will be called during Compute
  // execution.
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase** output) = 0;

 private:
  Metadata metadata_;
};

// Encapsulates the work required to plug unary Datasets into the core
// TensorFlow graph execution engine.
class UnaryDatasetOpKernel : public DatasetOpKernel {
 public:
  explicit UnaryDatasetOpKernel(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_95(mht_95_v, 1825, "", "./tensorflow/core/framework/dataset.h", "UnaryDatasetOpKernel");
}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) final;
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                           DatasetBase** output) = 0;
};

// Encapsulates the work required to plug binary Datasets into the core
// TensorFlow graph execution engine.
class BinaryDatasetOpKernel : public DatasetOpKernel {
 public:
  explicit BinaryDatasetOpKernel(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTh mht_96(mht_96_v, 1841, "", "./tensorflow/core/framework/dataset.h", "BinaryDatasetOpKernel");
}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) final;
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                           DatasetBase* another_input,
                           DatasetBase** output) = 0;
};

// A simple background worker that executes closures asynchronously and without
// blocking.
//
// A `BackgroundWorker` is used to offload blocking work from an `AsyncOpKernel`
// to avoid blocking an executor thread that may be required by the blocking
// work.
//
// NOTE(mrry): We do not use a regular `tensorflow::thread::ThreadPool` for this
// purpose because its current implementation (in Eigen) uses a finite-length
// queue and will block the caller when full. This can lead to deadlock under
// heavy load. Since the number of concurrent work items in each user of a
// `BackgroundWorker` is at most one per op invocation, the dynamic allocation
// overhead is tolerable.
class BackgroundWorker {
 public:
  BackgroundWorker(Env* env, const char* name);

  ~BackgroundWorker();

  void Schedule(std::function<void()> work_item);

 private:
  void WorkerLoop();

  Env* const env_;
  const char* const name_;

  std::unique_ptr<Thread> thread_;
  mutex mu_;
  condition_variable cond_var_;
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
  std::deque<std::function<void()>> work_queue_ TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_DATASET_H_
