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
class MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc() {
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
#include "tensorflow/core/framework/dataset.h"

#include <unordered_map>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/resource.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/public/version.h"

// On Windows, disable some macros that would break compile
#if defined(PLATFORM_WINDOWS)
#undef GetMessage
#endif

namespace tensorflow {
namespace data {
namespace {

static mutex* get_dataset_op_registry_lock() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/framework/dataset.cc", "get_dataset_op_registry_lock");

  static mutex dataset_op_registry_lock(LINKER_INITIALIZED);
  return &dataset_op_registry_lock;
}

static std::unordered_set<string>* get_dataset_op_registry() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/framework/dataset.cc", "get_dataset_op_registry");

  static std::unordered_set<string>* names = new std::unordered_set<string>;
  return names;
}

std::string UniqueNodeName(const std::string& base) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("base: \"" + base + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_2(mht_2_v, 233, "", "./tensorflow/core/framework/dataset.cc", "UniqueNodeName");

  static std::atomic<int64_t> counter(0);
  return strings::StrCat(base, "/", counter.fetch_add(1));
}

// A wrapper class for storing a `DatasetBase` instance in a DT_VARIANT tensor.
// Objects of the wrapper class own a reference on an instance of `DatasetBase`,
// and the wrapper's copy constructor and destructor take care of managing the
// reference count.
//
// NOTE(mrry): This is not a feature-complete implementation of the DT_VARIANT
// specification. In particular, we cannot currently serialize an arbitrary
// `DatasetBase` object, so the `Encode()` and `Decode()` methods are not
// implemented.
class DatasetVariantWrapper {
 public:
  DatasetVariantWrapper() : dataset_(nullptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/framework/dataset.cc", "DatasetVariantWrapper");
}

  // Transfers ownership of `dataset` to `*this`.
  explicit DatasetVariantWrapper(DatasetBase* dataset) : dataset_(dataset) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_4(mht_4_v, 258, "", "./tensorflow/core/framework/dataset.cc", "DatasetVariantWrapper");
}

  DatasetVariantWrapper(const DatasetVariantWrapper& other)
      : dataset_(other.dataset_) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_5(mht_5_v, 264, "", "./tensorflow/core/framework/dataset.cc", "DatasetVariantWrapper");

    if (dataset_) dataset_->Ref();
  }

  DatasetVariantWrapper& operator=(DatasetVariantWrapper&& other) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_6(mht_6_v, 271, "", "./tensorflow/core/framework/dataset.cc", "=");

    if (&other == this) return *this;
    std::swap(dataset_, other.dataset_);
    return *this;
  }

  DatasetVariantWrapper& operator=(const DatasetVariantWrapper& other) = delete;

  ~DatasetVariantWrapper() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_7(mht_7_v, 282, "", "./tensorflow/core/framework/dataset.cc", "~DatasetVariantWrapper");

    if (dataset_) dataset_->Unref();
  }

  DatasetBase* get() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_8(mht_8_v, 289, "", "./tensorflow/core/framework/dataset.cc", "get");
 return dataset_; }

  string TypeName() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_9(mht_9_v, 294, "", "./tensorflow/core/framework/dataset.cc", "TypeName");
 return "tensorflow::DatasetVariantWrapper"; }
  string DebugString() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_10(mht_10_v, 298, "", "./tensorflow/core/framework/dataset.cc", "DebugString");

    if (dataset_) {
      return dataset_->DebugString();
    } else {
      return "<Uninitialized DatasetVariantWrapper>";
    }
  }
  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_11(mht_11_v, 308, "", "./tensorflow/core/framework/dataset.cc", "Encode");

    LOG(ERROR) << "The Encode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
  }
  bool Decode(const VariantTensorData& data) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_12(mht_12_v, 315, "", "./tensorflow/core/framework/dataset.cc", "Decode");

    LOG(ERROR) << "The Decode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
    return false;
  }

 private:
  DatasetBase* dataset_;  // Owns one reference.
};

const char kWrappedDatasetVariantTypeName[] =
    "tensorflow::data::WrappedDatasetVariant";

class WrappedDatasetVariantWrapper {
 public:
  WrappedDatasetVariantWrapper() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_13(mht_13_v, 333, "", "./tensorflow/core/framework/dataset.cc", "WrappedDatasetVariantWrapper");
}

  explicit WrappedDatasetVariantWrapper(const Tensor& ds_tensor)
      : ds_tensor_(ds_tensor) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_14(mht_14_v, 339, "", "./tensorflow/core/framework/dataset.cc", "WrappedDatasetVariantWrapper");
}

  Tensor get() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_15(mht_15_v, 344, "", "./tensorflow/core/framework/dataset.cc", "get");
 return ds_tensor_; }

  string TypeName() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_16(mht_16_v, 349, "", "./tensorflow/core/framework/dataset.cc", "TypeName");
 return "tensorflow::WrappedDatasetVariantWrapper"; }

  string DebugString() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_17(mht_17_v, 354, "", "./tensorflow/core/framework/dataset.cc", "DebugString");

    return "tensorflow::WrappedDatasetVariantWrapper::DebugString";
  }

  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_18(mht_18_v, 361, "", "./tensorflow/core/framework/dataset.cc", "Encode");

    *(data->add_tensors()) = ds_tensor_;
  }

  bool Decode(const VariantTensorData& data) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_19(mht_19_v, 368, "", "./tensorflow/core/framework/dataset.cc", "Decode");

    ds_tensor_ = data.tensors(0);
    return true;
  }

 private:
  Tensor ds_tensor_;
};

class WrapDatasetVariantOp : public OpKernel {
 public:
  explicit WrapDatasetVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_20(mht_20_v, 382, "", "./tensorflow/core/framework/dataset.cc", "WrapDatasetVariantOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_21(mht_21_v, 387, "", "./tensorflow/core/framework/dataset.cc", "Compute");

    const Tensor& tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                tensor.dtype() == DT_VARIANT &&
                    TensorShapeUtils::IsScalar(tensor.shape()),
                errors::InvalidArgument(
                    "Dataset tensor must be a scalar of dtype DT_VARIANT."));
    DatasetBase* unused;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(tensor, &unused));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    output->scalar<Variant>()() = WrappedDatasetVariantWrapper(tensor);
  }
};

REGISTER_KERNEL_BUILDER(Name("WrapDatasetVariant").Device(DEVICE_CPU),
                        WrapDatasetVariantOp);
REGISTER_KERNEL_BUILDER(Name("WrapDatasetVariant")
                            .HostMemory("input_handle")
                            .HostMemory("output_handle")
                            .Device(DEVICE_GPU),
                        WrapDatasetVariantOp);

class UnwrapDatasetVariantOp : public OpKernel {
 public:
  explicit UnwrapDatasetVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_22(mht_22_v, 415, "", "./tensorflow/core/framework/dataset.cc", "UnwrapDatasetVariantOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_23(mht_23_v, 420, "", "./tensorflow/core/framework/dataset.cc", "Compute");

    const Tensor& tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                tensor.dtype() == DT_VARIANT &&
                    TensorShapeUtils::IsScalar(tensor.shape()),
                errors::InvalidArgument(
                    "Dataset tensor must be a scalar of dtype DT_VARIANT."));
    Variant variant = tensor.scalar<Variant>()();
    const WrappedDatasetVariantWrapper* wrapper =
        variant.get<WrappedDatasetVariantWrapper>();
    OP_REQUIRES(ctx, wrapper != nullptr,
                errors::InvalidArgument(
                    "Tensor must be a WrappedDataset variant object."));
    Tensor ds_tensor = wrapper->get();
    OP_REQUIRES_OK(ctx, ctx->set_output("output_handle", ds_tensor));
  }
};

REGISTER_KERNEL_BUILDER(Name("UnwrapDatasetVariant").Device(DEVICE_CPU),
                        UnwrapDatasetVariantOp);
REGISTER_KERNEL_BUILDER(Name("UnwrapDatasetVariant")
                            .HostMemory("input_handle")
                            .HostMemory("output_handle")
                            .Device(DEVICE_GPU),
                        UnwrapDatasetVariantOp);

static Status WrappedDatasetVariantDeviceCopy(
    const WrappedDatasetVariantWrapper& from, WrappedDatasetVariantWrapper* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_24(mht_24_v, 451, "", "./tensorflow/core/framework/dataset.cc", "WrappedDatasetVariantDeviceCopy");

  *to = WrappedDatasetVariantWrapper(from);
  return Status::OK();
}

#define REGISTER_OPTIONAL_COPY(DIRECTION)               \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      WrappedDatasetVariantWrapper, DIRECTION,          \
      WrappedDatasetVariantDeviceCopy)

REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(WrappedDatasetVariantWrapper,
                                       kWrappedDatasetVariantTypeName);

}  // namespace

Status GraphDefBuilderWrapper::AddDataset(const DatasetBase* dataset,
                                          const std::vector<Node*>& inputs,
                                          Node** output) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_25(mht_25_v, 475, "", "./tensorflow/core/framework/dataset.cc", "GraphDefBuilderWrapper::AddDataset");

  return AddDataset(dataset, inputs, {}, output);
}

Status GraphDefBuilderWrapper::AddDataset(
    const DatasetBase* dataset, const std::vector<Node*>& inputs,
    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
    Node** output) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_26(mht_26_v, 485, "", "./tensorflow/core/framework/dataset.cc", "GraphDefBuilderWrapper::AddDataset");

  std::vector<std::pair<size_t, Node*>> enumerated_inputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    enumerated_inputs[i] = std::make_pair(i, inputs[i]);
  }
  return AddDataset(dataset, enumerated_inputs, {}, attrs, output);
}

Status GraphDefBuilderWrapper::AddDataset(
    const DatasetBase* dataset,
    const std::vector<std::pair<size_t, Node*>>& inputs,
    const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
    Node** output) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_27(mht_27_v, 501, "", "./tensorflow/core/framework/dataset.cc", "GraphDefBuilderWrapper::AddDataset");

  return AddDataset(dataset, inputs, list_inputs, attrs,
                    /*use_dataset_name=*/false, output);
}

Status GraphDefBuilderWrapper::AddDataset(
    const DatasetBase* dataset,
    const std::vector<std::pair<size_t, Node*>>& inputs,
    const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
    bool use_dataset_name, Node** output) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_28(mht_28_v, 514, "", "./tensorflow/core/framework/dataset.cc", "GraphDefBuilderWrapper::AddDataset");

  auto& type_string = dataset->type_string();
  auto opts = absl::make_unique<GraphDefBuilder::Options>(b_->opts());
  // TODO(srbs|mrry): Not all datasets have output_types and output_shapes
  // attributes defined. It will be nice to have a consistent pattern.
  bool has_output_types_attr = HasAttr(type_string, "output_types");
  bool has_output_shapes_attr = HasAttr(type_string, "output_shapes");
  if (has_output_shapes_attr) {
    opts = absl::make_unique<GraphDefBuilder::Options>(
        opts->WithAttr("output_shapes", dataset->output_shapes()));
  }
  if (has_output_types_attr) {
    opts = absl::make_unique<GraphDefBuilder::Options>(
        opts->WithAttr("output_types", dataset->output_dtypes()));
  }
  bool has_metadata_attr = HasAttr(type_string, "metadata");
  if (has_metadata_attr) {
    std::string serialized_metadata;
    dataset->metadata().SerializeToString(&serialized_metadata);
    opts = absl::make_unique<GraphDefBuilder::Options>(
        opts->WithAttr("metadata", serialized_metadata));
  }
  for (const auto& attr : attrs) {
    opts = absl::make_unique<GraphDefBuilder::Options>(
        opts->WithAttr(attr.first, attr.second));
  }
  if (opts->HaveError()) {
    return errors::Internal("AddDataset: Failed to build Options with error ",
                            opts->StatusToString());
  }
  NodeBuilder node_builder(
      use_dataset_name ? dataset->node_name() : opts->GetNameForOp(type_string),
      type_string, opts->op_registry());
  {
    size_t total_size = inputs.size() + list_inputs.size();
    auto inputs_iter = inputs.begin();
    auto list_inputs_iter = list_inputs.begin();
    for (int i = 0; i < total_size; i++) {
      if (inputs_iter != inputs.end() && inputs_iter->first == i) {
        node_builder.Input(NodeBuilder::NodeOut(inputs_iter->second));
        inputs_iter++;
      } else if (list_inputs_iter != list_inputs.end() &&
                 list_inputs_iter->first == i) {
        std::vector<NodeBuilder::NodeOut> nodeout_inputs;
        nodeout_inputs.reserve(list_inputs_iter->second.size());
        for (Node* n : list_inputs_iter->second) {
          nodeout_inputs.emplace_back(n);
        }
        node_builder.Input(nodeout_inputs);
        list_inputs_iter++;
      } else {
        return errors::InvalidArgument("No input found for index ", i);
      }
    }
  }
  *output = opts->FinalizeBuilder(&node_builder);
  if (*output == nullptr) {
    return errors::Internal("AddDataset: Failed to build ", type_string,
                            " op with error ", opts->StatusToString());
  }
  return Status::OK();
}

Status GraphDefBuilderWrapper::AddFunction(
    SerializationContext* ctx, const string& function_name,
    const FunctionLibraryDefinition& lib_def) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_29(mht_29_v, 583, "", "./tensorflow/core/framework/dataset.cc", "GraphDefBuilderWrapper::AddFunction");

  if (b_->HasFunction(function_name)) {
    VLOG(1) << "Function with name " << function_name << "already exists in"
            << " the graph. It will not be added again.";
    return Status::OK();
  }
  const FunctionDef* f_def = lib_def.Find(function_name);
  if (f_def == nullptr) {
    return errors::InvalidArgument("Unable to find FunctionDef for ",
                                   function_name, " in the registry.");
  }
  FunctionDefLibrary def;
  *def.add_function() = *f_def;
  const string gradient_func = lib_def.FindGradient(function_name);
  if (!gradient_func.empty()) {
    GradientDef* g_def = def.add_gradient();
    g_def->set_function_name(function_name);
    g_def->set_gradient_func(gradient_func);
  }
  TF_RETURN_IF_ERROR(b_->AddFunctionLibrary(def));

  // Recursively add functions in inputs of function_name.
  for (const NodeDef& node_def : f_def->node_def()) {
    const OpRegistrationData* op_reg_data = nullptr;
    TF_RETURN_IF_ERROR(lib_def.LookUp(node_def.op(), &op_reg_data));
    if (op_reg_data->is_function_op) {
      TF_RETURN_IF_ERROR(AddFunction(ctx, op_reg_data->op_def.name(), lib_def));
    }
    // Recursively add functions in attrs of this NodeDef.
    for (const auto& pair : node_def.attr()) {
      TF_RETURN_IF_ERROR(AddAttrFunctions(ctx, pair.second, lib_def));
    }
  }

  // Recursively add functions in attrs of function_name.
  for (auto iter = f_def->attr().begin(); iter != f_def->attr().end(); iter++) {
    TF_RETURN_IF_ERROR(AddAttrFunctions(ctx, iter->second, lib_def));
  }
  return Status::OK();
}

void GraphDefBuilderWrapper::AddPlaceholderInternal(const Tensor& val,
                                                    Node** output) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_30(mht_30_v, 628, "", "./tensorflow/core/framework/dataset.cc", "GraphDefBuilderWrapper::AddPlaceholderInternal");

  *output = ops::SourceOp(
      "Placeholder",
      b_->opts().WithAttr("dtype", val.dtype()).WithAttr("shape", val.shape()));
}

void GraphDefBuilderWrapper::AddTensorInternal(const Tensor& val,
                                               Node** output) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_31(mht_31_v, 638, "", "./tensorflow/core/framework/dataset.cc", "GraphDefBuilderWrapper::AddTensorInternal");

  *output = ops::SourceOp(
      "Const",
      b_->opts().WithAttr("dtype", val.dtype()).WithAttr("value", val));
}

bool GraphDefBuilderWrapper::HasAttr(const string& name,
                                     const string& attr_name) const {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("name: \"" + name + "\"");
   mht_32_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_32(mht_32_v, 650, "", "./tensorflow/core/framework/dataset.cc", "GraphDefBuilderWrapper::HasAttr");

  const OpDef* op_def = nullptr;
  Status s = b_->opts().op_registry()->LookUpOpDef(name, &op_def);
  if (!s.ok() || op_def == nullptr) {
    return false;
  }
  return HasAttr(op_def, attr_name);
}

int32_t GetRunnerThreadpoolSizeFromOpKernelContext(OpKernelContext* ctx) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_33(mht_33_v, 662, "", "./tensorflow/core/framework/dataset.cc", "GetRunnerThreadpoolSizeFromOpKernelContext");

  thread::ThreadPool* thread_pool =
      ctx->device()->tensorflow_device_thread_pool();
  if (thread_pool) {
    return thread_pool->NumThreads();
  } else {
    static const int32_t kDefaultRunnerThreadpoolSize = port::MaxParallelism();
    return kDefaultRunnerThreadpoolSize;
  }
}

Status IteratorBase::InitializeBase(IteratorContext* ctx,
                                    const IteratorBase* parent) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_34(mht_34_v, 677, "", "./tensorflow/core/framework/dataset.cc", "IteratorBase::InitializeBase");

  parent_ = parent;
  id_ =
      Hash64CombineUnordered(Hash64(prefix()), reinterpret_cast<uint64>(this));
  if (parent_) {
    parent_id_ = Hash64CombineUnordered(Hash64(parent_->prefix()),
                                        reinterpret_cast<uint64>(parent_));
  }
  if (const auto& model = ctx->model()) {
    auto factory = [ctx, this](model::Node::Args args) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_35(mht_35_v, 689, "", "./tensorflow/core/framework/dataset.cc", "lambda");

      return CreateNode(ctx, std::move(args));
    };
    model->AddNode(std::move(factory), prefix(), parent->model_node(), &node_);
    cleanup_fns_.push_back([this, model]() { model->RemoveNode(node_); });
  }
  return Status::OK();
}

int64_t GetAllocatedBytes(const std::vector<Tensor>& element) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_36(mht_36_v, 701, "", "./tensorflow/core/framework/dataset.cc", "GetAllocatedBytes");

  int64_t allocated_bytes = 0;
  DatasetBase* dataset;
  for (auto& tensor : element) {
    if (tensor.dtype() == DT_VARIANT &&
        GetDatasetFromVariantTensor(tensor, &dataset).ok()) {
      allocated_bytes += dataset->AllocatedBytes();
    } else {
      allocated_bytes += tensor.AllocatedBytes();
    }
  }
  return allocated_bytes;
}

int64_t GetTotalBytes(const std::vector<Tensor>& element) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_37(mht_37_v, 718, "", "./tensorflow/core/framework/dataset.cc", "GetTotalBytes");

  int64_t total_bytes = 0;
  DatasetBase* dataset;
  for (auto& tensor : element) {
    if (tensor.dtype() == DT_VARIANT &&
        GetDatasetFromVariantTensor(tensor, &dataset).ok()) {
      total_bytes += dataset->TotalBytes();
    } else {
      total_bytes += tensor.TotalBytes();
    }
  }
  return total_bytes;
}

std::string FullName(const std::string& prefix, const std::string& name) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("prefix: \"" + prefix + "\"");
   mht_38_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_38(mht_38_v, 737, "", "./tensorflow/core/framework/dataset.cc", "FullName");

  if (str_util::StrContains(name, kColon)) {
    LOG(ERROR) << name << " should not contain " << kColon;
  }

  return strings::StrCat(kFullNameRandomHex, kPipe, prefix, kColon, name);
}

Status GetDatasetFromVariantTensor(const Tensor& tensor,
                                   DatasetBase** out_dataset) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_39(mht_39_v, 749, "", "./tensorflow/core/framework/dataset.cc", "GetDatasetFromVariantTensor");

  if (!(tensor.dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor.shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  const Variant& variant = tensor.scalar<Variant>()();
  const DatasetVariantWrapper* wrapper = variant.get<DatasetVariantWrapper>();
  if (wrapper == nullptr) {
    return errors::InvalidArgument("Tensor must be a Dataset object.");
  }
  *out_dataset = wrapper->get();
  if (*out_dataset == nullptr) {
    return errors::Internal("Read uninitialized Dataset variant.");
  }
  return Status::OK();
}

Status StoreDatasetInVariantTensor(DatasetBase* dataset, Tensor* tensor) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_40(mht_40_v, 770, "", "./tensorflow/core/framework/dataset.cc", "StoreDatasetInVariantTensor");

  if (!(tensor->dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor->shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  tensor->scalar<Variant>()() = DatasetVariantWrapper(dataset);
  return Status::OK();
}

namespace internal {

#define WARN_PROTO_FIELD_CONFLICT(reflection, field, field_type, src, dst)     \
  {                                                                            \
    auto source_value = reflection->Get##field_type(src, field);               \
    auto destination_value = reflection->Get##field_type(*dst, field);         \
    if (source_value != destination_value) {                                   \
      LOG(WARNING) << "Changing the value of option field " << field->name()   \
                   << " from " << destination_value << " to " << source_value; \
    }                                                                          \
  }

#define WARN_PROTO_ENUM_FIELD_CONFLICT(reflection, field, src, dst) \
  {                                                                 \
    auto source_value = reflection->GetEnum(src, field);            \
    auto destination_value = reflection->GetEnum(*dst, field);      \
    if (source_value != destination_value) {                        \
      LOG(WARNING) << "Changing the value of option enum field "    \
                   << field->name() << " from "                     \
                   << destination_value->full_name() << " to "      \
                   << source_value->full_name();                    \
    }                                                               \
  }

void WarnProtoConflicts(const protobuf::Message& src, protobuf::Message* dst) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_41(mht_41_v, 807, "", "./tensorflow/core/framework/dataset.cc", "WarnProtoConflicts");

  std::vector<const protobuf::FieldDescriptor*> set_src;
  std::vector<const protobuf::FieldDescriptor*> set_dst;
  const protobuf::Reflection* reflection = src.GetReflection();
  reflection->ListFields(src, &set_src);
  reflection->ListFields(*dst, &set_dst);
  std::sort(set_src.begin(), set_src.end());
  std::sort(set_dst.begin(), set_dst.end());

  std::vector<const protobuf::FieldDescriptor*> in_both;
  std::set_intersection(set_src.begin(), set_src.end(), set_dst.begin(),
                        set_dst.end(), std::back_inserter(in_both));

  for (auto field : in_both) {
    if (field->type() == protobuf::FieldDescriptor::TYPE_MESSAGE) {
      WarnProtoConflicts(reflection->GetMessage(src, field),
                         reflection->MutableMessage(dst, field));
    } else {
      switch (field->cpp_type()) {
        case protobuf::FieldDescriptor::CPPTYPE_INT32:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, Int32, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_INT64:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, Int64, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_UINT32:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, UInt32, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_UINT64:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, UInt64, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, Double, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_FLOAT:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, Float, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_BOOL:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, Bool, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_ENUM:
          WARN_PROTO_ENUM_FIELD_CONFLICT(reflection, field, src, dst);
          break;
        default: {
          LOG(ERROR) << "Unrecognized proto type for field "
                     << field->full_name();
        }
      }
    }
  }
}

#undef WARN_PROTO_ENUM_FIELD_CONFLICT
#undef WARN_PROTO_FIELD_CONFLICT

void MergeOptions(const protobuf::Message& source,
                  protobuf::Message* destination) {
  WarnProtoConflicts(source, destination);
  destination->MergeFrom(source);
}

void MergeOptions(const protobuf::MessageLite& source,
                  protobuf::MessageLite* destination) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_42(mht_42_v, 872, "", "./tensorflow/core/framework/dataset.cc", "MergeOptions");

  destination->CheckTypeAndMergeFrom(source);
}

}  // namespace internal

void DatasetBase::Initialize(const Metadata& metadata) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_43(mht_43_v, 881, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::Initialize");

  Status s = ComputeNumSources();
  if (!s.ok()) {
    LOG(ERROR) << s;
  }
  s = MergeOptionsFromInputs();
  if (!s.ok()) {
    LOG(ERROR) << s;
  }
  metadata_ = metadata;
  if (metadata_.name() == "") {
    static std::atomic<int64_t> id_counter(0);
    *metadata_.mutable_name() =
        strings::StrCat(type_string(), ":", id_counter.fetch_add(1));
  }
}

Status DatasetBase::ComputeNumSources() {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_44(mht_44_v, 901, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::ComputeNumSources");

  std::vector<const DatasetBase*> inputs;
  Status s = InputDatasets(&inputs);
  if (errors::IsUnimplemented(s)) {
    return errors::Unimplemented(
        "Cannot compute input sources for dataset of type ", type_string(),
        ", because the dataset does not implement `InputDatasets`.");
  }
  if (num_sources_ >= 0) {
    // Already computed.
    return Status::OK();
  }
  num_sources_ = 0;
  if (inputs.empty()) {
    num_sources_ = 1;
    return Status::OK();
  }
  for (const auto& input : inputs) {
    if (input->num_sources() < 0) {
      return errors::FailedPrecondition(
          "Cannot compute input sources for dataset of type ", type_string(),
          ", because sources could not be computed for input dataset of type ",
          input->type_string());
    }
    num_sources_ += input->num_sources();
  }
  return Status::OK();
}

Status DatasetBase::CheckRandomAccessCompatible(const int64 index) const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_45(mht_45_v, 933, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::CheckRandomAccessCompatible");

  CardinalityOptions options;
  options.set_compute_level(CardinalityOptions::CARDINALITY_COMPUTE_MODERATE);
  int64 cardinality = Cardinality(options);
  if (cardinality == kInfiniteCardinality ||
      cardinality == kUnknownCardinality) {
    return tensorflow::errors::FailedPrecondition(
        "Dataset of type ", this->DebugString(), "has cardinality ",
        cardinality, "which does not support random access.");
  }
  if (index < 0 || index >= cardinality) {
    return errors::OutOfRange("Index out of range [0, ", cardinality,
                              "):", index);
  }
  return Status::OK();
}

Status DatasetBase::Get(OpKernelContext* ctx, int64 index,
                        std::vector<Tensor>* out_tensors) const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_46(mht_46_v, 954, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::Get");

  return errors::Unimplemented(
      "Random access is not implemented for this dataset.");
}

StatusOr<DatasetBase*> DatasetBase::Finalize(
    OpKernelContext* ctx,
    std::function<StatusOr<core::RefCountPtr<DatasetBase>>()>
        make_finalized_dataset) const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_47(mht_47_v, 965, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::Finalize");

  mutex_lock l(mu_);
  if (!finalized_dataset_) {
    TF_ASSIGN_OR_RETURN(finalized_dataset_, make_finalized_dataset());
  }
  return finalized_dataset_.get();
}

Status DatasetBase::MergeOptionsFromInputs() {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_48(mht_48_v, 976, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::MergeOptionsFromInputs");

  std::vector<const DatasetBase*> inputs;
  Status s = InputDatasets(&inputs);
  if (errors::IsUnimplemented(s)) {
    return errors::Unimplemented(
        "Cannot merge options for dataset of type ", type_string(),
        ", because the dataset does not implement `InputDatasets`.");
  }
  if (inputs.empty()) {
    return Status::OK();
  }
  // Merge options from inputs sequentially before merging options from dataset.
  // Since the last options merged takes precedence, the options that may be set
  // for the current dataset through OptionsDataset takes precedence over those
  // set on the input datasets.
  Options merged_options = inputs[0]->options_;
  for (int i = 1; i < inputs.size(); ++i) {
    internal::MergeOptions(inputs[i]->options_, &merged_options);
  }
  internal::MergeOptions(options_, &merged_options);
  options_ = merged_options;
  return Status::OK();
}

Status DatasetBase::MakeIterator(
    IteratorContext* ctx, const IteratorBase* parent,
    const string& output_prefix,
    std::unique_ptr<IteratorBase>* iterator) const {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("output_prefix: \"" + output_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_49(mht_49_v, 1007, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::MakeIterator");

  if (type_string() == "OptionsDataset" || type_string() == "FinalizeDataset") {
    std::vector<const DatasetBase*> inputs;
    Status s = InputDatasets(&inputs);
    return inputs[0]->MakeIterator(ctx, parent, output_prefix, iterator);
  }
  profiler::TraceMe traceme(
      [&] {
        return profiler::TraceMeEncode(
            strings::StrCat("MakeIterator::", type_string()), {});
      },
      profiler::TraceMeLevel::kInfo);
  *iterator = MakeIteratorInternal(output_prefix);
  Status s = (*iterator)->InitializeBase(ctx, parent);
  if (s.ok()) {
    s.Update((*iterator)->Initialize(ctx));
  }
  if (!s.ok()) {
    // Reset the iterator to avoid returning an uninitialized iterator.
    iterator->reset();
  }
  return s;
}

Status DatasetBase::MakeSplitProviders(
    std::vector<std::unique_ptr<SplitProvider>>* split_providers) const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_50(mht_50_v, 1035, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::MakeSplitProviders");

  std::vector<const DatasetBase*> inputs;
  Status s = InputDatasets(&inputs);
  if (errors::IsUnimplemented(s)) {
    return errors::Unimplemented(
        "Cannot create split providers for dataset of type ", type_string(),
        ", because the dataset implements neither `InputDatasets` nor "
        "`MakeSplitProvider`.");
  }
  if (inputs.size() != 1) {
    return errors::Unimplemented(
        "Cannot create split providers for dataset of type ", type_string(),
        ", because the dataset is not unary (instead having arity ",
        inputs.size(),
        "), and no custom implementation of `MakeSplitProvider` is defined.");
  }
  return inputs[0]->MakeSplitProviders(split_providers);
}

int64_t DatasetBase::Cardinality() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_51(mht_51_v, 1057, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::Cardinality");

  mutex_lock l(cardinality_mu_);
  if (cardinality_ == kUnknownCardinality) {
    cardinality_ = CardinalityInternal();
  }
  return cardinality_;
}

int64_t DatasetBase::Cardinality(CardinalityOptions options) const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_52(mht_52_v, 1068, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::Cardinality");

  mutex_lock l(cardinality_mu_);
  if (cardinality_ == kUnknownCardinality) {
    cardinality_ = CardinalityInternal(options);
  }
  return cardinality_;
}

Status DatasetBase::InputDatasets(
    std::vector<const DatasetBase*>* inputs) const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_53(mht_53_v, 1080, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::InputDatasets");

  return errors::Unimplemented("InputDatasets not implemented for ",
                               type_string());
}

Status DatasetBase::DatasetGraphDefBuilder::AddInputDataset(
    SerializationContext* ctx, const DatasetBase* dataset, Node** output) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_54(mht_54_v, 1089, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::DatasetGraphDefBuilder::AddInputDataset");

  Status status = dataset->AsGraphDefInternal(ctx, this, output);
  if (ctx->is_graph_rewrite()) {
    if (status.ok()) {
      // Record cardinality in an unregistered attributes so that rewrites have
      // this information.
      (*output)->AddAttr(kCardinalityAttrForRewrite, dataset->Cardinality());
    } else if (errors::IsUnimplemented(status)) {
      Tensor t(DT_VARIANT, TensorShape({}));
      // `StoreDatasetInVariantTensor` will transfer ownership of `dataset`. We
      // increment the refcount of `dataset` here to retain ownership.
      dataset->Ref();
      TF_RETURN_IF_ERROR(
          StoreDatasetInVariantTensor(const_cast<DatasetBase*>(dataset), &t));
      TF_RETURN_IF_ERROR(AddPlaceholder(t, output));
      DCHECK_NE(ctx->input_list(), nullptr);
      ctx->input_list()->emplace_back((*output)->name(), std::move(t));
      LOG_EVERY_N_SEC(WARNING, 30)
          << "Input of " << dataset->DebugString()
          << " will not be optimized because the dataset does not implement "
             "the "
             "AsGraphDefInternal() method needed to apply optimizations.";
      return Status::OK();
    }
  }
  return status;
}

Status DatasetBase::DatasetGraphDefBuilder::AddDatasetOrTensor(
    SerializationContext* ctx, const Tensor& t, Node** output) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_55(mht_55_v, 1121, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::DatasetGraphDefBuilder::AddDatasetOrTensor");

  if (t.dtype() == DT_VARIANT) {
    // If the input tensor is a variant, it may represent a multi-dimensional
    // array of datasets. We attempt to decode each dataset so that we can use
    // their custom serialization logic and combine the result of their
    // individual serializations using the `Pack` operation.
    //
    // If this fails, we fallback to using its Variant::Encode() based
    // serialization.
    Status s = AddDatasetOrTensorHelper(ctx, t, output);
    if (s.ok()) {
      return s;
    }
  }
  if (t.dtype() == DT_RESOURCE && !ctx->is_graph_rewrite()) {
    Status s = AddResourceHelper(ctx, t, output);
    if (!errors::IsUnimplemented(s)) {
      // Fall through to AddTensor if AsGraphDef is not implemented for this
      // resource.
      return s;
    }
  }
  return AddTensor(t, output);
}

Status DatasetBase::DatasetGraphDefBuilder::AddIdentity(
    SerializationContext* ctx, const std::string& name_prefix, Node** input,
    Node** output) {
   std::vector<std::string> mht_56_v;
   mht_56_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_56(mht_56_v, 1152, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::DatasetGraphDefBuilder::AddIdentity");

  *output =
      ops::UnaryOp("Identity", *input,
                   builder()->opts().WithName(UniqueNodeName(name_prefix)));
  return Status::OK();
}

Status DatasetBase::DatasetGraphDefBuilder::AddDatasetOrTensorHelper(
    SerializationContext* ctx, const Tensor& t, Node** output) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_57(mht_57_v, 1163, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::DatasetGraphDefBuilder::AddDatasetOrTensorHelper");

  if (t.dims() == 0) {
    DatasetBase* dataset;
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(t, &dataset));
    return AddInputDataset(ctx, dataset, output);
  }
  std::vector<NodeBuilder::NodeOut> nodes;
  for (int i = 0; i < t.dim_size(0); ++i) {
    Node* node;
    TF_RETURN_IF_ERROR(AddDatasetOrTensorHelper(ctx, t.SubSlice(i), &node));
    nodes.emplace_back(node);
  }
  auto op_name = "Pack";
  auto opts = builder()->opts();
  NodeBuilder node_builder(opts.GetNameForOp(op_name), op_name,
                           opts.op_registry());
  node_builder.Input(std::move(nodes));
  *output = opts.FinalizeBuilder(&node_builder);
  return Status::OK();
}

Status DatasetBase::DatasetGraphDefBuilder::AddResourceHelper(
    SerializationContext* ctx, const Tensor& t, Node** output) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_58(mht_58_v, 1188, "", "./tensorflow/core/framework/dataset.cc", "DatasetBase::DatasetGraphDefBuilder::AddResourceHelper");

  const ResourceHandle& handle = t.flat<ResourceHandle>()(0);
  if (ctx->device_name() != handle.device()) {
    return errors::InvalidArgument("Trying to access resource ", handle.name(),
                                   " located in device ", handle.device(),
                                   " from device ", ctx->device_name());
  }
  ResourceBase* resource;
  TF_RETURN_IF_ERROR(ctx->resource_mgr()->Lookup(handle, &resource));
  core::ScopedUnref unref(resource);
  return resource->AsGraphDef(builder(), output);
}

DatasetBaseIterator::DatasetBaseIterator(const BaseParams& params)
    : params_(params) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_59(mht_59_v, 1205, "", "./tensorflow/core/framework/dataset.cc", "DatasetBaseIterator::DatasetBaseIterator");

  params_.dataset->Ref();
  VLOG(2) << prefix() << " constructor";
  strings::StrAppend(&traceme_metadata_, "name=", dataset()->metadata().name());
  strings::StrAppend(&traceme_metadata_, ",shapes=");
  auto& shapes = output_shapes();
  for (int i = 0; i < shapes.size(); ++i) {
    if (i > 0) {
      strings::StrAppend(&traceme_metadata_, " ");
    }
    strings::StrAppend(&traceme_metadata_, shapes.at(i).DebugString());
  }
  strings::StrAppend(&traceme_metadata_, ",types=");
  auto& types = output_dtypes();
  for (int i = 0; i < types.size(); ++i) {
    if (i > 0) {
      strings::StrAppend(&traceme_metadata_, " ");
    }
    strings::StrAppend(&traceme_metadata_, DataTypeString(types.at(i)));
  }
}

DatasetBaseIterator::~DatasetBaseIterator() {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_60(mht_60_v, 1230, "", "./tensorflow/core/framework/dataset.cc", "DatasetBaseIterator::~DatasetBaseIterator");

  VLOG(2) << prefix() << " destructor";
  params_.dataset->Unref();
}

string DatasetBaseIterator::BuildTraceMeName() {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_61(mht_61_v, 1238, "", "./tensorflow/core/framework/dataset.cc", "DatasetBaseIterator::BuildTraceMeName");

  string result =
      strings::StrCat(params_.prefix, "#", traceme_metadata_, ",id=", id_);
  if (parent_) {
    strings::StrAppend(&result, ",parent_id=", parent_id_);
  }
  TraceMeMetadata metadata = GetTraceMeMetadata();
  for (const auto& pair : metadata) {
    strings::StrAppend(&result, ",", pair.first, "=", pair.second);
  }
  strings::StrAppend(&result, "#");
  return result;
}

Status DatasetBaseIterator::GetNext(IteratorContext* ctx,
                                    std::vector<Tensor>* out_tensors,
                                    bool* end_of_sequence) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_62(mht_62_v, 1257, "", "./tensorflow/core/framework/dataset.cc", "DatasetBaseIterator::GetNext");

  profiler::TraceMe activity([&] { return BuildTraceMeName(); },
                             profiler::TraceMeLevel::kInfo);
  DVLOG(3) << prefix() << " GetNext enter";
  auto model = ctx->model();
  if (collect_resource_usage(ctx)) {
    int64_t now_nanos = EnvTime::NowNanos();
    auto output = node_->output();
    if (output) {
      output->record_stop(now_nanos);
    }
    node_->record_start(now_nanos);
  }
  out_tensors->clear();
  Status s = GetNextInternal(ctx, out_tensors, end_of_sequence);
  if (TF_PREDICT_TRUE(s.ok())) {
    if (TF_PREDICT_TRUE(!*end_of_sequence)) {
      DCHECK_EQ(out_tensors->size(), dataset()->output_dtypes().size());
      RecordElement(ctx, out_tensors);
    } else {
      out_tensors->clear();
    }
  }
  if (collect_resource_usage(ctx)) {
    int64_t now_nanos = EnvTime::NowNanos();
    node_->record_stop(now_nanos);
    auto output = node_->output();
    if (output) {
      output->record_start(now_nanos);
    }
  }
  if (TF_PREDICT_FALSE(errors::IsOutOfRange(s))) {
    s = errors::Internal("Iterator \"", params_.prefix,
                         "\" returned `OutOfRange`. This indicates an "
                         "implementation error as `OutOfRange` errors are not "
                         "expected to be returned here. Original message: ",
                         s.error_message());
    LOG(ERROR) << s;
  }
  DVLOG(3) << prefix() << " GetNext exit";
  return s;
}

Status DatasetBaseIterator::Skip(IteratorContext* ctx, int num_to_skip,
                                 bool* end_of_sequence, int* num_skipped) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_63(mht_63_v, 1304, "", "./tensorflow/core/framework/dataset.cc", "DatasetBaseIterator::Skip");

  profiler::TraceMe activity([&] { return BuildTraceMeName(); },
                             profiler::TraceMeLevel::kInfo);
  DVLOG(3) << prefix() << " Skip enter";
  auto model = ctx->model();
  if (collect_resource_usage(ctx)) {
    int64_t now_nanos = EnvTime::NowNanos();
    auto output = node_->output();
    if (output) {
      output->record_stop(now_nanos);
    }
    node_->record_start(now_nanos);
  }
  Status s = SkipInternal(ctx, num_to_skip, end_of_sequence, num_skipped);
  if (collect_resource_usage(ctx)) {
    int64_t now_nanos = EnvTime::NowNanos();
    node_->record_stop(now_nanos);
    auto output = node_->output();
    if (output) {
      output->record_start(now_nanos);
    }
  }
  if (TF_PREDICT_FALSE(errors::IsOutOfRange(s))) {
    s = errors::Internal("Iterator \"", params_.prefix,
                         "\" returned `OutOfRange`. This indicates an "
                         "implementation error as `OutOfRange` errors are not "
                         "expected to be returned here. Original message: ",
                         s.error_message());
    LOG(ERROR) << s;
  }
  DVLOG(3) << prefix() << " Skip exit";
  return s;
}

Status DatasetBaseIterator::SkipInternal(IteratorContext* ctx, int num_to_skip,
                                         bool* end_of_sequence,
                                         int* num_skipped) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_64(mht_64_v, 1343, "", "./tensorflow/core/framework/dataset.cc", "DatasetBaseIterator::SkipInternal");

  *num_skipped = 0;
  for (int i = 0; i < num_to_skip; ++i) {
    std::vector<Tensor> out_tensors;
    TF_RETURN_IF_ERROR(GetNextInternal(ctx, &out_tensors, end_of_sequence));
    if (*end_of_sequence) {
      return Status::OK();
    }
    // RecordElement is used to count the number of element computed and
    // help calculate the CPU time spent on a given iterator to do the
    // autotuning.
    // Here we only call RecordElement in the default implementation of
    // SkipInternal (which trivially calls GetNextInternal) and assume
    // that the overridden SkipInternal in the derived class will have
    // negligible cost compare to its GetNextInternal.
    RecordElement(ctx, &out_tensors);
    (*num_skipped)++;
  }
  return Status::OK();
}

void DatasetOpKernel::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_65(mht_65_v, 1367, "", "./tensorflow/core/framework/dataset.cc", "DatasetOpKernel::Compute");

  DatasetBase* dataset = nullptr;
  MakeDataset(ctx, &dataset);
  if (ctx->status().ok()) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(ctx, StoreDatasetInVariantTensor(dataset, output));
    dataset->Initialize(metadata_);
  }
}

string DatasetOpKernel::TraceString(const OpKernelContext& ctx,
                                    bool verbose) const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_66(mht_66_v, 1382, "", "./tensorflow/core/framework/dataset.cc", "DatasetOpKernel::TraceString");

  return profiler::TraceMeOp(name_view(), type_string_view());
}

// static
bool DatasetOpKernel::IsDatasetOp(const OpDef& op_def) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_67(mht_67_v, 1390, "", "./tensorflow/core/framework/dataset.cc", "DatasetOpKernel::IsDatasetOp");

  if (op_def.output_arg_size() != 1) return false;
  if (op_def.output_arg(0).type() != DT_VARIANT) return false;
  absl::string_view op_name = op_def.name();
  if (op_name == "DatasetFromGraph") return true;
  if (absl::EndsWith(op_name, "Dataset")) return true;
  // Check if the suffix matches "DatasetV[0-9]+".
  size_t index = op_name.length() - 1;
  while (index >= 0 && isdigit(op_name[index])) {
    index--;
  }
  constexpr absl::string_view kDatasetPrefix = "DatasetV";
  constexpr absl::string_view::size_type kPrefixLength = kDatasetPrefix.size();
  if (index < kPrefixLength - 1 || index == op_name.length() - 1) return false;
  return op_name.substr(index - kPrefixLength + 1, kPrefixLength) ==
         kDatasetPrefix;
}

void UnaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_68(mht_68_v, 1412, "", "./tensorflow/core/framework/dataset.cc", "UnaryDatasetOpKernel::MakeDataset");

  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  MakeDataset(ctx, input, output);
}

void BinaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                        DatasetBase** output) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_69(mht_69_v, 1422, "", "./tensorflow/core/framework/dataset.cc", "BinaryDatasetOpKernel::MakeDataset");

  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  DatasetBase* another_input;
  OP_REQUIRES_OK(ctx,
                 GetDatasetFromVariantTensor(ctx->input(1), &another_input));
  MakeDataset(ctx, input, another_input, output);
}

const char DatasetBase::kDatasetGraphKey[] = "_DATASET_GRAPH";
const char DatasetBase::kDatasetGraphOutputNodeKey[] =
    "_DATASET_GRAPH_OUTPUT_NODE";

BackgroundWorker::BackgroundWorker(Env* env, const char* name)
    : env_(env), name_(name) {
   std::vector<std::string> mht_70_v;
   mht_70_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_70(mht_70_v, 1440, "", "./tensorflow/core/framework/dataset.cc", "BackgroundWorker::BackgroundWorker");
}

BackgroundWorker::~BackgroundWorker() {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_71(mht_71_v, 1445, "", "./tensorflow/core/framework/dataset.cc", "BackgroundWorker::~BackgroundWorker");

  {
    mutex_lock l(mu_);
    cancelled_ = true;
  }
  cond_var_.notify_one();
  // Block until the background thread has terminated.
  //
  // NOTE(mrry): We explicitly free and join the thread here because
  // `WorkerLoop()` uses other members of this object, and so we must join
  // the thread before destroying them.
  thread_.reset();
}

void BackgroundWorker::Schedule(std::function<void()> work_item) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_72(mht_72_v, 1462, "", "./tensorflow/core/framework/dataset.cc", "BackgroundWorker::Schedule");

  {
    mutex_lock l(mu_);
    if (!thread_) {
      thread_ = absl::WrapUnique(env_->StartThread(
          {} /* thread_options */, name_, [this]() {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_73(mht_73_v, 1470, "", "./tensorflow/core/framework/dataset.cc", "lambda");
 WorkerLoop(); }));
    }
    work_queue_.push_back(std::move(work_item));
  }
  cond_var_.notify_one();
}

void BackgroundWorker::WorkerLoop() {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_74(mht_74_v, 1480, "", "./tensorflow/core/framework/dataset.cc", "BackgroundWorker::WorkerLoop");

  tensorflow::ResourceTagger tag(kTFDataResourceTag, "Background");
  while (true) {
    std::function<void()> work_item = nullptr;
    {
      mutex_lock l(mu_);
      while (!cancelled_ && work_queue_.empty()) {
        cond_var_.wait(l);
      }
      if (cancelled_) {
        return;
      }
      DCHECK(!work_queue_.empty());
      work_item = std::move(work_queue_.front());
      work_queue_.pop_front();
    }
    DCHECK(work_item != nullptr);
    work_item();
  }
}

namespace {
class RunnerImpl : public Runner {
 public:
  void Run(const std::function<void()>& f) override {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_75(mht_75_v, 1507, "", "./tensorflow/core/framework/dataset.cc", "Run");

    tensorflow::ResourceTagger tag(kTFDataResourceTag, "Runner");
    f();

    // NOTE: We invoke a virtual function to prevent `f` being tail-called, and
    // thus ensure that this function remains on the stack until after `f`
    // returns.
    PreventTailCall();
  }

 private:
  virtual void PreventTailCall() {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_76(mht_76_v, 1521, "", "./tensorflow/core/framework/dataset.cc", "PreventTailCall");
}
};
}  // namespace

/* static */
Runner* Runner::get() {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdatasetDTcc mht_77(mht_77_v, 1529, "", "./tensorflow/core/framework/dataset.cc", "Runner::get");

  static Runner* singleton = new RunnerImpl;
  return singleton;
}

}  // namespace data
}  // namespace tensorflow
