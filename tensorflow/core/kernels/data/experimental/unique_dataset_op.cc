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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/unique_dataset_op.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const UniqueDatasetOp::kDatasetType;
/* static */ constexpr const char* const UniqueDatasetOp::kInputDataset;
/* static */ constexpr const char* const UniqueDatasetOp::kOutputTypes;
/* static */ constexpr const char* const UniqueDatasetOp::kOutputShapes;

class UniqueDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)), input_(input) {
    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::Unique")});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "output_shapes");

    return input_->output_shapes();
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_3(mht_3_v, 231, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "DebugString");

    return strings::StrCat("UniqueDatasetOp::Dataset");
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_5(mht_5_v, 246, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_6(mht_6_v, 256, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const typename Iterator::Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_7(mht_7_v, 270, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_8(mht_8_v, 275, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "Initialize");

      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_9(mht_9_v, 284, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "GetNextInternal");

      mutex_lock l(mu_);
      bool saw_new_value;
      do {
        saw_new_value = false;
        out_tensors->clear();
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (*end_of_sequence) {
          break;
        }
        DCHECK_EQ(1, out_tensors->size());
        saw_new_value = unique_elements_.insert((*out_tensors)[0]).second;
      } while (!saw_new_value);
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeUnknownRatioNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_10(mht_10_v, 311, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      if (input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      } else {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("input_impl_empty"), ""));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("unique_elements_size"),
                                             unique_elements_.size()));
      size_t i = 0;
      for (const Tensor& t : unique_elements_) {
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            full_name(strings::StrCat("unique_elements[", i++, "]")), t));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_11(mht_11_v, 333, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      if (!reader->Contains(full_name("input_impl_empty"))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }
      int64_t num_unique_elements;
      unique_elements_.clear();
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("unique_elements_size"),
                                            &num_unique_elements));
      for (int64_t i = 0; i < num_unique_elements; ++i) {
        Tensor unique_element;
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            ctx->flr(), full_name(strings::StrCat("unique_elements[", i, "]")),
            &unique_element));
        auto insert_result = unique_elements_.insert(unique_element);
        if (!insert_result.second) {
          return errors::InvalidArgument(
              "Checkpoint contained two unique elements with the same "
              "value.");
        }
      }
      return Status::OK();
    }

   private:
    struct TensorHash {
      size_t operator()(const Tensor& t) const {
        if (t.dtype() == DT_INT32 || t.dtype() == DT_INT64) {
          return Hash64(t.tensor_data().data(), t.tensor_data().size());
        } else {
          DCHECK_EQ(DT_STRING, t.dtype());
          auto flat_t = t.flat<tstring>();
          uint64 hash = 0;
          for (int64_t i = 0; i < t.NumElements(); ++i) {
            hash = Hash64Combine(hash, Hash64(flat_t(i)));
          }
          return static_cast<size_t>(hash);
        }
      }
    };

    struct TensorKeyEqual {
      bool operator()(const Tensor& lhs, const Tensor& rhs) const {
        if (lhs.shape() != rhs.shape() || lhs.dtype() != rhs.dtype()) {
          return false;
        }
        switch (lhs.dtype()) {
#define HANDLE_TYPE(T)                                     \
  case T:                                                  \
    do {                                                   \
      auto lhs_flat = lhs.flat<EnumToDataType<T>::Type>(); \
      auto rhs_flat = rhs.flat<EnumToDataType<T>::Type>(); \
      for (int64_t i = 0; i < lhs.NumElements(); ++i) {    \
        if (lhs_flat(i) != rhs_flat(i)) {                  \
          return false;                                    \
        }                                                  \
      }                                                    \
      return true;                                         \
    } while (0)

          HANDLE_TYPE(DT_INT32);
          HANDLE_TYPE(DT_INT64);
          HANDLE_TYPE(DT_STRING);
          default:
            DCHECK(false) << "UniqueDataset unhandled data type: "
                          << DataTypeString(lhs.dtype());
            return false;
        }
      }
    };

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      std::unordered_set<Tensor, TensorHash, TensorKeyEqual> unique_elements_
          TF_GUARDED_BY(mu_);
  };

    const DatasetBase* const input_;
};

void UniqueDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                  DatasetBase** output) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_opDTcc mht_12(mht_12_v, 419, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op.cc", "UniqueDatasetOp::MakeDataset");

  OP_REQUIRES(ctx, input->output_dtypes().size() == 1,
              errors::InvalidArgument("UniqueDataset only supports "
                                      "inputs with a single component."));

  DataType input_dtype = input->output_dtypes()[0];
  OP_REQUIRES(ctx,
              input_dtype == DT_INT32 || input_dtype == DT_INT64 ||
                  input_dtype == DT_STRING,
              errors::InvalidArgument(
                  "UniqueDataset only supports inputs with a single "
                  "`tf.int32`, `tf.int64`, or `tf.string` component."));

  *output = new Dataset(ctx, input);
}

namespace {

REGISTER_KERNEL_BUILDER(Name("UniqueDataset").Device(DEVICE_CPU),
                        UniqueDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalUniqueDataset").Device(DEVICE_CPU),
                        UniqueDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
