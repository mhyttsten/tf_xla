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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc() {
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
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class DenseToSparseBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit DenseToSparseBatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "DenseToSparseBatchDatasetOp");
}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_1(mht_1_v, 203, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "MakeDataset");

    // Create a new DenseToSparseBatchDatasetOp::Dataset, insert it in the
    // step-local container, and return it as the output.
    OP_REQUIRES(
        ctx, input->output_dtypes().size() == 1,
        errors::InvalidArgument("DenseToSparseBatchDataset only supports "
                                "inputs with a single component."));

    int64_t batch_size;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64_t>(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("Batch size must be greater than zero."));

    const Tensor* row_shape_t;
    OP_REQUIRES_OK(ctx, ctx->input("row_shape", &row_shape_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(row_shape_t->shape()),
                errors::InvalidArgument("row_shape must be a vector"));
    PartialTensorShape row_shape;
    OP_REQUIRES_OK(ctx, PartialTensorShape::MakePartialShape(
                            row_shape_t->vec<int64_t>().data(),
                            row_shape_t->NumElements(), &row_shape));

    *output = nullptr;

#define HANDLE_TYPE(T)                                           \
  case DataTypeToEnum<T>::value: {                               \
    *output = new Dataset<T>(ctx, batch_size, row_shape, input); \
    break;                                                       \
  }

    switch (input->output_dtypes()[0]) {
      TF_CALL_DATASET_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
      default:
        OP_REQUIRES(ctx, false,
                    errors::Unimplemented(
                        "DenseToSparseBatchDataset unhandled data type: ",
                        input->output_dtypes()[0]));
    }
  }

 private:
  // TODO(mrry): Push the templated code down to the raw copying routine.
  template <class T>
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, int64_t batch_size,
            const PartialTensorShape& row_shape, const DatasetBase* input)
        : DatasetBase(DatasetContext(ctx)),
          batch_size_(batch_size),
          row_shape_(row_shape),
          input_(input) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_2(mht_2_v, 259, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "Dataset");

      input_->Ref();

      output_shapes_.reserve(1);
      PartialTensorShape output_shape({-1});
      output_shape.AppendShape(row_shape_);
      output_shapes_.push_back(output_shape);
    }

    ~Dataset() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_3(mht_3_v, 271, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(typename Iterator::Params{
          this, strings::StrCat(prefix, "::DenseToSparseBatch")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_4(mht_4_v, 282, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "output_dtypes");

      static DataTypeVector* output_dtypes = new DataTypeVector({DT_VARIANT});
      return *output_dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_5(mht_5_v, 290, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "output_shapes");

      return output_shapes_;
    }

    string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_6(mht_6_v, 297, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "DebugString");

      return strings::StrCat("DenseToSparseBatchDatasetOp(", batch_size_,
                             ")::Dataset");
    }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_7(mht_7_v, 305, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "CardinalityInternal");

      int64_t n = input_->Cardinality();
      if (n == kInfiniteCardinality || n == kUnknownCardinality) {
        return n;
      }
      return n / batch_size_ + (n % batch_size_ == 0 ? 0 : 1);
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_8(mht_8_v, 317, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_9(mht_9_v, 325, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "CheckExternalState");

      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_10(mht_10_v, 335, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "AsGraphDefInternal");

      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
      Node* batch_size_node;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size_node));
      Node* row_shape_node;
      std::vector<int64_t> row_shape;
      row_shape.reserve(
          row_shape_.dims());  // not an unknown rank PartialTensorShape
      for (int i = 0; i < row_shape_.dims(); i++)
        row_shape.emplace_back(row_shape_.dim_size(i));
      TF_RETURN_IF_ERROR(b->AddVector(row_shape, &row_shape_node));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_node, batch_size_node, row_shape_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset<T>> {
     public:
      explicit Iterator(const typename Iterator::Params& params)
          : DatasetIterator<Dataset<T>>(params) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_11(mht_11_v, 359, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_12(mht_12_v, 364, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "Initialize");

        return DatasetIterator<Dataset<T>>::dataset()->input_->MakeIterator(
            ctx, this, DatasetIterator<Dataset<T>>::prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_13(mht_13_v, 374, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "GetNextInternal");

        // Each row of the output SparseTensor is an individual tensor
        // from the input iterator.
        std::vector<Tensor> batch_elements;
        int64_t total_elements = 0;
        batch_elements.reserve(
            DatasetIterator<Dataset<T>>::dataset()->batch_size_);
        const PartialTensorShape& row_shape =
            DatasetIterator<Dataset<T>>::dataset()->row_shape_;
        const int row_ndims = row_shape.dims();

        // Determine the size of the output tensors:
        // * dense_shape will be [`row_shape + 1`].
        Tensor dense_shape(ctx->allocator({}), DT_INT64, {row_ndims + 1});
        auto dense_shape_vec = dense_shape.vec<int64_t>();
        for (size_t i = 0; i < row_ndims; ++i) {
          if (row_shape.dim_size(i) == -1) {
            dense_shape_vec(i + 1) = 0;
          } else {
            dense_shape_vec(i + 1) = row_shape.dim_size(i);
          }
        }

        {
          mutex_lock l(mu_);
          *end_of_sequence = false;
          for (int i = 0;
               i < DatasetIterator<Dataset<T>>::dataset()->batch_size_ &&
               !*end_of_sequence;
               ++i) {
            std::vector<Tensor> batch_element_tuple;
            TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &batch_element_tuple,
                                                    end_of_sequence));
            if (!*end_of_sequence) {
              DCHECK_EQ(1, batch_element_tuple.size());
              batch_elements.push_back(std::move(batch_element_tuple[0]));
              total_elements += batch_element_tuple[0].NumElements();

              // TODO(mrry): Investigate how to hoist this check when we
              // have static information that renders it unnecessary.
              if (batch_element_tuple[0].shape().dims() != row_ndims) {
                return errors::InvalidArgument(
                    "Input element had shape (",
                    batch_element_tuple[0].shape().DebugString(),
                    ") that is incompatible with the row shape (",
                    row_shape.DebugString(), ").");
              }
              for (int j = 0; j < row_ndims; ++j) {
                // Take the maximum in the dimension if -1 is given.
                if (row_shape.dim_size(j) == -1) {
                  dense_shape_vec(j + 1) =
                      std::max(batch_element_tuple[0].dim_size(j),
                               dense_shape_vec(j + 1));
                } else if (batch_element_tuple[0].dim_size(j) >
                           row_shape.dim_size(j)) {
                  return errors::DataLoss(
                      "Input element had shape (",
                      batch_element_tuple[0].shape().DebugString(),
                      ") that is larger than the row shape (",
                      row_shape.DebugString(), ").");
                }
              }
            }
          }
        }

        if (batch_elements.empty()) {
          DCHECK(*end_of_sequence);
          return Status::OK();
        }

        // * indices will be [`total_elements`, `row_shape + 1`].
        // * values will be [`total_elements`].
        Tensor indices(ctx->allocator({}), DT_INT64,
                       {total_elements, row_ndims + 1});
        Tensor values(
            ctx->allocator({}),
            DatasetIterator<Dataset<T>>::dataset()->input_->output_dtypes()[0],
            {total_elements});
        auto indices_matrix = indices.matrix<int64_t>();
        auto values_flat = values.flat<T>();

        int64_t current_position_in_values = 0;
        for (int64_t i = 0; i < batch_elements.size(); ++i) {
          const Tensor& t = batch_elements[i];
          const auto& t_flat = t.flat<T>();
          // TODO(mrry): Replace with a memcpy or something more
          // efficient. (Maybe an Eigen assign op?)
          gtl::InlinedVector<int64_t, 4> strides(row_ndims);
          if (!strides.empty()) {
            strides[row_ndims - 1] = 1;
            for (int64_t row_dim = strides.size() - 2; row_dim >= 0;
                 --row_dim) {
              strides[row_dim] =
                  strides[row_dim + 1] * t.shape().dim_size(row_dim + 1);
            }
          }

          for (int64_t j = 0; j < t.NumElements(); ++j) {
            values_flat(current_position_in_values) = t_flat(j);
            indices_matrix(current_position_in_values, 0) = i;
            int64_t index = j;
            for (size_t k = 0; k < strides.size(); ++k) {
              indices_matrix(current_position_in_values, k + 1) =
                  index / strides[k];
              index %= strides[k];
            }
            ++current_position_in_values;
          }
        }

        dense_shape_vec(0) = batch_elements.size();

        Tensor serialized_sparse(DT_VARIANT, TensorShape({3}));
        auto serialized_sparse_t = serialized_sparse.vec<Variant>();
        serialized_sparse_t(0) = std::move(indices);
        serialized_sparse_t(1) = std::move(values);
        serialized_sparse_t(2) = std::move(dense_shape);
        out_tensors->push_back(std::move(serialized_sparse));

        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(
            std::move(args),
            DatasetIterator<Dataset<T>>::dataset()->batch_size_);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_14(mht_14_v, 510, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "SaveInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(Iterator::SaveInput(ctx, writer, input_impl_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdense_to_sparse_batch_dataset_opDTcc mht_15(mht_15_v, 520, "", "./tensorflow/core/kernels/data/experimental/dense_to_sparse_batch_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(Iterator::RestoreInput(ctx, reader, input_impl_));
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    };

    const int64_t batch_size_;
    const PartialTensorShape row_shape_;
    const DatasetBase* const input_;
    std::vector<PartialTensorShape> output_shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("DenseToSparseBatchDataset").Device(DEVICE_CPU),
                        DenseToSparseBatchDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalDenseToSparseBatchDataset").Device(DEVICE_CPU),
    DenseToSparseBatchDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
