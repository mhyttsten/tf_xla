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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc() {
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
#include <numeric>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

template <typename T>
class Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx,
                   const sparse::SparseTensor& sparse_tensor)
      : DatasetBase(DatasetContext(ctx)),
        sparse_tensor_(sparse_tensor),
        dtypes_({DT_INT64, sparse_tensor.dtype(), DT_INT64}),
        shapes_({{-1, sparse_tensor.dims() - 1},
                 {-1},
                 {sparse_tensor.dims() - 1}}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "Dataset");
}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(typename Iterator::Params{
        this, strings::StrCat(prefix, "::SparseTensorSlice")});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "output_dtypes");
 return dtypes_; }
  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "output_shapes");

    return shapes_;
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_3(mht_3_v, 231, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "DebugString");

    return "SparseTensorSliceDatasetOp::Dataset";
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "CardinalityInternal");

    return sparse_tensor_.shape()[0];
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_5(mht_5_v, 245, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "InputDatasets");

    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_6(mht_6_v, 252, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "CheckExternalState");
 return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_7(mht_7_v, 260, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "AsGraphDefInternal");

    Node* indices_node;
    TF_RETURN_IF_ERROR(b->AddTensor(sparse_tensor_.indices(), &indices_node));
    Node* value_node;
    TF_RETURN_IF_ERROR(b->AddTensor(sparse_tensor_.values(), &value_node));
    Node* dense_shape_node;
    std::vector<int64_t> dense_shape;
    dense_shape.reserve(sparse_tensor_.shape().size());
    for (int i = 0; i < sparse_tensor_.shape().size(); i++)
      dense_shape.emplace_back(sparse_tensor_.shape()[i]);
    TF_RETURN_IF_ERROR(b->AddVector(dense_shape, &dense_shape_node));
    AttrValue val_dtype;
    b->BuildAttrValue(sparse_tensor_.dtype(), &val_dtype);
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {indices_node, value_node, dense_shape_node},
                      {{"Tvalues", val_dtype}}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset<T>> {
   public:
    explicit Iterator(const typename Iterator::Params& params)
        : DatasetIterator<Dataset<T>>(params),
          num_elements_(params.dataset->sparse_tensor_.shape()[0]),
          dense_shape_(DT_INT64, {params.dataset->sparse_tensor_.dims() - 1}),
          group_iterable_(params.dataset->sparse_tensor_.group({0})),
          iter_(group_iterable_.begin()) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_8(mht_8_v, 290, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "Iterator");

      for (size_t i = 0; i < dense_shape_.NumElements(); ++i) {
        dense_shape_.vec<int64_t>()(i) =
            params.dataset->sparse_tensor_.shape()[i + 1];
      }
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_9(mht_9_v, 302, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "GetNextInternal");

      mutex_lock l(mu_);
      if (i_ == num_elements_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      out_tensors->clear();
      out_tensors->reserve(3);
      const int rank = Iterator::dataset()->sparse_tensor_.dims();

      if (i_ > next_non_empty_i_ && iter_ != group_iterable_.end()) {
        // We still have elements to consume from `group_iterable_`
        // and we have emitted all elements up to and including the
        // current position.
        sparse::Group group = *iter_;
        const auto indices = group.indices();
        const auto values = group.values<T>();
        const int64_t num_entries = values.size();
        next_non_empty_i_ = indices(0, 0);

        next_indices_ = Tensor(DT_INT64, {num_entries, rank - 1});
        next_values_ = Tensor(DataTypeToEnum<T>::value, {num_entries});

        auto next_indices_t = next_indices_.matrix<int64_t>();
        auto next_values_t = next_values_.vec<T>();

        for (int64_t i = 0; i < num_entries; ++i) {
          for (int d = 1; d < rank; ++d) {
            next_indices_t(i, d - 1) = indices(i, d);
          }
          next_values_t(i) = values(i);
        }

        ++iter_;
      }
      if (i_ == next_non_empty_i_) {
        // The current position is non-empty in the input
        // `SparseTensor`, and we have already read the value from the
        // `GroupIterable`.
        out_tensors->push_back(std::move(next_indices_));
        out_tensors->push_back(std::move(next_values_));
        out_tensors->push_back(dense_shape_);
        next_non_empty_i_ = kNextNonEmptyUnknown;
      } else {
        DCHECK(i_ < next_non_empty_i_ || iter_ == group_iterable_.end());
        // The current position is empty in the input `SparseTensor`,
        // so emit empty indices and values.
        out_tensors->push_back(Tensor(DT_INT64, TensorShape({0, rank - 1})));
        out_tensors->push_back(Tensor(DataTypeToEnum<T>::value, {0}));
        out_tensors->push_back(dense_shape_);
      }

      ++i_;
      *end_of_sequence = false;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_10(mht_10_v, 370, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(Iterator::full_name("i"), i_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(Iterator::full_name("iter_loc"), iter_.loc()));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          Iterator::full_name("next_non_empty_i_"), next_non_empty_i_));
      if (i_ <= next_non_empty_i_) {
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            Iterator::full_name("next_indices_"), next_indices_));
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            Iterator::full_name("next_values_"), next_values_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_11(mht_11_v, 390, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(reader->ReadScalar(Iterator::full_name("i"), &i_));
      int64_t iter_loc;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(Iterator::full_name("iter_loc"), &iter_loc));
      iter_ = group_iterable_.at(iter_loc);
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          Iterator::full_name("next_non_empty_i_"), &next_non_empty_i_));
      if (i_ <= next_non_empty_i_) {
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            Iterator::full_name("next_indices_"), &next_indices_));
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            Iterator::full_name("next_values_"), &next_values_));
      }
      return Status::OK();
    }

   private:
    const int64_t num_elements_;

    Tensor dense_shape_;

    mutex mu_;
    sparse::GroupIterable group_iterable_ TF_GUARDED_BY(mu_);
    sparse::GroupIterable::IteratorStep iter_ TF_GUARDED_BY(mu_);
    int64_t i_ TF_GUARDED_BY(mu_) = 0;
    const int64_t kNextNonEmptyUnknown = -1;
    int64_t next_non_empty_i_ TF_GUARDED_BY(mu_) = kNextNonEmptyUnknown;
    Tensor next_indices_ TF_GUARDED_BY(mu_);
    Tensor next_values_ TF_GUARDED_BY(mu_);
  };

  const sparse::SparseTensor sparse_tensor_;
  const DataTypeVector dtypes_;
  const std::vector<PartialTensorShape> shapes_;
};

template <typename T>
class SparseTensorSliceDatasetOp : public DatasetOpKernel {
 public:
  explicit SparseTensorSliceDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_12(mht_12_v, 435, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "SparseTensorSliceDatasetOp");
}

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_opDTcc mht_13(mht_13_v, 440, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op.cc", "MakeDataset");

    // Create a new SparseTensorSliceDatasetOp::Dataset, insert it in
    // the step container, and return it as the output.
    const Tensor* indices;
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices));
    const Tensor* values;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values));
    const Tensor* dense_shape;
    OP_REQUIRES_OK(ctx, ctx->input("dense_shape", &dense_shape));

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(indices->shape()),
                errors::InvalidArgument("Input indices must be a matrix. Got: ",
                                        indices->shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(values->shape()),
                errors::InvalidArgument("Input values must be a vector. Got: ",
                                        values->shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(dense_shape->shape()),
                errors::InvalidArgument("Input shape must be a vector. Got: ",
                                        dense_shape->shape().DebugString()));
    OP_REQUIRES(
        ctx, values->shape().dim_size(0) == indices->shape().dim_size(0),
        errors::InvalidArgument(
            "Number of values must match first dimension of indices. ", "Got ",
            values->shape().dim_size(0),
            " values, indices shape: ", indices->shape().DebugString()));
    OP_REQUIRES(
        ctx, dense_shape->shape().dim_size(0) == indices->shape().dim_size(1),
        errors::InvalidArgument(
            "Number of dimensions must match second dimension of indices. ",
            "Got ", dense_shape->shape().dim_size(0),
            " dimensions, indices shape: ", indices->shape().DebugString()));
    OP_REQUIRES(ctx, dense_shape->NumElements() > 0,
                errors::InvalidArgument(
                    "The shape argument requires at least one element."));

    // We currently ensure that `sparse_tensor` is ordered in the
    // batch dimension.
    // TODO(mrry): Investigate ways to avoid this unconditional check
    // if we can be sure that the sparse tensor was produced in an
    // appropriate order (e.g. by `tf.parse_example()` or a Dataset
    // that batches elements into rows of a SparseTensor).
    int64_t previous_batch_index = -1;
    for (int64_t i = 0; i < indices->dim_size(0); ++i) {
      int64_t next_batch_index = indices->matrix<int64_t>()(i, 0);
      OP_REQUIRES(
          ctx, next_batch_index >= previous_batch_index,
          errors::Unimplemented("The SparseTensor must be ordered in the batch "
                                "dimension; handling arbitrarily ordered input "
                                "is not currently supported."));
      previous_batch_index = next_batch_index;
    }
    gtl::InlinedVector<int64_t, 8> std_order(dense_shape->NumElements(), 0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, TensorShape::BuildTensorShape(
                            dense_shape->vec<int64_t>(), &shape));
    sparse::SparseTensor tensor;
    OP_REQUIRES_OK(ctx, sparse::SparseTensor::Create(*indices, *values, shape,
                                                     std_order, &tensor));
    *output = new Dataset<T>(ctx, std::move(tensor));
  }

 private:
};

#define REGISTER_DATASET_KERNEL(type)                           \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorSliceDataset")      \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<type>("Tvalues"), \
                          SparseTensorSliceDatasetOp<type>);

TF_CALL_DATASET_TYPES(REGISTER_DATASET_KERNEL);
#undef REGISTER_DATASET_KERNEL

}  // namespace
}  // namespace data
}  // namespace tensorflow
