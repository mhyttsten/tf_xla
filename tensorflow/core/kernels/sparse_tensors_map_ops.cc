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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc() {
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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/overflow.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

using sparse::SparseTensor;

class SparseTensorsMap : public ResourceBase {
 public:
  explicit SparseTensorsMap(const string& name) : name_(name), counter_(0) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "SparseTensorsMap");
}

  string DebugString() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "DebugString");
 return "A SparseTensorsMap"; }

  typedef struct {
    Tensor indices;
    Tensor values;
    gtl::InlinedVector<int64_t, 8> shape;
  } PersistentSparseTensor;

  Status AddSparseTensor(OpKernelContext* ctx, const SparseTensor& sp,
                         int64_t* handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "AddSparseTensor");

    Tensor ix;
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(sp.indices().dtype(), sp.indices().shape(), &ix));
    ix = sp.indices();

    Tensor values;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(sp.indices().dtype(),
                                          sp.indices().shape(), &values));
    values = sp.values();
    {
      mutex_lock l(mu_);
      int64_t unique_st_handle = counter_++;  // increment is guarded on purpose
      sp_tensors_[unique_st_handle] = PersistentSparseTensor{
          ix, values,
          gtl::InlinedVector<int64_t, 8>(sp.shape().begin(), sp.shape().end())};
      *handle = unique_st_handle;
    }
    return Status::OK();
  }

  Status RetrieveAndClearSparseTensors(
      OpKernelContext* ctx, const TTypes<int64_t>::ConstVec& handles,
      std::vector<SparseTensor>* sparse_tensors) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_3(mht_3_v, 255, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "RetrieveAndClearSparseTensors");

    sparse_tensors->clear();
    sparse_tensors->reserve(handles.size());
    {
      mutex_lock l(mu_);
      for (size_t i = 0; i < handles.size(); ++i) {
        const int64_t handle = handles(i);
        auto sp_iter = sp_tensors_.find(handle);
        if (sp_iter == sp_tensors_.end()) {
          return errors::InvalidArgument(
              "Unable to find SparseTensor: ", handle, " in map: ", name_);
        }
        const Tensor* ix = &sp_iter->second.indices;
        const Tensor* values = &sp_iter->second.values;
        const auto& shape = sp_iter->second.shape;
        SparseTensor tensor;
        TF_RETURN_IF_ERROR(SparseTensor::Create(*ix, *values, shape, &tensor));
        sparse_tensors->push_back(std::move(tensor));
        sp_tensors_.erase(sp_iter);
      }
    }

    return Status::OK();
  }

 protected:
  ~SparseTensorsMap() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_4(mht_4_v, 284, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "~SparseTensorsMap");
}

 private:
  string name_;

  mutex mu_;
  int64_t counter_ TF_GUARDED_BY(mu_);
  std::unordered_map<int64_t, PersistentSparseTensor> sp_tensors_
      TF_GUARDED_BY(mu_);
};

class SparseTensorAccessingOp : public OpKernel {
 public:
  typedef std::function<Status(SparseTensorsMap**)> CreatorCallback;

  explicit SparseTensorAccessingOp(OpKernelConstruction* context)
      : OpKernel(context), sparse_tensors_map_(nullptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_5(mht_5_v, 303, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "SparseTensorAccessingOp");
}

 protected:
  ~SparseTensorAccessingOp() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_6(mht_6_v, 309, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "~SparseTensorAccessingOp");

    if (sparse_tensors_map_) sparse_tensors_map_->Unref();
  }

  Status GetMap(OpKernelContext* ctx, bool is_writing,
                SparseTensorsMap** sparse_tensors_map) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_7(mht_7_v, 317, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "GetMap");

    mutex_lock l(mu_);

    if (sparse_tensors_map_) {
      *sparse_tensors_map = sparse_tensors_map_;
      return Status::OK();
    }

    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def(),
                                   is_writing /* use_node_name_as_default */));

    CreatorCallback sparse_tensors_map_creator = [this](SparseTensorsMap** c) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_8(mht_8_v, 331, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "lambda");

      SparseTensorsMap* map = new SparseTensorsMap(cinfo_.name());
      *c = map;
      return Status::OK();
    };

    TF_RETURN_IF_ERROR(
        cinfo_.resource_manager()->LookupOrCreate<SparseTensorsMap>(
            cinfo_.container(), cinfo_.name(), &sparse_tensors_map_,
            sparse_tensors_map_creator));

    *sparse_tensors_map = sparse_tensors_map_;
    return Status::OK();
  }

 private:
  ContainerInfo cinfo_;

  mutex mu_;
  SparseTensorsMap* sparse_tensors_map_ TF_PT_GUARDED_BY(mu_);
};

class AddSparseToTensorsMapOp : public SparseTensorAccessingOp {
 public:
  explicit AddSparseToTensorsMapOp(OpKernelConstruction* context)
      : SparseTensorAccessingOp(context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_9(mht_9_v, 359, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "AddSparseToTensorsMapOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_10(mht_10_v, 364, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "Compute");

    const Tensor* input_indices;
    const Tensor* input_values;
    const Tensor* input_shape;
    SparseTensorsMap* map;

    OP_REQUIRES_OK(context, context->input("sparse_indices", &input_indices));
    OP_REQUIRES_OK(context, context->input("sparse_values", &input_values));
    OP_REQUIRES_OK(context, context->input("sparse_shape", &input_shape));
    OP_REQUIRES_OK(context, GetMap(context, true /* is_writing */, &map));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString()));

    TensorShape input_shape_object;
    OP_REQUIRES_OK(
        context, TensorShapeUtils::MakeShape(input_shape->vec<int64_t>().data(),
                                             input_shape->NumElements(),
                                             &input_shape_object));
    SparseTensor st;
    OP_REQUIRES_OK(context, SparseTensor::Create(*input_indices, *input_values,
                                                 input_shape_object, &st));
    int64_t handle;
    OP_REQUIRES_OK(context, map->AddSparseTensor(context, st, &handle));

    Tensor sparse_handle(DT_INT64, TensorShape({}));
    auto sparse_handle_t = sparse_handle.scalar<int64_t>();

    sparse_handle_t() = handle;

    context->set_output(0, sparse_handle);
  }
};

REGISTER_KERNEL_BUILDER(Name("AddSparseToTensorsMap").Device(DEVICE_CPU),
                        AddSparseToTensorsMapOp);

template <typename T>
class AddManySparseToTensorsMapOp : public SparseTensorAccessingOp {
 public:
  explicit AddManySparseToTensorsMapOp(OpKernelConstruction* context)
      : SparseTensorAccessingOp(context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_11(mht_11_v, 420, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "AddManySparseToTensorsMapOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_12(mht_12_v, 425, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "Compute");

    const Tensor* input_indices;
    const Tensor* input_values;
    const Tensor* input_shape;
    SparseTensorsMap* map;

    OP_REQUIRES_OK(context, context->input("sparse_indices", &input_indices));
    OP_REQUIRES_OK(context, context->input("sparse_values", &input_values));
    OP_REQUIRES_OK(context, context->input("sparse_shape", &input_shape));
    OP_REQUIRES_OK(context, GetMap(context, true /* is_writing */, &map));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString()));
    OP_REQUIRES(
        context,
        input_values->shape().dim_size(0) == input_indices->shape().dim_size(0),
        errors::InvalidArgument(
            "Number of values must match first dimension of indices. ", "Got ",
            input_values->shape().dim_size(0),
            " values, indices shape: ", input_indices->shape().DebugString()));
    OP_REQUIRES(
        context,
        input_shape->shape().dim_size(0) == input_indices->shape().dim_size(1),
        errors::InvalidArgument(
            "Number of dimensions must match second dimension of indices. ",
            "Got ", input_shape->shape().dim_size(0),
            " dimensions, indices shape: ",
            input_indices->shape().DebugString()));

    int rank = input_shape->NumElements();

    OP_REQUIRES(
        context, rank > 1,
        errors::InvalidArgument(
            "Rank of input SparseTensor should be > 1, but saw rank: ", rank));

    auto input_shape_vec = input_shape->vec<int64_t>();

    TensorShape tensor_input_shape;
    OP_REQUIRES_OK(context, TensorShape::BuildTensorShape(input_shape_vec,
                                                          &tensor_input_shape));
    gtl::InlinedVector<int64_t, 8> std_order(rank);
    std::iota(std_order.begin(), std_order.end(), 0);
    SparseTensor input_st;
    OP_REQUIRES_OK(context, SparseTensor::Create(*input_indices, *input_values,
                                                 tensor_input_shape, std_order,
                                                 &input_st));

    const int64_t N = input_shape_vec(0);

    Tensor sparse_handles(DT_INT64, TensorShape({N}));
    auto sparse_handles_t = sparse_handles.vec<int64_t>();

    OP_REQUIRES_OK(context, input_st.IndicesValid());

    // We can generate the output shape proto string now, for all
    // minibatch entries.
    TensorShape output_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_shape_vec.data() + 1,
                                input_shape->NumElements() - 1, &output_shape));

    // Get groups by minibatch dimension
    std::unordered_set<int64_t> visited;
    sparse::GroupIterable minibatch = input_st.group({0});
    for (const auto& subset : minibatch) {
      const int64_t b = subset.group()[0];
      visited.insert(b);
      OP_REQUIRES(
          context, b > -1 && b < N,
          errors::InvalidArgument(
              "Received unexpected column 0 value in input SparseTensor: ", b,
              " < 0 or >= N (= ", N, ")"));

      const auto indices = subset.indices();
      const auto values = subset.values<T>();
      const int64_t num_entries = values.size();

      Tensor output_indices = Tensor(DT_INT64, {num_entries, rank - 1});
      Tensor output_values = Tensor(DataTypeToEnum<T>::value, {num_entries});

      auto output_indices_t = output_indices.matrix<int64_t>();
      auto output_values_t = output_values.vec<T>();

      for (int i = 0; i < num_entries; ++i) {
        for (int d = 1; d < rank; ++d) {
          output_indices_t(i, d - 1) = indices(i, d);
        }
        output_values_t(i) = values(i);
      }

      SparseTensor st_i;
      OP_REQUIRES_OK(context,
                     SparseTensor::Create(output_indices, output_values,
                                          output_shape, &st_i));
      int64_t handle;
      OP_REQUIRES_OK(context, map->AddSparseTensor(context, st_i, &handle));
      sparse_handles_t(b) = handle;
    }

    // Fill in any gaps; we must provide an empty ST for batch entries
    // the grouper didn't find.
    if (visited.size() < N) {
      Tensor empty_indices(DT_INT64, {0, rank - 1});
      Tensor empty_values(DataTypeToEnum<T>::value, {0});
      SparseTensor empty_st;
      OP_REQUIRES_OK(context, SparseTensor::Create(empty_indices, empty_values,
                                                   output_shape, &empty_st));

      for (int64_t b = 0; b < N; ++b) {
        // We skipped this batch entry.
        if (visited.find(b) == visited.end()) {
          int64_t handle;
          OP_REQUIRES_OK(context,
                         map->AddSparseTensor(context, empty_st, &handle));
          sparse_handles_t(b) = handle;
        }
      }
    }

    context->set_output(0, sparse_handles);
  }
};

#define REGISTER_KERNELS(type)                              \
  REGISTER_KERNEL_BUILDER(Name("AddManySparseToTensorsMap") \
                              .Device(DEVICE_CPU)           \
                              .TypeConstraint<type>("T"),   \
                          AddManySparseToTensorsMapOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

template <typename T>
class TakeManySparseFromTensorsMapOp : public SparseTensorAccessingOp {
 public:
  explicit TakeManySparseFromTensorsMapOp(OpKernelConstruction* context)
      : SparseTensorAccessingOp(context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_13(mht_13_v, 575, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "TakeManySparseFromTensorsMapOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensors_map_opsDTcc mht_14(mht_14_v, 580, "", "./tensorflow/core/kernels/sparse_tensors_map_ops.cc", "Compute");

    SparseTensorsMap* map = nullptr;
    OP_REQUIRES_OK(context, GetMap(context, false /* is_writing */, &map));

    const Tensor& sparse_handles = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(sparse_handles.shape()),
                errors::InvalidArgument(
                    "sparse_handles should be a vector but received shape ",
                    sparse_handles.shape().DebugString()));

    int64_t N = sparse_handles.shape().dim_size(0);

    OP_REQUIRES(
        context, N > 0,
        errors::InvalidArgument("Must have at least 1 serialized SparseTensor, "
                                "but input matrix has 0 rows"));

    std::vector<Tensor> indices_to_concat;
    std::vector<Tensor> values_to_concat;
    std::vector<TensorShape> shapes_to_concat;

    const auto& sparse_handles_t = sparse_handles.vec<int64_t>();

    std::vector<SparseTensor> sparse_tensors;

    OP_REQUIRES_OK(context, map->RetrieveAndClearSparseTensors(
                                context, sparse_handles_t, &sparse_tensors));

    for (int64_t i = 0; i < N; ++i) {
      const SparseTensor& st = sparse_tensors[i];
      const Tensor& output_indices = st.indices();
      const Tensor& output_values = st.values();
      const auto output_shape = st.shape();

      OP_REQUIRES(context, TensorShapeUtils::IsMatrix(output_indices.shape()),
                  errors::InvalidArgument(
                      "Expected sparse_handles[", i,
                      "] to represent an index matrix but received shape ",
                      output_indices.shape().DebugString()));
      OP_REQUIRES(context, TensorShapeUtils::IsVector(output_values.shape()),
                  errors::InvalidArgument(
                      "Expected sparse_handles[", i,
                      "] to represent a values vector but received shape ",
                      output_values.shape().DebugString()));
      OP_REQUIRES(
          context, DataTypeToEnum<T>::value == output_values.dtype(),
          errors::InvalidArgument(
              "Requested SparseTensor of type ",
              DataTypeString(DataTypeToEnum<T>::value), " but SparseTensor[", i,
              "].values.dtype() == ", DataTypeString(output_values.dtype())));

      int64_t num_entries = output_indices.dim_size(0);
      OP_REQUIRES(context, num_entries == output_values.dim_size(0),
                  errors::InvalidArgument(
                      "Expected row counts of SparseTensor[", i,
                      "].indices and SparseTensor[", i,
                      "].values to match but they do not: ", num_entries,
                      " vs. ", output_values.dim_size(0)));
      int rank = output_indices.dim_size(1);
      OP_REQUIRES(
          context, rank == output_shape.size(),
          errors::InvalidArgument("Expected column counts of SparseTensor[", i,
                                  "].indices to match size of SparseTensor[", i,
                                  "].shape "
                                  "but they do not: ",
                                  rank, " vs. ", output_shape.size()));

      // Now we expand each SparseTensors' indices and shape by
      // prefixing a dimension
      Tensor expanded_indices(
          DT_INT64, TensorShape({num_entries, 1 + output_indices.dim_size(1)}));
      Tensor expanded_shape(DT_INT64, TensorShape({1 + rank}));
      const auto& output_indices_t = output_indices.matrix<int64_t>();
      auto expanded_indices_t = expanded_indices.matrix<int64_t>();
      auto expanded_shape_t = expanded_shape.vec<int64_t>();
      expanded_indices_t.chip<1>(0).setZero();
      Eigen::DSizes<Eigen::DenseIndex, 2> indices_start(0, 1);
      Eigen::DSizes<Eigen::DenseIndex, 2> indices_sizes(num_entries, rank);
      expanded_indices_t.slice(indices_start, indices_sizes) = output_indices_t;
      expanded_shape_t(0) = 1;
      // TODO: copy shape from TensorShape to &expanded_shape_t(1)
      // std::copy_n(&output_shape_t(0), rank, &expanded_shape_t(1));
      for (int i = 0; i < rank; ++i) {
        expanded_shape_t(i + 1) = output_shape[i];
      }
      TensorShape expanded_tensor_shape(expanded_shape_t);

      indices_to_concat.push_back(std::move(expanded_indices));
      values_to_concat.push_back(output_values);
      shapes_to_concat.push_back(std::move(expanded_tensor_shape));
    }

    int rank = -1;
    for (int i = 0; i < N; ++i) {
      if (rank < 0) rank = shapes_to_concat[i].dims();
      OP_REQUIRES(context, rank == shapes_to_concat[i].dims(),
                  errors::InvalidArgument(
                      "Inconsistent rank across SparseTensors: rank prior to "
                      "SparseTensor[",
                      i, "] was: ", rank, " but rank of SparseTensor[", i,
                      "] is: ", shapes_to_concat[i].dims()));
    }

    // SparseTensor::Concat requires consistent shape for all but the
    // primary order dimension (dimension 0 in this case).  So we get
    // the maximum value across all the input SparseTensors for each
    // dimension and use that.
    TensorShape preconcat_shape(shapes_to_concat[0]);
    for (int i = 0; i < N; ++i) {
      for (int d = 0; d < rank; ++d) {
        preconcat_shape.set_dim(d, std::max(preconcat_shape.dim_size(d),
                                            shapes_to_concat[i].dim_size(d)));
      }
    }

    // Dimension 0 is the primary dimension.
    gtl::InlinedVector<int64_t, 8> std_order(rank);
    std::iota(std_order.begin(), std_order.end(), 0);

    std::vector<SparseTensor> tensors_to_concat;
    tensors_to_concat.reserve(N);
    for (int i = 0; i < N; ++i) {
      SparseTensor tensor;
      OP_REQUIRES_OK(context,
                     SparseTensor::Create(std::move(indices_to_concat[i]),
                                          std::move(values_to_concat[i]),
                                          preconcat_shape, std_order, &tensor));
      tensors_to_concat.push_back(std::move(tensor));
    }

    auto output = SparseTensor::Concat<T>(tensors_to_concat);
    Tensor final_output_shape(DT_INT64, TensorShape({output.dims()}));

    std::copy_n(output.shape().data(), output.dims(),
                final_output_shape.vec<int64_t>().data());

    context->set_output(0, output.indices());
    context->set_output(1, output.values());
    context->set_output(2, final_output_shape);
  }
};

#define REGISTER_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("TakeManySparseFromTensorsMap") \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<type>("dtype"),  \
                          TakeManySparseFromTensorsMapOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow
