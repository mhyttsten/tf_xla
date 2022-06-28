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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_enqueue_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_enqueue_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_enqueue_opsDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/kernels/tpu_embedding_enqueue_ops.h"

#include <string>
#include <vector>

#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"

namespace tensorflow {

Status ValidateCombiners(absl::Span<const std::string> combiners) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_enqueue_opsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/tpu/kernels/tpu_embedding_enqueue_ops.cc", "ValidateCombiners");

  for (const std::string& combiner : combiners) {
    if (combiner != "sum" && combiner != "mean" && combiner != "sqrtn") {
      return errors::InvalidArgument(
          "Invalid combiner: only \"sum\", \"mean\", and "
          "\"sqrtn\" are supported.");
    }
  }
  return Status::OK();
}

Status GetValidatedModeOverride(const string& mode_override,
                                tpu::TPUEmbeddingConfiguration::Mode* mode) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("mode_override: \"" + mode_override + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_enqueue_opsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/tpu/kernels/tpu_embedding_enqueue_ops.cc", "GetValidatedModeOverride");

  if (mode_override == "train") {
    *mode = tpu::TPUEmbeddingConfiguration::TRAINING;
  } else if (mode_override == "inference") {
    *mode = tpu::TPUEmbeddingConfiguration::INFERENCE;
  } else if (mode_override == "unspecified") {
    *mode = tpu::TPUEmbeddingConfiguration::UNSPECIFIED;
  } else {
    return errors::InvalidArgument("Unsupported value ", mode_override,
                                   " specified for mode_override.");
  }
  return Status::OK();
}

namespace {

// Deallocates all tensors in `tf_tensors`.
void DeleteTensors(std::vector<TF_Tensor*>& tf_tensors) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_enqueue_opsDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/tpu/kernels/tpu_embedding_enqueue_ops.cc", "DeleteTensors");

  for (TF_Tensor* tf_tensor : tf_tensors) {
    TF_DeleteTensor(tf_tensor);
  }
  tf_tensors.clear();
}

// T1: The type of the sample_indices op input.
// T2: The type of the embedding_indices op input.
// T3: The type of the aggregation_weights op input.
template <typename T1, typename T2, typename T3>
class EnqueueTPUEmbeddingArbitraryTensorBatchOp : public OpKernel {
 public:
  explicit EnqueueTPUEmbeddingArbitraryTensorBatchOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_enqueue_opsDTcc mht_3(mht_3_v, 254, "", "./tensorflow/core/tpu/kernels/tpu_embedding_enqueue_ops.cc", "EnqueueTPUEmbeddingArbitraryTensorBatchOp");

    if (ctx->HasAttr("device_ordinal")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));
      device_ordinal_set_ = true;
    }

    std::vector<std::string> combiners;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiners", &combiners));
    OP_REQUIRES_OK(ctx, ValidateCombiners(combiners));
    TpuEmbedding_TensorBatchFixedState_Create_Params fixed_state_create_params;
    StatusHelper status;
    fixed_state_create_params.status = status.c_status;

    std::vector<char*> c_str_combiners;
    c_str_combiners.reserve(combiners.size());
    for (std::string combiner : combiners) {
      c_str_combiners.push_back(&combiner.front());
    }
    fixed_state_create_params.combiners_size = c_str_combiners.size();
    fixed_state_create_params.combiners = c_str_combiners.data();
    fixed_state_ = tpu::OpsApiFn()->TpuEmbeddingTensorBatchFixedState_CreateFn(
        &fixed_state_create_params);

    OP_REQUIRES_OK(ctx, status.status());
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_enqueue_opsDTcc mht_4(mht_4_v, 283, "", "./tensorflow/core/tpu/kernels/tpu_embedding_enqueue_ops.cc", "Compute");

    VLOG(2) << "EnqueueTPUEmbeddingArbitraryTensorBatchOp::Compute";

    OpInputList sample_indices_or_row_splits_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("sample_indices_or_row_splits",
                                        &sample_indices_or_row_splits_list));

    OpInputList embedding_indices_list;
    OP_REQUIRES_OK(
        ctx, ctx->input_list("embedding_indices", &embedding_indices_list));

    OpInputList aggregation_weights_list;
    OP_REQUIRES_OK(
        ctx, ctx->input_list("aggregation_weights", &aggregation_weights_list));

    const Tensor* mode_override;
    OP_REQUIRES_OK(ctx, ctx->input("mode_override", &mode_override));

    const string& mode_value = mode_override->scalar<tstring>()();
    tpu::TPUEmbeddingConfiguration::Mode mode;
    OP_REQUIRES_OK(ctx, GetValidatedModeOverride(mode_value, &mode));

    // Extract device_ordinal from input when op did not has it as attribute.
    if (!device_ordinal_set_) {
      const Tensor* device_ordinal_tensor;
      OP_REQUIRES_OK(ctx, ctx->input("device_ordinal", &device_ordinal_tensor));
      device_ordinal_ = device_ordinal_tensor->flat<int32>()(0);
      device_ordinal_set_ = true;
    }

    const int num_input_features = sample_indices_or_row_splits_list.size();

    std::vector<TF_Tensor*> sample_indices_or_row_splits_tensors(
        num_input_features);
    std::vector<TF_Tensor*> embedding_indices_tensors(num_input_features);
    std::vector<TF_Tensor*> aggregation_weights_tensors(num_input_features);

    for (int i = 0; i < num_input_features; ++i) {
      Status tf_status;
      sample_indices_or_row_splits_tensors[i] =
          TF_TensorFromTensor(sample_indices_or_row_splits_list[i], &tf_status);
      OP_REQUIRES_OK(ctx, tf_status);
      embedding_indices_tensors[i] =
          TF_TensorFromTensor(embedding_indices_list[i], &tf_status);
      OP_REQUIRES_OK(ctx, tf_status);
      aggregation_weights_tensors[i] =
          TF_TensorFromTensor(aggregation_weights_list[i], &tf_status);
      OP_REQUIRES_OK(ctx, tf_status);
    }

    TpuEmbeddingEngine_EnqueueTensorBatch_Params params;
    params.sample_indices_tensors = sample_indices_or_row_splits_tensors.data();
    params.sample_indices_tensors_size = num_input_features;
    params.embedding_indices_tensors = embedding_indices_tensors.data();
    params.embedding_indices_tensors_size = num_input_features;
    params.aggregation_weights_tensors = aggregation_weights_tensors.data();
    params.aggregation_weights_tensors_size = num_input_features;

    StatusHelper status;
    params.status = status.c_status;
    params.status = status.c_status;
    params.fixed_state = fixed_state_;
    params.local_device_ordinal = device_ordinal_;
    params.mode = mode;

    tpu::OpsApiFn()->TpuEmbeddingEngine_EnqueueTensorBatchFn(&params);
    OP_REQUIRES_OK(ctx, status.status());

    DeleteTensors(sample_indices_or_row_splits_tensors);
    DeleteTensors(embedding_indices_tensors);
    DeleteTensors(aggregation_weights_tensors);

    VLOG(2) << "EnqueueTPUEmbeddingArbitraryTensorBatchOp::Compute done";
  }

  ~EnqueueTPUEmbeddingArbitraryTensorBatchOp() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_enqueue_opsDTcc mht_5(mht_5_v, 361, "", "./tensorflow/core/tpu/kernels/tpu_embedding_enqueue_ops.cc", "~EnqueueTPUEmbeddingArbitraryTensorBatchOp");

    tpu::OpsApiFn()->TpuEmbeddingTensorBatchFixedState_DestroyFn(fixed_state_);
  }

 private:
  int device_ordinal_ = -1;
  bool device_ordinal_set_ = false;
  TpuEmbedding_TensorBatchFixedState* fixed_state_;

  TF_DISALLOW_COPY_AND_ASSIGN(EnqueueTPUEmbeddingArbitraryTensorBatchOp);
};

#ifdef LIBTPU_ON_GCE
// Arbitrary tensor batch.
REGISTER_KERNEL_BUILDER(
    Name("EnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int32>("T1")
        .TypeConstraint<int32>("T2")
        .TypeConstraint<float>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int32, int32, float>);
REGISTER_KERNEL_BUILDER(
    Name("EnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int64>("T1")
        .TypeConstraint<int64>("T2")
        .TypeConstraint<float>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int64, int64, float>);
REGISTER_KERNEL_BUILDER(
    Name("EnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int32>("T1")
        .TypeConstraint<int64>("T2")
        .TypeConstraint<float>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int32, int64, float>);
REGISTER_KERNEL_BUILDER(
    Name("EnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int64>("T1")
        .TypeConstraint<int32>("T2")
        .TypeConstraint<float>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int64, int32, float>);
REGISTER_KERNEL_BUILDER(
    Name("EnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int32>("T1")
        .TypeConstraint<int32>("T2")
        .TypeConstraint<double>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int32, int32, double>);
REGISTER_KERNEL_BUILDER(
    Name("EnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int64>("T1")
        .TypeConstraint<int64>("T2")
        .TypeConstraint<double>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int64, int64, double>);
REGISTER_KERNEL_BUILDER(
    Name("EnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int32>("T1")
        .TypeConstraint<int64>("T2")
        .TypeConstraint<double>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int32, int64, double>);
REGISTER_KERNEL_BUILDER(
    Name("EnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int64>("T1")
        .TypeConstraint<int32>("T2")
        .TypeConstraint<double>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int64, int32, double>);

// Arbitrary tensor batch dynamic op having device tensor as input.
REGISTER_KERNEL_BUILDER(
    Name("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int32>("T1")
        .TypeConstraint<int32>("T2")
        .TypeConstraint<float>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int32, int32, float>);
REGISTER_KERNEL_BUILDER(
    Name("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int64>("T1")
        .TypeConstraint<int64>("T2")
        .TypeConstraint<float>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int64, int64, float>);
REGISTER_KERNEL_BUILDER(
    Name("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int32>("T1")
        .TypeConstraint<int64>("T2")
        .TypeConstraint<float>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int32, int64, float>);
REGISTER_KERNEL_BUILDER(
    Name("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int64>("T1")
        .TypeConstraint<int32>("T2")
        .TypeConstraint<float>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int64, int32, float>);
REGISTER_KERNEL_BUILDER(
    Name("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int32>("T1")
        .TypeConstraint<int32>("T2")
        .TypeConstraint<double>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int32, int32, double>);
REGISTER_KERNEL_BUILDER(
    Name("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int64>("T1")
        .TypeConstraint<int64>("T2")
        .TypeConstraint<double>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int64, int64, double>);
REGISTER_KERNEL_BUILDER(
    Name("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int32>("T1")
        .TypeConstraint<int64>("T2")
        .TypeConstraint<double>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int32, int64, double>);
REGISTER_KERNEL_BUILDER(
    Name("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")
        .TypeConstraint<int64>("T1")
        .TypeConstraint<int32>("T2")
        .TypeConstraint<double>("T3")
        .Device(DEVICE_CPU),
    EnqueueTPUEmbeddingArbitraryTensorBatchOp<int64, int32, double>);
#endif  // LIBTPU_ON_GCE
}  // namespace
}  // namespace tensorflow
