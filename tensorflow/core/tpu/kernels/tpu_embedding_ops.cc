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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/cleanup/cleanup.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"

namespace tensorflow {

namespace {

// This TensorFlow op receives a batch of activations from the
// TpuEmbeddingEngine.
class RecvTPUEmbeddingActivationsOp : public XlaOpKernel {
 public:
  explicit RecvTPUEmbeddingActivationsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/tpu/kernels/tpu_embedding_ops.cc", "RecvTPUEmbeddingActivationsOp");

    string config_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string));

    OP_REQUIRES(
        ctx, tpu_embedding_config_.ParseFromString(config_string),
        xla::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                             "proto from config attr"));
  }

  ~RecvTPUEmbeddingActivationsOp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/tpu/kernels/tpu_embedding_ops.cc", "~RecvTPUEmbeddingActivationsOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/tpu/kernels/tpu_embedding_ops.cc", "Compile");

    ResourceMgr* rm = GetTPUConfigResourceMgr();

    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);
    OP_REQUIRES(
        ctx, ctx->num_inputs() == 1,
        errors::Internal("Kernel has ", ctx->num_inputs(),
                         " inputs but configuration expects one input"));

    xla::XlaOp deduplication_data = ctx->Input("deduplication_data");

    TpuEmbeddingEngine_RecvActivationsComputation_Params recv_activation_params;
    TpuSerializedProto xla_computation_serialized;
    auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
      StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
    });
    recv_activation_params.xla_computation = &xla_computation_serialized;
    StatusHelper status;
    recv_activation_params.status = status.c_status;
    recv_activation_params.tpu_mesh_state = mesh_state->data();
    auto builder = ctx->builder();
    OP_REQUIRES_VALUE(auto shape, ctx, builder->GetShape(deduplication_data));
    XLA_Shape c_shape;
    ApiConverter::ToC(shape, &c_shape);
    auto c_shape_cleanup =
        absl::MakeCleanup([&c_shape] { ApiConverter::Destroy(&c_shape); });
    recv_activation_params.deduplication_data_shape = &c_shape;
    tpu::OpsApiFn()->TpuEmbeddingEngine_RecvActivationsComputationFn(
        &recv_activation_params);
    OP_REQUIRES_OK(ctx, status.status());
    auto xla_computation =
        stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
            xla_computation_serialized);
    auto final_activations =
        xla::Call(builder, xla_computation, {deduplication_data});

    int32 output_count = tpu_embedding_config_.feature_descriptor_size();
    OP_REQUIRES(
        ctx, ctx->num_outputs() == output_count,
        xla::InvalidArgument(
            "Kernel has %d outputs but configuration expects %d outputs.",
            ctx->num_outputs(), output_count));

    for (int32 output_id = 0; output_id < output_count; ++output_id) {
      ctx->SetOutput(output_id,
                     xla::GetTupleElement(final_activations, output_id));
    }
  }

 private:
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvTPUEmbeddingActivationsOp);
};

REGISTER_XLA_OP(Name("_RecvTPUEmbeddingActivations").AllowVariantTypes(),
                RecvTPUEmbeddingActivationsOp);

// This TensorFlow op receives a batch of deduplication data from the
// TPUEmbeddingEngine. The output is a list of R1 tensors containing the weights
// and indices for gathering the embedding vectors.
class RecvTPUEmbeddingDeduplicationDataOp : public XlaOpKernel {
 public:
  explicit RecvTPUEmbeddingDeduplicationDataOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc mht_3(mht_3_v, 303, "", "./tensorflow/core/tpu/kernels/tpu_embedding_ops.cc", "RecvTPUEmbeddingDeduplicationDataOp");

    std::string config_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string));
    OP_REQUIRES(
        ctx, tpu_embedding_config_.ParseFromString(config_string),
        xla::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                             "proto from config attr"));
  }

  ~RecvTPUEmbeddingDeduplicationDataOp() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc mht_4(mht_4_v, 315, "", "./tensorflow/core/tpu/kernels/tpu_embedding_ops.cc", "~RecvTPUEmbeddingDeduplicationDataOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc mht_5(mht_5_v, 320, "", "./tensorflow/core/tpu/kernels/tpu_embedding_ops.cc", "Compile");

    VLOG(1) << "Compile RecvTPUDeduplicationDataOp";

    ResourceMgr* rm = GetTPUConfigResourceMgr();

    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);

    TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation_Params
        recv_deduplication_params;
    TpuSerializedProto xla_computation_serialized;
    auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
      StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
    });
    recv_deduplication_params.xla_computation = &xla_computation_serialized;
    StatusHelper status;
    recv_deduplication_params.status = status.c_status;
    recv_deduplication_params.tpu_mesh_state = mesh_state->data();

    tpu::OpsApiFn()
        ->TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputationFn(
            &recv_deduplication_params);
    OP_REQUIRES_OK(ctx, status.status());

    auto xla_computation =
        stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
            xla_computation_serialized);

    const xla::XlaOp deduplication_data =
        xla::Call(ctx->builder(), xla_computation, {});

    // Ensure that the number of outputs is equal to 1 (for deduplication data).
    OP_REQUIRES(ctx, ctx->num_outputs() == 1,
                xla::InvalidArgument(
                    "Kernel has %d outputs but configuration expects 1 output.",
                    ctx->num_outputs()));

    ctx->SetOutput(0, deduplication_data);
    VLOG(1) << "Compile RecvTPUDeduplicationDataOp done";
  }

 private:
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvTPUEmbeddingDeduplicationDataOp);
};

REGISTER_XLA_OP(Name("_RecvTPUEmbeddingDeduplicationData").AllowVariantTypes(),
                RecvTPUEmbeddingDeduplicationDataOp);

// This TensorFlow op sends a batch of gradient and learning rate updates to the
// TpuEmbeddingEngine.
class SendTPUEmbeddingGradientsOp : public XlaOpKernel {
 public:
  explicit SendTPUEmbeddingGradientsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc mht_6(mht_6_v, 382, "", "./tensorflow/core/tpu/kernels/tpu_embedding_ops.cc", "SendTPUEmbeddingGradientsOp");

    string config_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string));

    OP_REQUIRES(
        ctx, tpu_embedding_config_.ParseFromString(config_string),
        xla::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                             "proto from config attr"));
  }

  ~SendTPUEmbeddingGradientsOp() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc mht_7(mht_7_v, 395, "", "./tensorflow/core/tpu/kernels/tpu_embedding_ops.cc", "~SendTPUEmbeddingGradientsOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_opsDTcc mht_8(mht_8_v, 400, "", "./tensorflow/core/tpu/kernels/tpu_embedding_ops.cc", "Compile");

    VLOG(1) << "Compile SendTPUEmbeddingGradientsOp";

    ResourceMgr* rm = GetTPUConfigResourceMgr();

    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);

    std::vector<xla::XlaOp> gradients;
    std::vector<TensorShape> tf_gradient_shapes;
    OP_REQUIRES_OK(
        ctx, ctx->InputList("gradients", &gradients, &tf_gradient_shapes));
    std::vector<xla::Shape> gradient_shapes;
    auto builder = ctx->builder();
    gradient_shapes.reserve(gradients.size());
    for (xla::XlaOp op : gradients) {
      gradient_shapes.push_back(builder->GetShape(op).ValueOrDie());
    }

    std::vector<xla::XlaOp> learning_rates;
    std::vector<TensorShape> tf_learning_rate_shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("learning_rates", &learning_rates,
                                       &tf_learning_rate_shapes));
    std::vector<xla::Shape> learning_rate_shapes;
    learning_rate_shapes.reserve(learning_rates.size());
    for (xla::XlaOp op : learning_rates) {
      learning_rate_shapes.push_back(builder->GetShape(op).ValueOrDie());
    }

    xla::XlaOp deduplication_data = ctx->Input("deduplication_data");

    TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation_Params
        send_gradients_params;
    TpuSerializedProto xla_computation_serialized;
    auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
      StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
    });
    send_gradients_params.xla_computation = &xla_computation_serialized;
    StatusHelper status;
    send_gradients_params.status = status.c_status;
    send_gradients_params.tpu_mesh_state = mesh_state->data();
    OP_REQUIRES_VALUE(auto deduplication_shape, ctx,
                      builder->GetShape(deduplication_data));
    XLA_Shape gradient_tuple_c_shape;
    ApiConverter::ToC(xla::ShapeUtil::MakeTupleShape(gradient_shapes),
                      &gradient_tuple_c_shape);
    XLA_Shape learning_rate_tuple_c_shape;
    ApiConverter::ToC(xla::ShapeUtil::MakeTupleShape(learning_rate_shapes),
                      &learning_rate_tuple_c_shape);
    XLA_Shape deduplication_c_shape;
    ApiConverter::ToC(deduplication_shape, &deduplication_c_shape);

    auto c_shape_cleanup = absl::MakeCleanup([&gradient_tuple_c_shape,
                                              &learning_rate_tuple_c_shape,
                                              &deduplication_c_shape] {
      ApiConverter::Destroy(&gradient_tuple_c_shape);
      ApiConverter::Destroy(&learning_rate_tuple_c_shape);
      ApiConverter::Destroy(&deduplication_c_shape);
    });
    send_gradients_params.num_inputs = ctx->num_inputs();

    tpu::OpsApiFn()->TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputationFn(
        &send_gradients_params);
    OP_REQUIRES_OK(ctx, status.status());

    auto xla_computation =
        stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
            xla_computation_serialized);

    xla::Call(builder, xla_computation,
              {xla::Tuple(builder, gradients),
               xla::Tuple(builder, learning_rates), deduplication_data});

    VLOG(1) << "Compile SendTPUEmbeddingGradientsOp done";
  }

 private:
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;

  TF_DISALLOW_COPY_AND_ASSIGN(SendTPUEmbeddingGradientsOp);
};

REGISTER_XLA_OP(Name("_SendTPUEmbeddingGradients").AllowVariantTypes(),
                SendTPUEmbeddingGradientsOp);

}  // anonymous namespace
}  // namespace tensorflow
