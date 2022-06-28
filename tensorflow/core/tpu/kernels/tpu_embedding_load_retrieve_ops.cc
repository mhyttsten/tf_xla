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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc() {
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

#include "tensorflow/core/tpu/kernels/tpu_embedding_load_retrieve_ops.h"

#include <stddef.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/tpu/optimization_parameters.pb.h"
#include "tensorflow/core/tpu/ops/tpu_embedding_shape_util.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.h"
#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"


using tensorflow::tpu::TPUEmbeddingConfiguration;
using tensorflow::tpu::TpuEmbeddingShapeUtil;


namespace tensorflow {

// Computes (and VLOGs) the expected shapes of the embedding table shards.
Status ComputeExpectedTableShardShapes(const TPUEmbeddingConfiguration& config,
                                       int shard_id, int num_shards,
                                       const string& op_name,
                                       std::vector<TensorShape>* table_shapes) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/tpu/kernels/tpu_embedding_load_retrieve_ops.cc", "ComputeExpectedTableShardShapes");

  std::vector<TensorShapeProto> shape_protos;
  const int num_tables = config.table_descriptor_size();
  TF_RETURN_IF_ERROR(TpuEmbeddingShapeUtil::ComputeTableShapes(
      config, shard_id, num_shards, &shape_protos));
  if (num_tables != shape_protos.size()) {
    return errors::InvalidArgument(
        op_name, ": The size of the shape_protos vector ", shape_protos.size(),
        " must be the same as the number of tables ", num_tables);
  }
  for (int table_id = 0; table_id < num_tables; ++table_id) {
    const TensorShape& shape = TensorShape(shape_protos[table_id]);
    table_shapes->push_back(shape);

    const auto& table_descriptor = config.table_descriptor(table_id);
    VLOG(1) << "Table " << table_id << " (name " << table_descriptor.name()
            << ") has shape: " << shape.DebugString()
            << " on shard: " << shard_id << " (of " << num_shards << ").";
  }

  return Status::OK();
}

// Logs min/max/avg for the specified state_variable array.
void LogRangeStatistics(int32 table_id, int32 state_variable_index,
                        absl::Span<const float> state_variable) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/tpu/kernels/tpu_embedding_load_retrieve_ops.cc", "LogRangeStatistics");

  if (VLOG_IS_ON(5)) {
    float min = std::numeric_limits<float>::infinity();
    float max = -std::numeric_limits<float>::infinity();
    double avg = 0.0;
    for (int elt = 0; elt < state_variable.size(); ++elt) {
      if (state_variable[elt] < min) min = state_variable[elt];
      if (state_variable[elt] > max) max = state_variable[elt];
      avg += state_variable[elt];
    }
    LOG(INFO) << "Table " << table_id << " state_variable "
              << state_variable_index << " min " << min << " max " << max
              << " avg " << avg / state_variable.size() << " total elts "
              << state_variable.size();
  }
}


LoadAllTPUEmbeddingParametersOp::LoadAllTPUEmbeddingParametersOp(
    OpKernelConstruction* ctx)
    : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc mht_2(mht_2_v, 272, "", "./tensorflow/core/tpu/kernels/tpu_embedding_load_retrieve_ops.cc", "LoadAllTPUEmbeddingParametersOp::LoadAllTPUEmbeddingParametersOp");

  string config_string;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string));

  OP_REQUIRES(
      ctx, config_.ParseFromString(config_string),
      errors::InvalidArgument("LoadAllTPUEmbeddingParametersOp: Failed to "
                              "parse TPUEmbeddingConfiguration "
                              "proto from config attr"));
  // Auto populate the feature descriptor
  // TODO (b/201806244): remove this logic after the change to the
  // initialization to the config proto.
  OP_REQUIRES_OK(ctx, PopulateMissingFieldsInTPUEmbeddingConfig(&config_));

  int num_shards;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_shards", &num_shards));
  int shard_id;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shard_id", &shard_id));

  OP_REQUIRES_OK(ctx, ComputeExpectedTableShardShapes(
                          config_, shard_id, num_shards,
                          "LoadAllTPUEmbeddingParametersOp", &table_shapes_));
}

void LoadAllTPUEmbeddingParametersOp::GetStateVariables(
    OpKernelContext* ctx,
    std::array<std::vector<absl::Span<const float>>,
               tpu::kMaxAuxiliaryParameterCount + 1> &state_variable_vector) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc mht_3(mht_3_v, 302, "", "./tensorflow/core/tpu/kernels/tpu_embedding_load_retrieve_ops.cc", "LoadAllTPUEmbeddingParametersOp::GetStateVariables");

    std::array<OpInputList, tpu::kMaxAuxiliaryParameterCount + 1>
      state_variable;
  OP_REQUIRES_OK(ctx, ctx->input_list("parameters", &state_variable[0]));
  for (int i = 1; i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
    OP_REQUIRES_OK(ctx, ctx->input_list(absl::StrCat("auxiliary", i),
                                        &state_variable[i]));
  }
  const int num_tables = state_variable[0].size();
  // This should be enforced by Tensorflow's type system.
  for (int i = 1; i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
    CHECK_EQ(num_tables, state_variable[i].size());
  }

  OP_REQUIRES(ctx, num_tables == table_shapes_.size(),
              errors::InvalidArgument(
                  "LoadAllTPUEmbeddingParametersOp has ", num_tables,
                  " inputs in lists but config specifies ",
                  table_shapes_.size(), " embedding tables."));

  CHECK_EQ(num_tables, config_.table_descriptor_size());
  for (int table_id = 0; table_id < num_tables; ++table_id) {
    const auto& table_descriptor = config_.table_descriptor(table_id);
    std::vector<tpu::StateVariableSpecification> state_variable_specs;
    Status status = tpu::GetOptimizationAlgorithmStateVariables(
        table_descriptor.optimization_parameters(), &state_variable_specs);
    OP_REQUIRES(ctx, status.ok(),
                errors::InvalidArgument(
                    "LoadAllTPUEmbeddingParametersOp: No optimization "
                    "algorithm specified for table ",
                    table_id, " (named ", table_descriptor.name(), ")"));
    OP_REQUIRES(
        ctx, state_variable[0][table_id].shape() == table_shapes_[table_id],
        errors::InvalidArgument(
            "LoadAllTPUEmbeddingParametersOp: Embeddings for table ",
            table_id, " (named ", table_descriptor.name(), ") has shape ",
            state_variable[0][table_id].shape().DebugString(),
            " but config specifies table shape ",
            table_shapes_[table_id].DebugString()));
    for (int i = 1; i < state_variable_specs.size(); ++i) {
      OP_REQUIRES(
          ctx, state_variable[i][table_id].shape() == table_shapes_[table_id],
          errors::InvalidArgument(
              "LoadAllTPUEmbeddingParametersOp: Auxiliary ", i - 1,
              " for table ", table_id, " has shape ",
              state_variable[i][table_id].shape().DebugString(),
              " but config specifies table shape ",
              table_shapes_[table_id].DebugString()));
    }
    const int64 num_elements = state_variable[0][table_id].NumElements();
    VLOG(1) << "Table " << table_id << " (name " << table_descriptor.name()
            << ") has shape: " << table_shapes_[table_id].DebugString()
            << ", number of elements: " << num_elements;
    for (int i = 0; i < state_variable_specs.size(); ++i) {
      OP_REQUIRES(
          ctx, state_variable[i][table_id].NumElements() == num_elements,
          errors::InvalidArgument(
              "LoadAllTPUEmbeddingParametersOp: Embeddings/auxiliary ", i,
              " for table ", table_id, " has element count ",
              state_variable[i][table_id].NumElements(),
              " but config requires count ", num_elements));
      const float* state_variable_i_ptr =
          state_variable[i][table_id].flat<float>().data();
      state_variable_vector[i].push_back(absl::MakeConstSpan(
          state_variable_i_ptr, static_cast<size_t>(num_elements)));
      LogRangeStatistics(
          table_id, i,
          absl::MakeConstSpan(state_variable_i_ptr, num_elements));
    }
    for (int i = state_variable_specs.size();
         i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
      OP_REQUIRES(ctx, state_variable[i][table_id].NumElements() == 0,
                  errors::InvalidArgument(
                      "LoadAllTPUEmbeddingParametersOp: Auxiliary ", i,
                      " for table ", table_id, " has element count ",
                      state_variable[i][table_id].NumElements(),
                      " but config requires empty tensor"));
      state_variable_vector[i].push_back(absl::Span<const float>());
    }
  }
}

void LoadAllTPUEmbeddingParametersOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc mht_4(mht_4_v, 387, "", "./tensorflow/core/tpu/kernels/tpu_embedding_load_retrieve_ops.cc", "LoadAllTPUEmbeddingParametersOp::Compute");

    VLOG(1) << "LoadAllTPUEmbeddingParameters::Compute";

  std::array<std::vector<absl::Span<const float>>,
             tpu::kMaxAuxiliaryParameterCount + 1> state_variable_vector;

  GetStateVariables(ctx, state_variable_vector);
  const int num_tables = state_variable_vector[0].size();

  std::unique_ptr<ApiConverter::TpuEmbeddingEngineParametersData> params =
    ApiConverter::Create(num_tables);
  std::array<std::vector<FloatListRef>,
             tpu::kMaxAuxiliaryParameterCount + 1> params_data;
  for (size_t i = 0; i < tpu::kMaxAuxiliaryParameterCount + 1; i++) {
    params_data[i] = std::vector<FloatListRef>(num_tables);
    for (size_t table_id = 0; table_id < num_tables; table_id++) {
      params->c_params.parameters[i][table_id] = &(params_data[i][table_id]);
      params->c_params.parameters[i][table_id]->size =
          state_variable_vector[i][table_id].size();
      params->c_params.parameters[i][table_id]->ptr = const_cast<float_t*>(
          state_variable_vector[i][table_id].data());
    }
  }
  StatusHelper status;
  tpu::OpsApiFn()->TpuEmbeddingEngine_WriteParametersFn(&(params->c_params),
                                                        status.c_status);
  OP_REQUIRES_OK(ctx, status.status());

  VLOG(1) << "LoadAllTPUEmbeddingParameters::Compute done";
}


RetrieveAllTPUEmbeddingParametersOp::RetrieveAllTPUEmbeddingParametersOp(
    OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc mht_5(mht_5_v, 424, "", "./tensorflow/core/tpu/kernels/tpu_embedding_load_retrieve_ops.cc", "RetrieveAllTPUEmbeddingParametersOp::RetrieveAllTPUEmbeddingParametersOp");

  string config_string;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string));

  OP_REQUIRES(
      ctx, config_.ParseFromString(config_string),
      errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                              "proto from config attr"));

  // Auto populate the feature descriptor
  // TODO (b/201806244): remove this logic after the change to the
  // initialization to the config proto.
  OP_REQUIRES_OK(ctx, PopulateMissingFieldsInTPUEmbeddingConfig(&config_));

  int num_shards;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_shards", &num_shards));
  int shard_id;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shard_id", &shard_id));

  OP_REQUIRES_OK(ctx,
                 ComputeExpectedTableShardShapes(
                     config_, shard_id, num_shards,
                     "RetrieveAllTPUEmbeddingParametersOp", &table_shapes_));
}

void RetrieveAllTPUEmbeddingParametersOp::GetStateVariables(
    OpKernelContext* ctx,
    std::array<std::vector<absl::Span<float>>,
               tpu::kMaxAuxiliaryParameterCount + 1> &state_variable_vector,
    std::vector<int> & num_state_variables) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc mht_6(mht_6_v, 456, "", "./tensorflow/core/tpu/kernels/tpu_embedding_load_retrieve_ops.cc", "RetrieveAllTPUEmbeddingParametersOp::GetStateVariables");

  std::array<OpOutputList, tpu::kMaxAuxiliaryParameterCount + 1>
      state_variable;
  OP_REQUIRES_OK(ctx, ctx->output_list("parameters", &state_variable[0]));
  for (int i = 1; i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
    OP_REQUIRES_OK(ctx, ctx->output_list(absl::StrCat("auxiliary", i),
                                         &state_variable[i]));
  }
  const int num_tables = state_variable[0].size();
  // This should be enforced by Tensorflow's type system.
  for (int i = 1; i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
    CHECK_EQ(num_tables, state_variable[i].size());
  }

  OP_REQUIRES(ctx, num_tables == table_shapes_.size(),
              errors::InvalidArgument(
                  "RetrieveAllTPUEmbeddingParametersOp has ", num_tables,
                  " outputs in lists but config specifies ",
                  table_shapes_.size(), " embedding tables."));

  for (auto& v : state_variable_vector) {
    v.resize(num_tables);
  }
  num_state_variables.resize(num_tables);

  // Get locations to write returned data
  for (int table_id = 0; table_id < num_tables; ++table_id) {
    const auto& table_descriptor = config_.table_descriptor(table_id);

    std::vector<tpu::StateVariableSpecification> state_variable_specs;
    Status status = tpu::GetOptimizationAlgorithmStateVariables(
        table_descriptor.optimization_parameters(), &state_variable_specs);
    OP_REQUIRES(
        ctx, status.ok(),
        errors::InvalidArgument("RetrieveAllTPUEmbeddingParametersOp: No "
                                "optimization algorithm specified for table ",
                                table_id));
    num_state_variables[table_id] = state_variable_specs.size();
    const int64 num_elements = table_shapes_[table_id].num_elements();
    for (int i = 0; i < state_variable_specs.size(); ++i) {
      Tensor* state_variable_tensor;
      OP_REQUIRES_OK(
          ctx, state_variable[i].allocate(table_id, table_shapes_[table_id],
                                          &state_variable_tensor));
      float* state_variable_ptr = state_variable_tensor->flat<float>().data();
      state_variable_vector[i][table_id] =
          absl::MakeSpan(state_variable_ptr, num_elements);
    }
    // Fill in auxiliary values after the number actually used for table_id
    // with empty 2-D tensors.
    for (int i = state_variable_specs.size();
         i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
      Tensor* auxiliary_tensor;
      TensorShape shape;
      std::array<int32, 2> dims = {{0, 0}};
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(dims, &shape));
      OP_REQUIRES_OK(ctx, state_variable[i].allocate(table_id, shape,
                                                     &auxiliary_tensor));
      state_variable_vector[i][table_id] = absl::Span<float>();
    }
  }
}

void RetrieveAllTPUEmbeddingParametersOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_load_retrieve_opsDTcc mht_7(mht_7_v, 522, "", "./tensorflow/core/tpu/kernels/tpu_embedding_load_retrieve_ops.cc", "RetrieveAllTPUEmbeddingParametersOp::Compute");

  VLOG(1) << "RetrieveAllTPUEmbeddingParameters::Compute";

  std::array<std::vector<absl::Span<float>>,
             tpu::kMaxAuxiliaryParameterCount + 1> state_variable_vector;
  std::vector<int> num_state_variables;

  GetStateVariables(ctx, state_variable_vector, num_state_variables);
  const int num_tables = state_variable_vector[0].size();


  std::unique_ptr<ApiConverter::TpuEmbeddingEngineParametersData> params =
      ApiConverter::Create(num_tables);
  std::array<std::vector<FloatListRef>,
             tpu::kMaxAuxiliaryParameterCount + 1> params_data;
  for (size_t i = 0; i < tpu::kMaxAuxiliaryParameterCount + 1; i++) {
    params_data[i] = std::vector<FloatListRef>(num_tables);
    for (size_t table_id = 0; table_id < num_tables; table_id++) {
      params->c_params.parameters[i][table_id] =
          &(params_data[i][table_id]);
      params->c_params.parameters[i][table_id]->size =
          state_variable_vector[i][table_id].size(),
      params->c_params.parameters[i][table_id]->ptr =
          state_variable_vector[i][table_id].data();
    }
  }
  StatusHelper status;
  tpu::OpsApiFn()->TpuEmbeddingEngine_ReadParametersFn(&(params->c_params),
                                                       status.c_status);
  OP_REQUIRES_OK(ctx, status.status());

  if (VLOG_IS_ON(5)) {
    for (int table_id = 0; table_id < num_tables; ++table_id) {
      for (int i = 0; i < num_state_variables[table_id]; ++i) {
        LogRangeStatistics(table_id, i, state_variable_vector[i][table_id]);
      }
    }
  }
}

#ifdef LIBTPU_ON_GCE

REGISTER_KERNEL_BUILDER(
    Name("LoadAllTPUEmbeddingParameters").Device(DEVICE_CPU),
    LoadAllTPUEmbeddingParametersOp);
REGISTER_KERNEL_BUILDER(
    Name("RetrieveAllTPUEmbeddingParameters").Device(DEVICE_CPU),
    RetrieveAllTPUEmbeddingParametersOp);

#endif  // LIBTPU_ON_GCE
}  // namespace tensorflow
