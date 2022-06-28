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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifdef LIBTPU_ON_GCE

#include <string>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"

using ::tensorflow::tpu::TPUEmbeddingConfiguration;

namespace tensorflow {

namespace {
namespace se_tpu = ::stream_executor::tpu;
}

// The ExecuteTPUEmbeddingPartitioner Op is used to run the TPUEmbedding
// partitioner as well as calculate the HBM size (in bytes) required for
// TPUEmbedding operation. It takes as input a TPUEmbeddingConfiguration proto
// which describes all the embedding tables and metadata. It should be run on
// the TPU_SYSTEM device on only one task (by convention, task 0).
// Note that the _ConfigureDistributedTPU Op must have run before this Op so
// that the TpuTopology is added to the TpuMeshCommonState. Subsequent
// TPUEmbedding host configuration Ops (one per task) will use the output of
// this Op.

class ExecuteTPUEmbeddingPartitionerOp : public OpKernel {
 public:
  explicit ExecuteTPUEmbeddingPartitionerOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "ExecuteTPUEmbeddingPartitionerOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES(
        ctx, TPUEmbeddingConfiguration().ParseFromString(config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "Compute");

    VLOG(1) << "ExecuteTPUEmbeddingPartitioner::Compute";
    TpuEmbeddingEngine_ExecutePartitioner_Params params;
    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();
    ResourceMgr* rm = GetTPUConfigResourceMgr();
    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);
    params.tpu_mesh_state = mesh_state->data();

    char* common_config_output = nullptr;
    auto cleanup = absl::MakeCleanup([&common_config_output]() {
      tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(
          common_config_output);
    });
    size_t common_config_output_size;
    params.common_config_size = &common_config_output_size;
    params.common_config = &common_config_output;

    StatusHelper status;
    params.status = status.c_status;

    tpu::OpsApiFn()->TpuEmbeddingEngine_ExecutePartitionerFn(&params);
    if (!status.ok()) {
      VLOG(0) << "ExecuteTPUEmbeddingPartitioner::Compute failed"
              << status.status().ToString();
      return;
    }
    std::string common_config_string =
        std::string(common_config_output, common_config_output_size);
    Tensor* output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("common_config", TensorShape({}), &output));
    output->flat<tstring>()(0) = common_config_string;
    VLOG(1) << "ExecuteTPUEmbeddingPartitioner::Compute done";
  }

 private:
  // The embedding layer configuration for the TPUEmbedding host software.
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExecuteTPUEmbeddingPartitionerOp);
};

// Initialize the HBM memory addresses and segments on each host.
// The ConfigureTPUEmbeddingMemoryOp allocates HBM memory used by TPUEmbedding.
// It takes as input a TPUEmbeddingConfiguration proto, which describes all
// the embedding tables, and the output of the
// _ExecuteTPUEmbeddingPartitioner Op. It should be run on one TPU device
// on each task, by convention TPU:0.
class ConfigureTPUEmbeddingMemoryOp : public OpKernel {
 public:
  explicit ConfigureTPUEmbeddingMemoryOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_2(mht_2_v, 290, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "ConfigureTPUEmbeddingMemoryOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES(
        ctx, TPUEmbeddingConfiguration().ParseFromString(config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_3(mht_3_v, 301, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "Compute");

    VLOG(1) << "ConfigureTPUEmbeddingMemoryOp::Compute";
    std::string common_config_string = ctx->input(0).flat<tstring>()(0);
    std::string host_config;

    TpuEmbeddingEngine_ConfigureMemory_Params params;
    params.common_config_string = common_config_string.c_str();
    params.common_config_string_size = common_config_string.size();

    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();

    StatusHelper status;
    params.status = status.c_status;

    char* task_host_config_output = nullptr;
    auto task_host_config_cleanup =
        absl::MakeCleanup([&task_host_config_output]() {
          tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(
              task_host_config_output);
        });
    size_t task_host_config_output_size;
    params.task_host_config_size = &task_host_config_output_size;
    params.task_host_config = &task_host_config_output;
    params.num_inputs = ctx->num_inputs();

    tpu::OpsApiFn()->TpuEmbeddingEngine_ConfigureMemoryFn(&params);
    OP_REQUIRES_OK(ctx, status.status());
    std::string task_host_config =
        std::string(task_host_config_output, task_host_config_output_size);

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("task_host_config",
                                             TensorShape({}), &output));
    output->flat<tstring>()(0) = task_host_config;

    VLOG(1) << "ConfigureTPUEmbeddingMemoryOp::Compute done";
  }

  ~ConfigureTPUEmbeddingMemoryOp() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_4(mht_4_v, 343, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "~ConfigureTPUEmbeddingMemoryOp");
}

 private:
  // The embedding layer configuration for the TPUEmbedding host software.
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConfigureTPUEmbeddingMemoryOp);
};

// The ConfigureTPUEmbeddingHost Op is used to set up the TPUEmbedding
// software on a given host. It takes as input a TPUEmbeddingConfiguration
// proto which describes all the embedding tables as well as the output of
// the _ExecuteTPUEmbeddingPartitioner Op. It should be run on one TPU device
// on each task, by convention TPU:0.
class ConfigureTPUEmbeddingHostOp : public OpKernel {
 public:
  explicit ConfigureTPUEmbeddingHostOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_5(mht_5_v, 363, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "ConfigureTPUEmbeddingHostOp");

    OP_REQUIRES(ctx, ctx->num_inputs() > 0,
                errors::InvalidArgument("ConfigureTPUEmbeddingHostOp must "
                                        "have at least one input"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES(
        ctx, TPUEmbeddingConfiguration().ParseFromString(config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_6(mht_6_v, 378, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "Compute");

    VLOG(1) << "ConfigureTPUEmbeddingHostOp::Compute";
    std::string common_config_string = ctx->input(0).flat<tstring>()(0);

    // Retrieve per-task config received from each
    // ConfigureTPUEmbeddingMemoryOp node.
    OpInputList task_host_config;
    OP_REQUIRES_OK(ctx, ctx->input_list("task_host_config", &task_host_config));

    std::vector<std::string> task_host_config_string(task_host_config.size());
    std::vector<se_tpu::SerializedProto> task_hosts_config(
        task_host_config.size());
    for (int i = 0; i < task_host_config.size(); ++i) {
      task_host_config_string[i] = task_host_config[i].flat<tstring>()(0);
      task_hosts_config[i].bytes = task_host_config_string[i].c_str();
      task_hosts_config[i].size = task_host_config_string[i].size();
    }

    TpuEmbeddingEngine_ConfigureHost_Params params;
    params.common_config_string = common_config_string.c_str();
    params.common_config_string_size = common_config_string.size();

    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();

    char* host_config_output = nullptr;
    auto cleanup = absl::MakeCleanup([&host_config_output]() {
      tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(host_config_output);
    });

    size_t host_config_output_size;
    params.host_config_size = &host_config_output_size;
    params.host_config = &host_config_output;
    params.num_inputs = ctx->num_inputs();

    params.task_host_config = task_hosts_config.data();
    params.task_host_config_size = task_hosts_config.size();

    StatusHelper status;
    params.status = status.c_status;

    tpu::OpsApiFn()->TpuEmbeddingEngine_ConfigureHostFn(&params);

    OP_REQUIRES_OK(ctx, status.status());
    std::string host_config =
        std::string(host_config_output, host_config_output_size);

    Tensor* output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("host_config", TensorShape({}), &output));
    output->flat<tstring>()(0) = host_config;
    VLOG(1) << "ConfigureTPUEmbeddingHostOp::Compute done";
  }

  ~ConfigureTPUEmbeddingHostOp() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_7(mht_7_v, 435, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "~ConfigureTPUEmbeddingHostOp");
}

 private:
  // The embedding layer configuration for the TPUEmbedding host software.
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConfigureTPUEmbeddingHostOp);
};

// The ConnectInterTPUEmbeddingCommunication op is used to set up gRPC
// connections between instances of the TPUEmbedding host software on different
// hosts; it must be run after ConfigureTPUEmbeddingHostOp has been called on
// each host. It takes as input a string from each host which describes metadata
// about the TPUEmbedding configuration on that host. It should be run on one
// TPU device in the host, by convention TPU:0.
class ConnectInterTPUEmbeddingCommunicationOp : public OpKernel {
 public:
  explicit ConnectInterTPUEmbeddingCommunicationOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_8(mht_8_v, 456, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "ConnectInterTPUEmbeddingCommunicationOp");

    OP_REQUIRES(
        ctx, ctx->num_inputs() > 0,
        errors::InvalidArgument("ConnectInterTPUEmbeddingCommunication must "
                                "have at least one input"));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_9(mht_9_v, 466, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "Compute");

    VLOG(1) << "ConnectInterTPUEmbeddingCommunication::Compute";

    std::vector<std::string> hosts_config_string(ctx->num_inputs());
    std::vector<se_tpu::SerializedProto> hosts_config(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      hosts_config_string[i] = ctx->input(i).flat<tstring>()(0);
      hosts_config[i].bytes = hosts_config_string[i].c_str();
      hosts_config[i].size = hosts_config_string[i].size();
    }

    TpuEmbeddingEngine_ConfigureCommunication_Params params;
    StatusHelper status;
    params.status = status.c_status;

    params.host_config = hosts_config.data();
    params.host_config_size = hosts_config.size();

    tpu::OpsApiFn()->TpuEmbeddingEngine_ConfigureCommunicationFn(&params);
    OP_REQUIRES_OK(ctx, status.status());

    VLOG(1) << "ConnectInterTPUEmbeddingCommunication::Compute done";
  }

  ~ConnectInterTPUEmbeddingCommunicationOp() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_10(mht_10_v, 493, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "~ConnectInterTPUEmbeddingCommunicationOp");
}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConnectInterTPUEmbeddingCommunicationOp);
};

// The FinalizeTPUEmbeddingSystemConfiguration op is used to configure the
// system once ConfigureTPUEmbeddingHostOp has been called on each host. It
// takes as input a string from each host which describes metadata about the
// TPUEmbedding configuration on that host. It must be run on the TPU system
// device to which the TPUEmbedding hosts are attached.
class FinalizeTPUEmbeddingSystemConfigurationOp : public OpKernel {
 public:
  explicit FinalizeTPUEmbeddingSystemConfigurationOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_11(mht_11_v, 510, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "FinalizeTPUEmbeddingSystemConfigurationOp");

    OP_REQUIRES(
        ctx, ctx->num_inputs() > 0,
        errors::InvalidArgument("FinalizeTPUEmbeddingSystemConfiguration must "
                                "have at least one input"));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_12(mht_12_v, 520, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "Compute");

    VLOG(1) << "FinalizeTPUEmbeddingSystemConfiguration::Compute";

    std::vector<std::string> hosts_config_string(ctx->num_inputs());
    std::vector<se_tpu::SerializedProto> hosts_config(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      hosts_config_string[i] = ctx->input(i).flat<tstring>()(0);
      hosts_config[i].bytes = hosts_config_string[i].c_str();
      hosts_config[i].size = hosts_config_string[i].size();
    }

    TpuEmbeddingEngine_FinalizeConfiguration_Params params;
    StatusHelper status;
    params.status = status.c_status;

    params.host_config = hosts_config.data();
    params.host_config_size = hosts_config.size();

    ResourceMgr* rm = GetTPUConfigResourceMgr();
    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);
    params.tpu_mesh_state = mesh_state->data();

    tpu::OpsApiFn()->TpuEmbeddingEngine_FinalizeConfigurationFn(&params);
    OP_REQUIRES_OK(ctx, status.status());
  }

  ~FinalizeTPUEmbeddingSystemConfigurationOp() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_13(mht_13_v, 554, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "~FinalizeTPUEmbeddingSystemConfigurationOp");
}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FinalizeTPUEmbeddingSystemConfigurationOp);
};

// The IsTPUEmbeddingInitializedOp is used to check whether the TPU
// TPUEmbedding Embedding has been initialized. It takes no argument and outputs
// a boolean value which indicates the TPUEmbedding Embedding is initialized or
// not. It runs on the CPU device.
class IsTPUEmbeddingInitializedOp : public OpKernel {
 public:
  explicit IsTPUEmbeddingInitializedOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_14(mht_14_v, 570, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "IsTPUEmbeddingInitializedOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_15(mht_15_v, 577, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "Compute");

    VLOG(1) << "IsTPUEmbeddingInitializedOp::Compute";

    TpuEmbeddingEngine_IsInitialized_Params params;
    StatusHelper status;
    params.status = status.c_status;

    params.config_string = config_string_.c_str();
    params.config_string_size = config_string_.size();
    bool is_initialized = false;
    params.is_tpu_embedding_initialized = &is_initialized;

    tpu::OpsApiFn()->TpuEmbeddingEngine_IsInitializedFn(&params);

    OP_REQUIRES_OK(ctx, status.status());

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    output->flat<bool>()(0) = is_initialized;

    VLOG(1) << "IsTPUEmbeddingInitializedOp::Compute done";
  }
  ~IsTPUEmbeddingInitializedOp() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_embedding_configuration_opsDTcc mht_16(mht_16_v, 602, "", "./tensorflow/core/tpu/kernels/tpu_embedding_configuration_ops.cc", "~IsTPUEmbeddingInitializedOp");
}

 private:
  std::string config_string_;
  TF_DISALLOW_COPY_AND_ASSIGN(IsTPUEmbeddingInitializedOp);
};

// These ops execute on the TPU device, so that they can access
// the JF node interfaces stored in the JF device's resource manager.
REGISTER_KERNEL_BUILDER(Name("_ExecuteTPUEmbeddingPartitioner")
                            .Device(DEVICE_CPU)
                            .HostMemory("common_config"),
                        ExecuteTPUEmbeddingPartitionerOp);
REGISTER_KERNEL_BUILDER(Name("_ConfigureTPUEmbeddingMemory")
                            .Device(DEVICE_CPU)
                            .HostMemory("common_config")
                            .HostMemory("task_host_config"),
                        ConfigureTPUEmbeddingMemoryOp);
REGISTER_KERNEL_BUILDER(Name("_ConfigureTPUEmbeddingHost")
                            .Device(DEVICE_CPU)
                            .HostMemory("common_config")
                            .HostMemory("task_host_config")
                            .HostMemory("host_config"),
                        ConfigureTPUEmbeddingHostOp);
REGISTER_KERNEL_BUILDER(Name("_ConnectInterTPUEmbeddingCommunication")
                            .Device(DEVICE_CPU)
                            .HostMemory("host_config"),
                        ConnectInterTPUEmbeddingCommunicationOp);
REGISTER_KERNEL_BUILDER(Name("_FinalizeTPUEmbeddingSystemConfiguration")
                            .Device(DEVICE_CPU)
                            .HostMemory("host_config"),
                        FinalizeTPUEmbeddingSystemConfigurationOp);
REGISTER_KERNEL_BUILDER(Name("IsTPUEmbeddingInitialized").Device(DEVICE_CPU),
                        IsTPUEmbeddingInitializedOp);

}  // namespace tensorflow

#endif  // LIBTPU_ON_GCE
