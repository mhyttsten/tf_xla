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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdata_service_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdata_service_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdata_service_opsDTcc() {
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

#include "tensorflow/core/kernels/data/experimental/data_service_ops.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

namespace {
const int64_t kRetryTimeoutMicros = 1000LL * 1000 * 60 * 60;  // 60 minutes.
}

RegisterDatasetOp::RegisterDatasetOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdata_service_opsDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/data/experimental/data_service_ops.cc", "RegisterDatasetOp::RegisterDatasetOp");

  int64_t external_state_policy_int;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr(kExternalStatePolicy, &external_state_policy_int));
  external_state_policy_ =
      SerializationContext::ExternalStatePolicy(external_state_policy_int);

  if (ctx->HasAttr(kElementSpec)) {
    tstring element_spec;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kElementSpec, &element_spec));
    element_spec_ = element_spec;
  }

  if (ctx->HasAttr(kMetadata)) {
    tstring serialized_metadata;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kMetadata, &serialized_metadata));
    serialized_metadata_ = serialized_metadata;
  }
}

void RegisterDatasetOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdata_service_opsDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/kernels/data/experimental/data_service_ops.cc", "RegisterDatasetOp::Compute");

  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));

  tstring address;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kAddress, &address));
  OP_REQUIRES(ctx, !address.empty(),
              errors::InvalidArgument(kAddress, " must be non-empty."));

  tstring protocol;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kProtocol, &protocol));
  OP_REQUIRES(ctx, !protocol.empty(),
              errors::InvalidArgument(kProtocol, " must be non-empty."));

  SerializationContext::Params params(ctx);
  params.external_state_policy = external_state_policy_;
  SerializationContext serialization_ctx(params);
  DatasetDef dataset_def;
  Status s = AsGraphDef(dataset, std::move(serialization_ctx),
                        dataset_def.mutable_graph());
  if (!s.ok()) {
    OP_REQUIRES_OK(
        ctx,
        errors::FailedPrecondition(
            "Serialization error while trying to register a dataset with "
            "tf.data service. "
            "The dataset may depend on a resource located on a different "
            "device. "
            "To address this, call `register_dataset` from the device with the "
            "resource, then use `from_dataset_id` to create per-device "
            "datasets. "
            "Original error: ",
            s));
  }

  DataServiceMetadata metadata;
  if (!element_spec_.empty()) {
    metadata.set_element_spec(element_spec_);
  }
  if (!serialized_metadata_.empty()) {
    OP_REQUIRES(ctx, metadata.ParseFromString(serialized_metadata_),
                errors::InvalidArgument(
                    "Failed to parse DataServiceMetadata from string: ",
                    std::string(serialized_metadata_)));
  }
  metadata.set_cardinality(dataset->Cardinality());

  DataServiceDispatcherClient client(address, protocol);
  int64_t dataset_id;
  int64_t deadline_micros = EnvTime::NowMicros() + kRetryTimeoutMicros;
  OP_REQUIRES_OK(
      ctx, grpc_util::Retry(
               [&]() {
                 return client.RegisterDataset(dataset_def, metadata,
                                               dataset_id);
               },
               /*description=*/
               strings::StrCat("register dataset with dispatcher at ", address),
               deadline_micros));

  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));
  auto output_dataset_id = output->tensor<int64_t, 0>();
  output_dataset_id() = dataset_id;
}

REGISTER_KERNEL_BUILDER(Name("RegisterDataset").Device(DEVICE_CPU),
                        RegisterDatasetOp);

}  // namespace data
}  // namespace tensorflow
