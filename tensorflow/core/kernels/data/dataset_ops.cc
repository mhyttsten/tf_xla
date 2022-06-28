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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSdataset_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSdataset_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSdataset_opsDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/dataset_ops.h"

// On mobile we do not provide this functionality because not all of its
// dependencies are available there.
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_stateful_op_allowlist.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const DatasetToGraphOp::kAllowStateful;
/* static */ constexpr const char* const
    DatasetToGraphOp::kStripDeviceAssignment;
/* static */ constexpr const char* const DatasetToGraphOp::kExternalStatePolicy;
/* static */ constexpr const char* const DatasetToGraphOp::kDatasetToGraph;
/* static */ constexpr const char* const DatasetFromGraphOp::kGraphDef;
/* static */ constexpr const char* const DatasetFromGraphOp::kHandle;

namespace {
constexpr char kPyFunc[] = "PyFunc";
}  // namespace

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
DatasetToGraphOp::DatasetToGraphOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), op_version_(ctx->def().op() == kDatasetToGraph ? 1 : 2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSdataset_opsDTcc mht_0(mht_0_v, 222, "", "./tensorflow/core/kernels/data/dataset_ops.cc", "DatasetToGraphOp::DatasetToGraphOp");

  if (op_version_ == 2) {
    if (ctx->HasAttr(kExternalStatePolicy)) {
      int64_t state_change_option;
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr(kExternalStatePolicy, &state_change_option));
      external_state_policy_ =
          SerializationContext::ExternalStatePolicy(state_change_option);
    }
  } else {
    if (ctx->HasAttr(kAllowStateful)) {
      bool allow_stateful;
      OP_REQUIRES_OK(ctx, ctx->GetAttr(kAllowStateful, &allow_stateful));
      if (allow_stateful) {
        external_state_policy_ =
            SerializationContext::ExternalStatePolicy::kWarn;
      } else {
        external_state_policy_ =
            SerializationContext::ExternalStatePolicy::kFail;
      }
    }
  }

  if (ctx->HasAttr(kStripDeviceAssignment)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kStripDeviceAssignment, &strip_device_assignment_));
  }
}

void DatasetToGraphOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSdataset_opsDTcc mht_1(mht_1_v, 254, "", "./tensorflow/core/kernels/data/dataset_ops.cc", "DatasetToGraphOp::Compute");

  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  if (dataset->options().optional_external_state_policy_case() ==
      Options::kExternalStatePolicy) {
    switch (dataset->options().external_state_policy()) {
      case ExternalStatePolicy::POLICY_WARN:
        external_state_policy_ =
            SerializationContext::ExternalStatePolicy::kWarn;
        break;
      case ExternalStatePolicy::POLICY_IGNORE:
        external_state_policy_ =
            SerializationContext::ExternalStatePolicy::kIgnore;
        break;
      case ExternalStatePolicy::POLICY_FAIL:
        external_state_policy_ =
            SerializationContext::ExternalStatePolicy::kFail;
        break;
      default: {
        LOG(ERROR) << "Dataset " << dataset->type_string()
                   << " has an unknown external_state_policy enum value: "
                   << dataset->options().external_state_policy();
      }
    }
  }
  SerializationContext::Params params(ctx);
  params.external_state_policy = external_state_policy_;

  GraphDef graph_def;
  Status s = AsGraphDef(dataset, SerializationContext(params), &graph_def);
  if (!s.ok()) {
    ctx->CtxFailure(errors::FailedPrecondition(
        "Failed to serialize the input pipeline graph: ", s.error_message()));
    return;
  }
  if (strip_device_assignment_) {
    auto library = graph_def.mutable_library();
    for (auto& function : (*library->mutable_function())) {
      for (auto& node : (*function.mutable_node_def())) {
        // We do not strip the device assignment from `PyFunc` ops because they
        // need to be pinned to a host that is known to have Python interpreter.
        if (!node.device().empty() && node.op() != kPyFunc) {
          *node.mutable_device() = DeviceNameUtils::LocalName(node.device());
        }
      }
    }
  }

  Tensor* result;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &result));
  result->scalar<tstring>()() = graph_def.SerializeAsString();
}

void DatasetCardinalityOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSdataset_opsDTcc mht_2(mht_2_v, 310, "", "./tensorflow/core/kernels/data/dataset_ops.cc", "DatasetCardinalityOp::Compute");

  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  Tensor* result;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &result));
  result->scalar<int64_t>()() = dataset->Cardinality();
}

void DatasetFromGraphOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSdataset_opsDTcc mht_3(mht_3_v, 321, "", "./tensorflow/core/kernels/data/dataset_ops.cc", "DatasetFromGraphOp::Compute");

  tstring graph_def_string;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kGraphDef, &graph_def_string));
  GraphDef graph_def;
  OP_REQUIRES(ctx, graph_def.ParseFromString(graph_def_string),
              errors::InvalidArgument("Could not parse GraphDef"));
  string output_node;
  for (const auto& node : graph_def.node()) {
    if (node.op() == FunctionLibraryDefinition::kRetOp) {
      output_node = node.input(0);
    }
  }
  Graph graph(OpRegistry::Global());
  OP_REQUIRES_OK(ctx, ImportGraphDef({}, graph_def, &graph, nullptr));

  FunctionLibraryRuntime* flr;
  std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
  OP_REQUIRES_OK(ctx,
                 ctx->function_library()->Clone(&flib_def, &pflr, &flr, true));

  // Some function names may be duplicated (for example, if the serialized
  // graph has an optimized function that retains its original name). We
  // override functions in flib_def in the event of conflict. It is
  // safe to assume that any node in the serialized graph is referring to the
  // serialized function when there is a conflict.
  OP_REQUIRES_OK(ctx,
                 AddToFunctionLibrary(flib_def.get(), graph_def.library()));

  std::vector<Tensor> outputs;
  GraphRunner graph_runner(flr->device());
  OP_REQUIRES_OK(ctx,
                 graph_runner.Run(&graph, flr, {}, {output_node}, &outputs));
  OP_REQUIRES_OK(ctx, ctx->set_output(kHandle, outputs[0]));
}

REGISTER_KERNEL_BUILDER(Name("DatasetToGraph").Device(DEVICE_CPU),
                        DatasetToGraphOp);
REGISTER_KERNEL_BUILDER(Name("DatasetToGraphV2").Device(DEVICE_CPU),
                        DatasetToGraphOp);

REGISTER_KERNEL_BUILDER(Name("DatasetCardinality").Device(DEVICE_CPU),
                        DatasetCardinalityOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalDatasetCardinality").Device(DEVICE_CPU),
    DatasetCardinalityOp);

REGISTER_KERNEL_BUILDER(Name("DatasetFromGraph").Device(DEVICE_CPU),
                        DatasetFromGraphOp);

}  // namespace data
}  // namespace tensorflow
#endif  // !IS_MOBILE_PLATFORM
