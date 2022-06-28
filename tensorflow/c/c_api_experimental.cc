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
class MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc {
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
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc() {
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

#include "tensorflow/c/c_api_experimental.h"

#include "absl/strings/substitute.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_plugin_init.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/net.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

using tensorflow::FunctionDef;
using tensorflow::Node;
using tensorflow::NodeBuilder;
using tensorflow::Status;
using tensorflow::errors::InvalidArgument;

namespace {
typedef std::unique_ptr<TF_Function, decltype(&TF_DeleteFunction)>
    UniqueFuncPtr;
}

// struct TF_Operation { tensorflow::Node node; };
static TF_Operation* ToTF_Operation(Node* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_0(mht_0_v, 231, "", "./tensorflow/c/c_api_experimental.cc", "ToTF_Operation");

  return static_cast<TF_Operation*>(static_cast<void*>(node));
}

void TF_EnableXLACompilation(TF_SessionOptions* options, unsigned char enable) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("enable: '" + std::string(1, enable) + "'");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_1(mht_1_v, 239, "", "./tensorflow/c/c_api_experimental.cc", "TF_EnableXLACompilation");

  tensorflow::ConfigProto& config = options->options.config;
  auto* optimizer_options =
      config.mutable_graph_options()->mutable_optimizer_options();
  if (enable) {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::ON_1);

    // These XLA flags are needed to trigger XLA properly from C (more generally
    // non-Python) clients. If this API is called again with `enable` set to
    // false, it is safe to keep these flag values as is.
    tensorflow::MarkForCompilationPassFlags* flags =
        tensorflow::GetMarkForCompilationPassFlags();
    flags->tf_xla_cpu_global_jit = true;
    flags->tf_xla_min_cluster_size = 1;
  } else {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::OFF);
  }
}

unsigned char TF_SetXlaEnableLazyCompilation(unsigned char enable) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("enable: '" + std::string(1, enable) + "'");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_2(mht_2_v, 262, "", "./tensorflow/c/c_api_experimental.cc", "TF_SetXlaEnableLazyCompilation");

  tensorflow::BuildXlaOpsPassFlags* flags =
      tensorflow::GetBuildXlaOpsPassFlags();
  bool original = flags->tf_xla_enable_lazy_compilation;
  flags->tf_xla_enable_lazy_compilation = enable;
  return original;
}

unsigned char TF_SetTfXlaCpuGlobalJit(unsigned char enable) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("enable: '" + std::string(1, enable) + "'");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_3(mht_3_v, 274, "", "./tensorflow/c/c_api_experimental.cc", "TF_SetTfXlaCpuGlobalJit");

  tensorflow::MarkForCompilationPassFlags* flags =
      tensorflow::GetMarkForCompilationPassFlags();
  bool original = flags->tf_xla_cpu_global_jit;
  flags->tf_xla_cpu_global_jit = static_cast<bool>(enable);
  return static_cast<unsigned char>(original);
}

void TF_SetXlaAutoJitMode(const char* mode) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("mode: \"" + (mode == nullptr ? std::string("nullptr") : std::string((char*)mode)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_4(mht_4_v, 286, "", "./tensorflow/c/c_api_experimental.cc", "TF_SetXlaAutoJitMode");

  tensorflow::SetXlaAutoJitFlagFromFlagString(mode);
}

unsigned char TF_GetXlaAutoJitEnabled() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_5(mht_5_v, 293, "", "./tensorflow/c/c_api_experimental.cc", "TF_GetXlaAutoJitEnabled");

  tensorflow::XlaAutoJitFlag flag =
      tensorflow::GetMarkForCompilationPassFlags()->xla_auto_jit_flag;
  return static_cast<unsigned char>(flag.optimization_level_single_gpu > 0 ||
                                    flag.optimization_level_general > 0);
}

unsigned char TF_GetXlaConstantFoldingDisabled() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_6(mht_6_v, 303, "", "./tensorflow/c/c_api_experimental.cc", "TF_GetXlaConstantFoldingDisabled");

  return static_cast<unsigned char>(
      tensorflow::GetBuildXlaOpsPassFlags()->tf_xla_disable_constant_folding);
}

void TF_SetXlaConstantFoldingDisabled(unsigned char should_enable) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("should_enable: '" + std::string(1, should_enable) + "'");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_7(mht_7_v, 312, "", "./tensorflow/c/c_api_experimental.cc", "TF_SetXlaConstantFoldingDisabled");

  tensorflow::GetBuildXlaOpsPassFlags()->tf_xla_disable_constant_folding =
      static_cast<bool>(should_enable);
}

void TF_SetXlaMinClusterSize(int size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_8(mht_8_v, 320, "", "./tensorflow/c/c_api_experimental.cc", "TF_SetXlaMinClusterSize");

  tensorflow::MarkForCompilationPassFlags* flags =
      tensorflow::GetMarkForCompilationPassFlags();
  flags->tf_xla_min_cluster_size = size;
}

TF_Buffer* TF_CreateConfig(unsigned char enable_xla_compilation,
                           unsigned char gpu_memory_allow_growth,
                           unsigned int num_cpu_devices) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("enable_xla_compilation: '" + std::string(1, enable_xla_compilation) + "'");
   mht_9_v.push_back("gpu_memory_allow_growth: '" + std::string(1, gpu_memory_allow_growth) + "'");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_9(mht_9_v, 333, "", "./tensorflow/c/c_api_experimental.cc", "TF_CreateConfig");

  tensorflow::ConfigProto config;
  auto* optimizer_options =
      config.mutable_graph_options()->mutable_optimizer_options();
  if (enable_xla_compilation) {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::ON_1);

    // These XLA flags are needed to trigger XLA properly from C (more generally
    // non-Python) clients. If this API is called again with `enable` set to
    // false, it is safe to keep these flag values as is.
    tensorflow::MarkForCompilationPassFlags* flags =
        tensorflow::GetMarkForCompilationPassFlags();
    flags->tf_xla_cpu_global_jit = true;
    flags->tf_xla_min_cluster_size = 1;
  } else {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::OFF);
  }

  auto* gpu_options = config.mutable_gpu_options();
  gpu_options->set_allow_growth(gpu_memory_allow_growth);

  (*config.mutable_device_count())["CPU"] = num_cpu_devices;

  // TODO(b/113217601): This is needed for EagerContext::runner_ to use a
  // threadpool, so that we avoid the possibility of running the runner_ in the
  // threadpool of GPU event mgr, as that can trigger more callbacks to be
  // scheduled on that same threadpool, causing a deadlock in cases where the
  // caller of event_mgr->ThenExecute() blocks on the completion of the callback
  // (as in the case of ConstOp kernel creation on GPU, which involves copying a
  // CPU tensor to GPU).
  // Setting a larger thread pool does not help with the Swift caller, as we use
  // a different TFE context for each thread of execution (for running graph
  // functions, and their send/recvs corountines).
  config.set_inter_op_parallelism_threads(1);

  TF_Buffer* ret = TF_NewBuffer();
  TF_CHECK_OK(MessageToBuffer(config, ret));
  return ret;
}

TF_Buffer* TF_CreateRunOptions(unsigned char enable_full_trace) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("enable_full_trace: '" + std::string(1, enable_full_trace) + "'");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_10(mht_10_v, 377, "", "./tensorflow/c/c_api_experimental.cc", "TF_CreateRunOptions");

  tensorflow::RunOptions options;
  if (enable_full_trace) {
    options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
  } else {
    options.set_trace_level(tensorflow::RunOptions::NO_TRACE);
  }
  TF_Buffer* ret = TF_NewBuffer();
  TF_CHECK_OK(MessageToBuffer(options, ret));
  return ret;
}

const char* TF_GraphDebugString(TF_Graph* graph, size_t* len) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_11(mht_11_v, 392, "", "./tensorflow/c/c_api_experimental.cc", "TF_GraphDebugString");

  tensorflow::mutex_lock c(graph->mu);
  const auto& debug_str = graph->graph.ToGraphDefDebug().DebugString();
  *len = debug_str.size();
  char* ret = static_cast<char*>(malloc(*len + 1));
  memcpy(ret, debug_str.c_str(), *len + 1);
  return ret;
}

char* TF_FunctionDebugString(TF_Function* func, size_t* len) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_12(mht_12_v, 404, "", "./tensorflow/c/c_api_experimental.cc", "TF_FunctionDebugString");

  const auto& debug_str = DebugString(func->fdef);
  *len = debug_str.size();
  char* ret = static_cast<char*>(malloc(*len + 1));
  memcpy(ret, debug_str.c_str(), *len + 1);
  return ret;
}

// On success, returns a set of TF_Function instances from `text_proto` of
// GraphDef type. These functions must be deleted by calling TF_DeleteFunction.
//
// If `mutate_proto_func` is non-NULL, run it over each FunctionDef proto,
// before creating a TF_Function out of the possibly mutated proto.
static std::vector<UniqueFuncPtr> CreateFunctionsFromTextProto(
    const char* text_proto,
    std::function<void(FunctionDef*)>* mutate_proto_func, TF_Status* status) {
  tensorflow::GraphDef gdef;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(text_proto, &gdef)) {
    status->status = tensorflow::errors::Internal(
        "Invalid text proto for GraphDef: ", text_proto);
    return {};
  }
  const auto& fdef_lib = gdef.library();
  if (fdef_lib.gradient_size() > 0) {
    status->status = tensorflow::errors::Internal(
        "GradientDef is not supported in reading Dataset related functions: ",
        text_proto);
    return {};
  }
  std::vector<UniqueFuncPtr> ret;
  for (const FunctionDef& fdef : fdef_lib.function()) {
    // Make a copy so that we can mutate it.
    FunctionDef fdef_to_load = fdef;
    if (mutate_proto_func) {
      (*mutate_proto_func)(&fdef_to_load);
    }
    VLOG(1) << "Adding func to graph: " << fdef_to_load.DebugString();
    std::vector<char> binary_proto_buf(fdef_to_load.ByteSizeLong());
    fdef_to_load.SerializeToArray(binary_proto_buf.data(),
                                  binary_proto_buf.size());
    TF_Function* func = TF_FunctionImportFunctionDef(
        binary_proto_buf.data(), binary_proto_buf.size(), status);
    if (!status->status.ok()) return {};
    ret.push_back(UniqueFuncPtr(func, TF_DeleteFunction));
  }
  return ret;
}

TF_Tensor* TF_DequeueNamedTensor(TF_Session* session, int tensor_id,
                                 TF_Status* status) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_13(mht_13_v, 456, "", "./tensorflow/c/c_api_experimental.cc", "TF_DequeueNamedTensor");

  assert(session);
  {
    tensorflow::mutex_lock c(session->graph->mu);
    VLOG(1) << "Dequeuing named tensor with id " << tensor_id
            << ", with input graph: "
            << session->graph->graph.ToGraphDefDebug().DebugString();
  }

  TF_Operation* dequeue_op = TF_GraphOperationByName(
      session->graph,
      tensorflow::strings::StrCat("fifo_queue_dequeue_", tensor_id).c_str());
  if (dequeue_op == nullptr) {
    status->status = tensorflow::errors::Internal(
        "Unable to find the dequeue node in the TF graph.");
    return nullptr;
  }

  VLOG(1) << "Running the dequeue op";
  TF_Output output{dequeue_op, 0};
  TF_Tensor* ret;
  TF_SessionRun(session, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ nullptr, /*input_values*/ nullptr, /*ninputs*/ 0,
                // output related parameters
                /*outputs*/ &output, /*output_values*/ &ret,
                /*noutputs*/ 1,
                /*targets*/ nullptr, /*ntargets*/ 0,
                /*run_metadata*/ nullptr, status);
  if (VLOG_IS_ON(1) && status->status.ok()) {
    tensorflow::Tensor tensor;
    if (tensorflow::TF_TensorToTensor(ret, &tensor).ok()) {
      VLOG(1) << "Dequeued tensor content: " << tensor.DebugString();
    }
  }
  return ret;
}

void TF_EnqueueNamedTensor(TF_Session* session, int tensor_id,
                           TF_Tensor* tensor, TF_Status* status) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_14(mht_14_v, 498, "", "./tensorflow/c/c_api_experimental.cc", "TF_EnqueueNamedTensor");

  assert(session);
  {
    tensorflow::mutex_lock c(session->graph->mu);
    if (VLOG_IS_ON(1)) {
      VLOG(1) << "Enqueuing named tensor with id " << tensor_id
              << ", with input graph: "
              << session->graph->graph.ToGraphDefDebug().DebugString();
      tensorflow::Tensor internal_tensor;
      if (tensorflow::TF_TensorToTensor(tensor, &internal_tensor).ok()) {
        VLOG(1) << "Enqueu'ing tensor content: "
                << internal_tensor.DebugString();
      }
    }
  }

  TF_Operation* enqueue_op = TF_GraphOperationByName(
      session->graph,
      tensorflow::strings::StrCat("fifo_queue_enqueue_", tensor_id).c_str());
  if (enqueue_op == nullptr) {
    status->status = tensorflow::errors::Internal(
        "Unable to find the enqueue node in the TF graph.");
    return;
  }

  TF_Operation* placeholder_op = TF_GraphOperationByName(
      session->graph,
      tensorflow::strings::StrCat("arg_tensor_enqueue_", tensor_id).c_str());
  if (placeholder_op == nullptr) {
    status->status = tensorflow::errors::Internal(
        "Unable to find the placeholder node as input to enqueue in the TF "
        "graph.");
    return;
  }

  VLOG(1) << "Running the enqueue op";
  TF_Output input{placeholder_op, 0};
  TF_SessionRun(session, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ &input, /*input_values*/ &tensor, /*ninputs*/ 1,
                // output related parameters
                /*outputs*/ nullptr, /*output_values*/ nullptr, /*noutputs*/ 0,
                /*targets*/ &enqueue_op, /*ntargets*/ 1,
                /*run_metadata*/ nullptr, status);
  VLOG(1) << "Enqueuing is done.";
}

TF_Buffer* TFE_GetServerDef(const char* text_proto, TF_Status* status) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("text_proto: \"" + (text_proto == nullptr ? std::string("nullptr") : std::string((char*)text_proto)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_15(mht_15_v, 549, "", "./tensorflow/c/c_api_experimental.cc", "TFE_GetServerDef");

  tensorflow::ServerDef server_def;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(text_proto,
                                                         &server_def)) {
    status->status = tensorflow::errors::Internal(
        "Invalid text proto for ServerDef: ", text_proto);
    return nullptr;
  }
  status->status = tensorflow::Status();
  TF_Buffer* ret = TF_NewBuffer();
  TF_CHECK_OK(MessageToBuffer(server_def, ret));
  return ret;
}

void TF_MakeInternalErrorStatus(TF_Status* status, const char* errMsg) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("errMsg: \"" + (errMsg == nullptr ? std::string("nullptr") : std::string((char*)errMsg)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_16(mht_16_v, 567, "", "./tensorflow/c/c_api_experimental.cc", "TF_MakeInternalErrorStatus");

  status->status = tensorflow::errors::Internal(errMsg);
}

struct TF_CheckpointReader : public tensorflow::checkpoint::CheckpointReader {
  using tensorflow::checkpoint::CheckpointReader::CheckpointReader;
  std::vector<std::string> variable_list;
};

TF_CheckpointReader* TF_NewCheckpointReader(const char* filename,
                                            TF_Status* status) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("filename: \"" + (filename == nullptr ? std::string("nullptr") : std::string((char*)filename)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_17(mht_17_v, 581, "", "./tensorflow/c/c_api_experimental.cc", "TF_NewCheckpointReader");

  TF_CheckpointReader* reader = new TF_CheckpointReader(filename, status);
  if (!status->status.ok()) {
    TF_DeleteCheckpointReader(reader);
    return nullptr;
  }
  const auto& m = reader->GetVariableToDataTypeMap();
  for (auto it = m.begin(); it != m.end(); ++it)
    reader->variable_list.push_back(it->first);
  std::sort(reader->variable_list.begin(), reader->variable_list.end());
  return reader;
}

void TF_DeleteCheckpointReader(TF_CheckpointReader* reader) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_18(mht_18_v, 597, "", "./tensorflow/c/c_api_experimental.cc", "TF_DeleteCheckpointReader");
 delete reader; }

int TF_CheckpointReaderHasTensor(TF_CheckpointReader* reader,
                                 const char* name) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_19(mht_19_v, 604, "", "./tensorflow/c/c_api_experimental.cc", "TF_CheckpointReaderHasTensor");

  return reader->HasTensor(name);
}

const char* TF_CheckpointReaderGetVariable(TF_CheckpointReader* reader,
                                           int index) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_20(mht_20_v, 612, "", "./tensorflow/c/c_api_experimental.cc", "TF_CheckpointReaderGetVariable");

  return reader->variable_list[index].c_str();
}

int TF_CheckpointReaderSize(TF_CheckpointReader* reader) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_21(mht_21_v, 619, "", "./tensorflow/c/c_api_experimental.cc", "TF_CheckpointReaderSize");

  return reader->variable_list.size();
}

TF_DataType TF_CheckpointReaderGetVariableDataType(TF_CheckpointReader* reader,
                                                   const char* name) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_22(mht_22_v, 628, "", "./tensorflow/c/c_api_experimental.cc", "TF_CheckpointReaderGetVariableDataType");

  const auto& m = reader->GetVariableToDataTypeMap();
  return static_cast<TF_DataType>(m.at(name));
}

TF_Tensor* TF_CheckpointReaderGetTensor(TF_CheckpointReader* reader,
                                        const char* name, TF_Status* status) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_23(mht_23_v, 638, "", "./tensorflow/c/c_api_experimental.cc", "TF_CheckpointReaderGetTensor");

  std::unique_ptr<tensorflow::Tensor> tensor;
  reader->GetTensor(name, &tensor, status);
  if (!status->status.ok()) return nullptr;
  return tensorflow::TF_TensorFromTensor(*tensor, &status->status);
}

void TF_CheckpointReaderGetVariableShape(TF_CheckpointReader* reader,
                                         const char* name, int64_t* dims,
                                         int num_dims, TF_Status* status) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_24(mht_24_v, 651, "", "./tensorflow/c/c_api_experimental.cc", "TF_CheckpointReaderGetVariableShape");

  const auto& shape = reader->GetVariableToShapeMap().at(name);
  int rank = shape.dims();
  if (num_dims != rank) {
    status->status = InvalidArgument("Expected rank is ", num_dims,
                                     " but actual rank is ", rank);
    return;
  }
  for (int i = 0; i < num_dims; i++) {
    dims[i] = shape.dim_size(i);
  }
}

int TF_CheckpointReaderGetVariableNumDims(TF_CheckpointReader* reader,
                                          const char* name) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_25(mht_25_v, 669, "", "./tensorflow/c/c_api_experimental.cc", "TF_CheckpointReaderGetVariableNumDims");

  const auto& m = reader->GetVariableToShapeMap();
  return m.at(name).dims();
}

// This builder is used in the eager API to build a NodeDef.
struct TF_AttrBuilder : public tensorflow::AttrBuilder {
  using tensorflow::AttrBuilder::AttrBuilder;
  // The string buffers to make sure that any `attr_name` we pass into
  // `builder->Set()` will outlive the subsequent
  // `TF_AttrBuilderCheckCanRunOnDevice()` call(s) on the same `builder`.
  std::set<std::string> attr_names;
};

TF_AttrBuilder* TF_NewAttrBuilder(const char* op_name) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_26(mht_26_v, 687, "", "./tensorflow/c/c_api_experimental.cc", "TF_NewAttrBuilder");

  return new TF_AttrBuilder(op_name);
}

void TF_DeleteAttrBuilder(TF_AttrBuilder* builder) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_27(mht_27_v, 694, "", "./tensorflow/c/c_api_experimental.cc", "TF_DeleteAttrBuilder");
 delete builder; }

void TF_AttrBuilderSetType(TF_AttrBuilder* builder, const char* attr_name,
                           TF_DataType value) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_28(mht_28_v, 701, "", "./tensorflow/c/c_api_experimental.cc", "TF_AttrBuilderSetType");

  auto iter = builder->attr_names.insert(attr_name).first;
  builder->Set(*iter, static_cast<tensorflow::DataType>(value));
}

void TF_AttrBuilderSetTypeList(TF_AttrBuilder* builder, const char* attr_name,
                               const TF_DataType* values, int num_values) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_29(mht_29_v, 711, "", "./tensorflow/c/c_api_experimental.cc", "TF_AttrBuilderSetTypeList");

  auto iter = builder->attr_names.insert(attr_name).first;
  builder->Set(*iter, tensorflow::gtl::ArraySlice<const tensorflow::DataType>(
                          reinterpret_cast<const tensorflow::DataType*>(values),
                          num_values));
}

void TF_AttrBuilderCheckCanRunOnDevice(TF_AttrBuilder* builder,
                                       const char* device_type,
                                       TF_Status* status) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("device_type: \"" + (device_type == nullptr ? std::string("nullptr") : std::string((char*)device_type)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_30(mht_30_v, 724, "", "./tensorflow/c/c_api_experimental.cc", "TF_AttrBuilderCheckCanRunOnDevice");

  status->status = tensorflow::FindKernelDef(
      tensorflow::DeviceType(device_type), builder->BuildNodeDef(),
      /* def = */ nullptr, /* kernel_class_name = */ nullptr);
}

const char* TF_GetNumberAttrForOpListInput(const char* op_name, int input_index,
                                           TF_Status* status) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_31(mht_31_v, 735, "", "./tensorflow/c/c_api_experimental.cc", "TF_GetNumberAttrForOpListInput");

  const tensorflow::OpDef* op_def = nullptr;
  status->status =
      tensorflow::OpRegistry::Global()->LookUpOpDef(op_name, &op_def);
  if (!status->status.ok()) return nullptr;

  if (input_index >= op_def->input_arg_size() || input_index < 0) {
    status->status = tensorflow::errors::InvalidArgument(
        input_index, " out of range for ", op_name);
    return nullptr;
  }

  const tensorflow::OpDef_ArgDef& input_arg = op_def->input_arg()[input_index];

  if (input_arg.number_attr().empty()) {
    status->status = tensorflow::errors::NotFound(
        op_name, " does not have number_attr() defined.");
    return nullptr;
  }

  // The returned string is owned by OpRegistry, so liveness is not a concern.
  return input_arg.number_attr().c_str();
}

int TF_OpIsStateful(const char* op_type, TF_Status* status) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("op_type: \"" + (op_type == nullptr ? std::string("nullptr") : std::string((char*)op_type)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_32(mht_32_v, 763, "", "./tensorflow/c/c_api_experimental.cc", "TF_OpIsStateful");

  const tensorflow::OpRegistrationData* op_reg_data;
  status->status =
      tensorflow::OpRegistry::Global()->LookUp(op_type, &op_reg_data);
  if (!status->status.ok()) {
    return 0;
  }
  return op_reg_data->op_def.is_stateful();
}

void TF_InitMain(const char* usage, int* argc, char*** argv) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("usage: \"" + (usage == nullptr ? std::string("nullptr") : std::string((char*)usage)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_33(mht_33_v, 777, "", "./tensorflow/c/c_api_experimental.cc", "TF_InitMain");

  tensorflow::port::InitMain(usage, argc, argv);
}

int TF_PickUnusedPortOrDie() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_34(mht_34_v, 784, "", "./tensorflow/c/c_api_experimental.cc", "TF_PickUnusedPortOrDie");

  return tensorflow::internal::PickUnusedPortOrDie();
}

TFE_TensorHandle* TFE_NewTensorHandleFromScalar(TF_DataType data_type,
                                                void* data, size_t len,
                                                TF_Status* status) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_35(mht_35_v, 793, "", "./tensorflow/c/c_api_experimental.cc", "TFE_NewTensorHandleFromScalar");

  auto dtype = static_cast<tensorflow::DataType>(data_type);
  DCHECK(tensorflow::DataTypeCanUseMemcpy(dtype));

  tensorflow::Tensor tensor(dtype, tensorflow::TensorShape({}));
  std::memcpy(tensorflow::TensorCApi::Buffer(tensor)->data(), data, len);

  status->status = tensorflow::Status::OK();
  return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
}

// Set server_def on the context, possibly updating it.
TF_CAPI_EXPORT extern void TFE_EnableCollectiveOps(TFE_Context* ctx,
                                                   const void* proto,
                                                   size_t proto_len,
                                                   TF_Status* status) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_36(mht_36_v, 811, "", "./tensorflow/c/c_api_experimental.cc", "TFE_EnableCollectiveOps");

  tensorflow::ServerDef server_def;
  if (!server_def.ParseFromArray(proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Invalid tensorflow.ServerDef protocol buffer");
    return;
  }
  status->status = tensorflow::unwrap(ctx)->EnableCollectiveOps(server_def);
}

TF_CAPI_EXPORT extern void TFE_AbortCollectiveOps(TFE_Context* ctx,
                                                  TF_Status* status) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_37(mht_37_v, 825, "", "./tensorflow/c/c_api_experimental.cc", "TFE_AbortCollectiveOps");

  tensorflow::EagerContext* context =
      tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
  auto collective_executor_handle = context->GetCollectiveExecutorHandle();
  collective_executor_handle->get()->StartAbort(status->status);
}

TF_CAPI_EXPORT extern void TFE_CollectiveOpsCheckPeerHealth(
    TFE_Context* ctx, const char* task, int64_t timeout_in_ms,
    TF_Status* status) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("task: \"" + (task == nullptr ? std::string("nullptr") : std::string((char*)task)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_38(mht_38_v, 838, "", "./tensorflow/c/c_api_experimental.cc", "TFE_CollectiveOpsCheckPeerHealth");

  tensorflow::EagerContext* context =
      tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
  auto collective_executor_handle = context->GetCollectiveExecutorHandle();
  tensorflow::Notification done;
  collective_executor_handle->get()->remote_access()->CheckPeerHealth(
      task, timeout_in_ms, [&done, status](const Status& s) {
        status->status = s;
        done.Notify();
      });
  done.WaitForNotification();
}

TF_ShapeAndTypeList* TF_NewShapeAndTypeList(int num_items) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_39(mht_39_v, 854, "", "./tensorflow/c/c_api_experimental.cc", "TF_NewShapeAndTypeList");

  TF_ShapeAndTypeList* result = new TF_ShapeAndTypeList;
  result->num_items = num_items;
  result->items = (num_items == 0) ? nullptr : new TF_ShapeAndType[num_items]();
  return result;
}

void TF_ShapeAndTypeListSetShape(TF_ShapeAndTypeList* shape_list, int index,
                                 const int64_t* dims, int num_dims) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_40(mht_40_v, 865, "", "./tensorflow/c/c_api_experimental.cc", "TF_ShapeAndTypeListSetShape");

  DCHECK(index >= 0 && index < shape_list->num_items);
  TF_ShapeAndType& shape = shape_list->items[index];
  DCHECK(shape.dims == nullptr) << "Shape at " << index << " is already set!";
  DCHECK(num_dims >= 0) << "Number of dimensions cannot be negative!";
  shape.num_dims = num_dims;
  shape.dims = new int64_t[num_dims];
  memcpy(shape.dims, dims, sizeof(int64_t) * num_dims);
}

void TF_ShapeAndTypeListSetUnknownShape(TF_ShapeAndTypeList* shape_list,
                                        int index) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_41(mht_41_v, 879, "", "./tensorflow/c/c_api_experimental.cc", "TF_ShapeAndTypeListSetUnknownShape");

  DCHECK(index >= 0 && index < shape_list->num_items);
  TF_ShapeAndType& shape = shape_list->items[index];
  DCHECK(shape.dims == nullptr) << "Shape at " << index << " is already set!";
  shape.num_dims = -1;
  shape.dims = nullptr;
}

void TF_ShapeAndTypeListSetDtype(TF_ShapeAndTypeList* shape_list, int index,
                                 TF_DataType dtype) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_42(mht_42_v, 891, "", "./tensorflow/c/c_api_experimental.cc", "TF_ShapeAndTypeListSetDtype");

  DCHECK(index >= 0 && index < shape_list->num_items);
  TF_ShapeAndType& shape_and_type = shape_list->items[index];
  shape_and_type.dtype = dtype;
}

void TF_DeleteShapeAndTypeList(TF_ShapeAndTypeList* shape_list) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_43(mht_43_v, 900, "", "./tensorflow/c/c_api_experimental.cc", "TF_DeleteShapeAndTypeList");

  if (shape_list == nullptr) return;
  for (size_t i = 0; i < shape_list->num_items; ++i) {
    delete[] shape_list->items[i].dims;
  }
  delete[] shape_list->items;
  delete shape_list;
}

void TF_DeleteShapeAndTypeListArray(TF_ShapeAndTypeList** shape_list_array,
                                    int num_items) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_44(mht_44_v, 913, "", "./tensorflow/c/c_api_experimental.cc", "TF_DeleteShapeAndTypeListArray");

  if (shape_list_array == nullptr) return;
  for (int i = 0; i < num_items; ++i) {
    TF_DeleteShapeAndTypeList(shape_list_array[i]);
  }
  delete[] shape_list_array;
}

namespace tensorflow {
Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);

// Helpers for loadding a TensorFlow PluggableDevice plugin (a .so file).
Status LoadPluggableDeviceLibrary(const char* library_filename, void** result);
}  // namespace tensorflow

void TFE_InferShapes(TFE_Op* tfe_op, TF_ShapeAndTypeList* input_shapes,
                     TF_Tensor** input_tensors,
                     TF_ShapeAndTypeList* input_tensors_as_shapes,
                     TF_ShapeAndTypeList** input_resource_shapes_and_types,
                     TF_ShapeAndTypeList** output_shapes,
                     TF_ShapeAndTypeList*** output_resource_shapes_and_types,
                     TF_Status* status) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_45(mht_45_v, 937, "", "./tensorflow/c/c_api_experimental.cc", "TFE_InferShapes");

  using tensorflow::NodeDef;
  using tensorflow::OpRegistrationData;
  using tensorflow::Tensor;
  using tensorflow::shape_inference::DimensionHandle;
  using tensorflow::shape_inference::InferenceContext;
  using tensorflow::shape_inference::ShapeAndType;
  using tensorflow::shape_inference::ShapeHandle;

  const int num_inputs = input_shapes->num_items;
  NodeDef node_def;
  tensorflow::ImmediateExecutionOperation* op = tensorflow::unwrap(tfe_op);
  node_def.set_name(op->Name());
  node_def.set_op(op->Name());
  for (int i = 0; i < num_inputs; ++i) {
    node_def.add_input("dummy_input");
  }
  OperationFromInterface(op)->Attrs().FillAttrValueMap(node_def.mutable_attr());

  const tensorflow::OpRegistrationData* op_reg_data;
  status->status =
      tensorflow::OpRegistry::Global()->LookUp(node_def.op(), &op_reg_data);
  if (!status->status.ok()) return;

  // Initialize a input_tensor vector with `nullptr` values.
  std::vector<const Tensor*> input_tensors_vector(num_inputs, nullptr);
  // A vector to keep track of newly created `tf::Tensor` objects.
  std::vector<Tensor> all_input_tensors;
  // Update the vector with information from `input_tensors` if provided.
  if (input_tensors != nullptr) {
    // Note that we take the address of the elements in `all_input_tensors`
    // below. Allocate enough space so that no reallocation happens, which will
    // make the pointers invalid.
    all_input_tensors.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      if (input_tensors[i] == nullptr) continue;
      all_input_tensors.emplace_back();
      Tensor& input_tensor = all_input_tensors.back();
      status->status = TF_TensorToTensor(input_tensors[i], &input_tensor);
      if (!status->status.ok()) return;
      input_tensors_vector[i] = &input_tensor;
    }
  }

  // Create an inference context with dummy values, which will be updated later.
  InferenceContext c(TF_GRAPH_DEF_VERSION, node_def, op_reg_data->op_def,
                     std::vector<ShapeHandle>(num_inputs), input_tensors_vector,
                     {},
                     std::vector<std::unique_ptr<std::vector<ShapeAndType>>>());

  // Set input_shapes.
  for (int i = 0; i < num_inputs; ++i) {
    std::vector<DimensionHandle> dims;
    const TF_ShapeAndType& input_shape = input_shapes->items[i];
    if (input_shape.num_dims == InferenceContext::kUnknownRank) {
      c.SetInput(i, c.UnknownShape());
      continue;
    }
    dims.reserve(input_shape.num_dims);
    for (int j = 0; j < input_shape.num_dims; ++j) {
      dims.push_back(c.MakeDim(input_shape.dims[j]));
    }
    c.SetInput(i, c.MakeShape(dims));
  }

  // TODO(bgogul): Handle input_tensors_as_shapes.
  // TODO(bgogul): Handle input_resource_shapes_and_types.

  status->status = c.construction_status();
  if (!status->status.ok()) return;

  if (op_reg_data->shape_inference_fn == nullptr) {
    status->status =
        InvalidArgument("No shape inference function exists for op '",
                        node_def.op(), "', did you forget to define it?");
    return;
  }

  status->status = c.Run(op_reg_data->shape_inference_fn);
  if (!status->status.ok()) return;

  // Set output_shapes.
  TF_ShapeAndTypeList* output_shapes_result =
      TF_NewShapeAndTypeList(c.num_outputs());
  for (int i = 0; i < c.num_outputs(); ++i) {
    ShapeHandle shape_handle = c.output(i);
    TF_ShapeAndType& shape = output_shapes_result->items[i];
    shape.num_dims = c.Rank(shape_handle);
    if (shape.num_dims == InferenceContext::kUnknownRank) {
      shape.dims = nullptr;
      continue;
    }
    shape.dims = new int64_t[shape.num_dims];
    for (size_t j = 0; j < shape.num_dims; ++j) {
      shape.dims[j] = c.Value(c.Dim(shape_handle, j));
    }
  }
  if (output_shapes != nullptr) *output_shapes = output_shapes_result;

  // TODO(bgogul): Set output_resource_shapes_and_types.
}

void TF_ImportGraphDefOptionsSetValidateColocationConstraints(
    TF_ImportGraphDefOptions* opts, unsigned char enable) {
   std::vector<std::string> mht_46_v;
   mht_46_v.push_back("enable: '" + std::string(1, enable) + "'");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_46(mht_46_v, 1044, "", "./tensorflow/c/c_api_experimental.cc", "TF_ImportGraphDefOptionsSetValidateColocationConstraints");

  opts->opts.validate_colocation_constraints = enable;
}

// Load a Pluggable Device library.
// On success, returns the handle to library in result and return OK from the
// function. Otherwise return nullptr in result and error Status from the
// function.
//
// If `library_filename` has already been loaded, we return a cached handle.
// Device and Kernels/Ops are registered as globals when a library is loaded
// for the first time.
TF_Library* TF_LoadPluggableDeviceLibrary(const char* library_filename,
                                          TF_Status* status) {
   std::vector<std::string> mht_47_v;
   mht_47_v.push_back("library_filename: \"" + (library_filename == nullptr ? std::string("nullptr") : std::string((char*)library_filename)) + "\"");
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_47(mht_47_v, 1061, "", "./tensorflow/c/c_api_experimental.cc", "TF_LoadPluggableDeviceLibrary");

#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "PluggableDevice plugin functionality is not supported on mobile");
  return nullptr;
#else
  TF_Library* lib_handle = new TF_Library;
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  static std::unordered_map<std::string, void*>* loaded_libs =
      new std::unordered_map<std::string, void*>();
  tensorflow::Env* env = tensorflow::Env::Default();
  {
    tensorflow::mutex_lock lock(mu);
    auto it = loaded_libs->find(library_filename);
    if (it != loaded_libs->end()) {
      lib_handle->lib_handle = it->second;
    } else {
      status->status =
          env->LoadDynamicLibrary(library_filename, &lib_handle->lib_handle);
      if (status->status.ok()) {
        TF_CHECK_OK(
            tensorflow::RegisterPluggableDevicePlugin(lib_handle->lib_handle));
      } else {
        delete lib_handle;
        return nullptr;
      }
    }
    return lib_handle;
  }
#endif
}

void TF_DeletePluggableDeviceLibraryHandle(TF_Library* lib_handle) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScPSc_api_experimentalDTcc mht_48(mht_48_v, 1096, "", "./tensorflow/c/c_api_experimental.cc", "TF_DeletePluggableDeviceLibraryHandle");

  delete lib_handle;
}
