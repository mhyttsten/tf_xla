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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/loader.h"

#include <unordered_set>

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader_util.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"

namespace tensorflow {
namespace {

auto* load_attempt_count = monitoring::Counter<2>::New(
    "/tensorflow/cc/saved_model/load_attempt_count",
    "The number of times a SavedModel was successfully loaded.", "model_path",
    "status");
auto* load_latency = monitoring::Counter<1>::New(
    "/tensorflow/cc/saved_model/load_latency",
    "Latency in microseconds for SavedModels that were successfully loaded.",
    "model_path");
auto* load_latency_by_stage = monitoring::Sampler<2>::New(
    {
        "/tensorflow/cc/saved_model/load_latency_by_stage",  // metric name
        "Distribution of wall time spent (in microseconds) in each stage "
        "(restore graph from disk, run init graph op, etc) when loading the "
        "model",
        "model_path",
        "stage",
    },
    // Scale of 10, power of 1.8 with bucket count 33 (~20 minutes).
    monitoring::Buckets::Exponential(10, 1.8, 33));

constexpr char kLoadAttemptFail[] = "fail";
constexpr char kLoadAttemptSuccess[] = "success";
// `tensorflow::LoadSavedModel` API label.
constexpr char kCCLoadLabel[] = "cc_load";

uint64 GetLatencyMicroseconds(const uint64 start_microseconds) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_0(mht_0_v, 241, "", "./tensorflow/cc/saved_model/loader.cc", "GetLatencyMicroseconds");

  const uint64 end_microseconds = EnvTime::NowMicros();
  // Avoid clock skew.
  if (end_microseconds < start_microseconds) return 0;
  return end_microseconds - start_microseconds;
}

// Ensure that constant tensors loaded from the saved model have valid shape.
// Also ensure that constant nodes have a value assigned to them.
// TODO(b/154763635): this is temporary and will be replaced with a better audit
static Status ValidateNode(const NodeDef& node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_1(mht_1_v, 254, "", "./tensorflow/cc/saved_model/loader.cc", "ValidateNode");

  const auto node_iterator = node.attr().find("value");
  if (node_iterator != node.attr().end()) {
    AttrValue node_value = node_iterator->second;
    if (node_value.has_tensor()) {
      const PartialTensorShape node_shape(node_value.tensor().tensor_shape());
      if (node_shape.num_elements() < 0) {
        return errors::FailedPrecondition(
            "Saved model contains node \"", node.name(), "\" (op \"", node.op(),
            "\") which initializes from a tensor with ",
            node_shape.num_elements(), " elements");
      }
    }
  } else if (node.op() == "Const") {
    return errors::FailedPrecondition(
        "Saved model contains node \"", node.name(),
        "\" which is a constant tensor but no value has been provided");
  }
  return Status::OK();
}

static Status ValidateFunctionNotRecursive(const FunctionDef& function) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_2(mht_2_v, 278, "", "./tensorflow/cc/saved_model/loader.cc", "ValidateFunctionNotRecursive");

  const auto& function_name = function.signature().name();
  for (const auto& node : function.node_def()) {
    if (node.op() == function_name) {
      return errors::FailedPrecondition(
          "Function ", function_name,
          " is self recursive and TensorFlow does not support this scenario.");
    }
  }

  return Status::OK();
}

static Status ValidateSavedTensors(const GraphDef& graph_def) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_3(mht_3_v, 294, "", "./tensorflow/cc/saved_model/loader.cc", "ValidateSavedTensors");

  for (const auto& node : graph_def.node()) {
    TF_RETURN_IF_ERROR(ValidateNode(node));
  }

  if (graph_def.has_library()) {
    const FunctionDefLibrary& library = graph_def.library();
    for (const auto& function : library.function()) {
      for (const auto& node : function.node_def()) {
        TF_RETURN_IF_ERROR(ValidateNode(node));
      }

      // Also check that there is no recursivity in the library
      // TODO(mihaimaruseac): Do more than self-recursivity
      TF_RETURN_IF_ERROR(ValidateFunctionNotRecursive(function));
    }
  }

  return Status::OK();
}

Tensor CreateStringTensor(const string& value) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_4(mht_4_v, 319, "", "./tensorflow/cc/saved_model/loader.cc", "CreateStringTensor");

  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<tstring>()() = value;
  return tensor;
}

void AddAssetsTensorsToInputs(const StringPiece export_dir,
                              const std::vector<AssetFileDef>& asset_file_defs,
                              std::vector<std::pair<string, Tensor>>* inputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_5(mht_5_v, 330, "", "./tensorflow/cc/saved_model/loader.cc", "AddAssetsTensorsToInputs");

  if (asset_file_defs.empty()) {
    return;
  }
  for (auto& asset_file_def : asset_file_defs) {
    Tensor assets_file_path_tensor = CreateStringTensor(io::JoinPath(
        export_dir, kSavedModelAssetsDirectory, asset_file_def.filename()));
    inputs->push_back(
        {asset_file_def.tensor_info().name(), assets_file_path_tensor});
  }
}

// Like Session::Run(), but uses the Make/Run/ReleaseCallable() API to avoid
// leaving behind non-GC'ed state.
//
// Detailed motivation behind this approach, from ashankar@:
//
// Each call to Session::Run() that identifies a new subgraph (based on feeds
// and fetches) creates some datastructures that live as long as the session
// (the partitioned graph, associated executors etc.).
//
// A pathological case of this would be if say the initialization op
// (main_op/legacy_init_op) involves the use of a large constant. Then we
// allocate memory for that large constant that will just stick around till the
// session dies. With this Callable mechanism, that memory will be released
// right after ReleaseCallable returns.
//
// However, the resource manager state remains.
Status RunOnce(const RunOptions& run_options,
               const std::vector<std::pair<string, Tensor>>& inputs,
               const std::vector<string>& output_tensor_names,
               const std::vector<string>& target_node_names,
               std::vector<Tensor>* outputs, RunMetadata* run_metadata,
               Session* session) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_6(mht_6_v, 366, "", "./tensorflow/cc/saved_model/loader.cc", "RunOnce");

  CallableOptions callable_options;
  std::vector<Tensor> feed_tensors;
  *callable_options.mutable_run_options() = run_options;
  for (const auto& input : inputs) {
    const string& name = input.first;
    const Tensor& tensor = input.second;
    callable_options.add_feed(name);
    feed_tensors.push_back(tensor);
  }
  for (const string& output_tensor_name : output_tensor_names) {
    callable_options.add_fetch(output_tensor_name);
  }
  for (const string& target_node_name : target_node_names) {
    callable_options.add_target(target_node_name);
  }

  Session::CallableHandle callable_handle;
  TF_RETURN_IF_ERROR(session->MakeCallable(callable_options, &callable_handle));
  const Status run_status = session->RunCallable(callable_handle, feed_tensors,
                                                 outputs, run_metadata);
  // Be sure to call ReleaseCallable() regardless of the outcome of
  // RunCallable().
  session->ReleaseCallable(callable_handle).IgnoreError();
  return run_status;
}

// RunInitOp will return OK if the initialization op was run successfully.
// An empty init_op_name indicates that there are no init ops to run.
Status RunInitOp(const RunOptions& run_options, const string& export_dir,
                 const MetaGraphDef& meta_graph_def,
                 const std::vector<AssetFileDef>& asset_file_defs,
                 Session* session, const string& init_op_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("export_dir: \"" + export_dir + "\"");
   mht_7_v.push_back("init_op_name: \"" + init_op_name + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_7(mht_7_v, 403, "", "./tensorflow/cc/saved_model/loader.cc", "RunInitOp");

  if (!init_op_name.empty()) {
    LOG(INFO) << "Running initialization op on SavedModel bundle at path: "
              << export_dir;
    std::vector<std::pair<string, Tensor>> inputs;
    AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);
    RunMetadata run_metadata;
    return RunOnce(run_options, inputs, {}, {init_op_name},
                   nullptr /* outputs */, &run_metadata, session);
  }
  return Status::OK();
}

Status RunRestore(const RunOptions& run_options, const string& export_dir,
                  const StringPiece restore_op_name,
                  const StringPiece variable_filename_const_op_name,
                  const std::vector<AssetFileDef>& asset_file_defs,
                  Session* session) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_8(mht_8_v, 424, "", "./tensorflow/cc/saved_model/loader.cc", "RunRestore");

  LOG(INFO) << "Restoring SavedModel bundle.";
  // Find path to variables to be restored in export directory.
  const string variables_directory =
      io::JoinPath(export_dir, kSavedModelVariablesDirectory);
  // Check for saver checkpoints in v2 format. Models exported in the checkpoint
  // v2 format will have a variables.index file. The corresponding
  // variables are stored in the variables.data-?????-of-????? files.
  const string variables_index_path = io::JoinPath(
      variables_directory, MetaFilename(kSavedModelVariablesFilename));
  if (!Env::Default()->FileExists(variables_index_path).ok()) {
    LOG(INFO) << "The specified SavedModel has no variables; no checkpoints "
                 "were restored. File does not exist: "
              << variables_index_path;
    return Status::OK();
  }
  const string variables_path =
      io::JoinPath(variables_directory, kSavedModelVariablesFilename);

  // Add variables to the graph.
  Tensor variables_path_tensor(DT_STRING, TensorShape({}));
  variables_path_tensor.scalar<tstring>()() = variables_path;

  std::vector<std::pair<string, Tensor>> inputs = {
      {string(variable_filename_const_op_name), variables_path_tensor}};

  AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);

  RunMetadata run_metadata;
  return RunOnce(run_options, inputs, {}, {string(restore_op_name)},
                 nullptr /* outputs */, &run_metadata, session);
}

}  // namespace

SavedModelBundleInterface::~SavedModelBundleInterface() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_9(mht_9_v, 462, "", "./tensorflow/cc/saved_model/loader.cc", "SavedModelBundleInterface::~SavedModelBundleInterface");
}

Status LoadMetagraphIntoSession(const SessionOptions& session_options,
                                const MetaGraphDef& meta_graph,
                                std::unique_ptr<Session>* session) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_10(mht_10_v, 469, "", "./tensorflow/cc/saved_model/loader.cc", "LoadMetagraphIntoSession");

  Session* session_p = nullptr;
  TF_RETURN_IF_ERROR(NewSession(session_options, &session_p));
  session->reset(session_p);
  TF_RETURN_IF_ERROR(ValidateSavedTensors(meta_graph.graph_def()));
  return (*session)->Create(meta_graph.graph_def());
}

Status LoadSavedModelInternal(const SessionOptions& session_options,
                              const RunOptions& run_options,
                              const string& export_dir,
                              const std::unordered_set<string>& tags,
                              SavedModelBundle* const bundle) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_11(mht_11_v, 485, "", "./tensorflow/cc/saved_model/loader.cc", "LoadSavedModelInternal");

  TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(export_dir, tags,
                                                    &bundle->meta_graph_def));
  TF_RETURN_IF_ERROR(
      ReadSavedModelDebugInfoIfPresent(export_dir, &bundle->debug_info));
  TF_RETURN_IF_ERROR(LoadMetagraphIntoSession(
      session_options, bundle->meta_graph_def, &bundle->session));
  TF_RETURN_IF_ERROR(RestoreSession(run_options, bundle->meta_graph_def,
                                    export_dir, &bundle->session));
  return Status::OK();
}

Status LoadSavedModel(const SessionOptions& session_options,
                      const RunOptions& run_options, const string& export_dir,
                      const std::unordered_set<string>& tags,
                      SavedModelBundle* const bundle) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_12(mht_12_v, 504, "", "./tensorflow/cc/saved_model/loader.cc", "LoadSavedModel");

  metrics::SavedModelReadApi(kCCLoadLabel).IncrementBy(1);

  // TODO(robson): Add tests for the counters.
  const uint64 start_microseconds = Env::Default()->NowMicros();
  const Status status = LoadSavedModelInternal(session_options, run_options,
                                               export_dir, tags, bundle);
  auto log_and_count = [&](const string& status_str) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("status_str: \"" + status_str + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_13(mht_13_v, 515, "", "./tensorflow/cc/saved_model/loader.cc", "lambda");

    LOG(INFO) << "SavedModel load for tags { " << absl::StrJoin(tags, " ")
              << " }; Status: " << status_str << ": " << status << ". Took "
              << GetLatencyMicroseconds(start_microseconds) << " microseconds.";
    load_attempt_count->GetCell(export_dir, status_str)->IncrementBy(1);
  };
  if (status.ok()) {
    log_and_count(kLoadAttemptSuccess);
  } else {
    log_and_count(kLoadAttemptFail);
  }
  load_latency->GetCell(export_dir)
      ->IncrementBy(GetLatencyMicroseconds(start_microseconds));
  return status;
}

namespace {
// Session wrapper that prevents calls to Session::Create(), Session::Extend(),
// and the deprecated partial-run methods.
//
// Limiting the available methods on a returned Session gives us the option
// to replace the Session with a cut-down implementation, without breaking any
// users.
class LiteSessionWrapper : public Session {
 public:
  explicit LiteSessionWrapper(std::unique_ptr<Session> wrapped)
      : wrapped_(std::move(wrapped)) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_14(mht_14_v, 544, "", "./tensorflow/cc/saved_model/loader.cc", "LiteSessionWrapper");
}

  Status Create(const GraphDef& graph) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_15(mht_15_v, 549, "", "./tensorflow/cc/saved_model/loader.cc", "Create");

    return errors::Unimplemented("Session::Create()");
  }
  Status Create(GraphDef&& graph) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_16(mht_16_v, 555, "", "./tensorflow/cc/saved_model/loader.cc", "Create");

    return errors::Unimplemented("Session::Create()");
  }

  Status Extend(const GraphDef& graph) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_17(mht_17_v, 562, "", "./tensorflow/cc/saved_model/loader.cc", "Extend");

    return errors::Unimplemented("Session::Extend()");
  }
  Status Extend(GraphDef&& graph) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_18(mht_18_v, 568, "", "./tensorflow/cc/saved_model/loader.cc", "Extend");

    return errors::Unimplemented("Session::Extend()");
  }

  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_19(mht_19_v, 578, "", "./tensorflow/cc/saved_model/loader.cc", "Run");

    return wrapped_->Run(inputs, output_tensor_names, target_node_names,
                         outputs);
  }

  Status Create(const RunOptions& run_options, const GraphDef& graph) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_20(mht_20_v, 586, "", "./tensorflow/cc/saved_model/loader.cc", "Create");

    return errors::Unimplemented("Session::Create()");
  }
  Status Extend(const RunOptions& run_options, const GraphDef& graph) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_21(mht_21_v, 592, "", "./tensorflow/cc/saved_model/loader.cc", "Extend");

    return errors::Unimplemented("Session::Extend()");
  }
  Status Create(const RunOptions& run_options, GraphDef&& graph) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_22(mht_22_v, 598, "", "./tensorflow/cc/saved_model/loader.cc", "Create");

    return errors::Unimplemented("Session::Create()");
  }
  Status Extend(const RunOptions& run_options, GraphDef&& graph) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_23(mht_23_v, 604, "", "./tensorflow/cc/saved_model/loader.cc", "Extend");

    return errors::Unimplemented("Session::Extend()");
  }
  Status Close(const RunOptions& run_options) override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_24(mht_24_v, 610, "", "./tensorflow/cc/saved_model/loader.cc", "Close");

    return wrapped_->Close(run_options);
  }

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_25(mht_25_v, 621, "", "./tensorflow/cc/saved_model/loader.cc", "Run");

    return wrapped_->Run(run_options, inputs, output_tensor_names,
                         target_node_names, outputs, run_metadata);
  }

  Status PRunSetup(const std::vector<string>& input_names,
                   const std::vector<string>& output_names,
                   const std::vector<string>& target_nodes,
                   string* handle) override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_26(mht_26_v, 632, "", "./tensorflow/cc/saved_model/loader.cc", "PRunSetup");

    return errors::Unimplemented("Session::PRunSetup()");
  }

  Status PRun(const string& handle,
              const std::vector<std::pair<string, Tensor>>& inputs,
              const std::vector<string>& output_names,
              std::vector<Tensor>* outputs) override {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_27(mht_27_v, 643, "", "./tensorflow/cc/saved_model/loader.cc", "PRun");

    return errors::Unimplemented("Session::PRun()");
  }

  Status ListDevices(std::vector<DeviceAttributes>* response) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_28(mht_28_v, 650, "", "./tensorflow/cc/saved_model/loader.cc", "ListDevices");

    return wrapped_->ListDevices(response);
  }

  Status Close() override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_29(mht_29_v, 657, "", "./tensorflow/cc/saved_model/loader.cc", "Close");
 return wrapped_->Close(); }

  Status MakeCallable(const CallableOptions& callable_options,
                      CallableHandle* out_handle) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_30(mht_30_v, 663, "", "./tensorflow/cc/saved_model/loader.cc", "MakeCallable");

    return wrapped_->MakeCallable(callable_options, out_handle);
  }

  Status RunCallable(CallableHandle handle,
                     const std::vector<Tensor>& feed_tensors,
                     std::vector<Tensor>* fetch_tensors,
                     RunMetadata* run_metadata) override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_31(mht_31_v, 673, "", "./tensorflow/cc/saved_model/loader.cc", "RunCallable");

    return wrapped_->RunCallable(handle, feed_tensors, fetch_tensors,
                                 run_metadata);
  }

  Status RunCallable(
      CallableHandle handle, const std::vector<Tensor>& feed_tensors,
      std::vector<Tensor>* fetch_tensors, RunMetadata* run_metadata,
      const thread::ThreadPoolOptions& threadpool_options) override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_32(mht_32_v, 684, "", "./tensorflow/cc/saved_model/loader.cc", "RunCallable");

    return wrapped_->RunCallable(handle, feed_tensors, fetch_tensors,
                                 run_metadata, threadpool_options);
  }

  Status ReleaseCallable(CallableHandle handle) override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_33(mht_33_v, 692, "", "./tensorflow/cc/saved_model/loader.cc", "ReleaseCallable");

    return wrapped_->ReleaseCallable(handle);
  }

 private:
  const std::unique_ptr<Session> wrapped_;
};
}  // namespace

Status RestoreSession(const RunOptions& run_options,
                      const MetaGraphDef& meta_graph, const string& export_dir,
                      std::unique_ptr<Session>* session) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_34(mht_34_v, 707, "", "./tensorflow/cc/saved_model/loader.cc", "RestoreSession");

  const uint64 read_start_microseconds = Env::Default()->NowMicros();
  std::vector<AssetFileDef> asset_file_defs;
  TF_RETURN_IF_ERROR(internal::GetAssetFileDefs(meta_graph, &asset_file_defs));
  if (meta_graph.has_saver_def()) {
    TF_RETURN_IF_ERROR(RunRestore(run_options, export_dir,
                                  meta_graph.saver_def().restore_op_name(),
                                  meta_graph.saver_def().filename_tensor_name(),
                                  asset_file_defs, session->get()));
  }
  // Record walltime spent in restoring graph from disk, but postpone metric
  // increments until graph init finishes.
  const uint64 restore_graph_walltime =
      GetLatencyMicroseconds(read_start_microseconds);

  const uint64 graph_init_start_microseconds = Env::Default()->NowMicros();
  string init_op_name;
  TF_RETURN_IF_ERROR(
      internal::GetInitOp(export_dir, meta_graph, &init_op_name));
  TF_RETURN_IF_ERROR(RunInitOp(run_options, export_dir, meta_graph,
                               asset_file_defs, session->get(), init_op_name));
  load_latency_by_stage->GetCell(export_dir, "restore_graph")
      ->Add(restore_graph_walltime);
  // Record wall time spent in init op.
  load_latency_by_stage->GetCell(export_dir, "init_graph")
      ->Add(GetLatencyMicroseconds(graph_init_start_microseconds));
  return Status::OK();
}

Status LoadSavedModel(const SessionOptions& session_options,
                      const RunOptions& run_options, const string& export_dir,
                      const std::unordered_set<string>& tags,
                      SavedModelBundleLite* const bundle) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_35(mht_35_v, 743, "", "./tensorflow/cc/saved_model/loader.cc", "LoadSavedModel");

  SavedModelBundle legacy_bundle;
  SessionOptions rewritten_options(session_options);
  // We disallow calls to Session::Extend() on the returned session, so we can
  // reduce memory consumption by not storing the original GraphDef.
  rewritten_options.config.mutable_experimental()
      ->set_optimize_for_static_graph(true);
  // Disallowing the `RunOptions.output_partition_graphs` option (typically used
  // in debugging and tests) allows us to reduce memory consumption further by
  // not storing the rewritten subgraph for each signature.
  rewritten_options.config.mutable_experimental()
      ->set_disable_output_partition_graphs(true);
  // TODO(mrry): Consider specializing the session creation to reduce peak
  // RAM consumption by using `Session::Create(GraphDef&&)`.
  TF_RETURN_IF_ERROR(LoadSavedModel(rewritten_options, run_options, export_dir,
                                    tags, &legacy_bundle));
  *bundle = SavedModelBundleLite(
      absl::make_unique<LiteSessionWrapper>(std::move(legacy_bundle.session)),
      std::move(*legacy_bundle.meta_graph_def.mutable_signature_def()));
  return Status::OK();
}

bool MaybeSavedModelDirectory(const string& export_dir) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTcc mht_36(mht_36_v, 769, "", "./tensorflow/cc/saved_model/loader.cc", "MaybeSavedModelDirectory");

  const string saved_model_pb_path =
      io::JoinPath(export_dir, kSavedModelFilenamePb);
  const string saved_model_pbtxt_path =
      io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
  return Env::Default()->FileExists(saved_model_pb_path).ok() ||
         Env::Default()->FileExists(saved_model_pbtxt_path).ok();
}

}  // namespace tensorflow
