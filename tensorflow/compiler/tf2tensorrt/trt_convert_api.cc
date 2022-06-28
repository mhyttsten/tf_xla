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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc() {
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

#include "tensorflow/compiler/tf2tensorrt/trt_convert_api.h"

#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/cc/tools/freeze_saved_model.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {

namespace tensorrt {
namespace {

// Creates and provisions a new cluster. The caller must call Shutdown before
// the cluster is destroyed.
Status NewCluster(grappler::Cluster** cluster) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "NewCluster");

  int num_cpu_cores = grappler::GetNumAvailableLogicalCPUCores();
  int num_gpus = grappler::GetNumAvailableGPUs();
  int timeout_s = 60 * 10;
  *cluster = new grappler::SingleMachine(timeout_s, num_cpu_cores, num_gpus);
  (*cluster)->DisableDetailedStats(true);
  (*cluster)->AllowSoftPlacement(true);
  (*cluster)->SetNumWarmupSteps(10);
  TF_RETURN_IF_ERROR((*cluster)->Provision());
  return Status::OK();
}

Status RunGrappler(const MetaGraphDef& meta_graph_def,
                   const std::vector<std::string>& input_names,
                   const std::vector<std::string>& output_names,
                   const ConfigProto& config_proto, grappler::Cluster* cluster,
                   GraphDef* out_graph_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_1(mht_1_v, 240, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "RunGrappler");

  grappler::ItemConfig item_config;

  for (const string& name : input_names) {
    item_config.feed_nodes.insert(name);
  }
  for (const string& name : output_names) {
    item_config.fetch_nodes.insert(name);
  }

  std::unique_ptr<grappler::GrapplerItem> item =
      grappler::GrapplerItemFromMetaGraphDef("tf_graph", meta_graph_def,
                                             item_config);
  if (!item) {
    return tensorflow::errors::Internal(
        "Failed to create grappler item from MetaGraphDef.");
  }

  tensorflow::DeviceBase* cpu_device = nullptr;
  TF_RETURN_IF_ERROR(grappler::RunMetaOptimizer(
      std::move(*item), config_proto, cpu_device, cluster, out_graph_def));
  VLOG(2) << "Grappler finished\n";
  return Status::OK();
}

Status ImportGraphDefToSession(Session* session, const GraphDef& graph_def,
                               const string& prefix) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_2(mht_2_v, 270, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "ImportGraphDefToSession");

  ImportGraphDefOptions opts;
  opts.prefix = prefix;
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef(opts, graph_def, &graph, nullptr));
  GraphDef new_graph_def;
  graph.ToGraphDef(&new_graph_def);
  TF_RETURN_IF_ERROR(session->Extend(new_graph_def));
  return Status::OK();
}

Status GetTrtRewriterConfig(const TfTrtConversionParams& params,
                            const GraphDef& frozen_graph_def,
                            RewriterConfig* opt_config) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_3(mht_3_v, 286, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "GetTrtRewriterConfig");

  opt_config->set_meta_optimizer_iterations(tensorflow::RewriterConfig::ONE);
  opt_config->set_min_graph_nodes(-1);  // do not skip small graphs

  // Turn off remapping.
  opt_config->set_remapping(RewriterConfig_Toggle::RewriterConfig_Toggle_OFF);

  // If the graph has QDQ nodes, then we need to disable folding of the
  // QDQ with constants. Otherwise, the conversion will not work corectly.
  // Ideally, we do this after segmentation and outlining of TRT regions to
  // functions, but we currently lack that capability. Disabling QDQ-const
  // folding doesn't matter if you don't have QDQ nodes, so we always enable
  // this.
  opt_config->set_experimental_disable_folding_quantization_emulation(
      IS_TRT_VERSION_GE(8, 0, 0, 0));

  // Initial transformations before TensorRTOptimizer is called
  opt_config->add_optimizers("function");
  opt_config->add_optimizers("constfold");
  opt_config->add_optimizers("layout");
  opt_config->add_optimizers("constfold");

  // Parameters for TensorRTOptimizer
  auto trt_optimizer = opt_config->add_custom_optimizers();
  trt_optimizer->set_name("TensorRTOptimizer");

  auto trt_parameter_map = trt_optimizer->mutable_parameter_map();
  (*trt_parameter_map)["is_dynamic_op"].set_b(true);
  (*trt_parameter_map)["minimum_segment_size"].set_i(
      params.minimum_segment_size);
  string prec_string;
  TF_RETURN_IF_ERROR(
      TrtPrecisionModeToName(params.precision_mode, &prec_string));
  (*trt_parameter_map)["precision_mode"].set_s(prec_string);
  (*trt_parameter_map)["max_batch_size"].set_i(1);
  (*trt_parameter_map)["max_workspace_size_bytes"].set_i(
      params.max_workspace_size_bytes);
  (*trt_parameter_map)["max_cached_engines"].set_i(params.max_cached_engines);
  (*trt_parameter_map)["use_calibration"].set_b(params.use_calibration);
  (*trt_parameter_map)["profile_strategy"].set_s(
      ProfileStrategyToName(params.profile_strategy));
  (*trt_parameter_map)["use_implicit_batch"].set_b(!params.use_dynamic_shape);
  (*trt_parameter_map)["_allow_build_at_runtime"].set_b(
      params.allow_build_at_runtime);
  return Status::OK();
}

// Runs TRTOptimizer grappler pass.
Status RunTfTrt(const MetaGraphDef& meta_graph_def,
                const std::vector<std::string>& input_names,
                const std::vector<std::string>& output_names,
                const RewriterConfig& rewriter_config,
                GraphDef* segmented_graph_def) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_4(mht_4_v, 341, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "RunTfTrt");

  ConfigProto config_proto;
  config_proto.mutable_graph_options()->mutable_rewrite_options()->CopyFrom(
      rewriter_config);

  VLOG(4) << "Setting up Grappler parameters\n" << config_proto.DebugString();
  std::unique_ptr<grappler::Cluster> cluster;
  grappler::Cluster* p_cluster;
  mutex mu_cluster;  // There can be only one provisioned cluster per process.
  mutex_lock lock(mu_cluster);
  TF_RETURN_IF_ERROR(NewCluster(&p_cluster));
  cluster.reset(p_cluster);
  TF_RETURN_IF_ERROR(RunGrappler(meta_graph_def, input_names, output_names,
                                 config_proto, cluster.get(),
                                 segmented_graph_def));
  TF_RETURN_IF_ERROR(cluster->Shutdown());
  return Status::OK();
}

// Sets the _profile_generation mode attribute of all TRTEngineOp nodes in the
// graph to mode.
Status SetProfileGenerationMode(GraphDef* graph_def, bool mode) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_5(mht_5_v, 365, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "SetProfileGenerationMode");

  VLOG(3) << "Setting _profile_generation_mode=" << mode;
  std::string op{"TRTEngineOp"};
  for (auto& node : *(graph_def->mutable_node())) {
    if (!op.compare(node.op())) {
      auto* attr = node.mutable_attr();
      AttrValue profile_generation_mode;
      profile_generation_mode.set_b(mode);
      (*attr)["_profile_generation_mode"] = profile_generation_mode;
    }
  }
  return Status::OK();
}

Status RunSession(Session* session, const std::vector<std::string>& input_names,
                  const std::vector<std::string>& output_names,
                  const std::vector<Tensor>& input_tensors,
                  string prefix = "") {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_6(mht_6_v, 385, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "RunSession");

  TRT_ENSURE(!input_names.empty());
  TRT_ENSURE(!output_names.empty());
  TRT_ENSURE(!input_tensors.empty());

  std::vector<std::pair<std::string, tensorflow::Tensor>> input_pairs;
  std::vector<std::string> prefixed_output_names;
  auto prefixed_name = [](std::string prefix, std::string name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("prefix: \"" + prefix + "\"");
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_7(mht_7_v, 397, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "lambda");

    return prefix.size() > 0 ? absl::StrJoin({prefix, name}, "/") : name;
  };
  for (int i = 0; i < input_names.size(); i++) {
    input_pairs.push_back(
        {prefixed_name(prefix, input_names.at(i)), input_tensors.at(i)});
  }
  for (int i = 0; i < output_names.size(); i++) {
    prefixed_output_names.push_back(prefixed_name(prefix, output_names.at(i)));
  }
  std::vector<tensorflow::Tensor> output_tensors;
  for (int i = 0; i < output_names.size(); i++) {
    output_tensors.push_back({});
  }
  VLOG(3) << "TF-TRT Build mode: running inference\n";
  TF_RETURN_IF_ERROR(
      session->Run(input_pairs, prefixed_output_names, {}, &output_tensors));
  return Status::OK();
}

// Runs the model to create the engines. In dynamic shape mode, before creating
// the engines, we provide shapes to define optimization profiles.
Status Build(GraphDef& segmented_graph_def,
             const std::vector<std::string>& input_names,
             const std::vector<std::string>& output_names,
             const std::vector<std::vector<tensorflow::Tensor>>& inputs,
             Session* session, const TfTrtConversionParams params) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_8(mht_8_v, 426, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "Build");

  VLOG(2) << "Building the model";
  bool need_collect_profiles = params.use_dynamic_shape && inputs.size() > 1;
  if (need_collect_profiles) {
    TF_RETURN_IF_ERROR(SetProfileGenerationMode(&segmented_graph_def, true));
  }
  TF_RETURN_IF_ERROR(session->Create(segmented_graph_def));
  string prefix = "";
  if (need_collect_profiles) {
    for (auto const& input : inputs) {
      TF_RETURN_IF_ERROR(RunSession(session, input_names, output_names, input));
    }
    prefix = "TrtBuildStep";
    TF_RETURN_IF_ERROR(SetProfileGenerationMode(&segmented_graph_def, false));
    VLOG(3) << "Importing graph with _profile_generation_mode disabled";
    TF_RETURN_IF_ERROR(
        ImportGraphDefToSession(session, segmented_graph_def, prefix));
  }
  TF_RETURN_IF_ERROR(
      RunSession(session, input_names, output_names, *inputs.begin(), prefix));
  return Status::OK();
}

// Returns the resource manager associated with the node.
Status GetResourceManager(const NodeDef& node, Session* session,
                          ResourceMgr** rm) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_9(mht_9_v, 454, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "GetResourceManager");

  const DeviceMgr* device_mgr;
  TF_RETURN_IF_ERROR(session->LocalDeviceManager(&device_mgr));
  Device* device;
  string device_name = node.device().empty()
                           ? "/job:localhost/replica:0/task:0/device:GPU:0"
                           : node.device();
  TF_RETURN_IF_ERROR(device_mgr->LookupDevice(device_name, &device));
  *rm = device->resource_manager();
  return Status::OK();
}

// Looks up the cache resurce associated with the TRT node.
Status GetEngineCacheResource(const NodeDef& node, Session* session,
                              TRTEngineCacheResource** resource) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_10(mht_10_v, 471, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "GetEngineCacheResource");

  ResourceMgr* rm;
  TF_RETURN_IF_ERROR(GetResourceManager(node, session, &rm));

  absl::string_view resource_name = node.name();
  size_t last_slash = resource_name.find_last_of('/');
  if (last_slash != absl::string_view::npos) {
    resource_name.remove_prefix(last_slash + 1);
  }
  const std::string container(kTfTrtContainerName);
  *resource = nullptr;
  TF_RETURN_IF_ERROR(
      rm->Lookup(container, std::string(resource_name), resource));
  if (resource == nullptr || (*resource)->cache_.size() == 0) {
    return errors::Internal("Engine cache not found for", resource_name);
  }
  return Status::OK();
}

// Looks up the engine from the engine cache, and serializes the engine.
Status ReadSerializedEngine(
    const NodeDef& node, Session* session,
    TrtUniquePtrType<nvinfer1::IHostMemory>* engine_data) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_11(mht_11_v, 496, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "ReadSerializedEngine");

  TRTEngineCacheResource* resource;
  TF_RETURN_IF_ERROR(GetEngineCacheResource(node, session, &resource));
  core::ScopedUnref unref_cache_res(resource);
  if (resource->cache_.size() > 1) {
    return errors::Internal(
        "Multiple engines found, but we can only serialize one");
  }
  const std::unique_ptr<EngineContext>& engine =
      resource->cache_.begin()->second;
  if (!engine) {
    return errors::Internal("Engine not found for", node.name());
  }

  if (engine->cuda_engine) {
    // Serialize the engine.
    engine_data->reset(engine->cuda_engine->serialize());
  } else {
    LOG(WARNING) << "Engine cache contains nullptr";
  }

  return Status::OK();
}

// Saves the TRT engines as attributes of the TRTEngineOp nodes.
Status ConvertToStaticEngine(const GraphDef graph_def,
                             GraphDef* static_graph_def, Session* session) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_12(mht_12_v, 525, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "ConvertToStaticEngine");

  static_graph_def->CopyFrom(graph_def);
  VLOG(1) << "Saving TRT engines as static engine";
  std::string op{"TRTEngineOp"};
  for (auto& node : *(static_graph_def->mutable_node())) {
    if (!op.compare(node.op())) {
      VLOG(2) << "Saving TRT engine for " << node.name()
              << ", device: " << node.device();
      TrtUniquePtrType<nvinfer1::IHostMemory> engine_data;
      TF_RETURN_IF_ERROR(ReadSerializedEngine(node, session, &engine_data));
      auto* attr = node.mutable_attr();
      AttrValue static_engine;
      static_engine.set_b(true);
      AttrValue engine_string;
      if (engine_data) {
        engine_string.set_s(engine_data->data(), engine_data->size());
      }
      (*attr)["static_engine"] = static_engine;
      (*attr)["serialized_segment"] = engine_string;
    }
  }
  return Status::OK();
}

Status ValidateConversionParams(const TfTrtConversionParams& p, int n_inputs) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_13(mht_13_v, 552, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "ValidateConversionParams");

  if (p.precision_mode == TrtPrecisionMode::INT8 && p.use_calibration) {
    return errors::InvalidArgument(
        "Calibration not yet implemented through the C++ interface. Please use "
        "our Python API for calibration.");
  }
  if (p.convert_to_static_engine && n_inputs == 0) {
    return errors::InvalidArgument(
        "TRT Engine needs to be built before we can convert it to static "
        "engine. Please provide input data to build the model.");
  }
  if (!p.convert_to_static_engine && n_inputs >= 0) {
    // After the conversion, the session that was used to build the engines
    // will be destroyed. If we do not convert the engine to static engine,
    // then we loose the engines.
    //
    // TODO(tfeher): Provide a way to save dynamic engines and remove this
    // warning.
    LOG(WARNING)
        << "Skipping build mode because we cannot save the "
           "engines. Use convert_to_static_engines=true conversion "
           "parameter to enable build mode and save the engines in the graph.";
  }
  if (!p.allow_build_at_runtime && n_inputs == 0) {
    LOG(WARNING)
        << "TRT will not be used since allow_build_at_runtime is disabled and "
           "no inputs are provided to build during conversion.";
  }
  return Status::OK();
}

// Returns configuration used during the build step session run.
tensorflow::SessionOptions GetSessionConfg() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_14(mht_14_v, 587, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "GetSessionConfg");

  // We also need to disable constant folding because we already ran constant
  // folding and may have prevented quantization operation folding on purpose.
  tensorflow::SessionOptions opts;
  auto* rewriter_opts =
      opts.config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_opts->set_experimental_disable_folding_quantization_emulation(true);

  // It seems  that we need to disable the optimizer entirely to prevent the
  // folding.
  rewriter_opts->set_disable_meta_optimizer(true);
  return opts;
}

}  // namespace

StatusOr<GraphDef> ConvertAndBuild(
    const GraphDef& frozen_graph_def, const std::vector<string>& input_names,
    const std::vector<string>& output_names,
    const std::vector<std::vector<tensorflow::Tensor>>& inputs,
    const TfTrtConversionParams& conv_params) {
  TF_RETURN_IF_ERROR(ValidateConversionParams(conv_params, inputs.size()));
  MetaGraphDef meta_graph;
  meta_graph.mutable_graph_def()->CopyFrom(frozen_graph_def);

  RewriterConfig rewriter_config;
  TF_RETURN_IF_ERROR(
      GetTrtRewriterConfig(conv_params, frozen_graph_def, &rewriter_config));

  GraphDef segmented_graph_def;
  TF_RETURN_IF_ERROR(RunTfTrt(meta_graph, input_names, output_names,
                              rewriter_config, &segmented_graph_def));

  GraphDef output;

  if (inputs.size() > 0 && conv_params.convert_to_static_engine) {
    // The TRTOptimization pass has inserted placeholder TRTEngineOps. Here we
    // trigger conversion by inferring the graph.
    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(GetSessionConfg()));
    if (!session.get()) {
      return errors::Internal("Failed to create build session");
    }

    TF_RETURN_IF_ERROR(Build(segmented_graph_def, input_names, output_names,
                             inputs, session.get(), conv_params));

    TF_RETURN_IF_ERROR(
        ConvertToStaticEngine(segmented_graph_def, &output, session.get()));
  } else {
    output.CopyFrom(segmented_graph_def);
  }
  VLOG(1) << "TF-TRT conversion finished";
  return output;
}

Status InlineFunctions(const MetaGraphDef& meta_graph_def,
                       GraphDef* out_graph_def) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_15(mht_15_v, 647, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "InlineFunctions");

  ConfigProto config_proto;
  auto opt_config =
      config_proto.mutable_graph_options()->mutable_rewrite_options();

  opt_config->set_meta_optimizer_iterations(tensorflow::RewriterConfig::ONE);
  opt_config->set_min_graph_nodes(-1);  // do not skip small graphs
  opt_config->add_optimizers("function");

  TF_RETURN_IF_ERROR(RunGrappler(meta_graph_def, {}, {}, config_proto, nullptr,
                                 out_graph_def));

  VLOG(2) << "Graph is inlined";
  return Status::OK();
}

// Freezes the graph. It is assumed that the functions are inlined and the
// variables are initialized.
Status FreezeGraph(SavedModelBundle& bundle, MetaGraphDef* frozen_meta_graph) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_apiDTcc mht_16(mht_16_v, 668, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api.cc", "FreezeGraph");

  std::unordered_set<std::string> inputs;
  std::unordered_set<std::string> outputs;
  GraphDef frozen_graph_def;
  TF_RETURN_IF_ERROR(
      FreezeSavedModel(bundle, &frozen_graph_def, &inputs, &outputs));

  frozen_meta_graph->CopyFrom(bundle.meta_graph_def);
  GraphDef* gdef = frozen_meta_graph->mutable_graph_def();
  gdef->CopyFrom(frozen_graph_def);

  VLOG(2) << "Graph frozen";
  return Status::OK();
}

// Returns the name of nodes listed in the signature definition.
std::vector<std::string> GetNodeNames(
    const google::protobuf::Map<std::string, tensorflow::TensorInfo>& signature) {
  std::vector<std::string> names;
  for (auto const& item : signature) {
    absl::string_view name = item.second.name();
    // Remove tensor suffix like ":0".
    size_t last_colon = name.find_last_of(':');
    if (last_colon != absl::string_view::npos) {
      name.remove_suffix(name.size() - last_colon);
    }
    names.push_back(std::string(name));
  }
  return names;
}

StatusOr<GraphDef> ConvertAndBuild(
    SavedModelBundle* bundle, const std::string& signature_key,
    const std::vector<std::vector<tensorflow::Tensor>>& inputs,
    const TfTrtConversionParams& conversion_params) {
  // Inline the functions.
  GraphDef inlined_graph_def;
  TF_RETURN_IF_ERROR(
      InlineFunctions(bundle->meta_graph_def, &inlined_graph_def));

  // Replace the graph_def with the inlined graph. Note that bundle->session
  // still has the original graph.
  bundle->meta_graph_def.mutable_graph_def()->CopyFrom(inlined_graph_def);

  // Freeze variables.
  MetaGraphDef frozen_meta_graph;
  TF_RETURN_IF_ERROR(FreezeGraph(*bundle, &frozen_meta_graph));

  // Convert.
  auto signature_map = bundle->GetSignatures();
  const tensorflow::SignatureDef& signature = signature_map[signature_key];
  std::vector<std::string> input_names = GetNodeNames(signature.inputs());
  std::vector<std::string> output_names = GetNodeNames(signature.outputs());
  return ConvertAndBuild(frozen_meta_graph.graph_def(), input_names,
                         output_names, inputs, conversion_params);
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
