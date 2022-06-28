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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPStrt_optimization_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPStrt_optimization_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPStrt_optimization_passDTcc() {
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

#include "tensorflow/compiler/tf2tensorrt/convert/trt_optimization_pass.h"

#include <memory>

#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {
namespace convert {
using absl::AsciiStrToUpper;
using absl::StrAppend;
using absl::StrCat;

namespace {

bool ShouldUseExplicitPrecision(const GraphDef& gdef) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPStrt_optimization_passDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/tf2tensorrt/convert/trt_optimization_pass.cc", "ShouldUseExplicitPrecision");

  if (!IS_TRT_VERSION_GE(8, 0, 0, 0)) {
    return false;
  }
  return absl::c_any_of(gdef.node(), [](const auto& node) {
    return (absl::c_find(kExplicitQuantizationOpNames, node.op()) !=
            kExplicitQuantizationOpNames.end());
  });
}

StatusOr<bool> ShouldConvertFunction(const grappler::GrapplerItem& item) {
  if (item.id == "tf_graph") {
    return false;
  }
  const auto& func_item =
      tensorflow::down_cast<const grappler::GrapplerFunctionItem&>(item);
  const AttrSlice& attr = func_item.func_attr();
  const AttrValue* attr_value = attr.FindByString("_tftrt_convert_function");
  if (attr_value != nullptr) {
    bool result = false;
    TF_RETURN_IF_ERROR(GetNodeAttr(attr, "_tftrt_convert_function", &result));
    return result;
  }
  VLOG(1) << "Attribute _tftrt_convert_function was not found.";
  return false;
}

// Converts function conversion attributes to conversion parameters.
Status UpdateFunctionSpecificConversionParams(
    TRTOptimizationPass::ConversionParams& cp,
    const tensorflow::AttrSlice& attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPStrt_optimization_passDTcc mht_1(mht_1_v, 251, "", "./tensorflow/compiler/tf2tensorrt/convert/trt_optimization_pass.cc", "UpdateFunctionSpecificConversionParams");

  auto get_size_attr = [](const AttrSlice& attr, absl::string_view name,
                          size_t* dst) -> Status {
    int tmp = 0;
    TF_RETURN_IF_ERROR(GetNodeAttr(attr, name, &tmp));
    *dst = static_cast<size_t>(tmp);
    return Status::OK();
  };

  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_trt_logger_name", &cp.trt_logger_name));
  TF_RETURN_IF_ERROR(
      get_size_attr(attr, "_tftrt_max_batch_size", &cp.max_batch_size));
  TF_RETURN_IF_ERROR(get_size_attr(attr, "_tftrt_max_workspace_size_bytes",
                                   &cp.max_workspace_size_bytes));
  std::string precision_mode;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_precision_mode", &precision_mode));
  TF_RETURN_IF_ERROR(
      TrtPrecisionModeFromName(precision_mode, &cp.precision_mode));
  TF_RETURN_IF_ERROR(GetNodeAttr(attr, "_tftrt_minimum_segment_size",
                                 &cp.minimum_segment_size));
  TF_RETURN_IF_ERROR(GetNodeAttr(attr, "_tftrt_is_dyn_op", &cp.is_dynamic_op));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_max_cached_engines", &cp.max_cached_engines));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_use_calibration", &cp.use_calibration));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_use_implicit_batch", &cp.use_implicit_batch));
  std::string profile_strategy;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_profile_strategy", &profile_strategy));
  TF_RETURN_IF_ERROR(
      ProfileStrategyFromName(profile_strategy, &cp.profile_strategy));
  TF_RETURN_IF_ERROR(GetNodeAttr(attr, "_tftrt_allow_build_at_runtime",
                                 &cp.allow_build_at_runtime));
  return Status::OK();
}
}  // namespace

Status TRTOptimizationPass::Init(
    const RewriterConfig_CustomGraphOptimizer* config) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPStrt_optimization_passDTcc mht_2(mht_2_v, 295, "", "./tensorflow/compiler/tf2tensorrt/convert/trt_optimization_pass.cc", "TRTOptimizationPass::Init");

  if (config == nullptr) {
    return Status::OK();
  }
  const auto params = config->parameter_map();
  if (params.count("minimum_segment_size")) {
    params_.minimum_segment_size = params.at("minimum_segment_size").i();
  }
  if (params.count("max_batch_size")) {
    params_.max_batch_size = params.at("max_batch_size").i();
  }
  if (params.count("is_dynamic_op")) {
    params_.is_dynamic_op = params.at("is_dynamic_op").b();
  }
  if (params.count("maximum_cached_engines")) {
    params_.max_cached_engines = params.at("maximum_cached_engines").i();
  }
  if (params.count("max_workspace_size_bytes")) {
    params_.max_workspace_size_bytes =
        params.at("max_workspace_size_bytes").i();
  }
  if (params.count("precision_mode")) {
    TF_RETURN_IF_ERROR(TrtPrecisionModeFromName(
        AsciiStrToUpper(params.at("precision_mode").s()),
        &params_.precision_mode));
  }
  if (params.count("use_calibration")) {
    params_.use_calibration = params.at("use_calibration").b();
  }
  if (params.count("trt_logger")) {
    params_.trt_logger_name = params.at("trt_logger").s();
  }
  if (params.count("allow_build_at_runtime")) {
    params_.allow_build_at_runtime = params.at("allow_build_at_runtime").b();
  }
  if (params.count("use_implicit_batch")) {
    params_.use_implicit_batch = params.at("use_implicit_batch").b();
  }
  if (params.count("profile_strategy")) {
    TF_RETURN_IF_ERROR(ProfileStrategyFromName(
        params.at("profile_strategy").s(), &params_.profile_strategy));
  }
  return Status::OK();
}

static bool ExplicitPrecisionModePolicy() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPStrt_optimization_passDTcc mht_3(mht_3_v, 343, "", "./tensorflow/compiler/tf2tensorrt/convert/trt_optimization_pass.cc", "ExplicitPrecisionModePolicy");

  return IS_TRT_VERSION_GE(8, 0, 0, 0);
}

Status TRTOptimizationPass::Optimize(grappler::Cluster* cluster,
                                     const grappler::GrapplerItem& item,
                                     GraphDef* optimized_graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPStrt_optimization_passDTcc mht_4(mht_4_v, 352, "", "./tensorflow/compiler/tf2tensorrt/convert/trt_optimization_pass.cc", "TRTOptimizationPass::Optimize");

  VLOG(1) << "Called TRTOptimization Pass " << name_
          << " on a grappler item with id=" << item.id;
  TF_ASSIGN_OR_RETURN(bool do_function_conversion, ShouldConvertFunction(item));
  // Optimizing the main graph(identified with `item.id == "tf_graph"`) with
  // `minimim_segment_size == -1` indicates skipping main graph conversion.
  if ((params_.minimum_segment_size == -1 && item.id == "tf_graph") ||
      (item.id != "tf_graph" && !do_function_conversion)) {
    VLOG(1) << "Not optimizing this grappler item: " << item.id;
    *optimized_graph = item.graph;
    return Status::OK();
  }

  if (params_.use_calibration &&
      params_.precision_mode != TrtPrecisionMode::INT8) {
    LOG(WARNING) << "Calibration with FP32 or FP16 is not implemented. "
                 << "Falling back to use_calibration = False."
                 << "Note that the default value of use_calibration is True.";
    params_.use_calibration = false;
  }

  params_.use_explicit_precision = ShouldUseExplicitPrecision(item.graph);
  if (params_.use_explicit_precision) {
    LOG(INFO) << "[TF-TRT] Using explicit QDQ mode";
    if (params_.precision_mode != TrtPrecisionMode::INT8 ||
        params_.use_calibration) {
      LOG(WARNING)
          << "Explicit precision mode with calibration or FP32/FP16 mode is "
             "not supported."
          << " Setting precision mode to INT8 and calibration to false.";
      params_.precision_mode = TrtPrecisionMode::INT8;
      params_.use_calibration = false;
    }
  }

  // Create a copy of the graph to optimize.
  grappler::GrapplerItem optimized_item(item);

  std::vector<string> nodes_to_preserve;
  const auto& old_nodes_to_preserve = item.NodesToPreserve();
  nodes_to_preserve.reserve(old_nodes_to_preserve.size());
  for (const auto& n : old_nodes_to_preserve) {
    auto tokens = str_util::Split(n, ":");
    string s = tokens.at(0);
    for (int i = 1; i < tokens.size() - 1; ++i) {
      StrAppend(&s, ":", tokens.at(i));
    }
    int dumm_port = -1;
    // If the last token is not an integer, it must be part of the name.
    // Otherwise it is port number.
    if (tokens.size() > 1 &&
        !strings::safe_strto32(tokens.back(), &dumm_port)) {  // non-absl ok
      StrAppend(&s, ":", tokens.back());
    }
    nodes_to_preserve.push_back(s);
  }

  if (item.id != "tf_graph" && do_function_conversion) {
    const grappler::GrapplerFunctionItem& func_item =
        tensorflow::down_cast<const grappler::GrapplerFunctionItem&>(item);
    TF_RETURN_IF_ERROR(
        UpdateFunctionSpecificConversionParams(params_, func_item.func_attr()));
  }

  return ConvertGraph(params_, optimized_item, nodes_to_preserve, cluster,
                      optimized_graph);
}

static grappler::CustomGraphOptimizerRegistrar TRTOptimizationPass_Registrar(
    []() {
      VLOG(1)
          << "Instantiating CustomOptimizationPass object TensorRTOptimizer";
      return new TRTOptimizationPass("TensorRTOptimizer");
    },
    ("TensorRTOptimizer"));

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
