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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmeta_optimizerDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmeta_optimizerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmeta_optimizerDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/meta_optimizer.h"

#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace grappler {

namespace {

using ConfigMap =
    std::map<string, tensorflow::RewriterConfig_CustomGraphOptimizer>;

// tf.data optimizations, in the order we want to perform them.
constexpr std::array<const char*, 20> kTFDataOptimizations = {
    "noop_elimination",
    "disable_intra_op_parallelism",
    "use_private_thread_pool",
    "shuffle_and_repeat_fusion",
    "map_fusion",
    "filter_fusion",
    "map_and_filter_fusion",
    "map_parallelization",
    "map_and_batch_fusion",
    "batch_parallelization",
    "filter_parallelization",
    "make_sloppy",
    "parallel_batch",
    "slack",
    "autotune_buffer_sizes",
    "inject_prefetch_eligible",
    "inject_prefetch",
    "disable_prefetch_legacy_autotune",
    "enable_gradient_descent",
    "make_deterministic"};

// Parses a list of string optimizer configurations into a map from
// optimizer name -> rewriter config for that optimizer.
Status ToConfigMap(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config,
    ConfigMap* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmeta_optimizerDTcc mht_0(mht_0_v, 234, "", "./tensorflow/core/grappler/optimizers/data/meta_optimizer.cc", "ToConfigMap");

  auto found = gtl::FindOrNull(config->parameter_map(), "optimizer_configs");
  if (!found) return Status::OK();

  auto& options = found->list().s();
  for (const auto& option_string : options) {
    // The option string has the format
    // <optimizer_name>:<config_key>:<config_value>
    std::vector<string> split = absl::StrSplit(option_string, ':');
    if (split.size() != 3) {
      return errors::Internal(
          "Wrong format for optimizer options. Expect <optimizer name>:<config "
          "key>:<config value>, received: ",
          option_string);
    }

    const string& optimizer_name = split[0];
    const string& config_key = split[1];
    const string& config_value = split[2];

    auto optimizer_config = gtl::FindOrNull(*result, optimizer_name);
    if (!optimizer_config) {
      (*result)[optimizer_name] =
          tensorflow::RewriterConfig_CustomGraphOptimizer();
      optimizer_config = gtl::FindOrNull(*result, optimizer_name);
    }
    (*optimizer_config->mutable_parameter_map())[config_key].set_s(
        config_value);
  }

  return Status::OK();
}

}  // namespace

Status TFDataMetaOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                     GraphDef* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmeta_optimizerDTcc mht_1(mht_1_v, 273, "", "./tensorflow/core/grappler/optimizers/data/meta_optimizer.cc", "TFDataMetaOptimizer::Optimize");

  // Stores the optimized item so far.
  GrapplerItem optimized_item = item;

  // Perform optimizations in a meaningful order.
  for (const auto& optimization : kTFDataOptimizations) {
    tensorflow::metrics::ScopedCounter<2> timings(
        tensorflow::metrics::GetGraphOptimizationCounter(),
        {"TFData", optimization});
    Status status = ApplyOptimization(optimization, cluster, &optimized_item);
    timings.ReportAndStop();
    if (!status.ok()) return status;
  }

  // Store the final result of all the optimizations in `output`.
  output->Swap(&optimized_item.graph);

  // Optimize tf.data user-defined functions.
  FunctionLibraryDefinition flib =
      FunctionLibraryDefinition(OpRegistry::Global(), output->library())
          .ReachableDefinitions(*output);
  const auto producer = output->versions().producer();
  bool optimized_functions = false;
  for (const auto& name : flib.ListFunctionNames()) {
    auto* func = flib.Find(name);
    // Skip non tf.data functions.
    if (!data::IsTFDataFunction(*func)) continue;
    VLOG(3) << "Optimize function: function=" << func->signature().name();
    optimized_functions = true;

    // Make a GrapplerItem from a FunctionDef.
    GrapplerFunctionItem func_item;
    TF_RETURN_IF_ERROR(
        MakeGrapplerFunctionItem(*func, flib, producer, &func_item));

    GraphDef optimized_func_graph;
    TF_RETURN_IF_ERROR(Optimize(cluster, func_item, &optimized_func_graph));

    // Function body optimization might have created new functions. Add them to
    // the library.
    for (const FunctionDef& func_def :
         optimized_func_graph.library().function()) {
      if (flib.Find(func_def.signature().name()) == nullptr) {
        TF_RETURN_IF_ERROR(flib.AddFunctionDef(func_def));
      }
    }

    // Convert optimized graph back to FunctionDef.
    FunctionDef optimized_func;
    func_item.SwapFunctionBody(std::move(optimized_func_graph));
    TF_RETURN_IF_ERROR(MakeFunctionDef(func_item, flib, &optimized_func));

    // Replace optimized function with a new FunctionDef.
    TF_RETURN_IF_ERROR(
        flib.ReplaceFunction(func->signature().name(), optimized_func));
  }
  if (optimized_functions) {
    *output->mutable_library() = flib.ToProto();
  }
  return Status::OK();
}

Status TFDataMetaOptimizer::ApplyOptimization(const string& name,
                                              Cluster* cluster,
                                              GrapplerItem* item) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmeta_optimizerDTcc mht_2(mht_2_v, 341, "", "./tensorflow/core/grappler/optimizers/data/meta_optimizer.cc", "TFDataMetaOptimizer::ApplyOptimization");

  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  const auto* optimizer = gtl::FindOrNull(enabled_optimizers_, name);
  if (!optimizer) {
    return Status::OK();
  }

  GraphDef result;
  (*optimizer)->set_deadline_usec(this->deadline_usec());
  Status status = (*optimizer)->Optimize(cluster, *item, &result);
  if (status.ok()) {
    // The optimizer succeeded and wrote the optimized graph to result.
    item->graph.Swap(&result);
  } else if (errors::IsAborted(status)) {
    // A status of errors::Aborted just means that the optimizer was a no-op and
    // did not populate result. Swallow the error status and leave the original
    // graph in item.
    status = Status::OK();
  }

  return status;
}

Status TFDataMetaOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmeta_optimizerDTcc mht_3(mht_3_v, 369, "", "./tensorflow/core/grappler/optimizers/data/meta_optimizer.cc", "TFDataMetaOptimizer::Init");

  if (!config) return Status::OK();

  // Initialize custom tf.data optimizers based on config.
  auto& optimizers = config->parameter_map().at("optimizers").list().s();
  ConfigMap optimizer_configs;
  TF_RETURN_IF_ERROR(ToConfigMap(config, &optimizer_configs));

  for (const auto& optimizer_name : optimizers) {
    auto optimizer =
        CustomGraphOptimizerRegistry::CreateByNameOrNull(optimizer_name);
    if (optimizer) {
      TF_RETURN_IF_ERROR(
          optimizer->Init(gtl::FindOrNull(optimizer_configs, optimizer_name)));

      enabled_optimizers_[optimizer_name] = std::move(optimizer);
    } else {
      return errors::Internal(
          "Tried to register a dataset optimizer that doesn't exist: ",
          optimizer_name);
    }
  }

  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(TFDataMetaOptimizer, "tf_data_meta_optimizer");

}  // namespace grappler
}  // namespace tensorflow
