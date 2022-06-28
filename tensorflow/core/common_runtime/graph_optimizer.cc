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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_optimizerDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_optimizerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_optimizerDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/graph_optimizer.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/optimizer_cse.h"

namespace tensorflow {

GraphOptimizer::GraphOptimizer(const OptimizerOptions& opts) : opts_(opts) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_optimizerDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/common_runtime/graph_optimizer.cc", "GraphOptimizer::GraphOptimizer");

  if (opts_.opt_level() >= OptimizerOptions::L1) {
    opts_.set_do_common_subexpression_elimination(true);
    opts_.set_do_constant_folding(true);
  }
}

GraphOptimizer::~GraphOptimizer() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_optimizerDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/common_runtime/graph_optimizer.cc", "GraphOptimizer::~GraphOptimizer");
}

void GraphOptimizer::Optimize(FunctionLibraryRuntime* runtime, Env* env,
                              const Device* device,
                              std::unique_ptr<Graph>* graph,
                              const Options& options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_optimizerDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/common_runtime/graph_optimizer.cc", "GraphOptimizer::Optimize");

  static const char* kGraphOptimizerCategory = "GraphOptimizerPass";

  Graph* g = graph->get();
  DumpGraph("Initial", g);
  bool changed = true;
  const int kMaxRounds = 10;
  for (int rounds = 0; rounds < kMaxRounds; ++rounds) {
    changed = false;
    if (RemoveListArrayConverter(g)) {
      DumpGraph("RemoveListArrayConverter", g);
      changed = true;
    }

    tensorflow::metrics::ScopedCounter<2> inlining_timings(
        tensorflow::metrics::GetGraphOptimizationCounter(),
        {kGraphOptimizerCategory, "function_inlining"});
    if (opts_.do_function_inlining() && RemoveDeadNodes(g)) {
      DumpGraph("RemoveDeadNodes", g);
      changed = true;
    }
    if (opts_.do_function_inlining() && RemoveIdentityNodes(g)) {
      DumpGraph("RemoveIdentityNodes", g);
      changed = true;
    }
    if (opts_.do_function_inlining()) {
      inlining_timings.AccumulateAndStop();
    }

    if (opts_.do_constant_folding()) {
      tensorflow::metrics::ScopedCounter<2> timings(
          tensorflow::metrics::GetGraphOptimizationCounter(),
          {kGraphOptimizerCategory, "constant_folding"});

      ConstantFoldingOptions cf_opts;
      cf_opts.shape_map = options.shape_map;
      cf_opts.consider = options.cf_consider_fn;
      if (opts_.max_folded_constant_in_bytes() > 0) {
        cf_opts.max_constant_size_in_bytes =
            opts_.max_folded_constant_in_bytes();
      }
      bool was_mutated;
      ConstantFold(cf_opts, runtime, env, device, g, &was_mutated)
          .IgnoreError();
      if (was_mutated) {
        RemoveDeadNodes(g);
        DumpGraph("ConstFolding", g);
        changed = true;
      }
    }

    if (opts_.do_function_inlining()) {
      inlining_timings.Start();
      if (FixupSourceAndSinkEdges(g)) {
        DumpGraph("FixupSourceAndSinkEdges", g);
        changed = true;
      }
      inlining_timings.AccumulateAndStop();
    }

    if (opts_.do_common_subexpression_elimination()) {
      tensorflow::metrics::ScopedCounter<2> timings(
          tensorflow::metrics::GetGraphOptimizationCounter(),
          {kGraphOptimizerCategory, "common_subexpression_elimination"});
      if (OptimizeCSE(g, options.cse_consider_fn)) {
        DumpGraph("OptimizeCSE", g);
        changed = true;
      }
    }
    if (opts_.do_function_inlining()) {
      inlining_timings.Start();
      ExpandInlineFunctionsOptions expand_inline_opts;
      expand_inline_opts.native_options.inlined_function_body_placer =
          InlinedFunctionBodyPlacer::SingleDevice();

      // Force single device placement strategy for multi-device function body.
      if (options.inline_with_single_device_body_placer) {
        expand_inline_opts.multi_device_options.inlined_function_body_placer =
            InlinedFunctionBodyPlacer::SingleDevice();
      }

      if (!options.inline_multi_device_functions) {
        // GraphOptimizer is running:
        //   (1) After partitioning when executing with a Session API.
        //   (2) For a single device function body after instantiation.
        // We can't inline multi-device functions in these cases, because it
        // might lead to multiple device assignments.
        expand_inline_opts.multi_device_options.disable_inlining = true;
      }
      if (options.inline_impl_selection_group_functions) {
        expand_inline_opts.native_options
            .inline_impl_selection_group_functions = true;
        expand_inline_opts.multi_device_options
            .inline_impl_selection_group_functions = true;
      }

      if (options.ignore_noinline) {
        expand_inline_opts.multi_device_options.ignore_noinline = true;
        expand_inline_opts.native_options.ignore_noinline = true;
      }

      bool was_mutated = ExpandInlineFunctions(runtime, g, expand_inline_opts);
      if (was_mutated) {
        DumpGraph("ExpandInlineFunctions", g);
        changed = true;
      }

      inlining_timings.ReportAndStop();
    }
    if (!changed) break;
  }

  // Clone the graph to copy the input FunctionLibraryDefinition, since the
  // original lib def will go out of scope.
  *graph = g->Clone();

  DumpGraph("ReCopy", graph->get());
}

void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g,
                   const GraphOptimizer::Options& graph_optimizer_options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_optimizerDTcc mht_3(mht_3_v, 339, "", "./tensorflow/core/common_runtime/graph_optimizer.cc", "OptimizeGraph");

  OptimizerOptions opts;
  opts.set_do_common_subexpression_elimination(true);
  opts.set_do_function_inlining(true);
  opts.set_do_constant_folding(true);
  GraphOptimizer optimizer(opts);
  optimizer.Optimize(lib, lib->env(), lib->device(), g,
                     graph_optimizer_options);
}

void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_optimizerDTcc mht_4(mht_4_v, 352, "", "./tensorflow/core/common_runtime/graph_optimizer.cc", "OptimizeGraph");

  OptimizeGraph(lib, g, GraphOptimizer::Options());
}

}  // end namespace tensorflow
