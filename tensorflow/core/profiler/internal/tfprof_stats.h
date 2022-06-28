/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

// Core API of tfprof.
// 1. Load protos generated from a tensorflow model.
// 2. Build in-memory representations of the tensorflow model, annotate the
//    representation with various stats, such as params,times,memory,etc.
// 3. Accept command and options to selectively aggregate stats for analysis
//    and print out the results.

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_STATS_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_STATS_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_statsDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_statsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_statsDTh() {
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


#include <map>
#include <memory>
#include <set>
#include <string>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/internal/tfprof_code.h"
#include "tensorflow/core/profiler/internal/tfprof_graph.h"
#include "tensorflow/core/profiler/internal/tfprof_node.h"
#include "tensorflow/core/profiler/internal/tfprof_op.h"
#include "tensorflow/core/profiler/internal/tfprof_scope.h"
#include "tensorflow/core/profiler/internal/tfprof_show.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"
#include "tensorflow/core/profiler/tfprof_options.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {

class TFStats {
 public:
  TFStats(std::unique_ptr<GraphDef> graph,
          std::unique_ptr<RunMetadata> run_meta,
          std::unique_ptr<OpLogProto> op_log,
          std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader);

  TFStats(const string& filename,
          std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader);

  ~TFStats() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_statsDTh mht_0(mht_0_v, 230, "", "./tensorflow/core/profiler/internal/tfprof_stats.h", "~TFStats");
}

  const std::map<string, std::unique_ptr<TFGraphNode>>& nodes() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_statsDTh mht_1(mht_1_v, 235, "", "./tensorflow/core/profiler/internal/tfprof_stats.h", "nodes");

    return nodes_map_;
  }
  const std::set<int64_t>& steps() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_statsDTh mht_2(mht_2_v, 241, "", "./tensorflow/core/profiler/internal/tfprof_stats.h", "steps");
 return steps_; }
  bool has_code_traces() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_statsDTh mht_3(mht_3_v, 245, "", "./tensorflow/core/profiler/internal/tfprof_stats.h", "has_code_traces");
 return has_code_traces_; }
  double run_coverage() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_statsDTh mht_4(mht_4_v, 249, "", "./tensorflow/core/profiler/internal/tfprof_stats.h", "run_coverage");

    return covered_nodes_.size() / (nodes_map_.size() + 1e-10);
  }

  void BuildView(const string& cmd);
  void BuildAllViews();

  // Note: Must first BuildView(view_foo) before ShowXXX(view_foo) methods.
  //
  // Organize the TensorFlow model as different types of views, and generate
  // outputs for profiling.
  // TODO(xpan): Should it return reference here?
  const GraphNodeProto& ShowGraphNode(const string& cmd,
                                      const Options& opts) const;
  const MultiGraphNodeProto& ShowMultiGraphNode(const string& cmd,
                                                const Options& opts) const;

  // Add a (partial) graph to existing graph.
  void AddGraph(std::unique_ptr<GraphDef> graph);

  // Add a step of run time meta data.
  void AddRunMeta(int64_t step, std::unique_ptr<RunMetadata> run_meta);
  // Add tfprof operation meta data, such as customized op type, float_ops,
  // and code traces.
  void AddOpLogProto(std::unique_ptr<OpLogProto> op_log);

  void SerializeToString(string* content);
  void WriteProfile(const string& filename);

  // For test purpose only.
  void AddNodeForTest(int64_t step, std::unique_ptr<TFGraphNode> node);

 private:
  bool Validate(const Options& opts) const;
  string MaybeReportMissingTrace() const;

  std::set<int64_t> steps_;
  bool has_code_traces_;
  bool miss_accelerator_stream_;
  std::unique_ptr<TFScope> scope_view_;
  std::unique_ptr<TFGraph> graph_view_;
  std::unique_ptr<TFCode> code_view_;
  std::unique_ptr<TFOp> op_view_;
  std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader_;
  // TODO(xpan): Store TFGraphNode instead of TFGraphNode* to avoid large
  // number of dynamic alloc.
  // Maps from graph node name to TFGraphNode.
  std::map<string, std::unique_ptr<TFGraphNode>> nodes_map_;
  GraphNodeProto empty_graph_node_;
  MultiGraphNodeProto empty_multi_graph_node_;

  std::map<int64_t, string> id_to_string_;
  // Graph nodes covered by RunMetadata, that is traced with run time stats.
  std::set<int64_t> covered_nodes_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_STATS_H_
