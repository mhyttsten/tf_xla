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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_toolDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_toolDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_toolDTcc() {
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

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/graph_analyzer/graph_analyzer.h"
#include "tensorflow/core/grappler/utils/transitive_fanin.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

// Dies on failure.
static void LoadModel(const string& filename,
                      tensorflow::MetaGraphDef* metagraph) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_toolDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_tool.cc", "LoadModel");

  LOG(INFO) << "Loading model from " << filename;
  Status st;
  st = ReadBinaryProto(Env::Default(), filename, metagraph);
  if (!st.ok()) {
    LOG(WARNING) << "Failed to read a binary metagraph: " << st;
    st = ReadTextProto(Env::Default(), filename, metagraph);
    if (!st.ok()) {
      LOG(FATAL) << "Failed to read a text metagraph: " << st;
    }
  }
}

// Prune the graph to only keep the transitive fanin part with respect to a set
// of train ops (if provided).
void MaybePruneGraph(const tensorflow::MetaGraphDef& metagraph,
                     tensorflow::GraphDef* graph) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_toolDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_tool.cc", "MaybePruneGraph");

  std::vector<string> fetch_nodes;
  for (const auto& fetch :
       metagraph.collection_def().at("train_op").node_list().value()) {
    LOG(INFO) << "Fetch node: " << fetch;
    fetch_nodes.push_back(fetch);
  }
  if (fetch_nodes.empty()) {
    *graph = metagraph.graph_def();
  } else {
    std::vector<const tensorflow::NodeDef*> fanin_nodes;
    TF_CHECK_OK(tensorflow::grappler::ComputeTransitiveFanin(
        metagraph.graph_def(), fetch_nodes, &fanin_nodes));
    for (const tensorflow::NodeDef* node : fanin_nodes) {
      *(graph->add_node()) = *node;
    }
    LOG(INFO) << "Pruned "
              << metagraph.graph_def().node_size() - graph->node_size()
              << " nodes. Original graph size: "
              << metagraph.graph_def().node_size()
              << ". New graph size: " << graph->node_size() << ".";
  }
}

void GraphAnalyzerTool(const string& file_name, int n) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_toolDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_tool.cc", "GraphAnalyzerTool");

  if (n < 1) {
    LOG(FATAL) << "Invalid subgraph size " << n << ", must be at least 1";
  }

  tensorflow::MetaGraphDef metagraph;
  LoadModel(file_name, &metagraph);
  tensorflow::GraphDef graph;
  MaybePruneGraph(metagraph, &graph);
  tensorflow::grappler::graph_analyzer::GraphAnalyzer analyzer(graph, n);
  LOG(INFO) << "Running the analysis";
  tensorflow::Status st = analyzer.Run();
  if (!st.ok()) {
    LOG(FATAL) << "Analysis failed: " << st;
  }

  LOG(INFO) << "Printing the result";
  st = analyzer.OutputSubgraphs();
  if (!st.ok()) {
    LOG(FATAL) << "Failed to print the result: " << st;
  }

  LOG(INFO) << "Completed";
}

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
