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
class MHTracer_DTPStensorflowPScorePSgraphPScostmodel_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPScostmodel_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPScostmodel_testDTcc() {
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

#include "tensorflow/core/graph/costmodel.h"

#include <memory>
#include <string>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

static void InitGraph(const string& s, Graph* graph) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPScostmodel_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/graph/costmodel_test.cc", "InitGraph");

  GraphDef graph_def;

  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(s, &graph_def)) << s;
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
}

static void GenerateStepStats(Graph* graph, StepStats* step_stats,
                              const string& device_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPScostmodel_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/graph/costmodel_test.cc", "GenerateStepStats");

  // Fill RunMetadata's step_stats and partition_graphs fields.
  DeviceStepStats* device_stepstats = step_stats->add_dev_stats();
  device_stepstats->set_device(device_name);
  for (const auto& node_def : graph->nodes()) {
    NodeExecStats* node_stats = device_stepstats->add_node_stats();
    node_stats->set_node_name(node_def->name());
  }
}

REGISTER_OP("Input").Output("o: float").SetIsStateful();

TEST(CostModelTest, GlobalId) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  std::unique_ptr<Graph> graph1 =
      std::unique_ptr<Graph>(new Graph(OpRegistry::Global()));
  std::unique_ptr<Graph> graph2 =
      std::unique_ptr<Graph>(new Graph(OpRegistry::Global()));
  InitGraph(
      "node { name: 'A1' op: 'Input'}"
      "node { name: 'B1' op: 'Input'}"
      "node { name: 'C1' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A1', 'B1'] }"
      "node { name: 'D1' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A1', 'B1'] }",
      graph1.get());
  InitGraph(
      "node { name: 'A2' op: 'Input'}"
      "node { name: 'B2' op: 'Input'}"
      "node { name: 'C2' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A2', 'B2'] }"
      "node { name: 'D2' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A2', 'B2'] }",
      graph2.get());
  StepStats step_stats;
  GenerateStepStats(graph1.get(), &step_stats, "DummyDevice1");
  GenerateStepStats(graph2.get(), &step_stats, "DummyDevice2");
  StepStatsCollector collector(&step_stats);
  std::unordered_map<string, const Graph*> device_map;
  device_map["DummyDevice1"] = graph1.get();
  device_map["DummyDevice2"] = graph2.get();
  CostModelManager cost_model_manager;
  collector.BuildCostModel(&cost_model_manager, device_map);
  CostGraphDef cost_graph_def;
  TF_ASSERT_OK(
      cost_model_manager.AddToCostGraphDef(graph1.get(), &cost_graph_def));
  TF_ASSERT_OK(
      cost_model_manager.AddToCostGraphDef(graph2.get(), &cost_graph_def));
  ASSERT_EQ(cost_graph_def.node_size(), 12);
  absl::flat_hash_map<int32, const CostGraphDef::Node> ids;
  for (auto node : cost_graph_def.node()) {
    int32_t index = node.id();
    auto result = ids.insert({index, node});
    EXPECT_TRUE(result.second);
  }
}

}  // namespace
}  // namespace tensorflow
