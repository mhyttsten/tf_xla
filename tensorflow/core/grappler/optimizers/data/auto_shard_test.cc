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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSauto_shard_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSauto_shard_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSauto_shard_testDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/auto_shard.h"

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using ::tensorflow::grappler::graph_tests_utils::MakeBatchV2Node;
using ::tensorflow::grappler::graph_tests_utils::MakeMapAndBatchNode;
using ::tensorflow::grappler::graph_tests_utils::MakeParallelBatchNode;
using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;
using ::testing::UnorderedElementsAre;

// Adds a MapDataset, a RebatchDataset, a PrefetchDataset and a fake sink that
// are common to all graphs; and sets the fetch node to the fake sink.
void FinishItem(GrapplerItem* item, const string& input_node_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input_node_name: \"" + input_node_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSauto_shard_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/grappler/optimizers/data/auto_shard_test.cc", "FinishItem");

  *item->graph.add_node() =
      NDef("map_before_rebatch", "MapDataset", {input_node_name},
           {{"f", "__inference_Dataset_map_normalize_8232"},
            {"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}});
  *item->graph.add_node() =
      NDef("num_replicas", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}});
  *item->graph.add_node() =
      NDef("rebatch", "RebatchDataset", {"map_before_rebatch", "num_replicas"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}});
  *item->graph.add_node() =
      NDef("prefetch_count", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}});
  *item->graph.add_node() =
      NDef("prefetch", "PrefetchDataset", {"rebatch", "prefetch_count"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}});
  *item->graph.add_node() = NDef("Sink", "Identity", {"prefetch"}, {});
  item->fetch.push_back("Sink");
}

NodeDef AddCardinalityAttr(NodeDef node, int64_t cardinality) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSauto_shard_testDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/grappler/optimizers/data/auto_shard_test.cc", "AddCardinalityAttr");

  (*node.mutable_attr())[data::kCardinalityAttrForRewrite].set_i(cardinality);
  return node;
}

TEST(RewriteBatchTest, InfiniteSource) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("repeat_count", "Const", {}, {{"value", -1}, {"dtype", DT_INT32}}),
      NDef("repeat", "RepeatDataset", {"tf_record", "repeat_count"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "repeat", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kInfiniteCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

TEST(RewriteBatchTest, InfiniteSourceMapAndBatch) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("repeat_count", "Const", {}, {{"value", -1}, {"dtype", DT_INT32}}),
      NDef("repeat", "RepeatDataset", {"tf_record", "repeat_count"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("num_parallel_calls", "Const", {},
           {{"value", 2}, {"dtype", DT_INT64}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeMapAndBatchNode("batch", "repeat", "batch_size",
                              "num_parallel_calls", "drop_remainder"),
          data::kInfiniteCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

TEST(RewriteBatchTest, InfiniteSourceParallelBatch) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("repeat_count", "Const", {}, {{"value", -1}, {"dtype", DT_INT32}}),
      NDef("repeat", "RepeatDataset", {"tf_record", "repeat_count"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("num_parallel_calls", "Const", {},
           {{"value", 2}, {"dtype", DT_INT64}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeParallelBatchNode("batch", "repeat", "batch_size",
                                "num_parallel_calls", "drop_remainder",
                                /*deterministic=*/"true"),
          data::kInfiniteCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

TEST(RewriteBatchTest, FiniteSourceNoDropRemainder) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", false}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kUnknownCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

TEST(RewriteBatchTest, FiniteSourceDropRemainder) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          /*cardinality=*/1337),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_DROP_REMAINDER_NOT_INFINITE"));
}

TEST(RewriteBatchTest, UnknownCardinalitySourceDropRemainder) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kUnknownCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_DROP_REMAINDER_NOT_INFINITE"));
}

TEST(RewriteBatchTest, FiniteSourceDropRemainderUnknown) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "RandomBool", {}, {}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kUnknownCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_DROP_REMAINDER_UNKNOWN"));
}

TEST(RewriteBatchTest, DropRemainderCardinalityNotAvailable) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {}, {{"value", true}}),
      MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                      /*parallel_copy=*/false),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_CARDINALITY_NOT_AVAILABLE"));
}

TEST(RewriteBatchTest, OpNotSupported) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kUnknownCardinality),
      NDef("take_count", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      graph_tests_utils::MakeTakeNode("take", "batch", "take_count"),
  });
  FinishItem(&item, "take");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("OP_NOT_SUPPORTED_TakeDataset",
                                   "BATCH_DROP_REMAINDER_NOT_INFINITE"));
}

TEST(RewriteBatchTest, BatchNotFound) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      graph_tests_utils::MakeTakeNode("take", "tf_record", "take_count"),
      NDef("take_count", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
  });
  FinishItem(&item, "take");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason, UnorderedElementsAre("BATCH_NOT_FOUND"));
}

// This is a very rare case (OneDeviceStrategy).
TEST(RewriteBatchTest, InfiniteSourceNoRebatch) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("repeat_count", "Const", {}, {{"value", -1}, {"dtype", DT_INT32}}),
      NDef("repeat", "RepeatDataset", {"tf_record", "repeat_count"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "repeat", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kInfiniteCardinality),
      NDef("Sink", "Identity", {"batch"}, {}),
  });
  item.fetch.push_back("Sink");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
