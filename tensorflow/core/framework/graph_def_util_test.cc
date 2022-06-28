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
class MHTracer_DTPStensorflowPScorePSframeworkPSgraph_def_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_def_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSgraph_def_util_testDTcc() {
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

#include "tensorflow/core/framework/graph_def_util.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

Status FinalizeOpDef(const OpDefBuilder& b, OpDef* op_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_def_util_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/framework/graph_def_util_test.cc", "FinalizeOpDef");

  OpRegistrationData op_reg_data;
  const Status s = b.Finalize(&op_reg_data);
  *op_def = op_reg_data.op_def;
  return s;
}

// We can create a Graph containing a namespaced Op
TEST(AddToGraphTest, MakeGraphDefWithNamespacedOpName) {
  OpList op_list;
  TF_ASSERT_OK(FinalizeOpDef(OpDefBuilder("Project>SomeOp"), op_list.add_op()));
  OpListOpRegistry registry(&op_list);

  GraphDef graph_def;
  TF_ASSERT_OK(NodeDefBuilder("node", "Project>SomeOp", &registry)
                   .Finalize(graph_def.add_node()));
}

// Producer and consumer have default for an attr -> graph unchanged.
TEST(RemoveNewDefaultAttrsFromGraphDefTest, NoChangeWithDefault) {
  OpList op_list;
  TF_ASSERT_OK(
      FinalizeOpDef(OpDefBuilder("NoChangeWithDefault").Attr("a: int = 12"),
                    op_list.add_op()));
  OpListOpRegistry registry(&op_list);

  GraphDef graph_def;
  TF_ASSERT_OK(NodeDefBuilder("ncwd", "NoChangeWithDefault", &registry)
                   .Finalize(graph_def.add_node()));
  GraphDef expected_graph_def = graph_def;

  std::set<std::pair<string, string>> op_attr_removed;
  TF_ASSERT_OK(RemoveNewDefaultAttrsFromGraphDef(&graph_def, registry, registry,
                                                 &op_attr_removed));

  TF_EXPECT_GRAPH_EQ(expected_graph_def, graph_def);
  EXPECT_TRUE(op_attr_removed.empty());
}

// Producer and consumer both have an attr -> graph unchanged.
TEST(RemoveNewDefaultAttrsFromGraphDefTest, NoChangeNoDefault) {
  OpList op_list;
  TF_ASSERT_OK(FinalizeOpDef(OpDefBuilder("NoChangeNoDefault").Attr("a: int"),
                             op_list.add_op()));
  OpListOpRegistry registry(&op_list);

  GraphDef graph_def;
  TF_ASSERT_OK(NodeDefBuilder("ncnd", "NoChangeNoDefault", &registry)
                   .Attr("a", 42)
                   .Finalize(graph_def.add_node()));
  GraphDef expected_graph_def = graph_def;

  std::set<std::pair<string, string>> op_attr_removed;
  TF_ASSERT_OK(RemoveNewDefaultAttrsFromGraphDef(&graph_def, registry, registry,
                                                 &op_attr_removed));

  TF_EXPECT_GRAPH_EQ(expected_graph_def, graph_def);
  EXPECT_TRUE(op_attr_removed.empty());
}

// Producer has default for an attr that the consumer does not know
// about, and the produced graph has the default value for the attr ->
// attr removed from graph (and so able to be consumed).
TEST(RemoveNewDefaultAttrsFromGraphDefTest, UsesDefault) {
  OpList consumer_op_list;
  TF_ASSERT_OK(
      FinalizeOpDef(OpDefBuilder("UsesDefault"), consumer_op_list.add_op()));
  OpListOpRegistry consumer_registry(&consumer_op_list);

  OpList producer_op_list;
  TF_ASSERT_OK(FinalizeOpDef(OpDefBuilder("UsesDefault").Attr("a: int = 17"),
                             producer_op_list.add_op()));
  OpListOpRegistry producer_registry(&producer_op_list);

  GraphDef produced_graph_def;
  TF_ASSERT_OK(NodeDefBuilder("uses_default", "UsesDefault", &producer_registry)
                   .Finalize(produced_graph_def.add_node()));

  std::set<std::pair<string, string>> op_attr_removed;
  TF_ASSERT_OK(
      RemoveNewDefaultAttrsFromGraphDef(&produced_graph_def, consumer_registry,
                                        producer_registry, &op_attr_removed));

  GraphDef expected_graph_def;
  TF_ASSERT_OK(NodeDefBuilder("uses_default", "UsesDefault", &consumer_registry)
                   .Finalize(expected_graph_def.add_node()));
  TF_EXPECT_GRAPH_EQ(expected_graph_def, produced_graph_def);

  std::set<std::pair<string, string>> expected_removed({{"UsesDefault", "a"}});
  EXPECT_EQ(expected_removed, op_attr_removed);
}

// Producer has default for an attr that the consumer does not know
// about, graph sets the attr to a value different from the default ->
// graph unchanged (but not able to be consumed by consumer).
TEST(RemoveNewDefaultAttrsFromGraphDefTest, ChangedFromDefault) {
  OpList consumer_op_list;
  TF_ASSERT_OK(FinalizeOpDef(OpDefBuilder("ChangedFromDefault"),
                             consumer_op_list.add_op()));
  OpListOpRegistry consumer_registry(&consumer_op_list);

  OpList producer_op_list;
  TF_ASSERT_OK(
      FinalizeOpDef(OpDefBuilder("ChangedFromDefault").Attr("a: int = 17"),
                    producer_op_list.add_op()));
  OpListOpRegistry producer_registry(&producer_op_list);

  GraphDef produced_graph_def;
  TF_ASSERT_OK(NodeDefBuilder("changed_from_default", "ChangedFromDefault",
                              &producer_registry)
                   .Attr("a", 9)
                   .Finalize(produced_graph_def.add_node()));
  GraphDef expected_graph_def = produced_graph_def;

  std::set<std::pair<string, string>> op_attr_removed;
  TF_ASSERT_OK(
      RemoveNewDefaultAttrsFromGraphDef(&produced_graph_def, consumer_registry,
                                        producer_registry, &op_attr_removed));

  TF_EXPECT_GRAPH_EQ(expected_graph_def, produced_graph_def);
  EXPECT_TRUE(op_attr_removed.empty());
}

// Attrs starting with underscores should not be removed.
TEST(RemoveNewDefaultAttrsFromGraphDefTest, UnderscoreAttrs) {
  OpList consumer_op_list;
  TF_ASSERT_OK(
      FinalizeOpDef(OpDefBuilder("Underscore"), consumer_op_list.add_op()));
  OpListOpRegistry consumer_registry(&consumer_op_list);

  OpList producer_op_list;
  TF_ASSERT_OK(
      FinalizeOpDef(OpDefBuilder("Underscore"), producer_op_list.add_op()));
  // Add the _underscore attr manually since OpDefBuilder would complain
  OpDef::AttrDef* attr = producer_op_list.mutable_op(0)->add_attr();
  attr->set_name("_underscore");
  attr->set_type("int");
  attr->mutable_default_value()->set_i(17);
  OpListOpRegistry producer_registry(&producer_op_list);

  GraphDef produced_graph_def;
  TF_ASSERT_OK(NodeDefBuilder("node", "Underscore", &producer_registry)
                   .Attr("_underscore", 17)
                   .Finalize(produced_graph_def.add_node()));
  GraphDef expected_graph_def = produced_graph_def;

  std::set<std::pair<string, string>> op_attr_removed;
  TF_ASSERT_OK(
      RemoveNewDefaultAttrsFromGraphDef(&produced_graph_def, consumer_registry,
                                        producer_registry, &op_attr_removed));

  TF_EXPECT_GRAPH_EQ(expected_graph_def, produced_graph_def);
  EXPECT_EQ(op_attr_removed.size(), 0);
}

TEST(RemoveNewDefaultAttrsFromGraphDefTest, HasFunction) {
  OpList consumer_op_list;
  TF_ASSERT_OK(
      FinalizeOpDef(OpDefBuilder("UsesDefault"), consumer_op_list.add_op()));
  TF_ASSERT_OK(FinalizeOpDef(OpDefBuilder("ChangedFromDefault"),
                             consumer_op_list.add_op()));
  OpListOpRegistry consumer_registry(&consumer_op_list);

  OpList producer_op_list;
  TF_ASSERT_OK(FinalizeOpDef(OpDefBuilder("UsesDefault").Attr("a: int = 17"),
                             producer_op_list.add_op()));
  TF_ASSERT_OK(
      FinalizeOpDef(OpDefBuilder("ChangedFromDefault").Attr("a: int = 17"),
                    producer_op_list.add_op()));
  OpListOpRegistry producer_registry(&producer_op_list);

  GraphDef produced_graph_def;
  *produced_graph_def.mutable_library()->add_function() =
      FunctionDefHelper::Create(
          "my_func", {}, {}, {},
          {{{"x"}, "UsesDefault", {}, {{"a", 17}}},
           {{"y"}, "ChangedFromDefault", {}, {{"a", 99}}}},
          {});
  OpList function_op_list;
  *function_op_list.add_op() =
      produced_graph_def.library().function(0).signature();
  OpListOpRegistry function_registry(&function_op_list);
  TF_ASSERT_OK(NodeDefBuilder("call_func", "my_func", &function_registry)
                   .Finalize(produced_graph_def.add_node()));

  std::set<std::pair<string, string>> op_attr_removed;
  TF_ASSERT_OK(
      RemoveNewDefaultAttrsFromGraphDef(&produced_graph_def, consumer_registry,
                                        producer_registry, &op_attr_removed));

  GraphDef expected_graph_def;
  *expected_graph_def.mutable_library()->add_function() =
      FunctionDefHelper::Create(
          "my_func", {}, {}, {},
          {{{"x"}, "UsesDefault", {}, {}},
           {{"y"}, "ChangedFromDefault", {}, {{"a", 99}}}},
          {});
  TF_ASSERT_OK(NodeDefBuilder("call_func", "my_func", &function_registry)
                   .Finalize(expected_graph_def.add_node()));
  TF_EXPECT_GRAPH_EQ(expected_graph_def, produced_graph_def);
  EXPECT_EQ(expected_graph_def.library().DebugString(),
            produced_graph_def.library().DebugString());

  std::set<std::pair<string, string>> expected_removed({{"UsesDefault", "a"}});
  EXPECT_EQ(expected_removed, op_attr_removed);
}

TEST(StripDefaultAttributesTest, DefaultStripped) {
  OpList op_list;
  TF_ASSERT_OK(FinalizeOpDef(OpDefBuilder("OpName1").Attr("a: int = 12"),
                             op_list.add_op()));
  OpListOpRegistry registry(&op_list);

  GraphDef graph_def;
  // This adds the default attribute
  TF_ASSERT_OK(NodeDefBuilder("op1", "OpName1", &registry)
                   .Finalize(graph_def.add_node()));
  ASSERT_EQ(1, graph_def.node(0).attr_size());
  ASSERT_EQ(12, graph_def.node(0).attr().at("a").i());

  StripDefaultAttributes(registry, graph_def.mutable_node());
  ASSERT_EQ(1, graph_def.node_size());
  ASSERT_EQ(0, graph_def.node(0).attr_size());
}

TEST(StripDefaultAttributesTest, NonDefaultNotStripped) {
  OpList op_list;
  TF_ASSERT_OK(FinalizeOpDef(OpDefBuilder("OpName1").Attr("a: int = 12"),
                             op_list.add_op()));
  OpListOpRegistry registry(&op_list);

  GraphDef graph_def;
  TF_ASSERT_OK(NodeDefBuilder("op1", "OpName1", &registry)
                   .Attr("a", 9)
                   .Finalize(graph_def.add_node()));

  GraphDef expected = graph_def;
  StripDefaultAttributes(registry, graph_def.mutable_node());
  TF_EXPECT_GRAPH_EQ(expected, graph_def);
}

TEST(StrippedOpListForGraphTest, FlatTest) {
  // Make four ops
  OpList op_list;
  for (const string& op : {"A", "B", "C", "D"}) {
    OpDef* op_def = op_list.add_op();
    op_def->set_name(op);
    op_def->set_summary("summary");
    op_def->set_description("description");
    op_def->set_is_commutative(op == "B");
  }

  // Make a graph which uses two ops once and twice, respectively.
  // The result should be independent of the ordering.
  const string graph_ops[4][3] = {
      {"C", "B", "B"}, {"B", "C", "B"}, {"B", "B", "C"}, {"C", "C", "B"}};
  for (const bool use_function : {false, true}) {
    for (int order = 0; order < 4; order++) {
      GraphDef graph_def;
      if (use_function) {
        FunctionDef* function_def = graph_def.mutable_library()->add_function();
        function_def->mutable_signature()->set_name("F");
        for (const string& op : graph_ops[order]) {
          function_def->add_node_def()->set_op(op);
        }
        graph_def.add_node()->set_op("F");
      } else {
        for (const string& op : graph_ops[order]) {
          string name = strings::StrCat("name", graph_def.node_size());
          NodeDef* node = graph_def.add_node();
          node->set_name(name);
          node->set_op(op);
        }
      }

      // Strip the op list
      OpList stripped_op_list;
      TF_ASSERT_OK(StrippedOpListForGraph(graph_def, OpListOpRegistry(&op_list),
                                          &stripped_op_list));

      // We should have exactly two ops: B and C.
      ASSERT_EQ(stripped_op_list.op_size(), 2);
      for (int i = 0; i < 2; i++) {
        const OpDef& op = stripped_op_list.op(i);
        EXPECT_EQ(op.name(), i ? "C" : "B");
        EXPECT_EQ(op.summary(), "");
        EXPECT_EQ(op.description(), "");
        EXPECT_EQ(op.is_commutative(), !i);
      }

      // Should get the same result using OpsUsedByGraph().
      std::set<string> used_ops;
      OpsUsedByGraph(graph_def, &used_ops);
      ASSERT_EQ(std::set<string>({"B", "C"}), used_ops);
    }
  }
}

TEST(StrippedOpListForGraphTest, NestedFunctionTest) {
  // Make a primitive op A.
  OpList op_list;
  op_list.add_op()->set_name("A");

  for (const bool recursive : {false, true}) {
    // Call A from function B, and B from function C.
    GraphDef graph_def;
    FunctionDef* b = graph_def.mutable_library()->add_function();
    FunctionDef* c = graph_def.mutable_library()->add_function();
    b->mutable_signature()->set_name("B");
    c->mutable_signature()->set_name("C");
    b->add_node_def()->set_op("A");
    c->add_node_def()->set_op("B");
    if (recursive) {
      b->add_node_def()->set_op("B");
      c->add_node_def()->set_op("C");
    }

    // Use C in the graph.
    graph_def.add_node()->set_op("C");

    // The stripped op list should contain just A.
    OpList stripped_op_list;
    TF_ASSERT_OK(StrippedOpListForGraph(graph_def, OpListOpRegistry(&op_list),
                                        &stripped_op_list));
    ASSERT_EQ(stripped_op_list.op_size(), 1);
    ASSERT_EQ(stripped_op_list.op(0).name(), "A");

    // Should get the same result using OpsUsedByGraph().
    std::set<string> used_ops;
    OpsUsedByGraph(graph_def, &used_ops);
    ASSERT_EQ(std::set<string>({"A"}), used_ops);
  }
}

}  // namespace
}  // namespace tensorflow
