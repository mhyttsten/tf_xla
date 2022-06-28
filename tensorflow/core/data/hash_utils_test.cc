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
class MHTracer_DTPStensorflowPScorePSdataPShash_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPShash_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPShash_utils_testDTcc() {
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

#include "tensorflow/core/data/hash_utils.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {
using ::testing::ContainsRegex;

class DatasetHashUtilsTest : public ::testing::Test {
 protected:
  uint64 GetHash(const FunctionDefLibrary& library, const FunctionDef& fn) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPShash_utils_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/data/hash_utils_test.cc", "GetHash");

    // Construct a node with a function as an attr.
    GraphDef graph_def;
    *graph_def.mutable_library() = library;
    NodeDef* node = graph_def.add_node();
    node->set_op("RemoteCall");
    NameAttrList func;
    func.set_name(fn.signature().name());
    AddNodeAttr("f", func, node);
    uint64 hash = 0;
    TF_CHECK_OK(HashNode(graph_def, *node, &hash));
    return hash;
  }

  Status CheckEqual(const FunctionDefLibrary& library, const FunctionDef& fn1,
                    const FunctionDef& fn2) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPShash_utils_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/data/hash_utils_test.cc", "CheckEqual");

    // Construct nodes with a function as an attr.
    GraphDef graph_def;
    *graph_def.mutable_library() = library;

    NodeDef* node1 = graph_def.add_node();
    node1->set_name("RemoteCall");
    node1->set_op("RemoteCall");
    NameAttrList func1;
    func1.set_name(fn1.signature().name());
    AddNodeAttr("f", func1, node1);

    NodeDef* node2 = graph_def.add_node();
    node1->set_name("RemoteCall2");
    node2->set_op("RemoteCall");
    NameAttrList func2;
    func2.set_name(fn2.signature().name());
    AddNodeAttr("f", func2, node2);

    return CheckSubgraphsEqual(graph_def, node1, graph_def, node2);
  }

  uint64 GetHash(const GraphDef& graph, const NodeDef& node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPShash_utils_testDTcc mht_2(mht_2_v, 250, "", "./tensorflow/core/data/hash_utils_test.cc", "GetHash");

    uint64 hash = 0;
    TF_CHECK_OK(HashNode(graph, node, &hash));
    return hash;
  }

  uint64 GetHash(const Tensor& tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPShash_utils_testDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/data/hash_utils_test.cc", "GetHash");

    uint64 hash = 0;
    TF_CHECK_OK(HashTensor(tensor, &hash));
    return hash;
  }
};

TEST_F(DatasetHashUtilsTest, HashFunctionSameFunctionDifferentNames) {
  FunctionDefLibrary fl;

  FunctionDef* f1 = fl.add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  FunctionDef* f2 = fl.add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndMul2", {"input: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"input", "input"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"input", "input"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  EXPECT_EQ(GetHash(fl, *f1), GetHash(fl, *f2));
  TF_EXPECT_OK(CheckEqual(fl, *f1, *f2));
}

TEST_F(DatasetHashUtilsTest, HashFunctionDifferentFunctions) {
  FunctionDefLibrary fl;

  FunctionDef* f1 = fl.add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  FunctionDef* f2 = fl.add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndAdd", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  // The second op in `f2` is changed to "Add"
  EXPECT_NE(GetHash(fl, *f1), GetHash(fl, *f2));
  Status s = CheckEqual(fl, *f1, *f2);
  EXPECT_NE(s.code(), error::OK);
  EXPECT_THAT(s.error_message(), ContainsRegex("Add"));
}

TEST_F(DatasetHashUtilsTest, HashFunctionDifferentInternalNodeNames) {
  FunctionDefLibrary fl;

  FunctionDef* f1 = fl.add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float", "j: float", "k: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "j"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"add:z:0", "k"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "ret"}});

  FunctionDef* f2 = fl.add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndMul2", {"a: float", "b: float", "c: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"a", "b"}, {{"T", DT_FLOAT}}},
       {{"mul"}, "Mul", {"add:z:0", "c"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "mul:z:0"}},
      /*control_ret_def=*/{{"must_execute", "mul"}});

  EXPECT_EQ(GetHash(fl, *f1), GetHash(fl, *f2));
  TF_EXPECT_OK(CheckEqual(fl, *f1, *f2));
}

TEST_F(DatasetHashUtilsTest, HashGraphWithMultipleCycles) {
  uint64 hash = 0;
  for (int i = 0; i < 1000; ++i) {
    GraphDef g;
    NodeDef* output_node = g.add_node();
    TF_CHECK_OK(NodeDefBuilder("O", "Add")
                    .Input("A", 0, DT_FLOAT)
                    .Input("D", 0, DT_FLOAT)
                    .Finalize(output_node));
    TF_CHECK_OK(NodeDefBuilder("A", "Abs")
                    .Input("B", 0, DT_FLOAT)
                    .Finalize(g.add_node()));
    TF_CHECK_OK(NodeDefBuilder("B", "Add")
                    .Input("C", 0, DT_FLOAT)
                    .Input("D", 0, DT_FLOAT)
                    .Finalize(g.add_node()));
    TF_CHECK_OK(NodeDefBuilder("C", "Ceil")
                    .Input("A", 0, DT_FLOAT)
                    .Finalize(g.add_node()));
    TF_CHECK_OK(NodeDefBuilder("D", "Cos")
                    .Input("E", 0, DT_FLOAT)
                    .Finalize(g.add_node()));
    TF_CHECK_OK(NodeDefBuilder("E", "Floor")
                    .Input("B", 0, DT_FLOAT)
                    .Finalize(g.add_node()));
    uint64 t = GetHash(g, *output_node);
    if (hash == 0) {
      hash = t;
    } else {
      EXPECT_EQ(t, hash);
    }
  }
}

TEST_F(DatasetHashUtilsTest, HashNodeSameGraphDifferentNames) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  NodeDef* n4 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_3/node_7", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n4));

  NodeDef* n5 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_4/node_9", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n5));

  NodeDef* n6 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_5/node_11", "Add")
                  .Device("CPU:0")
                  .Input(n4->name(), 0, DT_INT32)
                  .Input(n5->name(), 0, DT_INT32)
                  .Finalize(n6));

  uint64 hash1 = GetHash(gd, *n3);
  uint64 hash2 = GetHash(gd, *n6);
  EXPECT_EQ(hash1, hash2);
  TF_EXPECT_OK(CheckSubgraphsEqual(gd, n3, gd, n6));
}

TEST_F(DatasetHashUtilsTest, HashNodeDifferentGraphs) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  NodeDef* n4 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Mul")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n4));

  uint64 hash1 = GetHash(gd, *n3);
  uint64 hash2 = GetHash(gd, *n4);
  // We expect different hashes because the op has changed.
  EXPECT_NE(hash1, hash2);
  Status s = CheckSubgraphsEqual(gd, n3, gd, n4);
  EXPECT_NE(s.code(), error::OK);
  EXPECT_THAT(s.error_message(), ContainsRegex("Add"));
  EXPECT_THAT(s.error_message(), ContainsRegex("Mul"));
}

TEST_F(DatasetHashUtilsTest, HashSameGraphDifferentSeeds) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* seed = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/seed", "Const")
                  .Attr("value", 123)
                  .Device("CPU:0")
                  .Finalize(seed));

  NodeDef* seed2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/seed2", "Const")
                  .Attr("value", 456)
                  .Device("CPU:0")
                  .Finalize(seed2));

  NodeDef* range_ds = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/range", "RangeDataset")
                  .Input(n1->name(), 0, DT_INT64)
                  .Input(n1->name(), 0, DT_INT64)
                  .Input(n1->name(), 0, DT_INT64)
                  .Device("CPU:0")
                  .Finalize(range_ds));

  NodeDef* shuffle_ds = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/shuffle", "ShuffleDataset")
                  .Input(range_ds->name(), 0, DT_VARIANT)
                  .Input(n1->name(), 0, DT_INT64)
                  .Input(seed->name(), 0, DT_INT64)
                  .Input(seed2->name(), 0, DT_INT64)
                  .Device("CPU:0")
                  .Finalize(shuffle_ds));

  NodeDef* different_seed = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/different_seed", "Const")
                  .Attr("value", 789)
                  .Device("CPU:0")
                  .Finalize(different_seed));
  NodeDef* different_seed2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/different_seed2", "Const")
                  .Attr("value", 654)
                  .Device("CPU:0")
                  .Finalize(different_seed2));

  NodeDef* range_ds_2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/range_2", "RangeDataset")
                  .Input(n1->name(), 0, DT_INT64)
                  .Input(n1->name(), 0, DT_INT64)
                  .Input(n1->name(), 0, DT_INT64)
                  .Device("CPU:0")
                  .Finalize(range_ds_2));

  NodeDef* shuffle_ds_2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/shuffle_2", "ShuffleDataset")
                  .Input(range_ds_2->name(), 0, DT_VARIANT)
                  .Input(n1->name(), 0, DT_INT64)
                  .Input(different_seed->name(), 0, DT_INT64)
                  .Input(different_seed2->name(), 0, DT_INT64)
                  .Device("CPU:0")
                  .Finalize(shuffle_ds_2));

  uint64 hash1 = GetHash(gd, *shuffle_ds);
  uint64 hash2 = GetHash(gd, *shuffle_ds_2);
  EXPECT_EQ(hash1, hash2);
  TF_EXPECT_OK(CheckSubgraphsEqual(gd, shuffle_ds, gd, shuffle_ds_2));
}

TEST_F(DatasetHashUtilsTest, HashNodeSameGraphDifferentColocationNames) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Attr("_class", {"graph_1/node_2"})
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  NodeDef* n4 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_3/node_7", "Const")
                  .Attr("value", 1)
                  .Attr("_class", {"graph_3/node_9"})
                  .Device("CPU:0")
                  .Finalize(n4));

  NodeDef* n5 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_4/node_9", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n5));

  NodeDef* n6 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_5/node_11", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n6));

  uint64 hash1 = GetHash(gd, *n3);
  uint64 hash2 = GetHash(gd, *n6);

  EXPECT_EQ(hash1, hash2);
  TF_EXPECT_OK(CheckSubgraphsEqual(gd, n3, gd, n6));
}

TEST_F(DatasetHashUtilsTest, HashNodeReversedOrder) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  NodeDef* n4 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Add")
                  .Device("CPU:0")
                  .Input(n2->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Finalize(n4));

  uint64 hash1 = GetHash(gd, *n3);
  uint64 hash2 = GetHash(gd, *n4);
  // We expect different hashes because the inputs of n3 are swapped.
  EXPECT_NE(hash1, hash2);
  Status s = CheckSubgraphsEqual(gd, n3, gd, n4);
  EXPECT_NE(s.code(), error::OK);
  EXPECT_THAT(s.error_message(), ContainsRegex("AttrValues are different"));
}

TEST_F(DatasetHashUtilsTest, HashNodeInputPortChanged) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  NodeDef* n4 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 1, DT_INT32)
                  .Input(n2->name(), 2, DT_INT32)
                  .Finalize(n4));

  uint64 hash1 = GetHash(gd, *n3);
  uint64 hash2 = GetHash(gd, *n4);
  // We expect different hashes because the input ports for nodes used by n3
  // has changed.
  EXPECT_NE(hash1, hash2);
  Status s = CheckSubgraphsEqual(gd, n3, gd, n4);
  EXPECT_NE(s.code(), error::OK);
  EXPECT_THAT(s.error_message(), ContainsRegex("Node inputs"));
}

TEST_F(DatasetHashUtilsTest, HashNodeSameFunctionDifferentNames) {
  GraphDef gd;
  FunctionDefLibrary* fl1 = gd.mutable_library();

  FunctionDef* f1 = fl1->add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  FunctionDef* f2 = fl1->add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndMul2", {"input: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"input", "input"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"input", "input"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  AttrValue a1;
  NameAttrList* nal1 = a1.mutable_func();
  nal1->set_name("AddAndMul");

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  std::vector<NodeDefBuilder::NodeOut> func_inputs;
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a1)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  AttrValue a2;
  NameAttrList* nal2 = a2.mutable_func();
  nal2->set_name("AddAndMul2");

  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a2)
                  .Device("CPU:0")
                  .Finalize(n3));

  uint64 hash1 = GetHash(gd, *n2);
  uint64 hash2 = GetHash(gd, *n3);
  EXPECT_EQ(hash1, hash2);
  TF_EXPECT_OK(CheckSubgraphsEqual(gd, n2, gd, n3));
}

TEST_F(DatasetHashUtilsTest, HashNodeSameFunctionListsDifferentNames) {
  GraphDef gd;
  FunctionDefLibrary* fl1 = gd.mutable_library();

  FunctionDef* f1 = fl1->add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  FunctionDef* f2 = fl1->add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndMul2", {"input: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"input", "input"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"input", "input"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  AttrValue a1;
  AttrValue_ListValue* list1 = a1.mutable_list();
  NameAttrList* nal1 = list1->add_func();
  nal1->set_name("AddAndMul");

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  std::vector<NodeDefBuilder::NodeOut> func_inputs;
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a1)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  AttrValue a2;
  AttrValue_ListValue* list2 = a2.mutable_list();
  NameAttrList* nal2 = list2->add_func();
  nal2->set_name("AddAndMul2");

  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a2)
                  .Device("CPU:0")
                  .Finalize(n3));

  uint64 hash1 = GetHash(gd, *n2);
  uint64 hash2 = GetHash(gd, *n3);
  EXPECT_EQ(hash1, hash2);
  TF_EXPECT_OK(CheckSubgraphsEqual(gd, n2, gd, n3));
}

TEST_F(DatasetHashUtilsTest, HashNodeSameFunctionsOps) {
  GraphDef gd;

  FunctionDefLibrary* fl1 = gd.mutable_library();
  FunctionDef* f1 = fl1->add_function();

  FunctionDef func = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});
  *f1 = func;

  FunctionDef* f2 = fl1->add_function();
  func = FunctionDefHelper::Create(
      "AddAndMul2", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});
  *f2 = func;
  FunctionLibraryDefinition flib(OpRegistry::Global(), gd.library());

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "AddAndMul", &flib)
                  .Input(n1->name(), 0, DT_FLOAT)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "AddAndMul2", &flib)
                  .Input(n1->name(), 0, DT_FLOAT)
                  .Device("CPU:0")
                  .Finalize(n3));

  uint64 hash1 = GetHash(gd, *n2);
  uint64 hash2 = GetHash(gd, *n3);
  EXPECT_EQ(hash1, hash2);
  TF_EXPECT_OK(CheckSubgraphsEqual(gd, n2, gd, n3));
}

TEST_F(DatasetHashUtilsTest, HashNodeDifferentFunctionsOps) {
  GraphDef gd;

  FunctionDefLibrary* fl1 = gd.mutable_library();
  FunctionDef* f1 = fl1->add_function();

  FunctionDef func = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});
  *f1 = func;

  FunctionDef* f2 = fl1->add_function();
  func = FunctionDefHelper::Create(
      "AddAndMul2", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "ret"}});
  *f2 = func;
  FunctionLibraryDefinition flib(OpRegistry::Global(), gd.library());

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "AddAndMul", &flib)
                  .Input(n1->name(), 0, DT_FLOAT)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "AddAndMul2", &flib)
                  .Input(n1->name(), 0, DT_FLOAT)
                  .Device("CPU:0")
                  .Finalize(n3));

  uint64 hash1 = GetHash(gd, *n2);
  uint64 hash2 = GetHash(gd, *n3);
  EXPECT_NE(hash1, hash2);
  Status s = CheckSubgraphsEqual(gd, n2, gd, n3);
  EXPECT_NE(s.code(), error::OK);
  EXPECT_THAT(
      s.error_message(),
      ContainsRegex("Functions AddAndMul and AddAndMul2 are not the same"));
}

TEST_F(DatasetHashUtilsTest, HashNodeDifferentFunctions) {
  GraphDef gd;

  FunctionDefLibrary* fl1 = gd.mutable_library();
  FunctionDef* f1 = fl1->add_function();

  FunctionDef func = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});
  *f1 = func;

  FunctionDef* f2 = fl1->add_function();
  func = FunctionDefHelper::Create(
      "AddAndMul2", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "ret"}});
  *f2 = func;

  AttrValue a1;
  NameAttrList* nal1 = a1.mutable_func();
  nal1->set_name("AddAndMul");

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  std::vector<NodeDefBuilder::NodeOut> func_inputs;
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a1)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  AttrValue a2;
  NameAttrList* nal2 = a2.mutable_func();
  nal2->set_name("AddAndMul2");

  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a2)
                  .Device("CPU:0")
                  .Finalize(n3));

  uint64 hash1 = GetHash(gd, *n2);
  uint64 hash2 = GetHash(gd, *n3);
  EXPECT_NE(hash1, hash2);
  Status s = CheckSubgraphsEqual(gd, n2, gd, n3);
  EXPECT_NE(s.code(), error::OK);
  EXPECT_THAT(
      s.error_message(),
      ContainsRegex("Functions AddAndMul and AddAndMul2 are not the same"));
}

TEST_F(DatasetHashUtilsTest, HashNodeDifferentFunctionLists) {
  GraphDef gd;

  FunctionDefLibrary* fl1 = gd.mutable_library();
  FunctionDef* f1 = fl1->add_function();

  FunctionDef func = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});
  *f1 = func;

  FunctionDef* f2 = fl1->add_function();
  func = FunctionDefHelper::Create(
      "AddAndMul2", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "ret"}});
  *f2 = func;

  AttrValue a1;
  AttrValue_ListValue* list1 = a1.mutable_list();
  NameAttrList* nal1 = list1->add_func();
  nal1->set_name("AddAndMul");

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  std::vector<NodeDefBuilder::NodeOut> func_inputs;
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a1)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  AttrValue a2;
  AttrValue_ListValue* list2 = a2.mutable_list();
  NameAttrList* nal2 = list2->add_func();
  nal2->set_name("AddAndMul2");

  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a2)
                  .Device("CPU:0")
                  .Finalize(n3));

  uint64 hash1 = GetHash(gd, *n2);
  uint64 hash2 = GetHash(gd, *n3);
  EXPECT_NE(hash1, hash2);
  Status s = CheckSubgraphsEqual(gd, n2, gd, n3);
  EXPECT_NE(s.code(), error::OK);
  EXPECT_THAT(
      s.error_message(),
      ContainsRegex("Functions AddAndMul and AddAndMul2 are not the same"));
}

TEST_F(DatasetHashUtilsTest, HashNodeDifferentControlInputs) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Const")
                  .Attr("value", 10)
                  .Device("CPU:0")
                  .Finalize(n3));

  NodeDef* n4 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Identity")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .ControlInput(n2->name())
                  .Finalize(n4));

  NodeDef* n5 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_5", "Identity")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .ControlInput(n3->name())
                  .Finalize(n5));

  // Control inputs are different between these two graphs.
  uint64 hash1 = GetHash(gd, *n4);
  uint64 hash2 = GetHash(gd, *n5);
  EXPECT_NE(hash1, hash2);
  Status s = CheckSubgraphsEqual(gd, n4, gd, n5);
  EXPECT_NE(s.code(), error::OK);
  EXPECT_THAT(s.error_message(),
              ContainsRegex("Control dependencies are different"));
}

TEST_F(DatasetHashUtilsTest, HashNodeControlInputDifferentOrdering) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Const")
                  .Attr("value", 10)
                  .Device("CPU:0")
                  .Finalize(n3));

  NodeDef* n4 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Identity")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .ControlInput(n2->name())
                  .ControlInput(n3->name())
                  .Finalize(n4));

  NodeDef* n5 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_5", "Identity")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .ControlInput(n3->name())
                  .ControlInput(n2->name())
                  .Finalize(n5));

  uint64 hash1 = GetHash(gd, *n4);
  uint64 hash2 = GetHash(gd, *n5);
  EXPECT_EQ(hash1, hash2);
  TF_EXPECT_OK(CheckSubgraphsEqual(gd, n4, gd, n5));
}

TEST_F(DatasetHashUtilsTest, HashNodeDifferentGraphSamePartialGraph) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();

  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash1 = GetHash(gd, *n1);

  n3->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Mul")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash2 = GetHash(gd, *n1);

  EXPECT_EQ(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashNodeWithManyControlDependencies) {
  GraphDef gd;
  NodeDef* n;

  for (int i = 0; i < 1000; ++i) {
    n = gd.add_node();
    NodeDefBuilder ndb(absl::StrCat("graph_1/node_", i), "Const");
    ndb.Attr("value", 1);
    ndb.Device("CPU:0");
    for (int j = 0; j < i; ++j) {
      ndb.ControlInput(absl::StrCat("graph_1/node_", j));
    }
    TF_CHECK_OK(ndb.Finalize(n));
  }

  // No checks here, because so long as this does not time out, we are OK.
  GetHash(gd, *n);
}

TEST_F(DatasetHashUtilsTest, HashFunctionsWithControlDependencyLoop) {
  GraphDef gd;

  FunctionDefLibrary* fl1 = gd.mutable_library();
  FunctionDef* f1 = fl1->add_function();

  AttrValue a1;
  NameAttrList* nal1 = a1.mutable_func();
  nal1->set_name("AddAndMul");

  std::pair<string, FunctionDefHelper::AttrValueWrapper> func_attr = {
      "body", FunctionDefHelper::AttrValueWrapper(*nal1)};

  FunctionDef func = FunctionDefHelper::Create(
      /*function_name=*/"AddAndMul",
      /*in_def=*/{"i: float", "j: int32"},
      /*out_def=*/{"o: float"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}, {"ret"}},
       // This creates a dependency on the same function.
       {{"for"}, "For", {"j", "j", "j"}, {func_attr, {"T", DT_FLOAT}}, {"ret"}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});
  *f1 = func;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  std::vector<NodeDefBuilder::NodeOut> func_inputs;
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .ControlInput("graph_1/node_2")
                  .Attr("body", a1)
                  .Device("CPU:0")
                  .Finalize(n2));

  // No checks in the test, the fact that it runs and doesn't timeout or exhaust
  // the stack means it is successful.
  GetHash(gd, *n2);
}

TEST_F(DatasetHashUtilsTest, HashNodeWithControlDependencyLoop) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_2")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_1")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .ControlInput("graph_1/node_1")
                  .ControlInput("graph_1/node_2")
                  .Finalize(n3));

  // No checks in the test, the fact that it runs and doesn't timeout or exhaust
  // the stack means it is successful.
  GetHash(gd, *n3);
}

TEST_F(DatasetHashUtilsTest, HashNodeWithControlDependencyLoopDifferentNames) {
  GraphDef gd1;

  NodeDef* n1 = gd1.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_2")
                  .Finalize(n1));

  NodeDef* n2 = gd1.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_1")
                  .Finalize(n2));

  NodeDef* n3 = gd1.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .ControlInput("graph_1/node_1")
                  .ControlInput("graph_1/node_2")
                  .Finalize(n3));

  GraphDef gd2;

  NodeDef* n4 = gd2.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_5")
                  .Finalize(n4));

  NodeDef* n5 = gd2.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_5", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_4")
                  .Finalize(n5));

  NodeDef* n6 = gd2.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_6", "Add")
                  .Device("CPU:0")
                  .Input(n4->name(), 0, DT_INT32)
                  .Input(n5->name(), 0, DT_INT32)
                  .ControlInput("graph_1/node_4")
                  .ControlInput("graph_1/node_5")
                  .Finalize(n6));

  EXPECT_EQ(GetHash(gd1, *n3), GetHash(gd2, *n6));
}

TEST_F(DatasetHashUtilsTest, HashInt32Tensor) {
  Tensor s1(42);
  Tensor s2(42);
  Tensor s3(43);

  EXPECT_EQ(GetHash(s1), GetHash(s2));
  EXPECT_NE(GetHash(s1), GetHash(s3));

  Tensor v1(DT_INT32, TensorShape({2}));
  v1.vec<int32>()(0) = 0;
  v1.vec<int32>()(1) = 1;
  Tensor v2(DT_INT32, TensorShape({2}));
  v2.vec<int32>()(0) = 0;
  v2.vec<int32>()(1) = 1;
  Tensor v3(DT_INT32, TensorShape({2}));
  v3.vec<int32>()(0) = 0;
  v3.vec<int32>()(1) = 2;

  EXPECT_EQ(GetHash(v1), GetHash(v2));
  EXPECT_NE(GetHash(v1), GetHash(v3));
}

TEST_F(DatasetHashUtilsTest, HashStringTensor) {
  Tensor s1("hello");
  Tensor s2("hello");
  Tensor s3("world");

  EXPECT_EQ(GetHash(s1), GetHash(s2));
  EXPECT_NE(GetHash(s1), GetHash(s3));

  Tensor v1(DT_STRING, TensorShape({2}));
  v1.vec<tstring>()(0) = "hello";
  v1.vec<tstring>()(1) = "world";
  Tensor v2(DT_STRING, TensorShape({2}));
  v2.vec<tstring>()(0) = "hello";
  v2.vec<tstring>()(1) = "world";
  Tensor v3(DT_STRING, TensorShape({2}));
  v3.vec<tstring>()(0) = "hello";
  v3.vec<tstring>()(1) = "universe";

  EXPECT_EQ(GetHash(v1), GetHash(v2));
  EXPECT_NE(GetHash(v1), GetHash(v3));
}

// Benchmark that simulates a shallow and wide graph.
static void BM_ParallelFunctionCallsGraph(benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPShash_utils_testDTcc mht_4(mht_4_v, 1361, "", "./tensorflow/core/data/hash_utils_test.cc", "BM_ParallelFunctionCallsGraph");

  GraphDef graph_def;
  FunctionDefLibrary* fl = graph_def.mutable_library();

  FunctionDef* fd = fl->add_function();
  *fd = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  NodeDef* input = graph_def.add_node();
  input->set_name("InputPlaceholder");
  input->set_op("Placeholder");
  AddNodeAttr("dtype", DT_FLOAT, input);

  // Equivalent of a `tf.group()`.
  NodeDef* target = graph_def.add_node();
  target->set_name("Target");
  target->set_op("NoOp");

  // Create 100 parallel PartitionedCalls that all depend on input. Generate a
  // NodeDef that has close to similar attributes that TensorFlow will generate.
  ConfigProto config_pb;
  config_pb.mutable_device_count()->insert({"CPU", 1});
  config_pb.mutable_device_count()->insert({"GPU", 1});
  config_pb.set_allow_soft_placement(true);
  for (int i = 0; i < 100; ++i) {
    NodeDef* node = graph_def.add_node();
    node->set_name(absl::StrCat("PartitionedCall_", i));
    node->set_op("PartitionedCall");
    *node->add_input() = input->name();
    AddNodeAttr("Tin", DT_FLOAT, node);
    AddNodeAttr("Tout", DT_FLOAT, node);
    AddNodeAttr("config", "", node);
    AddNodeAttr("config_proto", config_pb.SerializeAsString(), node);
    NameAttrList func;
    func.set_name(fd->signature().name());
    AddNodeAttr("f", func, node);
    *target->add_input() = absl::StrCat("^", node->name());
  }

  uint64 hash_value;
  for (auto _ : state) {
    CHECK(HashNode(graph_def, *target, &hash_value).ok());
  }
}
BENCHMARK(BM_ParallelFunctionCallsGraph);

// Benchmark that simulates a narrow and deep graph.
static void BM_ChainedFunctionCallsGraph(benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPShash_utils_testDTcc mht_5(mht_5_v, 1415, "", "./tensorflow/core/data/hash_utils_test.cc", "BM_ChainedFunctionCallsGraph");

  GraphDef graph_def;
  FunctionDefLibrary* fl = graph_def.mutable_library();

  FunctionDef* fd = fl->add_function();
  *fd = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  NodeDef* input = graph_def.add_node();
  input->set_name("InputPlaceholder");
  input->set_op("Placeholder");
  AddNodeAttr("dtype", DT_FLOAT, input);

  // Create 100 chained PartitionedCalls, each depending on the previous.
  // Generate a NodeDef that has close to similar attributes that TensorFlow
  // will generate.
  ConfigProto config_pb;
  config_pb.mutable_device_count()->insert({"CPU", 1});
  config_pb.mutable_device_count()->insert({"GPU", 1});
  config_pb.set_allow_soft_placement(true);
  for (int i = 0; i < 100; ++i) {
    NodeDef* node = graph_def.add_node();
    node->set_name(absl::StrCat("PartitionedCall_", i));
    node->set_op("PartitionedCall");
    if (i > 0) {
      *node->add_input() = absl::StrCat("PartitionedCall_", i - 1);
    } else {
      *node->add_input() = input->name();
    }
    AddNodeAttr("Tin", DT_FLOAT, node);
    AddNodeAttr("Tout", DT_FLOAT, node);
    AddNodeAttr("config", "", node);
    AddNodeAttr("config_proto", config_pb.SerializeAsString(), node);
    NameAttrList func;
    func.set_name(fd->signature().name());
    AddNodeAttr("f", func, node);
  }

  const NodeDef& target = graph_def.node(graph_def.node_size() - 1);

  uint64 hash_value;
  for (auto _ : state) {
    CHECK(HashNode(graph_def, target, &hash_value).ok());
  }
}
BENCHMARK(BM_ChainedFunctionCallsGraph);

// Benchmark that simulates many nested function calls.
static void BM_ComposedFunctionCallsGraph(benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPShash_utils_testDTcc mht_6(mht_6_v, 1470, "", "./tensorflow/core/data/hash_utils_test.cc", "BM_ComposedFunctionCallsGraph");

  GraphDef graph_def;
  FunctionDefLibrary* fl = graph_def.mutable_library();

  // AddAndMul will be the last function, all others will be calls up to this.
  FunctionDef* fd = fl->add_function();
  *fd = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  ConfigProto config_pb;
  config_pb.mutable_device_count()->insert({"CPU", 1});
  config_pb.mutable_device_count()->insert({"GPU", 1});
  config_pb.set_allow_soft_placement(true);
  for (int i = 0; i < 99; ++i) {
    // Get the name fo the previous function
    NameAttrList func;
    func.set_name(fd->signature().name());

    FunctionDef* fd = fl->add_function();
    *fd = FunctionDefHelper::Create(
        /*function_name=*/absl::StrCat("F_", i),
        /*in_def=*/{"i: float"},
        /*out_def=*/{"o: float"},
        /*attr_def=*/{},
        /*node_def=*/
        {
            {
                {"inner_call"},
                "PartitionedCall",
                {"i"},
                {{"Ti", DT_FLOAT},
                 {"Tout", DT_FLOAT},
                 {"config", ""},
                 {"config_proto", config_pb.SerializeAsString()},
                 {"f", func}},
            },
        },
        /*ret_def=*/{{"o", "inner_call:o:0"}},
        /*control_ret_def=*/{{"must_execute", "inner_call"}});
  }

  NodeDef* input = graph_def.add_node();
  input->set_name("InputPlaceholder");
  input->set_op("Placeholder");
  AddNodeAttr("dtype", DT_FLOAT, input);

  // Create call to the outer most function.
  NodeDef* node = graph_def.add_node();
  node->set_name("PartitionedCall_start");
  node->set_op("PartitionedCall");
  *node->add_input() = input->name();
  AddNodeAttr("Tin", DT_FLOAT, node);
  AddNodeAttr("Tout", DT_FLOAT, node);
  AddNodeAttr("config", "", node);
  AddNodeAttr("config_proto", config_pb.SerializeAsString(), node);
  NameAttrList func;
  func.set_name(fd->signature().name());
  AddNodeAttr("f", func, node);

  const NodeDef& target = graph_def.node(graph_def.node_size() - 1);

  uint64 hash_value;
  for (auto _ : state) {
    CHECK(HashNode(graph_def, target, &hash_value).ok());
  }
}
BENCHMARK(BM_ComposedFunctionCallsGraph);

}  // namespace
}  // namespace data
}  // namespace tensorflow
