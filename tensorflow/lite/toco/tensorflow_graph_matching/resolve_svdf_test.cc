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
class MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_svdf_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_svdf_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_svdf_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/tensorflow_graph_matching/resolve_svdf.h"

#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster_utils.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/resolve_cluster.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"

using tensorflow::GraphDef;
using tensorflow::NodeDef;

namespace toco {

class ResolveSvdfTest : public ::testing::Test {
 public:
  ResolveSvdfTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_svdf_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/toco/tensorflow_graph_matching/resolve_svdf_test.cc", "ResolveSvdfTest");

    AddNewNode("Input1", "Const", {});
    AddNewNode("Svdf1/SVDF_weights_feature/part_0", "Const", {},
               {0.1, 0.2, 0.3});
    AddNewNode("Svdf1/SVDF_weights_feature/part_0/read", "Identity",
               {"Svdf1/SVDF_weights_feature/part_0"});
    AddNewNode("Svdf1/SVDF_weights_time/part_0", "Const", {}, {0.1, 0.2, 0.3});
    AddNewNode("Svdf1/SVDF_weights_time/part_0/read", "Identity",
               {"Svdf1/SVDF_weights_time/part_0"});

    AddNewNode("Svdf1/f1", "SVDF_F1",
               {"Input1", "Svdf1/SVDF_weights_feature/part_0/read"});
    AddNewNode("Svdf1/f2", "SVDF_F2",
               {"Svdf1/SVDF_weights_time/part_0/read", "Svdf1/f1"});
    AddNewNode("Svdf1/Relu", "Relu", {"Svdf1/f2"});
    AddShapeNode("Svdf1/Reshape/shape", {10, 1, -1});
    AddNewNode("Output1", "Const", {"Svdf1/Relu"});

    AddNewNode("Input2", "Const", {});
    AddNewNode("Svdf2/SVDF_weights_feature/part_0", "Const", {},
               {0.1, 0.2, 0.3});
    AddNewNode("Svdf2/SVDF_weights_feature/part_0/read", "Identity",
               {"Svdf2/SVDF_weights_feature/part_0"});
    AddNewNode("Svdf2/SVDF_weights_time/part_0", "Const", {}, {0.1, 0.2, 0.3});
    AddNewNode("Svdf2/SVDF_weights_time/part_0/read", "Identity",
               {"Svdf2/SVDF_weights_time/part_0"});

    AddNewNode("Svdf2/f1", "SVDF_F1",
               {"Input1", "Svdf2/SVDF_weights_feature/part_0/read"});
    AddNewNode("Svdf2/f2", "SVDF_F2",
               {"Svdf2/SVDF_weights_time/part_0/read", "Svdf2/f1"});
    AddNewNode("Svdf2/Relu", "Relu", {"Svdf2/f2"});
    AddShapeNode("Svdf2/Reshape/shape", {10, 2, -1});
    AddNewNode("Output2", "Const", {"Svdf2/Relu"});
  }

  ~ResolveSvdfTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_svdf_testDTcc mht_1(mht_1_v, 249, "", "./tensorflow/lite/toco/tensorflow_graph_matching/resolve_svdf_test.cc", "~ResolveSvdfTest");
}

 protected:
  void AddNewNode(const std::string& name, const std::string& op,
                  const std::vector<std::string>& inputs) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_svdf_testDTcc mht_2(mht_2_v, 258, "", "./tensorflow/lite/toco/tensorflow_graph_matching/resolve_svdf_test.cc", "AddNewNode");

    NodeDef* node = graph_.add_node();
    node->set_name(name);
    node->set_op(op);
    node->set_device("");
    for (int i = 0; i < inputs.size(); i++) {
      node->add_input();
      node->set_input(i, inputs[i]);
    }
  }

  void AddNewNode(const std::string& name, const std::string& op,
                  const std::vector<std::string>& inputs,
                  const std::vector<float>& values) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   mht_3_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_svdf_testDTcc mht_3(mht_3_v, 276, "", "./tensorflow/lite/toco/tensorflow_graph_matching/resolve_svdf_test.cc", "AddNewNode");

    NodeDef* node = graph_.add_node();
    node->set_name(name);
    node->set_op(op);
    node->set_device("");
    for (int i = 0; i < inputs.size(); i++) {
      node->add_input();
      node->set_input(i, inputs[i]);
    }
    // Add the float vector as an attribute to the node.
    (*node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    tensorflow::TensorProto* allocated_tensor = new tensorflow::TensorProto;
    tensorflow::TensorShapeProto* allocated_tensor_shape =
        new tensorflow::TensorShapeProto;
    auto tensor_shape_dim0 = allocated_tensor_shape->add_dim();
    tensor_shape_dim0->set_size(values.size());
    allocated_tensor->set_allocated_tensor_shape(allocated_tensor_shape);
    allocated_tensor->set_tensor_content(
        std::string(reinterpret_cast<const char*>(values.data()),
                    values.size() * sizeof(float)));
    (*node->mutable_attr())["value"].set_allocated_tensor(allocated_tensor);
  }

  void AddShapeNode(const std::string& name, const std::vector<int>& values) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_svdf_testDTcc mht_4(mht_4_v, 303, "", "./tensorflow/lite/toco/tensorflow_graph_matching/resolve_svdf_test.cc", "AddShapeNode");

    NodeDef* node = graph_.add_node();
    node->set_name(name);
    node->set_op("Const");
    node->set_device("");
    // Add the float vector as an attribute to the node.
    (*node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
    tensorflow::TensorProto* allocated_tensor = new tensorflow::TensorProto;
    tensorflow::TensorShapeProto* allocated_tensor_shape =
        new tensorflow::TensorShapeProto;
    auto tensor_shape_dim0 = allocated_tensor_shape->add_dim();
    tensor_shape_dim0->set_size(values.size());
    allocated_tensor->set_allocated_tensor_shape(allocated_tensor_shape);
    allocated_tensor->set_tensor_content(
        std::string(reinterpret_cast<const char*>(values.data()),
                    values.size() * sizeof(int)));
    (*node->mutable_attr())["value"].set_allocated_tensor(allocated_tensor);
  }

  GraphDef graph_;
  SvdfClusterFactory svdf_cluster_factory_;
  std::vector<std::unique_ptr<Cluster>> clusters_;
};

TEST_F(ResolveSvdfTest, TestTranspose2DTensor) {
  static float matrix[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
  static float expected_transposed_matrix[] = {1., 5., 9.,  2., 6., 10.,
                                               3., 7., 11., 4., 8., 12.};
  float* transposed_matrix = new float[12];
  Transpose2DTensor(matrix, 3, 4, transposed_matrix);

  std::vector<float> actual;
  actual.insert(
      actual.end(), transposed_matrix,
      transposed_matrix + sizeof(expected_transposed_matrix) / sizeof(float));
  std::vector<float> expected;
  expected.insert(expected.end(), expected_transposed_matrix,
                  expected_transposed_matrix +
                      sizeof(expected_transposed_matrix) / sizeof(float));
  delete[] transposed_matrix;
}

TEST_F(ResolveSvdfTest, TestResolveSvdfFlow) {
  std::unordered_map<std::string, bool> is_node_in_cluster;
  for (const NodeDef& node : graph_.node()) {
    is_node_in_cluster[node.name()] = false;
  }

  std::vector<std::string> cluster_names;
  CHECK(FindCluster(svdf_cluster_factory_, graph_, &is_node_in_cluster,
                    &clusters_));

  for (const std::unique_ptr<Cluster>& cluster : clusters_) {
    cluster_names.push_back(cluster->GetName());
    cluster->CreateNodes();
  }

  EXPECT_THAT(cluster_names,
              testing::UnorderedElementsAreArray({"Svdf1", "Svdf2"}));

  std::vector<std::string> new_node_names;
  std::vector<float> content_array(3);
  for (const std::unique_ptr<Cluster>& cluster : clusters_) {
    // After CreateNodes in each cluster we have three nodes: Svdf,
    // weights_feature and weights_time.
    CHECK_EQ(cluster->GetNewNodes().size(), 3);
    for (const std::unique_ptr<tensorflow::NodeDef>& node :
         cluster->GetNewNodes()) {
      new_node_names.push_back(node->name());
      if (node->op() == "Const") {
        CHECK_EQ(node->attr().at("dtype").type(), tensorflow::DT_FLOAT);
        toco::port::CopyToBuffer(
            node->attr().at("value").tensor().tensor_content(),
            reinterpret_cast<char*>(content_array.data()));
        EXPECT_THAT(content_array,
                    testing::UnorderedElementsAreArray({0.1, 0.2, 0.3}));
      } else {
        // Checking the Svdf node attributes (rank and activation type) are
        // correct.
        if (node->name() == "Svdf1") {
          CHECK_EQ(node->attr().at("Rank").i(), 1);
        } else if (node->name() == "Svdf2") {
          CHECK_EQ(node->attr().at("Rank").i(), 2);
        }
        CHECK_EQ(node->attr().at("ActivationFunction").s(), "Relu");
      }
    }
  }
  EXPECT_THAT(new_node_names, testing::UnorderedElementsAreArray(
                                  {"Svdf2/SVDF_weights_feature/part_0",
                                   "Svdf2/SVDF_weights_time/part_0", "Svdf2",
                                   "Svdf1/SVDF_weights_feature/part_0",
                                   "Svdf1/SVDF_weights_time/part_0", "Svdf1"}));
}

}  // end namespace toco
