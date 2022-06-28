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
class MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_cluster_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_cluster_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_cluster_testDTcc() {
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

#include "tensorflow/core/grappler/clusters/virtual_cluster.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

class VirtualClusterTest : public ::testing::Test {
 public:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_cluster_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/grappler/clusters/virtual_cluster_test.cc", "SetUp");

    // Invent a CPU so that predictions remain the same from machine to machine.
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    cpu_device.set_frequency(1000);
    cpu_device.set_num_cores(4);
    cpu_device.set_bandwidth(32);
    cpu_device.set_l1_cache_size(32 * 1024);
    cpu_device.set_l2_cache_size(256 * 1024);
    cpu_device.set_l3_cache_size(4 * 1024 * 1024);
    cpu_device.set_memory_size(1024 * 1024);
    std::unordered_map<string, DeviceProperties> devices;
    devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
    cluster_.reset(new VirtualCluster(devices));
    TF_CHECK_OK(cluster_->Provision());
  }

  void TearDown() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_cluster_testDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/grappler/clusters/virtual_cluster_test.cc", "TearDown");

    TF_CHECK_OK(cluster_->Shutdown());
    cluster_.reset();
  }

 protected:
  std::unique_ptr<VirtualCluster> cluster_;
};

TEST_F(VirtualClusterTest, ClusterType) {
  CHECK_EQ("virtual", cluster_->type());
}

TEST_F(VirtualClusterTest, CostModel) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TF_CHECK_OK(cluster_->Initialize(item));

  RunMetadata metadata;
  TF_CHECK_OK(cluster_->Run(item.graph, item.feed, item.fetch, &metadata));

  // There should be at least 4 nodes corresponding to the 4 stages we created
  // in the fake input.
  EXPECT_LE(4, metadata.cost_graph().node_size());
  for (const auto& node : metadata.cost_graph().node()) {
    // Skip the constant node that configures the random number generator.
    if (node.name().find("Const/Const") != string::npos) {
      continue;
    }
    EXPECT_EQ(1, node.output_info_size());
    EXPECT_EQ(40, node.output_info(0).size());
    const TensorShapeProto& shape = node.output_info(0).shape();
    EXPECT_EQ(2, shape.dim_size());
    EXPECT_EQ(10, shape.dim(0).size());
    EXPECT_EQ(1, shape.dim(1).size());
    if (node.name() == "x") {
      EXPECT_EQ(1500, node.compute_cost());
    } else {
      EXPECT_EQ(2500, node.compute_cost());
    }
  }

  for (const auto& dev_stat : metadata.step_stats().dev_stats()) {
    EXPECT_EQ("/job:localhost/replica:0/task:0/cpu:0", dev_stat.device());
    for (const auto& node : dev_stat.node_stats()) {
      if (node.node_name() == "AddN") {
        EXPECT_EQ(2500, node.op_end_rel_micros());
      }
    }
  }
}

TEST_F(VirtualClusterTest, OutOfMemory) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  // Create a large variable that can't fit in memory.
  auto zero = ops::Variable(root.WithOpName("zero"), {1024, 1024}, DT_FLOAT);
  auto identity = ops::Identity(root.WithOpName("i"), zero);
  auto identity2 = ops::Identity(root.WithOpName("i2"), identity);
  GrapplerItem item;
  TF_CHECK_OK(root.ToGraphDef(&item.graph));
  item.fetch.push_back("i2");

  TF_CHECK_OK(cluster_->Initialize(item));
  Status s = cluster_->Run(item.graph, item.feed, item.fetch, nullptr);
  EXPECT_EQ(error::RESOURCE_EXHAUSTED, s.code());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
