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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc() {
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

#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"

#include <regex>  // NOLINT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_testutils.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"  // NOLINT
#include "tensorflow/core/public/session.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace convert {

class FakeCluster : public grappler::Cluster {
 public:
  FakeCluster() : Cluster(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/tf2tensorrt/convert/convert_graph_test.cc", "FakeCluster");
}

  void SetDeviceSet(const DeviceSet* device_set) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/compiler/tf2tensorrt/convert/convert_graph_test.cc", "SetDeviceSet");
 device_set_ = device_set; }

  const DeviceSet* GetDeviceSet() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/tf2tensorrt/convert/convert_graph_test.cc", "GetDeviceSet");
 return device_set_; }

  string type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/compiler/tf2tensorrt/convert/convert_graph_test.cc", "type");
 return ""; }
  Status Provision() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc mht_4(mht_4_v, 234, "", "./tensorflow/compiler/tf2tensorrt/convert/convert_graph_test.cc", "Provision");
 return Status::OK(); }
  Status Initialize(const grappler::GrapplerItem& item) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc mht_5(mht_5_v, 238, "", "./tensorflow/compiler/tf2tensorrt/convert/convert_graph_test.cc", "Initialize");

    return Status::OK();
  }
  Status Run(const GraphDef& graph_def,
             const std::vector<std::pair<string, Tensor>>& feed,
             const std::vector<string>& fetch, RunMetadata* metadata) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc mht_6(mht_6_v, 246, "", "./tensorflow/compiler/tf2tensorrt/convert/convert_graph_test.cc", "Run");

    return Status::OK();
  }

 private:
  const DeviceSet* device_set_ = nullptr;
};

TEST(GetDeviceAndAllocatorTest, GetDeviceAndAllocator) {
  TRTOptimizationPass::ConversionParams params;
  EngineInfo engine_info;
  {
    // cluster is not set, and no gpu device is available.
    auto result = GetDeviceAndAllocator(nullptr, engine_info);
    EXPECT_EQ(-1, result.first);
    EXPECT_EQ(nullptr, result.second);
  }

  // Create a session with two (virtual) gpu device.
  SessionOptions options;
  ConfigProto* config = &options.config;
  GPUOptions* gpu_options = config->mutable_gpu_options();
  auto virtual_devices =
      gpu_options->mutable_experimental()->add_virtual_devices();
  virtual_devices->add_memory_limit_mb(200);
  virtual_devices->add_memory_limit_mb(200);
  std::unique_ptr<Session> session(NewSession(options));

  {
    // cluster is not set, should find and return first gpu id and
    // corresponding allocator.
    auto result = GetDeviceAndAllocator(nullptr, engine_info);
    EXPECT_EQ(0, result.first);
    EXPECT_NE(nullptr, result.second);
    EXPECT_EQ("GPU_0_bfc", result.second->Name());
  }

  FakeCluster cluster;
  {
    // params.cluster->GetDeviceSet() returns null, should find and return first
    // gpu id and corresponding allocator.
    auto result = GetDeviceAndAllocator(&cluster, engine_info);
    EXPECT_EQ(0, result.first);
    EXPECT_NE(nullptr, result.second);
    EXPECT_EQ("GPU_0_bfc", result.second->Name());
  }

  // Build the DeviceSet.
  DeviceSet device_set;
  const DeviceMgr* device_mgr = nullptr;
  TF_ASSERT_OK(session->LocalDeviceManager(&device_mgr));
  for (auto d : device_mgr->ListDevices()) {
    device_set.AddDevice(d);
  }
  cluster.SetDeviceSet(&device_set);
  {
    // engine_info.device is not set, should find and return first gpu id and
    // corresponding allocator.
    auto result = GetDeviceAndAllocator(&cluster, engine_info);
    EXPECT_EQ(0, result.first);
    EXPECT_NE(nullptr, result.second);
    EXPECT_EQ("GPU_0_bfc", result.second->Name());
  }

  engine_info.device = "/GPU:1";
  {
    // Set to use second device.
    auto result = GetDeviceAndAllocator(&cluster, engine_info);
    EXPECT_EQ(0, result.first);
    EXPECT_NE(nullptr, result.second);
    EXPECT_EQ("GPU_1_bfc", result.second->Name());
  }

  engine_info.device = "/GPU:3";
  {
    // Set to use nonexistent device.
    auto result = GetDeviceAndAllocator(&cluster, engine_info);
    EXPECT_EQ(-1, result.first);
    EXPECT_EQ(nullptr, result.second);
  }
}

class ConvertGraphTest : public ::testing::Test {
 public:
  Status RunConvertGraph(Scope s, GraphDef* output_graph_def,
                         int maximum_batch_size = 1000) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc mht_7(mht_7_v, 334, "", "./tensorflow/compiler/tf2tensorrt/convert/convert_graph_test.cc", "RunConvertGraph");

    // Create GraphProperties.
    grappler::GrapplerItem item;
    TF_EXPECT_OK(s.ToGraphDef(&item.graph));
    grappler::GraphProperties graph_properties(item);
    TF_EXPECT_OK(graph_properties.InferStatically(true));

    // Construct ConversionParams.
    const std::vector<string> input_output_names{"output"};
    TRTOptimizationPass::ConversionParams params;
    params.max_batch_size = maximum_batch_size;
    params.max_workspace_size_bytes = 8 << 20;
    params.minimum_segment_size = 1;
    params.use_calibration = false;
    params.trt_logger_name = "DefaultLogger";
    return ConvertGraph(params, item, input_output_names, nullptr,
                        output_graph_def);
  }
};

TEST_F(ConvertGraphTest, DirectlyConnectedEngines) {
  // Create the graph. There will be two TRTEngineOps after the conversion, and
  // the upstream TRTEngineOp will have two output connections from the same
  // node:port inside the op to the downstream TRTEngineOp. Then, if it adds the
  // downstream TRTEngineOp first, when adding the upstream op it'll need to
  // update the same output connection twice. This test ensures the correctness
  // of the conversion under such condition.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT,
                                ops::Placeholder::Shape({2, 1}));
  // We purposefully choose the name of the root node of each segment, so it'll
  // process the segment in the downstream first, then, when it tries to update
  // the edge between the two TRTEngineOps, it'll try to add the same edge
  // multiple times.
  auto segment_root_1 = ops::Identity(s.WithOpName("segment_root_b"), input);
  auto add1 = ops::Add(s.WithOpName("add1"), segment_root_1, segment_root_1);
  // Add incompatible reshapes that change the batch dimension.
  auto incompatible =
      ops::Reshape(s.WithOpName("reshape1"), add1, Input({1, 2}));
  incompatible =
      ops::Reshape(s.WithOpName("reshape2"), incompatible, Input({2, 1}));

  auto add2 = ops::Add(s.WithOpName("add2"), incompatible, add1);
  auto segment_root_2 = ops::Identity(s.WithOpName("segment_root_a"), add1);
  auto add3 = ops::Add(s.WithOpName("add3"), add2, segment_root_2);
  ops::Identity(s.WithOpName("output"), add3);

  GraphDef output_graph_def;
  TF_EXPECT_OK(RunConvertGraph(s, &output_graph_def));

  auto remove_graph_sequence_number = [](std::string node_name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSconvert_graph_testDTcc mht_8(mht_8_v, 388, "", "./tensorflow/compiler/tf2tensorrt/convert/convert_graph_test.cc", "lambda");

    const std::regex pattern("TRTEngineOp_[0-9]+_");
    return std::regex_replace(node_name, pattern, "TRTEngineOp_");
  };
  int num_trt_ops = 0;
  for (const NodeDef& node : output_graph_def.node()) {
    std::string node_name = node.name();
    if (node.op() != "TRTEngineOp") continue;
    node_name = remove_graph_sequence_number(node_name);
    if (node_name == "TRTEngineOp_001") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("input", node.input(0));
      ++num_trt_ops;
    } else if (node_name == "TRTEngineOp_000") {
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("TRTEngineOp_001", remove_graph_sequence_number(node.input(0)));
      EXPECT_EQ("reshape2", node.input(1));
      ++num_trt_ops;
    }
  }
  EXPECT_EQ(2, num_trt_ops);
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
