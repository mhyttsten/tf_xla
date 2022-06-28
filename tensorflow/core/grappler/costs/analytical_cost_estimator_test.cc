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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimator_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimator_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimator_testDTcc() {
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

#include "tensorflow/core/grappler/costs/virtual_scheduler.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/analytical_cost_estimator.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class AnalyticalCostEstimatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimator_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/grappler/costs/analytical_cost_estimator_test.cc", "SetUp");

    // Initializes cluster_ and placer_.
    std::unordered_map<string, DeviceProperties> devices;
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    cpu_device.set_num_cores(4);
    cpu_device.set_frequency(2600);
    cpu_device.set_bandwidth(24 * 1024 * 1024);
    devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
    DeviceProperties gpu_device;
    gpu_device.set_type("GPU");
    gpu_device.set_num_cores(12);
    gpu_device.set_frequency(1100);
    gpu_device.set_bandwidth(180 * 1024 * 1024);
    (*gpu_device.mutable_environment())["architecture"] = "6";
    devices["/job:localhost/replica:0/task:0/device:GPU:0"] = gpu_device;

    cluster_.reset(new VirtualCluster(devices));
  }

  GrapplerItem CreateMiniGraph() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimator_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/grappler/costs/analytical_cost_estimator_test.cc", "CreateMiniGraph");

    const int batch = 1;
    const int width = 28;
    const int height = 28;
    const int num_channels = 1;
    const int num_labels = 10;
    const int kernel_size = 3;
    const int conv_filters = 32;

    Scope s = Scope::NewRootScope();
    auto images = ops::RandomUniform(
        s.WithOpName("image"), {batch, width, height, num_channels}, DT_FLOAT);
    auto labels = ops::RandomUniform(s.WithOpName("label"), {batch, num_labels},
                                     DT_FLOAT);
    auto w = ops::Variable(
        s.WithOpName("W"),
        {kernel_size, kernel_size, num_channels, conv_filters}, DT_FLOAT);
    auto b = ops::Variable(s.WithOpName("B"), {conv_filters}, DT_FLOAT);
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), images, w, {1, 1, 1, 1}, "SAME");
    auto bias = ops::Add(s.WithOpName("bias"), conv, b);
    auto relu = ops::Relu(s.WithOpName("relu"), bias);
    auto flat_shape = ops::Const(s.WithOpName("flat_shape"),
                                 {batch, width * height * conv_filters});
    auto flat = ops::Reshape(s.WithOpName("flat"), relu, flat_shape);

    auto w2 =
        ops::Variable(s.WithOpName("W2"),
                      {width * height * conv_filters, num_labels}, DT_FLOAT);
    auto b2 = ops::Variable(s.WithOpName("B2"), {num_labels}, DT_FLOAT);
    auto matmul = ops::MatMul(s.WithOpName("matmul"), flat, w2);
    auto logits = ops::Add(s.WithOpName("logits"), matmul, b2);
    auto softmax = ops::Softmax(s.WithOpName("softmax"), logits);
    auto lsm = ops::Log(s.WithOpName("lsm"), softmax);

    GrapplerItem item;
    item.fetch.push_back("lsm");
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    return item;
  }

  std::unique_ptr<VirtualCluster> cluster_;
};

TEST_F(AnalyticalCostEstimatorTest, SimpleTest) {
  GrapplerItem item = CreateMiniGraph();

  AnalyticalCostEstimator estimator(cluster_.get(), /*use_static_shapes=*/true,
                                    /*use_aggressive_shape_inference=*/true);
  TF_ASSERT_OK(estimator.Initialize(item));

  RunMetadata run_metadata;
  Costs summary;
  TF_ASSERT_OK(estimator.PredictCosts(item.graph, &run_metadata, &summary));

  EXPECT_EQ(Costs::NanoSeconds(9158), summary.execution_time);
  // Note there are totally 17 nodes (RandomUniform creates 2 nodes), but
  // grappler will not process "label", therefore we have 15 here instead
  EXPECT_EQ(15, summary.num_ops_total);

  // Make this estimate accurate:
  // TODO(http://b/70031255): Accurate estimator for RandomUniform op needed
  //
  // Change to EXPECT_FALSE when the above TODOs are done:
  EXPECT_TRUE(summary.inaccurate);
  EXPECT_EQ(0, summary.num_ops_with_unknown_shapes);
}

}  // end namespace grappler
}  // end namespace tensorflow
