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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc() {
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

#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"

#include <atomic>

#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kDevice[] = "/device:CPU:0";

class TestOptimizer : public CustomGraphOptimizer {
 public:
  static void SetOptimized(const bool flag_value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "SetOptimized");
 optimized_ = flag_value; }
  static bool IsOptimized() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "IsOptimized");
 return optimized_; }

  TestOptimizer() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "TestOptimizer");
}
  string name() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "name");
 return "test_optimizer"; }
  bool UsesFunctionLibrary() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_4(mht_4_v, 233, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "UsesFunctionLibrary");
 return false; }

  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config =
                  nullptr) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_5(mht_5_v, 239, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "Init");

    return Status::OK();
  }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_6(mht_6_v, 247, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "Optimize");

    optimized_ = true;
    *optimized_graph = item.graph;
    return Status::OK();
  }

 private:
  static bool optimized_;
};

bool TestOptimizer::optimized_;

REGISTER_GRAPH_OPTIMIZER(TestOptimizer);

class TestGraphOptimizer : public TestOptimizer {
 public:
  string name() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_7(mht_7_v, 266, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "name");
 return "test_graph_optimizer"; }
};

REGISTER_GRAPH_OPTIMIZER(TestGraphOptimizer);

class TestOptimizerWithParams : public TestOptimizer {
 public:
  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_8(mht_8_v, 277, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "Init");

    CHECK(config != nullptr);
    return Status::OK();
  }
};

REGISTER_GRAPH_OPTIMIZER(TestOptimizerWithParams);

// Record various properties of the GrapplerItems passed for optimization.
class GrapplerItemPropertiesAccumulator : public CustomGraphOptimizer {
 public:
  static void SetOptimizationOptions(
      gtl::FlatMap<string, GrapplerItem::OptimizationOptions>*
          optimization_options) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_9(mht_9_v, 293, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "SetOptimizationOptions");

    optimization_options_ = optimization_options;
  }
  static void ResetOptimizationOptions() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_10(mht_10_v, 299, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "ResetOptimizationOptions");
 optimization_options_ = nullptr; }

  GrapplerItemPropertiesAccumulator() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_11(mht_11_v, 304, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "GrapplerItemPropertiesAccumulator");
}
  string name() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_12(mht_12_v, 308, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "name");

    return "grappler_item_properties_accumulator";
  }
  bool UsesFunctionLibrary() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_13(mht_13_v, 314, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "UsesFunctionLibrary");
 return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_14(mht_14_v, 320, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "Init");

    return Status::OK();
  }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_15(mht_15_v, 328, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "Optimize");

    *optimized_graph = item.graph;
    if (optimization_options_) {
      optimization_options_->insert({item.id, item.optimization_options()});
    }
    return Status::OK();
  }

 private:
  static gtl::FlatMap<string, GrapplerItem::OptimizationOptions>*
      optimization_options_;
};

gtl::FlatMap<string, GrapplerItem::OptimizationOptions>*
    GrapplerItemPropertiesAccumulator::optimization_options_;

REGISTER_GRAPH_OPTIMIZER(GrapplerItemPropertiesAccumulator);

class MetaOptimizerTest : public GrapplerTest {};

TEST_F(MetaOptimizerTest, RunsCustomOptimizer) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("TestOptimizer");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunsCustomOptimizerWithParams) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("TestOptimizerWithParams");
  auto* custom_config = rewriter_config.add_custom_optimizers();
  custom_config->set_name("TestOptimizerWithParams");
  (*custom_config->mutable_parameter_map())["foo"] = AttrValue();

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunsCustomOptimizerAndCustomGraphOptimizer) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  TestGraphOptimizer::SetOptimized(false);
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("TestOptimizer");
  auto customGraphOptimizer = rewriter_config.add_custom_optimizers();
  customGraphOptimizer->set_name("TestGraphOptimizer");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
  EXPECT_TRUE(TestGraphOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunsPluginOptimizer) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"/device:GPU:0"});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.set_min_graph_nodes(-1);

  const auto creator = []() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_16(mht_16_v, 425, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "lambda");
 return new TestOptimizer; };
  ConfigList config_list;
  config_list.disable_model_pruning = true;
  PluginGraphOptimizerRegistry::RegisterPluginOptimizerOrDie(creator, "GPU",
                                                             config_list);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunOptimizersTwice) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

TEST_F(MetaOptimizerTest, RunToggleOptimizersAndCustomGraphOptimizerTwice) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  auto customGraphOptimizer = rewriter_config.add_custom_optimizers();
  customGraphOptimizer->set_name("TestGraphOptimizer");
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestGraphOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibrary) {
  using test::function::NDef;

  // Enable only function optimization.
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();

  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_function_optimization(RewriterConfig::ON);
  rewriter_config.add_optimizers("function");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);

  // Define function library:
  //
  //   MyMul(x, y)    = x * y
  //  *MySquare(x)    = MyMul(x, x)
  //  *MyQuadratic(x) = MySquare(MySquare(x))
  //
  //  * - marked as noinline

  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /*ret_def=*/
      {{"z", "mul:z:0"}});

  FunctionDef square_func = FunctionDefHelper::Create(
      "MySquare", {"x:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"my_mul"}, "MyMul", {"x", "x"}, {{"T", "$T"}}}},
      /*ret_def=*/
      {{"z", "my_mul:z:0"}});
  (*square_func.mutable_attr())["_noinline"].set_b(true);

  FunctionDef quadratic_func = FunctionDefHelper::Create(
      "MyQuadratic", {"x:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"square"}, "MySquare", {"x"}, {{"T", "$T"}}},
       {{"quadratic"}, "MySquare", {"square:z"}, {{"T", "$T"}}}},
      /*ret_def=*/
      {{"z", "quadratic:z:0"}});
  (*quadratic_func.mutable_attr())["_noinline"].set_b(true);

  // Tensorflow graph:
  //
  //   a = tf.Placeholder(tf.float);
  //   b = tf.Placeholder(tf.int32);
  //
  //   square = MySquare(a);        // a^2
  //   quadratic = MyQuadratic(b);  // b^4
  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_INT32}}, kDevice),
       // Calls into function library
       NDef("square", "MySquare", {"a"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("quadratic", "MyQuadratic", {"b"}, {{"T", DT_INT32}}, kDevice),
       // Forward outputs
       NDef("out_s", "Identity", {"square:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("out_q", "Identity", {"quadratic:0"}, {{"T", DT_INT32}}, kDevice)},
      /*funcs=*/
      {mul_func, square_func, quadratic_func});

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  FunctionLibraryDefinition optimized_flib(OpRegistry::Global(),
                                           output.library());

  // Specialized and optimized functions should be added to the graph.
  EXPECT_EQ(5, optimized_flib.num_functions());

  // Get a specialized function name.
  const auto specialized_name = [](const string& fn, const string& node,
                                   const string& id) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("fn: \"" + fn + "\"");
   mht_17_v.push_back("node: \"" + node + "\"");
   mht_17_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_17(mht_17_v, 557, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "lambda");

    return absl::Substitute("$0_specialized_for_$1_at_$2", fn, node, id);
  };

  // MyQuadratic should be specialized once:
  //   0. 'quadratic' node in the main graph
  const string optimized_0 =
      specialized_name("MyQuadratic", "quadratic", "tf_graph");

  // MySquare should be specialized and optimized for 3 instantiations:
  //   1.  'square' node in the main graph
  //   2.  'square' node in the MyQuadratic specialization
  //   3*. 'quadratic' node in the MyQuadratic specialization
  //        has identical instantiation context to #2

  const string optimized_1 = specialized_name("MySquare", "square", "tf_graph");
  const string optimized_2 =
      specialized_name("MySquare", "square", optimized_0);

  const FunctionDef* optimized_func_0 = optimized_flib.Find(optimized_0);
  const FunctionDef* optimized_func_1 = optimized_flib.Find(optimized_1);
  const FunctionDef* optimized_func_2 = optimized_flib.Find(optimized_2);

  ASSERT_NE(optimized_func_0, nullptr);
  ASSERT_NE(optimized_func_1, nullptr);
  ASSERT_NE(optimized_func_2, nullptr);

  // Graph should call optimized function.
  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "square" && ++count) {
      EXPECT_EQ(optimized_1, node.op());
    } else if (node.name() == "quadratic" && ++count) {
      EXPECT_EQ(optimized_0, node.op());
    }
  }
  EXPECT_EQ(2, count);

  // Specialized MySquare should call specialized functions.
  count = 0;
  for (const NodeDef& node : optimized_func_0->node_def()) {
    if (node.name() == "square" && ++count) {
      EXPECT_EQ(optimized_2, node.op());
    } else if (node.name() == "quadratic" && ++count) {
      EXPECT_EQ(optimized_2, node.op());
    }
  }
  EXPECT_EQ(2, count);

  const std::vector<const FunctionDef*> optimized_funcs = {optimized_func_1,
                                                           optimized_func_2};

  // MyMul should be inlined into all optimized versions of MySquare.
  for (const FunctionDef* optimized_func : optimized_funcs) {
    count = 0;
    for (const NodeDef& node : optimized_func->node_def()) {
      if (node.name() == "Func/my_mul/input/_0" && ++count) {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ(1, node.input_size());
        EXPECT_EQ("x", node.input(0));
      } else if (node.name() == "Func/my_mul/input/_1" && ++count) {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ(1, node.input_size());
        EXPECT_EQ("x", node.input(0));
      } else if (node.name() == "my_mul/mul" && ++count) {
        EXPECT_EQ("Mul", node.op());
        EXPECT_EQ(2, node.input_size());
        EXPECT_EQ("Func/my_mul/input/_0:output:0", node.input(0));
        EXPECT_EQ("Func/my_mul/input/_1:output:0", node.input(1));
      }
      EXPECT_TRUE(node.device().empty());
    }
    EXPECT_EQ(3, count);
    ASSERT_EQ(1, optimized_func->ret().size());
    EXPECT_EQ("Func/my_mul/output/_2:output:0", optimized_func->ret().at("z"));
  }

  item.fetch = {"out_s", "out_q"};
  item.feed.emplace_back("a", test::AsScalar<float>(2.0f));
  item.feed.emplace_back("b", test::AsScalar<int>(4));
  auto tensors_expected = EvaluateFetchNodes(item);

  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<int>(tensors_expected[1], tensors[1]);
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibraryPruneUnusedOutputs) {
  using test::function::NDef;

  ConfigProto config_proto;
  MetaOptimizer optimizer(nullptr, config_proto);

  // MyMul computes x*y three times and has three output values.
  FunctionDef my_mul = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z0:T", "z1:T", "z2:T"}, {"T: {float, int32}"},
      {{{"output0"}, "Mul", {"x", "y"}, {{"T", "$T"}}},
       {{"output1"}, "Mul", {"x", "y"}, {{"T", "$T"}}},
       {{"output2"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /*ret_def=*/
      {{"z0", "output0:z:0"}, {"z1", "output1:z:0"}, {"z2", "output2:z:0"}});

  // Call MyMyl and forward all three outputs.
  FunctionDef my_fwd = FunctionDefHelper::Create(
      "Fwd", {"x:T", "y:T"}, {"z0:T", "z1:T", "z2:T"}, {"T: {float, int32}"},
      {{{"output"}, "MyMul", {"x", "y"}, {{"T", "$T"}}}},
      /*ret_def=*/
      {{"z0", "output:z0:0"}, {"z1", "output:z1:0"}, {"z2", "output:z2:0"}});

  // Mark both functions as `_noinline` to trigger specialization.
  (*my_mul.mutable_attr())["_noinline"].set_b(true);
  (*my_fwd.mutable_attr())["_noinline"].set_b(true);
  /*funcs=*/
  std::vector<FunctionDef> function_library = {my_mul, my_fwd};

  // Tensorflow graph:
  //   a = Placeholder[T=float]
  //   b = Placeholder[T=float]
  //   fwd = Fwd(a, b)
  //
  // Fetch fwd:2 via Identity node.
  GrapplerItem item;
  item.id = "tf_graph";
  item.fetch = {"ret"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("fwd", "Fwd", {"a", "b"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("ret", "Identity", {"fwd:2"}, {{"T", DT_FLOAT}}, kDevice)},
      function_library);

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  FunctionLibraryDefinition optimized_flib(OpRegistry::Global(),
                                           output.library());

  // Specialized functions should be added to the graph.
  EXPECT_EQ(3, optimized_flib.num_functions());

  // Expected names of the specialized functions.
  const string specialized_my_fwd = "Fwd_specialized_for_fwd_at_tf_graph";
  const string specialized_my_mul =
      absl::StrCat("MyMul_specialized_for_output_at_", specialized_my_fwd);

  // Specialized MyMul should have just one output argument.
  FunctionDef expected_my_mul = FunctionDefHelper::Create(
      specialized_my_mul, {"x:float", "y:float"}, {"z2:float"}, {},
      {{{"output2"}, "Mul", {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z2", "output2:z:0"}});

  // Specialized Fwd should also have just one output argument.
  FunctionDef expected_my_fwd = FunctionDefHelper::Create(
      specialized_my_fwd, {"x:float", "y:float"}, {"z2:float"}, {},
      {{{"output"}, specialized_my_mul, {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z2", "output:z2:0"}});

  const FunctionDef* my_mul_spec = optimized_flib.Find(specialized_my_mul);
  const FunctionDef* my_fwd_spec = optimized_flib.Find(specialized_my_fwd);

  ASSERT_NE(my_mul_spec, nullptr);
  ASSERT_NE(my_fwd_spec, nullptr);

  CompareFunctions(expected_my_mul, *my_mul_spec);
  CompareFunctions(expected_my_fwd, *my_fwd_spec);

  item.feed.emplace_back("a", test::AsScalar<float>(2.0f));
  item.feed.emplace_back("b", test::AsScalar<float>(4.0f));
  auto tensors_expected = EvaluateFetchNodes(item);

  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibraryPruneFunctionBody) {
  using test::function::NDef;

  // Enable function optimization and pruning.
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();

  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_function_optimization(RewriterConfig::ON);
  rewriter_config.add_optimizers("function");
  rewriter_config.add_optimizers("pruning");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);

  // MyFunc defines two Mul nodes inside function body and two corresponding
  // function outputs.
  FunctionDef my_func = FunctionDefHelper::Create(
      "MyFunc", {"x:T", "y:T"}, {"z1:T", "z2:T"}, {"T: {float, double}"},
      {{{"mul1"}, "Mul", {"x", "y"}, {{"T", "$T"}}},
       {{"mul2"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /*ret_def=*/
      {{"z1", "mul1:z:0"}, {"z2", "mul2:z:0"}});
  (*my_func.mutable_attr())["_noinline"].set_b(true);

  // Tensorflow graph:
  //
  //   a = tf.Placeholder(tf.float);
  //   b = tf.Placeholder(tf.int32);
  //
  //   fn1 = MyFunc(a, b);
  //   fn2 = MyFunc(a, b);
  //
  // Fetch: fn1:0 and fn2:1 via Identity nodes.
  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("fn1", "MyFunc", {"a", "b"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("fn2", "MyFunc", {"a", "b"}, {{"T", DT_FLOAT}}, kDevice),
       // Read outputs of function call nodes
       NDef("out_fn1", "Identity", {"fn1:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("out_fn2", "Identity", {"fn2:1"}, {{"T", DT_FLOAT}}, kDevice)},
      /*funcs=*/
      {my_func});

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  FunctionLibraryDefinition optimized_flib(OpRegistry::Global(),
                                           output.library());

  // Specialized and optimized functions should be added to the graph.
  EXPECT_EQ(2, optimized_flib.num_functions());

  // Expected names of the specialized and optimized functions.
  const string optimized_fn1 = "MyFunc_specialized_for_fn1_at_tf_graph";
  const string optimized_fn2 = "MyFunc_specialized_for_fn2_at_tf_graph";

  const FunctionDef* optimized_func_fn1 = optimized_flib.Find(optimized_fn1);
  const FunctionDef* optimized_func_fn2 = optimized_flib.Find(optimized_fn2);

  ASSERT_NE(optimized_func_fn1, nullptr);
  ASSERT_NE(optimized_func_fn2, nullptr);

  // Graph should call optimized function.
  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "fn1" && ++count) {
      EXPECT_EQ(optimized_fn1, node.op());
    } else if (node.name() == "fn2" && ++count) {
      EXPECT_EQ(optimized_fn2, node.op());
    }
  }
  EXPECT_EQ(2, count);

  // Specialized MyFuncs should have just one Mul node and single output arg.

  // 1. Specialized for fn1:0.
  ASSERT_EQ(1, optimized_func_fn1->node_def_size());
  EXPECT_EQ(1, optimized_func_fn1->signature().output_arg_size());
  EXPECT_EQ("z1", optimized_func_fn1->signature().output_arg(0).name());
  EXPECT_EQ("mul1", optimized_func_fn1->node_def(0).name());

  // 2. Specialized for fn2:1.
  ASSERT_EQ(1, optimized_func_fn2->node_def_size());
  EXPECT_EQ(1, optimized_func_fn2->signature().output_arg_size());
  EXPECT_EQ("z2", optimized_func_fn2->signature().output_arg(0).name());
  EXPECT_EQ("mul2", optimized_func_fn2->node_def(0).name());

  // Verify that output tensors are equal.
  item.fetch = {"out_fn1", "out_fn2"};
  item.feed.emplace_back("a", test::AsScalar<float>(2.0f));
  item.feed.emplace_back("b", test::AsScalar<float>(3.123f));
  auto tensors_expected = EvaluateFetchNodes(item);

  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<float>(tensors_expected[1], tensors[1]);
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibraryWithRestrictions) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  // We will record what type of optimizations meta optimizer allows for each
  // GrapplerItem (main graph and graphs for each function).
  gtl::FlatMap<string, GrapplerItem::OptimizationOptions> optimization_options;
  GrapplerItemPropertiesAccumulator::SetOptimizationOptions(
      &optimization_options);

  // Just record properties of optimized Grappler items.
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();

  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.add_optimizers("GrapplerItemPropertiesAccumulator");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);

  // Define simple function library with two identical mul functions.
  FunctionDef mul_func_1 = FunctionDefHelper::Create(
      "MyMul1", {"x:float", "y:float"}, {"z:float"}, {},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "mul:z:0"}});

  FunctionDef mul_func_2 = FunctionDefHelper::Create(
      "MyMul2", {"x:float", "y:float"}, {"z:float"}, {},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "mul:z:0"}});

  // Tensorflow graph:
  //
  //   x0 = tf.Placeholder(tf.float);
  //   x1 = tf.Placeholder(tf.float);
  //   dy = tf.Placeholder(tf.float);
  //
  //   mul_1 = MyMul1(x0, x1);
  //   mul_2 = MyMul2(x0, x1);
  //   dx = SymbolicGradient({x0, x1, dy}, f=MyMul2)
  GrapplerItem item;
  item.id = "main";
  item.graph = test::function::GDef(
      {NDef("x0", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("x1", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("dy", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("mul_1", "MyMul1", {"x0", "x1"}, {}, kDevice),
       NDef("mul_2", "MyMul2", {"x0", "x1"}, {}, kDevice),
       // Symbolic gradient of a MyMul2
       NDef("dx", "SymbolicGradient", {"x0", "x1", "dy"},
            {{"f", FDH::FunctionRef("MyMul2", {})},
             {"Tin", DataTypeSlice{DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT}}},
            kDevice)},
      /*funcs=*/
      {mul_func_1, mul_func_2});
  item.fetch = {"mul_1", "mul_2", "dx"};

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Our custom optimizer must be called for the main graph and for the two
  // functions.
  ASSERT_EQ(optimization_options.size(), 3);

  auto optimization_options_main =
      gtl::FindOrNull(optimization_options, "main");
  ASSERT_NE(optimization_options_main, nullptr);
  EXPECT_TRUE(optimization_options_main->allow_non_differentiable_rewrites);

  auto optimization_options_my_mul_1 =
      gtl::FindOrNull(optimization_options, "MyMul1");
  ASSERT_NE(optimization_options_my_mul_1, nullptr);
  EXPECT_TRUE(optimization_options_my_mul_1->allow_non_differentiable_rewrites);

  auto optimization_options_my_mul_2 =
      gtl::FindOrNull(optimization_options, "MyMul2");
  ASSERT_NE(optimization_options_my_mul_2, nullptr);
  EXPECT_FALSE(
      optimization_options_my_mul_2->allow_non_differentiable_rewrites);
}

class SleepingOptimizer : public CustomGraphOptimizer {
 public:
  SleepingOptimizer() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_18(mht_18_v, 934, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "SleepingOptimizer");
}
  string name() const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_19(mht_19_v, 938, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "name");
 return "test_optimizer"; }
  bool UsesFunctionLibrary() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_20(mht_20_v, 942, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "UsesFunctionLibrary");
 return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_21(mht_21_v, 948, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "Init");

    return Status::OK();
  }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_22(mht_22_v, 956, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "Optimize");

    *optimized_graph = item.graph;
    Env::Default()->SleepForMicroseconds(1000000);
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    optimized_graph->add_node();
    return Status::OK();
  }
};

REGISTER_GRAPH_OPTIMIZER(SleepingOptimizer);

TEST_F(MetaOptimizerTest, OptimizerTimesOut) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config;
  RewriterConfig& rewriter_config =
      *config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("SleepingOptimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_timeout_ms(500);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);

  GraphDef output;
  GraphDef original = item.graph;
  const Status status =
      RunMetaOptimizer(std::move(item), config, nullptr, nullptr, &output);
  EXPECT_EQ(status.error_message(), "meta_optimizer exceeded deadline.");
  // Make sure the graph was reverted to the original regardless of when the
  // optimizer timed out.
  CompareGraphs(original, output);
}

TEST_F(MetaOptimizerTest, MetaOptimizerTimesOut) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config;
  RewriterConfig& rewriter_config =
      *config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("SleepingOptimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_timeout_ms(1500);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);

  GraphDef output;
  const int original_node_size = item.graph.node_size();
  const Status status =
      RunMetaOptimizer(std::move(item), config, nullptr, nullptr, &output);
  EXPECT_EQ(status.error_message(), "meta_optimizer exceeded deadline.");
  // The meta optimizer should manage to finish one iteration.
  EXPECT_EQ(original_node_size + 1, output.node_size());
}

TEST_F(MetaOptimizerTest, OptimizerDoesNotTimeOut) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config;
  RewriterConfig& rewriter_config =
      *config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("SleepingOptimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_timeout_ms(2500);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  GraphDef output;
  const int original_node_size = item.graph.node_size();
  const Status status =
      RunMetaOptimizer(std::move(item), config, nullptr, nullptr, &output);
  TF_EXPECT_OK(status);
  // The meta optimizer should manage to finish two iterations.
  EXPECT_EQ(original_node_size + 2, output.node_size());
}

TEST_F(MetaOptimizerTest, RunPostOptimizationVerifiersOnValidGraph) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config_proto;
  auto& post_optimization_verifier_config =
      *config_proto.mutable_graph_options()
           ->mutable_rewrite_options()
           ->mutable_post_optimization_verifier_config();
  post_optimization_verifier_config.set_structure_verifier(VerifierConfig::ON);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

TEST_F(MetaOptimizerTest, RunInterOptimizerVerifiersOnValidGraph) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config_proto;
  auto& inter_optimizer_verifier_config =
      *config_proto.mutable_graph_options()
           ->mutable_rewrite_options()
           ->mutable_inter_optimizer_verifier_config();
  inter_optimizer_verifier_config.set_structure_verifier(VerifierConfig::ON);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

TEST_F(MetaOptimizerTest, RunPostOptimizationVerifiersOnInvalidGraph) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  gtl::FlatMap<string, GrapplerItem::OptimizationOptions> optimization_options;
  GrapplerItemPropertiesAccumulator::SetOptimizationOptions(
      &optimization_options);

  // Define simple function library with two identical mul functions.
  FunctionDef mul_func_1 =
      FunctionDefHelper::Create("MyMul1", {"x:float", "y:float"}, {"z:float"},
                                {}, {{{"mul"}, "Mul", {"x", "y"}, {}}},
                                /*ret_def=*/
                                {{"z", "mul:z:0"}});

  FunctionDef mul_func_2 =
      FunctionDefHelper::Create("MyMul2", {"x:float", "y:float"}, {"z:float"},
                                {}, {{{"mul"}, "Mul", {"x", "y"}, {}}},
                                /*ret_def=*/
                                {{"z", "mul:z:0"}});

  // Tensorflow graph:
  //
  //   x0 = tf.Placeholder(tf.float);
  //   x1 = tf.Placeholder(tf.float);
  //   dy = tf.Placeholder(tf.float);
  //
  //   mul_1 = MyMul1(x0, x1);
  //   mul_2 = MyMul2(x0, x1);
  //   dx = SymbolicGradient({x0, x1, dy}, f=MyMul2)
  GrapplerItem item;
  item.id = "main";
  item.graph = test::function::GDef(
      {NDef("x0", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("x1", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("dy", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("mul_1", "MyMul1", {"x0", "x1"}, {}, kDevice),
       NDef("mul_2", "MyMul2", {"x0", "x1"}, {}, kDevice),
       // Symbolic gradient of a MyMul2
       NDef("dx", "SymbolicGradient", {"x0", "x1", "dy"},
            {{"f", FDH::FunctionRef("MyMul2", {})},
             {"Tin", DataTypeSlice{DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT}}},
            kDevice)},
      /*funcs=*/
      {mul_func_1, mul_func_2});
  item.fetch = {"mul_1", "mul_2", "dx"};

  GraphDef output;

  // Call Optimize with post optimization verifiers.
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();

  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.add_optimizers("GrapplerItemPropertiesAccumulator");
  rewriter_config.set_min_graph_nodes(-1);
  auto& post_optimization_verifier_config =
      *config_proto.mutable_graph_options()
           ->mutable_rewrite_options()
           ->mutable_post_optimization_verifier_config();
  post_optimization_verifier_config.set_structure_verifier(VerifierConfig::ON);

  MetaOptimizer optimizer_with_post_verifiers(nullptr, config_proto);
  Status status =
      optimizer_with_post_verifiers.Optimize(nullptr, item, &output);
  EXPECT_EQ(status.code(), errors::Code::INVALID_ARGUMENT);
  EXPECT_TRUE(absl::StrContains(
      status.error_message(),
      "NodeDef expected inputs 'float' do not match 3 inputs specified"));
}

TEST_F(MetaOptimizerTest, RunInterOptimizerVerifiersOnInvalidGraph) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  gtl::FlatMap<string, GrapplerItem::OptimizationOptions> optimization_options;
  GrapplerItemPropertiesAccumulator::SetOptimizationOptions(
      &optimization_options);

  // Define simple function library with two identical mul functions.
  FunctionDef mul_func_1 =
      FunctionDefHelper::Create("MyMul1", {"x:float", "y:float"}, {"z:float"},
                                {}, {{{"mul"}, "Mul", {"x", "y"}, {}}},
                                /*ret_def=*/
                                {{"z", "mul:z:0"}});

  FunctionDef mul_func_2 =
      FunctionDefHelper::Create("MyMul2", {"x:float", "y:float"}, {"z:float"},
                                {}, {{{"mul"}, "Mul", {"x", "y"}, {}}},
                                /*ret_def=*/
                                {{"z", "mul:z:0"}});

  // Tensorflow graph:
  //
  //   x0 = tf.Placeholder(tf.float);
  //   x1 = tf.Placeholder(tf.float);
  //   dy = tf.Placeholder(tf.float);
  //
  //   mul_1 = MyMul1(x0, x1);
  //   mul_2 = MyMul2(x0, x1);
  //   dx = SymbolicGradient({x0, x1, dy}, f=MyMul2)
  GrapplerItem item;
  item.id = "main";
  item.graph = test::function::GDef(
      {NDef("x0", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("x1", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("dy", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("x1", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("mul_1", "MyMul1", {"x0", "x1"}, {}, kDevice),
       NDef("mul_2", "MyMul2", {"x0", "x1"}, {}, kDevice),
       // Symbolic gradient of a MyMul2
       NDef("dx", "SymbolicGradient", {"x0", "x1", "dy"},
            {{"f", FDH::FunctionRef("MyMul2", {})},
             {"Tin", DataTypeSlice{DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT}}},
            kDevice)},
      /*funcs=*/
      {mul_func_1, mul_func_2});
  item.fetch = {"mul_1", "mul_2", "dx"};

  GraphDef output;

  // Call Optimize with post optimization verifiers.
  ConfigProto config_proto;
  // Call Optimize with inter optimizer verifiers.
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.add_optimizers("GrapplerItemPropertiesAccumulator");
  rewriter_config.set_min_graph_nodes(-1);
  auto& inter_optimizer_verifier_config =
      *config_proto.mutable_graph_options()
           ->mutable_rewrite_options()
           ->mutable_inter_optimizer_verifier_config();
  inter_optimizer_verifier_config.set_structure_verifier(VerifierConfig::ON);

  MetaOptimizer optimizer_with_inter_verifiers(nullptr, config_proto);
  Status status =
      optimizer_with_inter_verifiers.Optimize(nullptr, item, &output);
  EXPECT_EQ(status.code(), errors::Code::INVALID_ARGUMENT);
  EXPECT_TRUE(absl::StrContains(
      status.error_message(),
      "NodeDef expected inputs 'float' do not match 3 inputs specified"));
}

TEST_F(MetaOptimizerTest, CompressConstants) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Tensor zeros_t(DT_FLOAT, TensorShape({64}));
  Tensor ones_t(DT_FLOAT, TensorShape({64}));
  for (int i = 0; i < 64; ++i) {
    zeros_t.flat<float>()(i) = 0.0f;
    ones_t.flat<float>()(i) = 1.0f;
  }
  Output zeros = ops::Const(scope.WithOpName("zeros"), zeros_t);
  Output host_ones = ops::Const(scope.WithOpName("host_ones"), ones_t);
  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  ASSERT_EQ(item.graph.node(1).name(), "host_ones");
  // There is not C++ api for HostConst, so we manually change the node type
  // here.
  item.graph.mutable_node(1)->set_op("HostConst");
  item.fetch = {"zeros", "host_ones"};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, {});

  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.set_min_graph_nodes(-1);
  MetaOptimizer optimizer(/*cpu_device=*/nullptr, config_proto);
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(/*cluster=*/nullptr, item, &output));

  bool found_zeros = false;
  bool found_host_ones = false;
  ASSERT_EQ(output.node_size(), 2);
  for (const auto& node : output.node()) {
    if (node.name() == "zeros") {
      found_zeros = true;
      EXPECT_EQ(node.op(), "Const");
      const TensorProto& zeroes_t = node.attr().at("value").tensor();
      EXPECT_EQ(zeroes_t.float_val_size(), 0);
    } else if (node.name() == "host_ones") {
      found_host_ones = true;
      EXPECT_EQ(node.op(), "HostConst");
      const TensorProto& ones_t = node.attr().at("value").tensor();
      EXPECT_EQ(ones_t.float_val_size(), 1);
      EXPECT_EQ(ones_t.float_val(0), 1.0f);
    }
  }

  EXPECT_TRUE(found_zeros);
  EXPECT_TRUE(found_host_ones);

  auto tensors = EvaluateNodes(output, item.fetch, {});
  ASSERT_EQ(tensors.size(), 2);
  ASSERT_EQ(tensors_expected.size(), 2);
  for (int i = 0; i < 2; ++i) {
    test::ExpectTensorEqual<float>(tensors[i], tensors_expected[i]);
  }
}

// Tests for checking expected behavior when skipping tf.data functions in
// meta optimizer.

// Custom optimizer which counts the number of calls of its method `Optimize`
// across all class instances.
class TfDataTestOptimizer : public CustomGraphOptimizer {
 public:
  static void InitCount() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_23(mht_23_v, 1284, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "InitCount");
 count_ = 0; }
  static int GetCount() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_24(mht_24_v, 1288, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "GetCount");
 return count_; }

  TfDataTestOptimizer() = default;
  ~TfDataTestOptimizer() override = default;
  TfDataTestOptimizer(const TfDataTestOptimizer&) = delete;
  TfDataTestOptimizer& operator=(const TfDataTestOptimizer& other) = delete;

  std::string name() const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_25(mht_25_v, 1298, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "name");
 return "tf_data_test_optimizer"; }
  bool UsesFunctionLibrary() const override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_26(mht_26_v, 1302, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "UsesFunctionLibrary");
 return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_27(mht_27_v, 1308, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "Init");

    return Status::OK();
  }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_28(mht_28_v, 1316, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "Optimize");

    ++count_;
    *optimized_graph = item.graph;
    return Status::OK();
  }

 private:
  static std::atomic<int> count_;
};

std::atomic<int> TfDataTestOptimizer::count_;

REGISTER_GRAPH_OPTIMIZER(TfDataTestOptimizer);

// Type for specifying how the inner function is nested inside the outer
// function.
enum class FuncNestingType {
  CallFromNode = 0,
  CallFromAttr = 1,
  CallFromList = 2
};

// Test fixture for parametrized testing.
class TfDataTestFixture
    : public ::testing::TestWithParam<std::tuple<bool, bool, FuncNestingType>> {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_29(mht_29_v, 1345, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "SetUp");

    is_inner_func_tf_data_ = std::get<0>(GetParam());
    is_outer_func_tf_data_ = std::get<1>(GetParam());
    func_nesting_type_ = std::get<2>(GetParam());
  }
  // Controls which of the functions is flagged as tf.data function.
  bool is_inner_func_tf_data_ = false;
  bool is_outer_func_tf_data_ = false;
  // Controls how the inner function is nested inside the outer function.
  FuncNestingType func_nesting_type_ = FuncNestingType::CallFromNode;
};

// Helper functions for setting up the call of `inner_func` inside of
// `outer_func`.

void SetUpCallFromNode(FunctionDef& outer_func) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_30(mht_30_v, 1363, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "SetUpCallFromNode");

  // Call `inner_func` from a node in `outer_func`.
  outer_func = FunctionDefHelper::Create(
      "outer_func", {"x:float"}, {"z:float"}, {},
      /*node_def=*/
      {{{"inner_func"}, "inner_func", {"x", "x"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "inner_func:z:0"}});
}

void SetUpCallFromAttr(FunctionDef& outer_func) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_31(mht_31_v, 1376, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "SetUpCallFromAttr");

  // Call `inner_func` from an attribute in a node in `outer_func`.
  outer_func = FunctionDefHelper::Create(
      "outer_func", {"x:float"}, {"z:float"}, {},
      /*node_def=*/
      {{{"identity"},
        "Identity",
        {"x"},
        {{"T", DT_FLOAT},
         {"f", FunctionDefHelper::FunctionRef("inner_func", {})}}}},
      /*ret_def=*/
      {{"z", "x"}});
}

void SetUpCallFromList(FunctionDef& outer_func) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizer_testDTcc mht_32(mht_32_v, 1393, "", "./tensorflow/core/grappler/optimizers/meta_optimizer_test.cc", "SetUpCallFromList");

  // Call `inner_func` from a list attribute in a node in `outer_func`.
  outer_func = FunctionDefHelper::Create(
      "outer_func", {"x:float"}, {"z:float"}, {},
      /*node_def=*/
      {{{"identity"}, "Identity", {"x"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "x"}});

  // Add a list containing `inner_func` to the `identity` node.
  // `list_value` will be deallocated automatically since it is passed as
  // allocated list below.
  AttrValue_ListValue* list_value =
      (*outer_func.mutable_node_def(0)->mutable_attr())["list"].mutable_list();
  NameAttrList* entry = list_value->add_func();
  entry->set_name("inner_func");
}

TEST_P(TfDataTestFixture, TfDataTests) {
  using test::function::NDef;

  // Define function library with `outer_func` and `inner_func`.

  FunctionDef inner_func = FunctionDefHelper::Create(
      "inner_func", {"x:float", "y:float"}, {"z:float"}, {},
      /*node_def=*/
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "mul:z:0"}});
  (*inner_func.mutable_attr())[data::kTFDataFunction].set_b(
      is_inner_func_tf_data_);

  FunctionDef outer_func;
  switch (func_nesting_type_) {
    case FuncNestingType::CallFromNode:
      SetUpCallFromNode(outer_func);
      break;
    case FuncNestingType::CallFromAttr:
      SetUpCallFromAttr(outer_func);
      break;
    case FuncNestingType::CallFromList:
      SetUpCallFromList(outer_func);
      break;
    default:
      break;
  }
  (*outer_func.mutable_attr())[data::kTFDataFunction].set_b(
      is_outer_func_tf_data_);

  // Tensorflow graph:
  //
  //   a = tf.Placeholder(tf.float);
  //   result = outer_func(a);
  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("outer_func_node", "outer_func", {"a"}, {{"T", DT_FLOAT}}, kDevice),
       // Forward outputs
       NDef("out_s", "Identity", {"outer_func_node:0"}, {{"T", DT_FLOAT}},
            kDevice)},
      /*funcs=*/
      {inner_func, outer_func});

  // Use only custom optimizer which counts its calls.
  TfDataTestOptimizer::InitCount();
  ConfigProto config_proto;
  auto& rewriter_config =
      *(config_proto.mutable_graph_options()->mutable_rewrite_options());
  rewriter_config.add_optimizers("TfDataTestOptimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // We expect one graph optimization + one optimization for each non-tf.data
  // function. Note that if `outer_func` is flagged as a tf.data function, then
  // `inner_func` is implicitly also considered a tf.data function because it is
  // called from `outer_func`.
  int expected_count = 3;
  if (is_outer_func_tf_data_)
    expected_count = 1;
  else if (is_inner_func_tf_data_)
    expected_count = 2;
  EXPECT_EQ(TfDataTestOptimizer::GetCount(), expected_count);

  // We expect that the tf.data-attribute has been propagated from `outer_func`
  // to its callee `inner_func` if the value is `true`. Otherwise, the attribute
  // values should be unchanged.
  FunctionLibraryDefinition flib(OpRegistry::Global(), output.library());
  const FunctionDef* outer_func_after_opt = flib.Find("outer_func");
  const FunctionDef* inner_func_after_opt = flib.Find("inner_func");

  EXPECT_EQ(data::IsTFDataFunction(*outer_func_after_opt),
            is_outer_func_tf_data_);
  if (is_outer_func_tf_data_ || is_inner_func_tf_data_) {
    EXPECT_EQ(data::IsTFDataFunction(*inner_func_after_opt), true);
  } else {
    EXPECT_EQ(data::IsTFDataFunction(*inner_func_after_opt), false);
  }
}

INSTANTIATE_TEST_SUITE_P(
    MetaOptimizerTest, TfDataTestFixture,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                       ::testing::Values(FuncNestingType::CallFromNode,
                                         FuncNestingType::CallFromAttr,
                                         FuncNestingType::CallFromList)),
    [](const ::testing::TestParamInfo<TfDataTestFixture::ParamType>& info) {
      bool is_inner_func_tf_data = std::get<0>(info.param);
      bool is_outer_func_tf_data = std::get<1>(info.param);
      FuncNestingType func_nesting_type = std::get<2>(info.param);

      std::string test_name;
      if (is_inner_func_tf_data && is_outer_func_tf_data)
        test_name = "both_funcs_tf_data";
      else if (is_inner_func_tf_data)
        test_name = "inner_func_tf_data";
      else if (is_outer_func_tf_data)
        test_name = "outer_func_tf_data";
      else
        test_name = "no_func_tf_data";
      switch (func_nesting_type) {
        case FuncNestingType::CallFromNode:
          test_name += "_call_from_node";
          break;
        case FuncNestingType::CallFromAttr:
          test_name += "_call_from_attribute";
          break;
        case FuncNestingType::CallFromList:
          test_name += "_call_from_list";
          break;
        default:
          break;
      }
      return test_name;
    });

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
