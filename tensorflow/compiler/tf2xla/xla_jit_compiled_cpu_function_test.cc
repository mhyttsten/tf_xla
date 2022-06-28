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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc() {
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

#include "tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function.h"

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"

namespace tensorflow {
namespace {

using ::testing::HasSubstr;

PLATFORM_DEFINE_ID(kFakePlatformId);

AttrValue TypeAttrValue(DataType type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "TypeAttrValue");

  AttrValue attr_value;
  SetAttrValue(type, &attr_value);
  return attr_value;
}

GraphDef SumGraph() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "SumGraph");

  GraphDef graph_def;
  NodeDef* x = graph_def.add_node();
  x->set_name("x");
  x->set_op("Placeholder");
  (*x->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  NodeDef* y = graph_def.add_node();
  y->set_name("y");
  y->set_op("Placeholder");
  (*y->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  NodeDef* sum = graph_def.add_node();
  sum->set_name("sum");
  sum->set_op("Add");
  sum->add_input("x");
  sum->add_input("y");
  (*sum->mutable_attr())["T"] = TypeAttrValue(DT_INT32);
  return graph_def;
}

tf2xla::Config SumConfig() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "SumConfig");

  tf2xla::Config config;
  tf2xla::Feed* x = config.add_feed();
  x->mutable_id()->set_node_name("x");
  x->set_name("x_name");
  tf2xla::Feed* y = config.add_feed();
  y->mutable_id()->set_node_name("y");
  y->set_name("y_name");
  tf2xla::Fetch* sum = config.add_fetch();
  sum->mutable_id()->set_node_name("sum");
  sum->set_name("sum_name");
  return config;
}

GraphDef SumGraphVariable() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_3(mht_3_v, 264, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "SumGraphVariable");

  constexpr char text_proto[] = R"pb(
    node {
      name: "x"
      op: "VarHandleOp"
      attr {
        key: "dtype"
        value { type: DT_INT32 }
      }
      attr {
        key: "shared_name"
        value { s: "myvar" }
      }
      attr {
        key: "shape"
        value { shape { dim { size: 1 } } }
      }
    }
    node {
      name: "read"
      op: "ReadVariableOp"
      input: "x"
      attr {
        key: "dtype"
        value { type: DT_INT32 }
      }
    }
    node {
      name: "y"
      op: "Placeholder"
      attr {
        key: "dtype"
        value { type: DT_INT32 }
      }
    }
    node {
      name: "sum"
      op: "Add"
      input: "read"
      input: "y"
      attr {
        key: "T"
        value { type: DT_INT32 }
      }
    }
    node {
      name: "assign"
      op: "AssignVariableOp"
      input: "x"
      input: "sum"
      attr {
        key: "dtype"
        value { type: DT_INT32 }
      }
    }
    # We use this identity op to make sure assign doesn't get pruned away.
    node {
      name: "out"
      op: "Identity"
      input: "y"
      input: "^assign"
      attr {
        key: "T"
        value { type: DT_INT32 }
      }
    })pb";
  GraphDef graph;
  CHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph));
  return graph;
}

tf2xla::Config SumConfigVariable() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_4(mht_4_v, 338, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "SumConfigVariable");

  constexpr char text_proto[] = R"pb(feed { id { node_name: "y" } }
                                     variable {
                                       node_name: "myvar"
                                       shape { dim { size: 1 } }
                                       type: DT_INT32
                                     }
                                     fetch { id { node_name: "out" } })pb";
  tf2xla::Config config;
  CHECK(protobuf::TextFormat::ParseFromString(text_proto, &config));
  return config;
}

TEST(XlaJitCompiledCpuFunction, Sum) {
  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XlaJitCompiledCpuFunction> jit,
      XlaJitCompiledCpuFunction::Compile(graph_def, config,
                                         xla::ExecutableBuildOptions()));
  XlaCompiledCpuFunction function(jit->StaticData());

  // Run the function and check results.
  *static_cast<int32*>(function.arg_data(0)) = 10;
  *static_cast<int32*>(function.arg_data(1)) = 32;
  EXPECT_TRUE(function.Run());
  EXPECT_EQ(function.error_msg(), "");
  EXPECT_EQ(*static_cast<int32*>(function.result_data(0)), 42);

  // Run the function again.
  *static_cast<int32*>(function.arg_data(0)) = 100;
  *static_cast<int32*>(function.arg_data(1)) = 320;
  EXPECT_TRUE(function.Run());
  EXPECT_EQ(function.error_msg(), "");
  EXPECT_EQ(*static_cast<int32*>(function.result_data(0)), 420);

  // Check name to index lookups.
  EXPECT_TRUE(function.HasNameIndices());

  EXPECT_EQ(function.LookupArgIndex("x_name"), 0);
  EXPECT_EQ(function.LookupArgIndex("y_name"), 1);
  EXPECT_EQ(function.LookupArgIndex(""), -1);
  EXPECT_EQ(function.LookupArgIndex("x"), -1);
  EXPECT_EQ(function.LookupArgIndex("y"), -1);
  EXPECT_EQ(function.LookupArgIndex("sum"), -1);
  EXPECT_EQ(function.LookupArgIndex("sum_name"), -1);

  EXPECT_EQ(function.LookupResultIndex("sum_name"), 0);
  EXPECT_EQ(function.LookupResultIndex(""), -1);
  EXPECT_EQ(function.LookupResultIndex("x"), -1);
  EXPECT_EQ(function.LookupResultIndex("y"), -1);
  EXPECT_EQ(function.LookupResultIndex("sum"), -1);
  EXPECT_EQ(function.LookupResultIndex("x_name"), -1);
  EXPECT_EQ(function.LookupResultIndex("y_name"), -1);

  EXPECT_EQ(0, function.num_variables());
  EXPECT_EQ(function.LookupVariableIndex("x"), -1);

  // Check program shape.
  using xla::ShapeUtil;
  const xla::Shape s32 = ShapeUtil::MakeShape(xla::S32, {});
  ASSERT_TRUE(function.ProgramShape() != nullptr);
  const xla::ProgramShape program_shape(*function.ProgramShape());
  ASSERT_EQ(program_shape.parameters_size(), 2);
  EXPECT_TRUE(ShapeUtil::Compatible(program_shape.parameters(0), s32));
  EXPECT_TRUE(ShapeUtil::Compatible(program_shape.parameters(1), s32));

  const xla::Shape& result = program_shape.result();
  ASSERT_EQ(result.element_type(), xla::TUPLE);
  ASSERT_EQ(ShapeUtil::TupleElementCount(result), 1);
  const xla::Shape& result0 = ShapeUtil::GetTupleElementShape(result, 0);
  EXPECT_TRUE(ShapeUtil::Compatible(result0, s32));
}

TEST(XlaJitCompiledCpuFunction, SumVariable) {
  GraphDef graph_def = SumGraphVariable();
  tf2xla::Config config = SumConfigVariable();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XlaJitCompiledCpuFunction> jit,
      XlaJitCompiledCpuFunction::Compile(graph_def, config,
                                         xla::ExecutableBuildOptions()));
  XlaCompiledCpuFunction function(jit->StaticData());

  // Run the function and check results.
  *static_cast<int32*>(function.arg_data(0)) = 10;
  *static_cast<int32*>(function.arg_data(1)) = 32;
  EXPECT_TRUE(function.Run());
  EXPECT_EQ(function.error_msg(), "");
  EXPECT_EQ(*static_cast<int32*>(function.result_data(0)), 10);
  EXPECT_EQ(*static_cast<int32*>(function.result_data(1)), 42);

  // Run the function again.
  *static_cast<int32*>(function.arg_data(0)) = 100;
  *static_cast<int32*>(function.arg_data(1)) = 320;
  EXPECT_TRUE(function.Run());
  EXPECT_EQ(function.error_msg(), "");
  EXPECT_EQ(*static_cast<int32*>(function.result_data(0)), 100);
  EXPECT_EQ(*static_cast<int32*>(function.result_data(1)), 420);

  // Check name to index lookups.
  EXPECT_TRUE(function.HasNameIndices());

  EXPECT_EQ(2, function.num_args());

  EXPECT_EQ(1, function.num_variables());
  EXPECT_EQ(function.LookupVariableIndex("myvar"), 1);

  // Check program shape.
  using xla::ShapeUtil;
  const xla::Shape s32 = ShapeUtil::MakeShape(xla::S32, {});
  const xla::Shape s32_1 = ShapeUtil::MakeShape(xla::S32, {1});
  ASSERT_TRUE(function.ProgramShape() != nullptr);
  const xla::ProgramShape program_shape(*function.ProgramShape());
  ASSERT_EQ(program_shape.parameters_size(), 2);
  EXPECT_TRUE(ShapeUtil::Compatible(program_shape.parameters(0), s32));
  EXPECT_TRUE(ShapeUtil::Compatible(program_shape.parameters(1), s32_1));

  const xla::Shape& result = program_shape.result();
  ASSERT_EQ(result.element_type(), xla::TUPLE);
  ASSERT_EQ(ShapeUtil::TupleElementCount(result), 2);
  const xla::Shape& result0 = ShapeUtil::GetTupleElementShape(result, 0);
  EXPECT_TRUE(ShapeUtil::Compatible(result0, s32));
}

TEST(XlaJitCompiledCpuFunction, CanCompileWithAdditionalPlatform) {
  class FakePlatform : public se::Platform {
   public:
    FakePlatform() : name_("FakePlatform") {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_5(mht_5_v, 470, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "FakePlatform");
}
    ~FakePlatform() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_6(mht_6_v, 474, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "~FakePlatform");
}

    se::Platform::Id id() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_7(mht_7_v, 479, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "id");
 return kFakePlatformId; }

    int VisibleDeviceCount() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_8(mht_8_v, 484, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "VisibleDeviceCount");
 return 0; }

    const string& Name() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_9(mht_9_v, 489, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "Name");
 return name_; }

    se::port::StatusOr<std::unique_ptr<se::DeviceDescription>>
    DescriptionForDevice(int ordinal) const override {
      return std::unique_ptr<se::DeviceDescription>(nullptr);
    }

    se::port::StatusOr<se::StreamExecutor*> ExecutorForDevice(
        int ordinal) override {
      return nullptr;
    }

    se::port::StatusOr<se::StreamExecutor*> ExecutorForDeviceWithPluginConfig(
        int ordinal, const se::PluginConfig& config) override {
      return nullptr;
    }

    se::port::StatusOr<se::StreamExecutor*> GetExecutor(
        const se::StreamExecutorConfig& config) override {
      return nullptr;
    }

    se::port::StatusOr<std::unique_ptr<se::StreamExecutor>> GetUncachedExecutor(
        const se::StreamExecutorConfig& config) override {
      return std::unique_ptr<se::StreamExecutor>(nullptr);
    }

    void RegisterTraceListener(
        std::unique_ptr<se::TraceListener> listener) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_10(mht_10_v, 520, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "RegisterTraceListener");
}

    void UnregisterTraceListener(se::TraceListener* listener) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_jit_compiled_cpu_function_testDTcc mht_11(mht_11_v, 525, "", "./tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function_test.cc", "UnregisterTraceListener");
}

   private:
    string name_;
  };

  TF_EXPECT_OK(se::MultiPlatformManager::RegisterPlatform(
      absl::make_unique<FakePlatform>()));
  xla::Compiler::RegisterCompilerFactory(kFakePlatformId, []() {
    return std::unique_ptr<xla::Compiler>(nullptr);
  });

  EXPECT_THAT(xla::PlatformUtil::GetDefaultPlatform().status().error_message(),
              HasSubstr("FakePlatform"));

  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XlaJitCompiledCpuFunction> jit,
      XlaJitCompiledCpuFunction::Compile(graph_def, config,
                                         xla::ExecutableBuildOptions()));
}

}  // namespace
}  // namespace tensorflow
