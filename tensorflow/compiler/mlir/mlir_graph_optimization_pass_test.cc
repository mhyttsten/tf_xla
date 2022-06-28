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
class MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_pass_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_pass_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_pass_testDTcc() {
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

#include "tensorflow/compiler/mlir/mlir_graph_optimization_pass.h"

#include <memory>

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::Test;

class MockMlirOptimizationPass : public MlirOptimizationPass {
 public:
  // MOCK_METHOD does not work on Windows build, using MOCK_CONST_METHODX
  // instead.
  MOCK_CONST_METHOD0(name, llvm::StringRef());
  MOCK_CONST_METHOD4(GetPassState,
                     MlirOptimizationPassState(
                         const DeviceSet* device_set,
                         const ConfigProto& config_proto, const Graph& graph,
                         const FunctionLibraryDefinition& function_library));
  MOCK_METHOD4(Run, Status(const ConfigProto& config_proto,
                           mlir::ModuleOp module, const Graph& graph,
                           const FunctionLibraryDefinition& function_library));
};

class MockMlirV1CompatOptimizationPass : public MlirV1CompatOptimizationPass {
 public:
  // MOCK_METHOD does not work on Windows build, using MOCK_CONST_METHODX
  // instead.
  MOCK_CONST_METHOD0(name, llvm::StringRef());
  MOCK_CONST_METHOD4(GetPassState,
                     MlirOptimizationPassState(
                         const DeviceSet* device_set,
                         const ConfigProto& config_proto, const Graph& graph,
                         const FunctionLibraryDefinition& function_library));
  MOCK_METHOD2(Run, Status(const GraphOptimizationPassOptions& options,
                           mlir::ModuleOp module));
};

class ModifyMlirModulePass : public MlirOptimizationPass {
 public:
  explicit ModifyMlirModulePass(Status run_status) : run_status_(run_status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_pass_testDTcc mht_0(mht_0_v, 230, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass_test.cc", "ModifyMlirModulePass");
}
  // MOCK_METHOD does not work on Windows build, using MOCK_CONST_METHODX
  // instead.
  MOCK_CONST_METHOD0(name, llvm::StringRef());
  MOCK_CONST_METHOD4(GetPassState,
                     MlirOptimizationPassState(
                         const DeviceSet* device_set,
                         const ConfigProto& config_proto, const Graph& graph,
                         const FunctionLibraryDefinition& function_library));

  // Just modify MLIR module so that we can check whether original TF graph
  // has changed or not.
  Status Run(const ConfigProto& config_proto, mlir::ModuleOp module,
             const Graph& graph,
             const FunctionLibraryDefinition& function_library) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_pass_testDTcc mht_1(mht_1_v, 247, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass_test.cc", "Run");

    mlir::Builder b(module.getContext());
    auto producer = b.getNamedAttr("producer", b.getI32IntegerAttr(0));
    auto min_consumer = b.getNamedAttr("min_consumer", b.getI32IntegerAttr(0));
    auto bad_consumers =
        b.getNamedAttr("bad_consumers", b.getI32ArrayAttr({1, 2, 3, 4}));

    module->setAttr("tf.versions",
                    b.getDictionaryAttr(llvm::ArrayRef<mlir::NamedAttribute>(
                        {producer, min_consumer, bad_consumers})));

    return run_status_;
  }

  Status run_status_;
};

class MlirGraphOptimizationPassTest : public Test {
 public:
  void Init(Status pass_run_result,
            const std::vector<MlirOptimizationPassState>& pass_states) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_pass_testDTcc mht_2(mht_2_v, 270, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass_test.cc", "Init");

    graph_ = std::make_unique<Graph>(OpRegistry::Global());

    int pass_priority = 0;
    for (const MlirOptimizationPassState& pass_state : pass_states) {
      auto optimization_pass =
          std::make_unique<NiceMock<MockMlirOptimizationPass>>();

      ON_CALL(*optimization_pass, GetPassState(_, _, _, _))
          .WillByDefault(Return(pass_state));
      ON_CALL(*optimization_pass, Run(_, _, _, _))
          .WillByDefault(Return(pass_run_result));
      MlirOptimizationPassRegistry::Global().Add(pass_priority++,
                                                 std::move(optimization_pass));
    }

    flib_.reset(new FunctionLibraryDefinition(graph_->flib_def()));
  }

  void AddModuleModificationPass(MlirOptimizationPassState pass_state,
                                 Status run_status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_pass_testDTcc mht_3(mht_3_v, 293, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass_test.cc", "AddModuleModificationPass");

    // Add FallbackEnabled pass that modifies the graph.
    auto optimization_pass =
        std::make_unique<NiceMock<ModifyMlirModulePass>>(run_status);
    ON_CALL(*optimization_pass, GetPassState(_, _, _, _))
        .WillByDefault(Return(pass_state));
    MlirOptimizationPassRegistry::Global().Add(10,
                                               std::move(optimization_pass));
  }

  void TearDown() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_pass_testDTcc mht_4(mht_4_v, 306, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass_test.cc", "TearDown");

    MlirOptimizationPassRegistry::Global().ClearPasses();
  }

  void verifyGraph(const GraphDef& original_graph_def, bool changed = false) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_pass_testDTcc mht_5(mht_5_v, 313, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass_test.cc", "verifyGraph");

// Proto matchers might be unavailable in the OSS.
#if defined(PLATFORM_GOOGLE)
    GraphDef resulted_graph_def;
    graph_->ToGraphDef(&resulted_graph_def);

    if (changed)
      EXPECT_THAT(resulted_graph_def,
                  Not(::testing::proto::IgnoringRepeatedFieldOrdering(
                      ::testing::EquivToProto(original_graph_def))));
    else
      EXPECT_THAT(resulted_graph_def,
                  ::testing::proto::IgnoringRepeatedFieldOrdering(
                      ::testing::EquivToProto(original_graph_def)));
#endif
  }

  ConfigProto config_proto_;
  MlirFunctionOptimizationPass function_optimization_pass_;
  DeviceSet device_set_;
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<FunctionLibraryDefinition> flib_;
  std::vector<std::string> control_ret_node_names_;
  bool control_rets_updated_{false};
};

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassFailsNoFallback) {
  Init(Status(error::Code::ABORTED, "aborted"),
       {MlirOptimizationPassState::Enabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  EXPECT_EQ(function_optimization_pass_.Run(
                device_set_, config_proto_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            Status(error::Code::ABORTED, "aborted"));
  verifyGraph(original_graph_def);
}

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassFailsDisabledFallback) {
  Init(Status(error::Code::ABORTED, "aborted"),
       {MlirOptimizationPassState::Disabled,
        MlirOptimizationPassState::FallbackEnabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);
  AddModuleModificationPass(MlirOptimizationPassState::FallbackEnabled,
                            Status(error::Code::ABORTED, "aborted"));

  EXPECT_EQ(function_optimization_pass_.Run(
                device_set_, config_proto_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            Status::OK());
  verifyGraph(original_graph_def);
}

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassDoesNotFailFallback) {
  Init(Status::OK(), {MlirOptimizationPassState::FallbackEnabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  AddModuleModificationPass(MlirOptimizationPassState::FallbackEnabled,
                            Status::OK());
  EXPECT_EQ(function_optimization_pass_.Run(
                device_set_, config_proto_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            Status::OK());

  verifyGraph(original_graph_def, true);
}

TEST(MlirOptimizationPassRegistry, RegisterPassesWithTheSamePriorityFails) {
  MlirOptimizationPassRegistry::Global().Add(
      0, std::make_unique<NiceMock<MockMlirOptimizationPass>>());
  EXPECT_DEATH(MlirOptimizationPassRegistry::Global().Add(
                   0, std::make_unique<NiceMock<MockMlirOptimizationPass>>()),
               "Pass priority must be unique.");
}

TEST(MlirV1CompatOptimizationPassRegistry, RegisterMultiplePassesFails) {
  MlirV1CompatOptimizationPassRegistry::Global().Add(
      std::make_unique<NiceMock<MockMlirV1CompatOptimizationPass>>());
  EXPECT_DEATH(
      MlirV1CompatOptimizationPassRegistry::Global().Add(
          std::make_unique<NiceMock<MockMlirV1CompatOptimizationPass>>()),
      "Only a single pass can be registered");
}

}  // namespace tensorflow
