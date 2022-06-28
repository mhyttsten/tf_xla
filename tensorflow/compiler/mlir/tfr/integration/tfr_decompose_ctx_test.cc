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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctx_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctx_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctx_testDTcc() {
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
#include "tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.h"

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using testing::ElementsAreArray;
using testing::Test;
using NodeAndType = std::pair<std::string, tensorflow::DataType>;

namespace tensorflow {
namespace {

REGISTER_OP("MyAddN")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
    .Attr("T: {numbertype, variant}")
    .SetIsCommutative()
    .SetIsAggregate()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscAddDummy")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {bfloat16, half, float, double, uint8, int8, int16, int32, int64, "
        "complex64, complex128, string}")
    .SetShapeFn(shape_inference::UnchangedShape);

constexpr char tfr_raw_text[] = R"(

tfr.func @tf__my_add_n(%values: !tfr.tensor_list,
                       %n: i64 {tfr.name="N"}) -> !tfr.tensor {
  %index = arith.constant 0 : index
  %cst = arith.constant 1 : i64
  %eq = arith.cmpi "eq", %n, %cst : i64
  %v1 = tfr.get_element %values[%index] : (!tfr.tensor_list, index) -> !tfr.tensor
  %res = scf.if %eq -> !tfr.tensor {
    scf.yield %v1 : !tfr.tensor
  } else {
    %step = arith.index_cast %cst : i64 to index
    %end = arith.index_cast %n : i64 to index
    %reduce = scf.for %i = %step to %end step %step iter_args(%reduce_iter=%v1) -> !tfr.tensor {
      %v = tfr.get_element %values[%i] : (!tfr.tensor_list, index) -> !tfr.tensor
      %reduce_next =  tfr.call @tf__risc_add_dummy(%reduce_iter, %v) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor
      scf.yield %reduce_next : !tfr.tensor
    }
    scf.yield %reduce : !tfr.tensor
  }
  tfr.return %res : !tfr.tensor
}

tfr.func @tf__my_add_n_(!tfr.tensor_list<N,T>, i64 {tfr.name="N"}) -> !tfr.tensor attributes{N,T}
tfr.func @tf__risc_add_dummy_(!tfr.tensor<T>, !tfr.tensor<T>) -> !tfr.tensor<T> attributes{T}
)";

class TFRDecomposeContextTest : public Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctx_testDTcc mht_0(mht_0_v, 260, "", "./tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx_test.cc", "SetUp");

    test_ctx_ = tfr::TFRDecomposeContext::GetFromText(tfr_raw_text, &ctx_);
  }

  void TearDown() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctx_testDTcc mht_1(mht_1_v, 267, "", "./tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx_test.cc", "TearDown");
 test_ctx_->Destroy(); }

  mlir::MLIRContext ctx_;
  std::unique_ptr<tfr::TFRDecomposeContext> test_ctx_;
};

std::vector<NodeAndType> NodesSequenceOf(const FunctionDef& graph) {
  std::vector<NodeAndType> nodes;
  for (auto& node : graph.node_def()) {
    nodes.push_back({node.op(), node.attr().at("T").type()});
  }
  return nodes;
}

TEST_F(TFRDecomposeContextTest, FLOAT_1_ins) {
  std::vector<NodeDefBuilder::NodeOut> src_list;
  src_list.emplace_back("input", 0, DT_FLOAT);
  NodeDef test_node;
  auto status = NodeDefBuilder("float_add", "MyAddN")
                    .Input(src_list)
                    .Finalize(&test_node);
  EXPECT_TRUE(status.ok());
  auto decomposed = test_ctx_->ExpandNode(test_node, "test");
  EXPECT_TRUE(decomposed.ok());
  std::vector<NodeAndType> expected_results{{"Identity", DT_FLOAT}};
  EXPECT_THAT(NodesSequenceOf(decomposed.ValueOrDie()),
              ElementsAreArray(expected_results));
}

TEST_F(TFRDecomposeContextTest, FLOAT_3_ins) {
  std::vector<NodeDefBuilder::NodeOut> src_list;
  src_list.emplace_back("in0", 0, DT_FLOAT);
  src_list.emplace_back("in1", 0, DT_FLOAT);
  src_list.emplace_back("in2", 0, DT_FLOAT);
  NodeDef test_node;
  auto status = NodeDefBuilder("float_add_3", "MyAddN")
                    .Input(src_list)
                    .Finalize(&test_node);
  EXPECT_TRUE(status.ok());
  auto decomposed = test_ctx_->ExpandNode(test_node, "test");
  EXPECT_TRUE(decomposed.ok());

  std::vector<NodeAndType> expected_results{{"RiscAddDummy", DT_FLOAT},
                                            {"RiscAddDummy", DT_FLOAT}};
  EXPECT_THAT(NodesSequenceOf(decomposed.ValueOrDie()),
              ElementsAreArray(expected_results));
}

TEST_F(TFRDecomposeContextTest, INT32_3_ins) {
  std::vector<NodeDefBuilder::NodeOut> src_list;
  src_list.emplace_back("in0", 0, DT_INT32);
  src_list.emplace_back("in1", 0, DT_INT32);
  src_list.emplace_back("in2", 0, DT_INT32);
  NodeDef test_node;
  auto status =
      NodeDefBuilder("int_add", "MyAddN").Input(src_list).Finalize(&test_node);
  EXPECT_TRUE(status.ok());
  auto decomposed = test_ctx_->ExpandNode(test_node, "test");
  EXPECT_TRUE(decomposed.ok());

  std::vector<NodeAndType> expected_results{{"RiscAddDummy", DT_INT32},
                                            {"RiscAddDummy", DT_INT32}};
  EXPECT_THAT(NodesSequenceOf(decomposed.ValueOrDie()),
              ElementsAreArray(expected_results));
}

}  // namespace
}  // namespace tensorflow
