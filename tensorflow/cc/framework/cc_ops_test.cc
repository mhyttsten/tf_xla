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
class MHTracer_DTPStensorflowPSccPSframeworkPScc_ops_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSframeworkPScc_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSframeworkPScc_ops_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/test_op.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace ops {
namespace {

Output Linear(const Scope& scope, Input x, Input w, Input b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSframeworkPScc_ops_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/cc/framework/cc_ops_test.cc", "Linear");

  auto cop_scopes = scope.GetCompositeOpScopes("linear");
  auto m = MatMul(cop_scopes.child, x, w);
  return BiasAdd(cop_scopes.last, m, b);
}

void GetColocationConstraints(const Output& tensor,
                              std::vector<string>* constraints) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSframeworkPScc_ops_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/cc/framework/cc_ops_test.cc", "GetColocationConstraints");

  constraints->clear();
  TF_EXPECT_OK(GetNodeAttr(tensor.op().node()->attrs(), kColocationAttrName,
                           constraints));
}

TEST(CCOpTest, Basic) {
  Scope root = Scope::NewRootScope();
  auto c = Const(root, {{1, 1}});
  // NOTE: The recommended style for constructing ops is
  // auto v = OpConstructor(t0, t1, ..);
  // Since the wrappers are implemented as one class per op, the following
  // style is also possible :
  // PrimitiveOp p(t0, t1, ...);
  // It's being used here ONLY to ensure that, that style is tested.
  MatMul m(root, c, {{41}, {1}});
  TF_EXPECT_OK(root.status());
  Tensor out;
  test::GetTensor(root, m, &out);
  test::ExpectTensorEqual<int>(out, test::AsTensor<int>({42}, {1, 1}));
}

TEST(CCOpTest, Attrs) {
  Scope root = Scope::NewRootScope();
  auto m = MatMul(root, {{1}, {1}}, {{41}, {1}}, MatMul::TransposeA(true));
  TF_EXPECT_OK(root.status());
  Tensor out;
  test::GetTensor(root, m, &out);
  test::ExpectTensorEqual<int>(out, test::AsTensor<int>({42}, {1, 1}));
}

TEST(CCOpTest, SplitConcat) {
  Scope root = Scope::NewRootScope();
  Split p(root, 0, {{1}, {2}}, 2);
  auto c = Concat(root, {p[0], p[1]}, 0);
  TF_EXPECT_OK(root.status());
  Tensor out;
  test::GetTensor(root, c, &out);
  test::ExpectTensorEqual<int>(out, test::AsTensor<int>({1, 2}, {2, 1}));
}

TEST(CCOpTest, CompositeOp) {
  Scope root = Scope::NewRootScope();
  auto l = Linear(root.WithOpName("layer0"), {{10.0f, -3.0f}},
                  {{.8f, .5f}, {.1f, .6f}}, {-8.0f, 31.0f});
  TF_EXPECT_OK(root.status());
  EXPECT_EQ(l.node()->name(), "layer0");
  Tensor out;
  test::GetTensor(root, l, &out);
  test::ExpectClose(out, test::AsTensor<float>({-0.3, 34.2}, {1, 2}));
}

TEST(CCOpTest, MultiOutput) {
  Scope root = Scope::NewRootScope();
  auto u = Unique(root, {1, 2, 2, 4, 3, 2});
  std::vector<Tensor> outputs;
  test::GetTensors(root, {u.y, u.idx}, &outputs);
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({1, 2, 4, 3}));
  test::ExpectTensorEqual<int>(outputs[1],
                               test::AsTensor<int>({0, 1, 1, 2, 3, 1}));
}

TEST(CCOpTest, ExampleTrainer) {
  Scope root = Scope::NewRootScope();
  // a = [3 2; -1 0]
  auto a = Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  // x = [1.0; 1.0]
  auto x = Const(root.WithOpName("x"), {{1.f}, {1.f}});
  // y = a * x
  auto y = MatMul(root.WithOpName("y"), a, x);
  // y2 = y.^2
  auto y2 = Square(root, y);
  // y2_sum = sum(y2)
  auto y2_sum = Sum(root, y2, 0);
  // y_norm = sqrt(y2_sum)
  auto y_norm = Sqrt(root, y2_sum);
  // y_normalized = y ./ y_norm
  auto y_normalized = Div(root.WithOpName("y_normalized"), y, y_norm);
  Tensor out;
  test::GetTensor(root, y_normalized, &out);
  test::ExpectTensorNear<float>(
      out, test::AsTensor<float>({0.98058069, -0.19611613}, {2, 1}), 1e-5);
}

TEST(CCOpTest, ThrowAwayOp) {
  Scope root = Scope::NewRootScope();
  ThrowAway1(root, 1, 2.3f, 1, 1, 1, ThrowAway1::Builder(42));
  ThrowAway2(root, ThrowAway2::ThrowAway2_(3).Scope(1));
  TF_EXPECT_OK(root.status());
}

TEST(CCOpTest, ControlDeps) {
  Scope root = Scope::NewRootScope();
  auto v = Variable(root, {}, DT_FLOAT);
  auto assign = Assign(root, v, 41.0f);
  Scope with_control_deps = root.WithControlDependencies(assign);
  auto add = Add(with_control_deps, v, 1.0f);
  Scope no_control_deps = with_control_deps.WithNoControlDependencies();
  auto sub = Sub(no_control_deps, 3.0f, 2.0f);
  auto is_inited =
      IsVariableInitialized(no_control_deps.WithControlDependencies(sub), v);

  TF_EXPECT_OK(root.status());

  std::vector<Tensor> out;

  test::GetTensors(root, {add}, &out);
  test::ExpectTensorNear<float>(out[0], test::AsTensor<float>({42.0f}, {}),
                                1e-5);

  out.clear();
  // Note : GetTensors creates a new session, so 'v' is uninitialized.
  // sub should have no control deps, so it should not cause the assign to run.
  // Hence is_inited should be false.
  test::GetTensors(root, {sub, is_inited}, &out);
  test::ExpectTensorNear<float>(out[0], test::AsTensor<float>({1.0f}, {}),
                                1e-5);
  test::ExpectTensorEqual<bool>(out[1], test::AsTensor<bool>({false}, {}));
}

TEST(CCOpTest, KernelLabel) {
  Scope root = Scope::NewRootScope();
  auto add = Add(root.WithKernelLabel("AddWithKernelLabel"), 1.0f, 2.0f);
  TF_EXPECT_OK(root.status());
  AttrSlice attrs = add.z.op().node()->attrs();
  const auto* kernel_attr = attrs.Find("_kernel");
  ASSERT_TRUE(kernel_attr);
  TF_EXPECT_OK(AttrValueHasType(*kernel_attr, "string"));
  EXPECT_EQ(kernel_attr->s(), "AddWithKernelLabel");
}

TEST(CCOpTest, ColocateWith) {
  Scope root = Scope::NewRootScope();
  auto c1 = Const(root.WithOpName("c1"), 1);
  auto c2 = Const(root.WithOpName("c2").ColocateWith(c1), 2);
  std::vector<string> constraints;
  GetColocationConstraints(c2, &constraints);
  EXPECT_EQ(constraints[0], "loc:@c1");

  auto c3 = Const(root.WithOpName("c3").ColocateWith(c2), 3);
  GetColocationConstraints(c3, &constraints);
  EXPECT_EQ(constraints[0], "loc:@c1");

  auto a = Const(root.WithOpName("a"), 4);
  auto c4 = Const(root.WithOpName("c4").ColocateWith(a), 5);
  GetColocationConstraints(c4, &constraints);
  EXPECT_EQ(constraints[0], "loc:@a");

  auto c5 = Const(root.WithOpName("c5").ColocateWith(c3).ColocateWith(c4), 6);
  GetColocationConstraints(c5, &constraints);
  EXPECT_EQ(constraints[0], "loc:@a");
  EXPECT_EQ(constraints[1], "loc:@c1");

  Scope with_colocate = root.ColocateWith(c3).ColocateWith(c4);
  auto c6 = Const(with_colocate.WithOpName("c6").ClearColocation(), 7);
  EXPECT_FALSE(c6.op().node()->attrs().Find("_class"));
}

TEST(CCOpTest, TemplatedConst) {
  Scope root = Scope::NewRootScope();
  auto c1 = ops::Const<float>(root, {{3, 2}, {-1, 0}});
  TF_EXPECT_OK(root.status());

  Tensor out;
  test::GetTensor(root, c1, &out);
  test::ExpectTensorEqual<float>(
      out, test::AsTensor<float>({3.f, 2.f, -1.f, 0.f}, {2, 2}));

  auto c2 = ops::Const<tstring>(root, {{"this"}, {"is"}, {"a"}, {"constant"}});
  test::GetTensor(root, c2, &out);
  test::ExpectTensorEqual<tstring>(
      out, test::AsTensor<tstring>({"this", "is", "a", "constant"}, {4, 1}));
}

TEST(CCOpTest, EmptyConst) {
  Scope root = Scope::NewRootScope();

  auto c1 = ops::Const(root, {});
  TF_CHECK_OK(root.status());

  Tensor out;
  test::GetTensor(root, c1, &out);
  test::ExpectTensorEqual<float>(out, Tensor(DT_FLOAT, {0}));

  auto c2 = ops::Const(root, {{}});
  TF_CHECK_OK(root.status());
  test::GetTensor(root, c2, &out);
  test::ExpectTensorEqual<float>(out, Tensor(DT_FLOAT, {1, 0}));

  auto c3 = ops::Const(root, {{{}, {}}});
  TF_CHECK_OK(root.status());
  test::GetTensor(root, c3, &out);
  test::ExpectTensorEqual<float>(out, Tensor(DT_FLOAT, {1, 2, 0}));

  auto c4 = ops::Const<int>(root, {{{}}});
  TF_CHECK_OK(root.status());
  test::GetTensor(root, c4, &out);
  test::ExpectTensorEqual<int>(out, Tensor(DT_INT32, {1, 1, 0}));

  ops::Const(root, {{}, {{}}});
  EXPECT_FALSE(root.status().ok());
}

TEST(CCOpTest, InvalidFinalize) {
  Scope root = Scope::NewRootScope();
  auto read_up_to =
      ops::ReaderReadUpTo(root, Variable(root, {}, DT_STRING),
                          Variable(root, {}, DT_STRING), static_cast<int32>(2));
  EXPECT_FALSE(root.status().ok());
  auto err_msg = root.status().error_message();
  EXPECT_NE(err_msg.find("'num_records' passed int32 expected int64"),
            string::npos);
}

}  // namespace
}  // namespace ops
}  // namespace tensorflow
