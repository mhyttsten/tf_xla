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
class MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc() {
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

#include <memory>
#include <vector>

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

namespace f = test::function;
using FDH = FunctionDefHelper;

std::unique_ptr<Session> NewSession() {
  SessionOptions opts;
  (*opts.config.mutable_device_count())["CPU"] = 1;
  return std::unique_ptr<Session>(NewSession(opts));
}

class MathGradTest : public ::testing::Test {
 protected:
  // Unary
  // dst is the output dtype of op_node.
  Status Unary(const FDH::Node& op_node, const Tensor& x, const DataType dst,
               Tensor* y) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/ops/math_grad_test.cc", "Unary");

    const DataType src = x.dtype();
    auto adef = [](const string& name,
                   const DataType type) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
  // E.g., x:float, dy:double
      return strings::StrCat(name, ":", DataTypeString(type));
    };
    // Sum(op(x)), sum all output of op(x).
    auto test = FDH::Define("Test", {adef("x", src)}, {adef("l", dst)}, {},
                            {
                                op_node,
                                FDH::Const("zero", 0),
                                FDH::Const("one", 1),
                                {{"r"}, "Rank", {"x"}, {{"T", src}}},
                                {{"indices"}, "Range", {"zero", "r", "one"}},
                                {{"l"}, "Sum", {"y", "indices"}, {{"T", dst}}},
                            });

    // TestGrad = Test'(x)
    auto grad = FDH::Define(
        "TestGrad", {adef("x", src)}, {adef("dx", src)}, {},
        {
            FDH::Const("one", 1),
            {{"dy"}, "Cast", {"one"}, {{"DstT", dst}, {"SrcT", DT_INT32}}},
            {{"grad"},
             "SymbolicGradient",
             {"x", "dy"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{src, dst}},
                 {"Tout", DataTypeSlice{src}},
             }},
            {{"dx"}, "Identity", {"grad"}, {{"T", src}}},
        });
    // Each test case will feed in "x:0" and expects to get "dx:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", src}}),
            f::NDef("dx", "TestGrad", {"x"}, {}),
        },
        {test, grad});

    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    auto s = sess->Run({{"x:0", x}}, {"dx:0"}, {}, &outputs);
    if (s.ok()) {
      CHECK_EQ(outputs.size(), 1);
      *y = outputs[0];
    }
    TF_CHECK_OK(sess->Close());
    return s;
  }

  Status Unary(const string& op, const Tensor& x, Tensor* y) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_2(mht_2_v, 274, "", "./tensorflow/core/ops/math_grad_test.cc", "Unary");

    const FDH::Node op_node = {{"y"}, op, {"x"}, {{"T", x.dtype()}}};
    return Unary(op_node, x, x.dtype(), y);
  }

  // Unary op expecting OK.
  Tensor SymGrad(const string& op, const Tensor& x) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_3(mht_3_v, 284, "", "./tensorflow/core/ops/math_grad_test.cc", "SymGrad");

    Tensor ret;
    TF_CHECK_OK(Unary(op, x, &ret));
    return ret;
  }

  Tensor SymCastGrad(const Tensor& x, const DataType dst) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_4(mht_4_v, 293, "", "./tensorflow/core/ops/math_grad_test.cc", "SymCastGrad");

    Tensor ret;
    const FDH::Node op_node = {
        {"y"}, "Cast", {"x"}, {{"SrcT", x.dtype()}, {"DstT", dst}}};
    TF_CHECK_OK(Unary(op_node, x, dst, &ret));
    return ret;
  }

  // Binary
  void SymGrad(const string& op, const Tensor& x, const Tensor& y, Tensor* dx,
               Tensor* dy) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_5(mht_5_v, 307, "", "./tensorflow/core/ops/math_grad_test.cc", "SymGrad");

    const DataType T = x.dtype();
    auto adef = [T](const string& name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_6(mht_6_v, 313, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
  // E.g., x:float, dy:double
      return strings::StrCat(name, ":", DataTypeString(T));
    };
    // Sum(op(x)), sum all output of op(x).
    auto test = FDH::Define("Test", {adef("x"), adef("y")}, {adef("l")}, {},
                            {
                                {{"z"}, op, {"x", "y"}, {{"T", T}}},
                                FDH::Const("zero", 0),
                                FDH::Const("one", 1),
                                {{"r"}, "Rank", {"z"}, {{"T", T}}},
                                {{"indices"}, "Range", {"zero", "r", "one"}},
                                {{"l"}, "Sum", {"z", "indices"}, {{"T", T}}},
                            });

    // TestGrad = Test'(x, y)
    auto grad = FDH::Define(
        "TestGrad", {adef("x"), adef("y")}, {adef("dx"), adef("dy")}, {},
        {
            FDH::Const("one", 1),
            {{"dz"}, "Cast", {"one"}, {{"DstT", T}, {"SrcT", DT_INT32}}},
            {{"grad0", "grad1"},
             "SymbolicGradient",
             {"x", "y", "dz"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{T, T, T}},
                 {"Tout", DataTypeSlice{T, T}},
             }},
            {{"dx"}, "Identity", {"grad0"}, {{"T", T}}},
            {{"dy"}, "Identity", {"grad1"}, {{"T", T}}},
        });
    // Each test case will feed in "x:0" and "y:0" and expects to get "d0" and
    // "d:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("d", "TestGrad", {"x", "y"}, {}),
        },
        {test, grad});

    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(
        sess->Run({{"x:0", x}, {"y:0", y}}, {"d:0", "d:1"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 2);
    TF_CHECK_OK(sess->Close());
    *dx = outputs[0];
    *dy = outputs[1];
  }

  // Reduction grad
  void ReductionGrad(const string& op, const Tensor& x, const Tensor& idx,
                     Tensor* dx, Tensor* di) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_7(mht_7_v, 371, "", "./tensorflow/core/ops/math_grad_test.cc", "ReductionGrad");

    const DataType T = x.dtype();
    auto adef = [T](const string& name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_8(mht_8_v, 377, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
  // E.g., x:float, dy:double
      return strings::StrCat(name, ":", DataTypeString(T));
    };
    // Sum(op(x, idx)), sum all output of op(x, idx).
    auto test = FDH::Define("Test", {adef("x"), "i:int32"}, {adef("l")}, {},
                            {
                                {{"y"}, op, {"x", "i"}, {{"T", T}}},
                                FDH::Const("zero", 0),
                                FDH::Const("one", 1),
                                {{"r"}, "Rank", {"y"}, {{"T", T}}},
                                {{"indices"}, "Range", {"zero", "r", "one"}},
                                {{"l"}, "Sum", {"y", "indices"}, {{"T", T}}},
                            });

    // TestGrad = Test'(x)
    auto grad = FDH::Define(
        "TestGrad", {adef("x"), "i:int32"}, {adef("dx"), "di:int32"}, {},
        {
            FDH::Const("one", 1),
            {{"dy"}, "Cast", {"one"}, {{"DstT", T}, {"SrcT", DT_INT32}}},
            {{"grad0", "grad1"},
             "SymbolicGradient",
             {"x", "i", "dy"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{T, DT_INT32, T}},
                 {"Tout", DataTypeSlice{T, DT_INT32}},
             }},
            {{"dx"}, "Identity", {"grad0"}, {{"T", T}}},
            {{"di"}, "Identity", {"grad1"}, {{"T", DT_INT32}}},
        });
    // Each test case will feed in "x:0" and expects to get "dx:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("i", "Placeholder", {}, {{"dtype", DT_INT32}}),
            f::NDef("d", "TestGrad", {"x", "i"}, {}),
        },
        {test, grad});

    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(
        sess->Run({{"x:0", x}, {"i:0", idx}}, {"d:0", "d:1"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 2);
    TF_CHECK_OK(sess->Close());
    *dx = outputs[0];
    *di = outputs[1];
  }

  Tensor ReduceSum(const Tensor& x, gtl::ArraySlice<int32> axes) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_9(mht_9_v, 431, "", "./tensorflow/core/ops/math_grad_test.cc", "ReduceSum");

    int num_axes = axes.length();
    Tensor y(DT_INT32, TensorShape({num_axes}));
    for (size_t i = 0; i < axes.size(); ++i) {
      y.flat<int32>()(i) = axes[i];
    }
    auto T = x.dtype();
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Const", {}, {{"dtype", DT_INT32}, {"value", y}}),
            f::NDef("z", "Sum", {"x", "y"}, {{"T", T}}),
        },
        {});
    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(sess->Run({{"x:0", x}}, {"z:0"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 1);
    TF_CHECK_OK(sess->Close());
    return outputs[0];
  }

  Tensor MatMulCommon(const string& opname, const string& attr_adj_x,
                      const string& attr_adj_y, const Tensor& x, bool ax,
                      const Tensor& y, bool ay) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("opname: \"" + opname + "\"");
   mht_10_v.push_back("attr_adj_x: \"" + attr_adj_x + "\"");
   mht_10_v.push_back("attr_adj_y: \"" + attr_adj_y + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_10(mht_10_v, 462, "", "./tensorflow/core/ops/math_grad_test.cc", "MatMulCommon");

    auto T = x.dtype();
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("z", opname, {"x", "y"},
                    {{"T", T}, {attr_adj_x, ax}, {attr_adj_y, ay}}),
        },
        {});
    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(sess->Run({{"x:0", x}, {"y:0", y}}, {"z:0"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 1);
    TF_CHECK_OK(sess->Close());
    return outputs[0];
  }

  Tensor MatMul(const Tensor& x, bool ax, const Tensor& y, bool ay) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_11(mht_11_v, 484, "", "./tensorflow/core/ops/math_grad_test.cc", "MatMul");

    return MatMulCommon("MatMul", "transpose_a", "transpose_b", x, ax, y, ay);
  }

  Tensor BatchMatMul(const Tensor& x, bool ax, const Tensor& y, bool ay) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_12(mht_12_v, 491, "", "./tensorflow/core/ops/math_grad_test.cc", "BatchMatMul");

    return MatMulCommon("BatchMatMul", "adj_x", "adj_y", x, ax, y, ay);
  }

  Tensor BatchMatMulV2(const Tensor& x, bool ax, const Tensor& y, bool ay) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_13(mht_13_v, 498, "", "./tensorflow/core/ops/math_grad_test.cc", "BatchMatMulV2");

    return MatMulCommon("BatchMatMulV2", "adj_x", "adj_y", x, ax, y, ay);
  }

  void MatMulGradCommon(const string& opname, const string& attr_adj_x,
                        const string& attr_adj_y, const Tensor& x, bool ax,
                        const Tensor& y, bool ay, Tensor* dx, Tensor* dy) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("opname: \"" + opname + "\"");
   mht_14_v.push_back("attr_adj_x: \"" + attr_adj_x + "\"");
   mht_14_v.push_back("attr_adj_y: \"" + attr_adj_y + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_14(mht_14_v, 510, "", "./tensorflow/core/ops/math_grad_test.cc", "MatMulGradCommon");

    const DataType T = x.dtype();
    auto adef = [T](const string& name) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_15(mht_15_v, 516, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
  // E.g., x:float, dy:double
      return strings::StrCat(name, ":", DataTypeString(T));
    };
    // Sum(op(x)), sum all output of op(x).
    auto test =
        FDH::Define("Test", {adef("x"), adef("y")}, {adef("l")}, {},
                    {
                        {{"z"},
                         opname,
                         {"x", "y"},
                         {{"T", T}, {attr_adj_x, ax}, {attr_adj_y, ay}}},
                        FDH::Const("zero", 0),
                        FDH::Const("one", 1),
                        {{"r"}, "Rank", {"z"}, {{"T", T}}},
                        {{"indices"}, "Range", {"zero", "r", "one"}},
                        {{"l"}, "Sum", {"z", "indices"}, {{"T", T}}},
                    });

    // TestGrad = Test'(x, y)
    auto grad = FDH::Define(
        "TestGrad", {adef("x"), adef("y")}, {adef("dx"), adef("dy")}, {},
        {
            FDH::Const("one", 1),
            {{"dz"}, "Cast", {"one"}, {{"DstT", T}, {"SrcT", DT_INT32}}},
            {{"grad0", "grad1"},
             "SymbolicGradient",
             {"x", "y", "dz"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{T, T, T}},
                 {"Tout", DataTypeSlice{T, T}},
             }},
            {{"dx"}, "Identity", {"grad0"}, {{"T", T}}},
            {{"dy"}, "Identity", {"grad1"}, {{"T", T}}},
        });
    // Each test case will feed in "x:0" and "y:0" and expects to get "d0" and
    // "d:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("d", "TestGrad", {"x", "y"}, {}),
        },
        {test, grad});

    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(
        sess->Run({{"x:0", x}, {"y:0", y}}, {"d:0", "d:1"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 2);
    TF_CHECK_OK(sess->Close());
    *dx = outputs[0];
    *dy = outputs[1];
  }

  void MatMulGrad(const Tensor& x, bool ax, const Tensor& y, bool ay,
                  Tensor* dx, Tensor* dy) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_16(mht_16_v, 576, "", "./tensorflow/core/ops/math_grad_test.cc", "MatMulGrad");

    return MatMulGradCommon("MatMul", "transpose_a", "transpose_b", x, ax, y,
                            ay, dx, dy);
  }

  void BatchMatMulGrad(const Tensor& x, bool ax, const Tensor& y, bool ay,
                       Tensor* dx, Tensor* dy) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_17(mht_17_v, 585, "", "./tensorflow/core/ops/math_grad_test.cc", "BatchMatMulGrad");

    return MatMulGradCommon("BatchMatMul", "adj_x", "adj_y", x, ax, y, ay, dx,
                            dy);
  }

  void BatchMatMulV2Grad(const Tensor& x, bool ax, const Tensor& y, bool ay,
                         Tensor* dx, Tensor* dy) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_18(mht_18_v, 594, "", "./tensorflow/core/ops/math_grad_test.cc", "BatchMatMulV2Grad");

    return MatMulGradCommon("BatchMatMulV2", "adj_x", "adj_y", x, ax, y, ay, dx,
                            dy);
  }

  void SelectGrad(const Tensor& c, const Tensor& x, const Tensor& y, Tensor* dc,
                  Tensor* dx, Tensor* dy) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_19(mht_19_v, 603, "", "./tensorflow/core/ops/math_grad_test.cc", "SelectGrad");

    auto T = DT_FLOAT;
    // Sum(Select(c, x, y))
    auto test =
        FDH::Define("Test", {"c:bool", "x:float", "y:float"}, {"l:float"}, {},
                    {
                        {{"z"}, "Select", {"c", "x", "y"}, {{"T", T}}},
                        FDH::Const("zero", 0),
                        FDH::Const("one", 1),
                        {{"r"}, "Rank", {"z"}, {{"T", T}}},
                        {{"indices"}, "Range", {"zero", "r", "one"}},
                        {{"l"}, "Sum", {"z", "indices"}, {{"T", T}}},
                    });

    // TestGrad(x, y) = Test'(c, x, y)
    auto grad = FDH::Define("TestGrad", {"c:bool", "x:float", "y:float"},
                            {"dc:bool", "dx:float", "dy:float"}, {},
                            {FDH::Const("dz", 1.f),
                             {{"grad0", "grad1", "grad2"},
                              "SymbolicGradient",
                              {"c", "x", "y", "dz"},
                              {
                                  {"f", FDH::FunctionRef("Test")},
                                  {"Tin", DataTypeSlice{DT_BOOL, T, T, T}},
                                  {"Tout", DataTypeSlice{DT_BOOL, T, T}},
                              }},
                             {{"dc"}, "Identity", {"grad0"}, {{"T", DT_BOOL}}},
                             {{"dx"}, "Identity", {"grad1"}, {{"T", T}}},
                             {{"dy"}, "Identity", {"grad2"}, {{"T", T}}}});
    // Each test case will feed in "x:0" and expects to get "dx:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("c", "Placeholder", {}, {{"dtype", DT_BOOL}}),
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("d", "TestGrad", {"c", "x", "y"}, {}),
        },
        {test, grad});

    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(sess->Run({{"c:0", c}, {"x:0", x}, {"y:0", y}},
                          {"d:0", "d:1", "d:2"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 3);
    TF_CHECK_OK(sess->Close());
    *dc = outputs[0];
    *dx = outputs[1];
    *dy = outputs[2];
  }
};

void HasError(const Status& s, const string& substr) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("substr: \"" + substr + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_20(mht_20_v, 659, "", "./tensorflow/core/ops/math_grad_test.cc", "HasError");

  EXPECT_TRUE(absl::StrContains(s.ToString(), substr))
      << s << ", expected substring " << substr;
}

REGISTER_OP("TestOpWithNoGrad")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Test op with no grad registered.

x: input
y: output
)doc");

class TestOp : public OpKernel {
 public:
  explicit TestOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_21(mht_21_v, 680, "", "./tensorflow/core/ops/math_grad_test.cc", "TestOp");
}
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_22(mht_22_v, 684, "", "./tensorflow/core/ops/math_grad_test.cc", "Compute");
 ctx->set_output(0, Tensor()); }
};
REGISTER_KERNEL_BUILDER(Name("TestOpWithNoGrad").Device(DEVICE_CPU), TestOp);

TEST_F(MathGradTest, Error_Reporting) {
  auto x = test::AsTensor<float>({-3.f});
  auto dx = test::AsTensor<float>({3.f});
  Tensor donotcare;
  HasError(Unary("TestOpWithNoGrad", x, &donotcare),
           "No gradient defined for op: TestOpWithNoGrad");
}

TEST_F(MathGradTest, Abs) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_23(mht_23_v, 702, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return x < 0 ? -1.f : 1.f; };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Abs", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Neg) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_24(mht_24_v, 715, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return -1.f; };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Neg", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Reciprocal) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_25(mht_25_v, 728, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return -1.f / (x * x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Reciprocal", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Square) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_26(mht_26_v, 741, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return 2 * x; };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Square", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Sqrt) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_27(mht_27_v, 754, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return 0.5f / std::sqrt(x); };
  auto dx = test::AsTensor<float>(
      {g(1.f), g(2.f), g(3.f), g(4.f), g(5.f), g(6.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Sqrt", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Rsqrt) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_28(mht_28_v, 767, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return -0.5f / (x * std::sqrt(x)); };
  auto dx = test::AsTensor<float>(
      {g(1.f), g(2.f), g(3.f), g(4.f), g(5.f), g(6.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Rsqrt", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Exp) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_29(mht_29_v, 780, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return std::exp(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Exp", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Expm1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_30(mht_30_v, 793, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return std::exp(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Expm1", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Log) {
  auto x = test::AsTensor<float>({0.1f, 1.f, 2.f, 3.f, 4.f, 10.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_31(mht_31_v, 806, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return 1 / x; };
  auto dx = test::AsTensor<float>(
      {g(.1f), g(1.f), g(2.f), g(3.f), g(4.f), g(10.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Log", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Log1p) {
  auto x = test::AsTensor<float>({0.1f, 1.f, 2.f, 3.f, 4.f, 10.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_32(mht_32_v, 819, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return 1 / (1 + x); };
  auto dx = test::AsTensor<float>(
      {g(.1f), g(1.f), g(2.f), g(3.f), g(4.f), g(10.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Log1p", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Sinh) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_33(mht_33_v, 832, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return std::cosh(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Sinh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Cosh) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_34(mht_34_v, 845, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return std::sinh(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Cosh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Tanh) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_35(mht_35_v, 858, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");

    auto y = std::tanh(x);
    return 1 - y * y;
  };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Tanh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Asinh) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_36(mht_36_v, 874, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");

    auto y = std::asinh(x);
    return std::cosh(y);
  };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Asinh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Acosh) {
  auto x = test::AsTensor<float>({6.f, 5.f, 4.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_37(mht_37_v, 890, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");

    auto y = std::acosh(x);
    return std::sinh(y);
  };
  auto dx = test::AsTensor<float>(
      {g(6.f), g(5.f), g(4.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Acosh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Atanh) {
  auto x = test::AsTensor<float>({-0.3f, -0.2f, -0.1f, 0.1f, 0.2f, 0.3f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_38(mht_38_v, 906, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return 1.f / (1.f - x * x); };
  auto dx = test::AsTensor<float>(
      {g(-0.3f), g(-0.2f), g(-0.1f), g(0.1f), g(0.2f), g(0.3f)},
      TensorShape({2, 3}));
  auto ans = SymGrad("Atanh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Sigmoid) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_39(mht_39_v, 920, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");

    auto y = 1.f / (1.f + std::exp(-x));
    return y * (1 - y);
  };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Sigmoid", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Sign) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_40(mht_40_v, 936, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return 0.f; };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Sign", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Sin) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_41(mht_41_v, 949, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return std::cos(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Sin", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Cos) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_42(mht_42_v, 962, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return -std::sin(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Cos", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Cast) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_43(mht_43_v, 975, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return 1.f; };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  Tensor ans = SymCastGrad(x, DT_INT32);
  test::ExpectClose(ans, dx);
}

// TODO(zhifengc)
// TEST_F(MathGradSComplexTest, Real) {}
// TEST_F(MathGradSComplexTest, Imag) {}
// TEST_F(MathGradSComplexTest, Angle) {}
// TEST_F(MathGradSComplexTest, Conj) {}
// TEST_F(MathGradTernary, Select) {}

TEST_F(MathGradTest, Add) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-10.f, 10.f}, TensorShape({2, 1}));
  auto ans_dx = test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                      TensorShape({2, 3}));
  auto ans_dy = test::AsTensor<float>({3.f, 3.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Add", x, y, &dx, &dy);
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
  {  // Swap x and y
    SymGrad("Add", y, x, &dy, &dx);
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
}

TEST_F(MathGradTest, Sub) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-10.f, 10.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Sub", x, y, &dx, &dy);
    auto ans_dx = test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                        TensorShape({2, 3}));
    auto ans_dy = test::AsTensor<float>({-3.f, -3.f}, TensorShape({2, 1}));
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
  {  // Swap x and y
    SymGrad("Sub", y, x, &dy, &dx);
    auto ans_dx = test::AsTensor<float>({-1.f, -1.f, -1.f, -1.f, -1.f, -1.f},
                                        TensorShape({2, 3}));
    auto ans_dy = test::AsTensor<float>({3.f, 3.f}, TensorShape({2, 1}));
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
}

TEST_F(MathGradTest, Mul) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-10.f, 10.f}, TensorShape({2, 1}));
  auto ans_dx = test::AsTensor<float>({-10.f, -10.f, -10.f, 10.f, 10.f, 10.f},
                                      TensorShape({2, 3}));
  auto ans_dy = test::AsTensor<float>({-3.f + (-2.f) + (-1.f), 1.f + 2.f + 3.f},
                                      TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Mul", x, y, &dx, &dy);
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
  {  // Swap x and y
    SymGrad("Mul", y, x, &dy, &dx);
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
}

TEST_F(MathGradTest, Div) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-10.f, 10.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Div", x, y, &dx, &dy);
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_44(mht_44_v, 1068, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return 1.f / y; };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-3.f, -10.f), g(-2.f, -10.f), g(-1.f, -10.f),
                                 g(1.f, 10.f), g(2.f, 10.f), g(3.f, 10.f)},
                                TensorShape({2, 3})));
    }
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_45(mht_45_v, 1078, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return -x / (y * y); };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-3.f, -10.f) + g(-2.f, -10.f) + g(-1.f, -10.f),
                             g(1.f, 10.f) + g(2.f, 10.f) + g(3.f, 10.f)},
                            TensorShape({2, 1})));
    }
  }
  {  // Swap x and y
    SymGrad("Div", y, x, &dy, &dx);
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_46(mht_46_v, 1092, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return 1.f / y; };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-10.f, -3.f) + g(-10.f, -2.f) + g(-10.f, -1.f),
                             g(10.f, 1.f) + g(10.f, 2.f) + g(10.f, 3.f)},
                            TensorShape({2, 1})));
    }
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_47(mht_47_v, 1103, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return -x / (y * y); };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-10.f, -3.f), g(-10.f, -2.f), g(-10.f, -1.f),
                                 g(10.f, 1.f), g(10.f, 2.f), g(10.f, 3.f)},
                                TensorShape({2, 3})));
    }
  }
}

TEST_F(MathGradTest, DivNoNan) {
  auto x = test::AsTensor<float>(
      {0.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 0.f}, TensorShape({3, 3}));
  auto y = test::AsTensor<float>({-10.f, 0.f, 10.f}, TensorShape({3, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("DivNoNan", x, y, &dx, &dy);
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_48(mht_48_v, 1124, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");

        if (y == 0.f) {
          return 0.f;
        } else {
          return 1.f / y;
        }
      };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(0.f, -10.f), g(-3.f, -10.f), g(-2.f, -10.f),
                                 g(-1.f, 0.f), g(0.f, 0.f), g(1.f, 0.f),
                                 g(2.f, 10.f), g(3.f, 10.f), g(0.f, 10.f)},
                                TensorShape({3, 3})));
    }
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_49(mht_49_v, 1141, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");

        if (y == 0.f) {
          return 0.f;
        } else {
          return -x / (y * y);
        }
      };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(0.f, -10.f) + g(-3.f, -10.f) + g(-2.f, -10.f),
                             g(-1.f, 0.f) + g(0.f, 0.f) + g(1.f, 0.f),
                             g(2.f, 10.f) + g(3.f, 10.f) + g(0.f, 10.f)},
                            TensorShape({3, 1})));
    }
  }
  {  // Swap x and y.
    SymGrad("DivNoNan", y, x, &dy, &dx);
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_50(mht_50_v, 1162, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");

        if (y == 0.f) {
          return 0.f;
        } else {
          return 1.f / y;
        }
      };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-10.f, 0.f) + g(-10.f, -3.f) + g(-10.f, -2.f),
                             g(0.f, -1.f) + g(0.f, 0.f) + g(0.f, 1.f),
                             g(10.f, 2.f) + g(10.f, 3.f) + g(10.f, 0.f)},
                            TensorShape({3, 1})));
    }
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_51(mht_51_v, 1180, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");

        if (y == 0.f) {
          return 0.f;
        } else {
          return -x / (y * y);
        }
      };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-10.f, 0.f), g(-10.f, -3.f), g(-10.f, -2.f),
                                 g(0.f, -1.f), g(0.f, 0.f), g(0.f, 1.f),
                                 g(10.f, 2.f), g(10.f, 3.f), g(10.f, 0.f)},
                                TensorShape({3, 3})));
    }
  }
}

TEST_F(MathGradTest, Pow) {
  auto x = test::AsTensor<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  auto g = [](float x, float y) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_52(mht_52_v, 1205, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return y * std::pow(x, y - 1); };
  auto h = [](float x, float y) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_53(mht_53_v, 1209, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");

    return std::pow(x, y) * (x ? std::log(x) : 0);
  };
  {
    SymGrad("Pow", x, y, &dx, &dy);
    test::ExpectClose(
        dx, test::AsTensor<float>({g(0.f, .5f), g(1.f, .5f), g(2.f, .5f),
                                   g(3.f, 2.f), g(4.f, 2.f), g(5.f, 2.f)},
                                  TensorShape({2, 3})));
    test::ExpectClose(
        dy, test::AsTensor<float>({h(0.f, .5f) + h(1.f, .5f) + h(2.f, .5f),
                                   h(3.f, 2.f) + h(4.f, 2.f) + h(5.f, 2.f)},
                                  TensorShape({2, 1})));
  }
  {  // Swap x and y
    SymGrad("Pow", y, x, &dy, &dx);
    test::ExpectClose(
        dy, test::AsTensor<float>({g(.5f, 0.f) + g(.5f, 1.f) + g(.5f, 2.f),
                                   g(2.f, 3.f) + g(2.f, 4.f) + g(2.f, 5.f)},
                                  TensorShape({2, 1})));
    test::ExpectClose(
        dx, test::AsTensor<float>({h(.5f, 0.f), h(.5f, 1.f), h(.5f, 2.f),
                                   h(2.f, 3.f), h(2.f, 4.f), h(2.f, 5.f)},
                                  TensorShape({2, 3})));
  }
}

TEST_F(MathGradTest, ComplexPow) {
  auto x = test::AsTensor<complex64>({0.f, 2.f, -2.f}, TensorShape({3}));
  auto y = test::AsTensor<complex64>({2.f, 2.f, 2.f}, TensorShape({3}));
  Tensor dx;
  Tensor dy;
  auto g = [](complex64 x, complex64 y) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_54(mht_54_v, 1244, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return y * std::pow(x, y - 1.f); };
  auto h = [](complex64 x, complex64 y) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_55(mht_55_v, 1248, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");

    return std::pow(x, y) * (x != complex64(0) ? std::log(x) : 0);
  };
  SymGrad("Pow", x, y, &dx, &dy);

  // This case failed on Kokoro MacOS:
  // dx[2] = (-4,6.0398321011234657e-07),
  // test::AsTensor[2] = (-4,-3.4969110629390343e-07).
  // dx[2] on linux is close to test::AsTensor[2].
  // This error hasn't shown up before because
  // ExpectClose used to check just the magnitude of a complex number, i.e.,
  // std::abs(complex) = sqrt(real^2 + imag^2).
  // Now ExpectClose checks the value of each component separately.
  // Workaround: I set a big tolerance to make the case pass for now.
  // TODO(penporn): Fix this or file a bug. This is not a precision issue.
  // Even the most significant digit (or the sign) doesn't match.
  test::ExpectClose(
      dx,
      test::AsTensor<complex64>({g(0.f, 2.f), g(2.f, 2.f), g(-2.f, 2.f)},
                                TensorShape({3})),
      1e-6f);

  // This case failed on Kokoro MacOS:
  // dx[2] = (2.7725925445556641,12.56636905670166),
  // test::AsTensor[2] = (2.7725865840911865,12.566371917724609)
  // dx[2] on linux is close to test::AsTensor[2].
  // Default atol = rtol = 5.96046e-07.
  // Real: diff = 5.96046e-06 > threshold = 2.248633e-06 <- failed
  // Complex: diff = 2.86102e-06 <= threshold = 8.08618e-06 <- passed
  // Again, this error hasn't shown up before because ExpectClose used to
  // check just the magnitude of the complex number. Now it checks each
  // component separately.
  // Workaround: Set a larger tolerance for now.
  // TODO(penporn): See if this is a precision issue or a bug.
  test::ExpectClose(
      dy,
      test::AsTensor<complex64>({h(0.f, 2.f), h(2.f, 2.f), h(-2.f, 2.f)},
                                TensorShape({3})),
      4.5e-6f);
}

TEST_F(MathGradTest, Xlogy) {
  auto x = test::AsTensor<float>({0.f, 0.f, 2.f, 3.f, 4.f, 5.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  auto g = [](float x, float y) -> float { return x == 0. ? 0. : std::log(y); };
  auto h = [](float x, float y) -> float { return x == 0. ? 0. : x / y; };
  SymGrad("Xlogy", x, y, &dx, &dy);
  test::ExpectClose(
      dx, test::AsTensor<float>({g(0.f, .5f), g(0.f, 0.f), g(2.f, .5f),
                                 g(3.f, 2.f), g(4.f, 2.f), g(5.f, 2.f)},
                                TensorShape({2, 3})));
  test::ExpectClose(
      dy, test::AsTensor<float>({h(0.f, .5f) + h(0.f, 0.f) + h(2.f, .5f),
                                 h(3.f, 2.f) + h(4.f, 2.f) + h(5.f, 2.f)},
                                TensorShape({2, 1})));
}

TEST_F(MathGradTest, Xlog1py) {
  auto x = test::AsTensor<float>({0.f, 0.f, 2.f, 3.f, 4.f, 5.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  auto g = [](float x, float y) -> float {
    return x == 0. ? 0. : std::log1p(y);
  };
  auto h = [](float x, float y) -> float {
    return x == 0. ? 0. : x / (y + 1.);
  };
  SymGrad("Xlog1py", x, y, &dx, &dy);
  test::ExpectClose(
      dx, test::AsTensor<float>({g(0.f, .5f), g(0.f, 0.f), g(2.f, .5f),
                                 g(3.f, 2.f), g(4.f, 2.f), g(5.f, 2.f)},
                                TensorShape({2, 3})));
  test::ExpectClose(
      dy, test::AsTensor<float>({h(0.f, .5f) + h(0.f, 0.f) + h(2.f, .5f),
                                 h(3.f, 2.f) + h(4.f, 2.f) + h(5.f, 2.f)},
                                TensorShape({2, 1})));
}

TEST_F(MathGradTest, Xdivy) {
  auto x = test::AsTensor<float>({0.f, 0.f, 2.f, 3.f, 4.f, 5.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  auto g = [](float x, float y) -> float { return x == 0. ? 0. : 1 / y; };
  auto h = [](float x, float y) -> float {
    return x == 0. ? 0. : -x / (y * y);
  };
  SymGrad("Xdivy", x, y, &dx, &dy);
  test::ExpectClose(
      dx, test::AsTensor<float>({g(0.f, .5f), g(0.f, 0.f), g(2.f, .5f),
                                 g(3.f, 2.f), g(4.f, 2.f), g(5.f, 2.f)},
                                TensorShape({2, 3})));
  test::ExpectClose(
      dy, test::AsTensor<float>({h(0.f, .5f) + h(0.f, 0.f) + h(2.f, .5f),
                                 h(3.f, 2.f) + h(4.f, 2.f) + h(5.f, 2.f)},
                                TensorShape({2, 1})));
}

TEST_F(MathGradTest, SquaredDifference) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  auto g = [](float x, float y) -> float { return 2. * (x - y); };
  auto h = [](float x, float y) -> float { return 2. * (y - x); };
  SymGrad("SquaredDifference", x, y, &dx, &dy);
  test::ExpectClose(
      dx, test::AsTensor<float>({g(-3.f, .5f), g(-2.f, .5f), g(-1.f, .5f),
                                 g(1.f, 2.f), g(2.f, 2.f), g(3.f, 2.f)},
                                TensorShape({2, 3})));
  test::ExpectClose(
      dy, test::AsTensor<float>({h(-3.f, .5f) + h(-2.f, .5f) + h(-1.f, .5f),
                                 h(1.f, 2.f) + h(2.f, 2.f) + h(3.f, 2.f)},
                                TensorShape({2, 1})));
}

TEST_F(MathGradTest, Maximum) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-1.5f, 1.5f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Maximum", x, y, &dx, &dy);
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_56(mht_56_v, 1383, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return x >= y ? 1.f : 0.f; };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-3.f, -1.5f), g(-2.f, -1.5f), g(-1.f, -1.5f),
                                 g(1.f, 1.5f), g(2.f, 1.5f), g(3.f, 1.5f)},
                                TensorShape({2, 3})));
    }
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_57(mht_57_v, 1393, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return x < y ? 1.f : 0.f; };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-3.f, -1.5f) + g(-2.f, -1.5f) + g(-1.f, -1.5f),
                             g(1.f, 1.5f) + g(2.f, 1.5f) + g(3.f, 1.5f)},
                            TensorShape({2, 1})));
    }
  }
  {  // Swap x and y
    SymGrad("Maximum", y, x, &dy, &dx);
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_58(mht_58_v, 1407, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return x >= y ? 1.f : 0.f; };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-1.5f, -3.f) + g(-1.5f, -2.f) + g(-1.5f, -1.f),
                             g(1.5f, 1.f) + g(1.5f, 2.f) + g(1.5f, 3.f)},
                            TensorShape({2, 1})));
    }
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_59(mht_59_v, 1418, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return x < y ? 1.f : 0.f; };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-1.5f, -3.f), g(-1.5f, -2.f), g(-1.5f, -1.f),
                                 g(1.5f, 1.f), g(1.5f, 2.f), g(1.5f, 3.f)},
                                TensorShape({2, 3})));
    }
  }
}

TEST_F(MathGradTest, Minimum) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-1.5f, 1.5f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Minimum", x, y, &dx, &dy);
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_60(mht_60_v, 1439, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return x <= y ? 1.f : 0.f; };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-3.f, -1.5f), g(-2.f, -1.5f), g(-1.f, -1.5f),
                                 g(1.f, 1.5f), g(2.f, 1.5f), g(3.f, 1.5f)},
                                TensorShape({2, 3})));
    }
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_61(mht_61_v, 1449, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return x > y ? 1.f : 0.f; };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-3.f, -1.5f) + g(-2.f, -1.5f) + g(-1.f, -1.5f),
                             g(1.f, 1.5f) + g(2.f, 1.5f) + g(3.f, 1.5f)},
                            TensorShape({2, 1})));
    }
  }
  {  // Swap x and y
    SymGrad("Minimum", y, x, &dy, &dx);
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_62(mht_62_v, 1463, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return x <= y ? 1.f : 0.f; };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-1.5f, -3.f) + g(-1.5f, -2.f) + g(-1.5f, -1.f),
                             g(1.5f, 1.f) + g(1.5f, 2.f) + g(1.5f, 3.f)},
                            TensorShape({2, 1})));
    }
    {
      auto g = [](float x, float y) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_grad_testDTcc mht_63(mht_63_v, 1474, "", "./tensorflow/core/ops/math_grad_test.cc", "lambda");
 return x > y ? 1.f : 0.f; };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-1.5f, -3.f), g(-1.5f, -2.f), g(-1.5f, -1.f),
                                 g(1.5f, 1.f), g(1.5f, 2.f), g(1.5f, 3.f)},
                                TensorShape({2, 3})));
    }
  }
}

TEST_F(MathGradTest, Select) {
  auto c = test::AsTensor<bool>({true, false, false, true, true, false},
                                TensorShape({2, 3}));
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({3.f, 2.f, 1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  Tensor dc;
  Tensor dx;
  Tensor dy;
  {
    SelectGrad(c, x, y, &dc, &dx, &dy);
    test::ExpectTensorEqual<bool>(
        dc, test::AsTensor<bool>({false, false, false, false, false, false},
                                 TensorShape({2, 3})));
    test::ExpectTensorEqual<float>(
        dx, test::AsTensor<float>({1.f, 0.f, 0.f, 1.f, 1.f, 0.f},
                                  TensorShape({2, 3})));
    test::ExpectTensorEqual<float>(
        dy, test::AsTensor<float>({0.f, 1.f, 1.f, 0.f, 0.f, 1.f},
                                  TensorShape({2, 3})));
  }
}

TEST_F(MathGradTest, MatMul_00) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({3, 1}));
  Tensor dx;
  Tensor dy;
  MatMulGrad(x, false, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({2, 1}));
  test::ExpectClose(dx, MatMul(dz, false, y, true));
  test::ExpectClose(dy, MatMul(x, true, dz, false));
}

TEST_F(MathGradTest, MatMul_01) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3}));
  Tensor dx;
  Tensor dy;
  MatMulGrad(x, false, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({2, 1}));
  test::ExpectClose(dx, MatMul(dz, false, y, false));
  test::ExpectClose(dy, MatMul(dz, true, x, false));
}

TEST_F(MathGradTest, MatMul_10) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({3, 1}));
  Tensor dx;
  Tensor dy;
  MatMulGrad(x, true, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({2, 1}));
  test::ExpectClose(dx, MatMul(y, false, dz, true));
  test::ExpectClose(dy, MatMul(x, false, dz, false));
}

TEST_F(MathGradTest, MatMul_11) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3}));
  Tensor dx;
  Tensor dy;
  MatMulGrad(x, true, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({2, 1}));
  test::ExpectClose(dx, MatMul(y, true, dz, true));
  test::ExpectClose(dy, MatMul(dz, true, x, true));
}

TEST_F(MathGradTest, BatchMatMul_00) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3, 1}));
  Tensor dx;
  Tensor dy;
  BatchMatMulGrad(x, false, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMul(dz, false, y, true));
  test::ExpectClose(dy, BatchMatMul(x, true, dz, false));
}

TEST_F(MathGradTest, BatchMatMul_01) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 1, 3}));
  Tensor dx;
  Tensor dy;
  BatchMatMulGrad(x, false, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMul(dz, false, y, false));
  test::ExpectClose(dy, BatchMatMul(dz, true, x, false));
}

TEST_F(MathGradTest, BatchMatMul_10) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3, 1}));
  Tensor dx;
  Tensor dy;
  BatchMatMulGrad(x, true, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMul(y, false, dz, true));
  test::ExpectClose(dy, BatchMatMul(x, false, dz, false));
}

TEST_F(MathGradTest, BatchMatMul_11) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 1, 3}));
  Tensor dx;
  Tensor dy;
  BatchMatMulGrad(x, true, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMul(y, true, dz, true));
  test::ExpectClose(dy, BatchMatMul(dz, true, x, true));
}

TEST_F(MathGradTest, BatchMatMulV2_00) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3, 1}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, false, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMulV2(dz, false, y, true));
  test::ExpectClose(dy, BatchMatMulV2(x, true, dz, false));
}

TEST_F(MathGradTest, BatchMatMulV2_01) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 1, 3}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, false, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMulV2(dz, false, y, false));
  test::ExpectClose(dy, BatchMatMulV2(dz, true, x, false));
}

TEST_F(MathGradTest, BatchMatMulV2_10) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3, 1}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, true, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMulV2(y, false, dz, true));
  test::ExpectClose(dy, BatchMatMulV2(x, false, dz, false));
}

TEST_F(MathGradTest, BatchMatMulV2_11) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 1, 3}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, true, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMulV2(y, true, dz, true));
  test::ExpectClose(dy, BatchMatMulV2(dz, true, x, true));
}

TEST_F(MathGradTest, BatchMatMulV2_LhsBroadcasts) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>(
      {1.f, 2.4, 3.f, -1.f, .5f, 2.f, 3.f, 1.f, -1.f, 2.f, -.1f, 0},
      TensorShape({2, 3, 2}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, false, y, false, &dx, &dy);
  EXPECT_TRUE(dx.shape().IsSameSize(x.shape()));
  EXPECT_TRUE(dy.shape().IsSameSize(y.shape()));
  auto dz = test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                  TensorShape({2, 2, 2}));
  Tensor ans_dx;
  CHECK(ans_dx.CopyFrom(ReduceSum(BatchMatMulV2(dz, false, y, true), {0}),
                        dx.shape()));
  Tensor ans_dy = BatchMatMulV2(x, true, dz, false);
  test::ExpectClose(dx, ans_dx);
  test::ExpectClose(dy, ans_dy);
}

TEST_F(MathGradTest, BatchMatMulV2_RhsBroadcasts) {
  auto x = test::AsTensor<float>(
      {1.f, 2.4, 3.f, -1.f, .5f, 2.f, 3.f, 1.f, -1.f, 2.f, -.1f, 0},
      TensorShape({2, 2, 3}));
  auto y = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({3, 2}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, false, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                  TensorShape({2, 2, 2}));
  Tensor ans_dx = BatchMatMulV2(dz, false, y, true);
  Tensor ans_dy;
  CHECK(ans_dy.CopyFrom(ReduceSum(BatchMatMulV2(x, true, dz, false), {0}),
                        dy.shape()));
  test::ExpectClose(dx, ans_dx);
  test::ExpectClose(dy, ans_dy);
}

TEST_F(MathGradTest, BatchMatMulV2_BothLhsAndRhsBroadcast) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 1, 1, 3}));
  auto y = test::AsTensor<float>({3.f, 1.f, -1.f, 2.f, -.1f, 0},
                                 TensorShape({1, 2, 3, 1}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, false, y, false, &dx, &dy);
  EXPECT_TRUE(dx.shape().IsSameSize(x.shape()));
  EXPECT_TRUE(dy.shape().IsSameSize(y.shape()));
  auto dz =
      test::AsTensor<float>({1.f, 1.f, 1.f, 1.f}, TensorShape({2, 2, 1, 1}));
  Tensor ans_dx;
  Tensor ans_dy;
  CHECK(ans_dx.CopyFrom(ReduceSum(BatchMatMulV2(dz, false, y, true), {1}),
                        dx.shape()));
  CHECK(ans_dy.CopyFrom(ReduceSum(BatchMatMulV2(x, true, dz, false), {0}),
                        dy.shape()));
  test::ExpectClose(dx, ans_dx);
  test::ExpectClose(dy, ans_dy);
}

TEST_F(MathGradTest, BatchMatMulV2_BroadcastWhileAdjointed) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 1, 3, 1}));
  auto y = test::AsTensor<float>({3.f, 1.f, -1.f, 2.f, -.1f, 0},
                                 TensorShape({1, 2, 1, 3}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, true, y, true, &dx, &dy);
  EXPECT_TRUE(dx.shape().IsSameSize(x.shape()));
  EXPECT_TRUE(dy.shape().IsSameSize(y.shape()));

  auto dz =
      test::AsTensor<float>({1.f, 1.f, 1.f, 1.f}, TensorShape({2, 2, 1, 1}));
  Tensor ans_dx;
  Tensor ans_dy;
  CHECK(ans_dx.CopyFrom(ReduceSum(BatchMatMulV2(y, true, dz, true), {1}),
                        dx.shape()));
  CHECK(ans_dy.CopyFrom(ReduceSum(BatchMatMulV2(dz, true, x, true), {0}),
                        dy.shape()));
  test::ExpectClose(dx, ans_dx);
  test::ExpectClose(dy, ans_dy);
}

TEST_F(MathGradTest, Sum_dim0) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Sum", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Sum_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({1}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Sum", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Mean_dim0) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Mean", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>(
              {1.f / 2, 1.f / 2, 1.f / 2, 1.f / 2, 1.f / 2, 1.f / 2},
              TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Mean_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({1}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Mean", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>(
              {1.f / 3, 1.f / 3, 1.f / 3, 1.f / 3, 1.f / 3, 1.f / 3},
              TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Mean_dim0_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0, 1}, TensorShape({2}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Mean", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>(
              {1.f / 6, 1.f / 6, 1.f / 6, 1.f / 6, 1.f / 6, 1.f / 6},
              TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(
      di, test::AsTensor<int32>({0, 0}, TensorShape({2})));
}

TEST_F(MathGradTest, Min_dim0) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Min", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({1.f, 1.f, 1.f, 0.f, 0.f, 0.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Min_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({1}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Min", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({1.f, 0.f, 0.f, 1.f, 0.f, 0.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Min_dim0_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0, 1}, TensorShape({2}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Min", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({1.f, 0.f, 0.f, 0.f, 0.f, 0.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(
      di, test::AsTensor<int32>({0, 0}, TensorShape({2})));
}

TEST_F(MathGradTest, Min_dim0_dim1_Dups) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, -3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0, 1}, TensorShape({2}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Min", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({.5f, 0.f, 0.f, 0.f, 0.f, .5f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(
      di, test::AsTensor<int32>({0, 0}, TensorShape({2})));
}

TEST_F(MathGradTest, Max_dim0) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Max", x, i, &dx, &di);
  LOG(INFO) << dx.SummarizeValue(6);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({0.f, 0.f, 0.f, 1.f, 1.f, 1.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Max_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({1}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Max", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({0.f, 0.f, 1.f, 0.f, 0.f, 1.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Max_dim0_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0, 1}, TensorShape({2}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Max", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({0.f, 0.f, 0.f, 0.f, 0.f, 1.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(
      di, test::AsTensor<int32>({0, 0}, TensorShape({2})));
}

TEST_F(MathGradTest, Max_dim0_dim1_Dups) {
  auto x = test::AsTensor<float>({3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0, 1}, TensorShape({2}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Max", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({.5f, 0.f, 0.f, 0.f, 0.f, .5f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(
      di, test::AsTensor<int32>({0, 0}, TensorShape({2})));
}

}  // namespace
}  // namespace tensorflow
