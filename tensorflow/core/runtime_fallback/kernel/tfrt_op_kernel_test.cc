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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernel_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernel_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernel_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h"

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/padding.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace tensorflow {
namespace {

std::unique_ptr<tfrt::HostContext> CreateTestHostContext(int num_threads) {
  return std::make_unique<tfrt::HostContext>(
      [](const tfrt::DecodedDiagnostic&) {}, tfrt::CreateMallocAllocator(),
      tfrt::CreateSingleThreadedWorkQueue());
}

TEST(TFRTOpKernelTest, TestGetBoolAttr) {
  tfrt::OpAttrs attrs;
  attrs.Set<bool>("foo", true);
  attrs.Set<bool>("bar", false);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  bool value;
  TF_ASSERT_OK(ctx.GetAttr("foo", &value));
  ASSERT_TRUE(value);
  TF_ASSERT_OK(ctx.GetAttr("bar", &value));
  ASSERT_FALSE(value);
}

TEST(TFRTOpKernelTest, TestGetIntAttr) {
  tfrt::OpAttrs attrs;
  attrs.Set<int32>("foo", -2);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  int32_t value;
  TF_ASSERT_OK(ctx.GetAttr("foo", &value));
  ASSERT_EQ(value, -2);
}

TEST(TFRTOpKernelTest, TestGetIntListAttr) {
  tfrt::OpAttrs attrs;
  attrs.SetArray<int32>("foo", {});
  attrs.SetArray<int32>("bar", {1});
  attrs.SetArray<int32>("baz", {1, 2, 3});
  attrs.SetString("bar", "test");
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  std::vector<int32> v1, v2, v3;
  std::vector<int32> expected_v1;
  std::vector<int32> expected_v2 = {1};
  std::vector<int32> expected_v3 = {1, 2, 3};
  TF_ASSERT_OK(ctx.GetAttr("foo", &v1));
  ASSERT_EQ(v1, expected_v1);
  TF_ASSERT_OK(ctx.GetAttr("bar", &v2));
  ASSERT_EQ(v2, expected_v2);
  TF_ASSERT_OK(ctx.GetAttr("baz", &v3));
  ASSERT_EQ(v3, expected_v3);
}

TEST(TFRTOpKernelTest, TestGetStrAttr) {
  tfrt::OpAttrs attrs;
  attrs.SetString("foo", "");
  attrs.SetString("bar", "test");
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  std::string value;
  TF_ASSERT_OK(ctx.GetAttr("foo", &value));
  ASSERT_EQ(value, "");
  TF_ASSERT_OK(ctx.GetAttr("bar", &value));
  ASSERT_EQ(value, "test");
}

TEST(TFRTOpKernelTest, TestGetPaddingAttr) {
  tfrt::OpAttrs attrs;
  attrs.SetString("foo", "VALID");
  attrs.SetString("bar", "SAME");
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  Padding value;
  TF_ASSERT_OK(ctx.GetAttr("foo", &value));
  ASSERT_EQ(value, Padding::VALID);
  TF_ASSERT_OK(ctx.GetAttr("bar", &value));
  ASSERT_EQ(value, Padding::SAME);
}

TEST(TFRTOpKernelTest, TestMissingAttr) {
  tfrt::OpAttrs attrs;
  attrs.Set<bool>("foo", true);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  bool value;
  auto status = ctx.GetAttr("bar", &value);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
}

class TestKernel : public TFRTOpKernel {
 public:
  explicit TestKernel(TFRTOpKernelConstruction* construction)
      : TFRTOpKernel(construction) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernel_testDTcc mht_0(mht_0_v, 301, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel_test.cc", "TestKernel");
}

  void Compute(TFRTOpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernel_testDTcc mht_1(mht_1_v, 306, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel_test.cc", "Compute");
}
};

TEST(TFRTOpKernelTest, TestKernelMatchesTypeConstraints) {
  tfrt::OpAttrs attrs;
  attrs.Set<tfrt::OpAttrType>("foo", tfrt::OpAttrType::F32);
  attrs.Set<tfrt::OpAttrType>("bar", tfrt::OpAttrType::I32);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);
  TFRTOpKernelReg reg([](TFRTOpKernelConstruction* construction)
                          -> std::unique_ptr<TFRTOpKernel> {
    return std::make_unique<TestKernel>(construction);
  });
  reg.type_constraints["foo"] = DT_FLOAT;
  reg.type_constraints["bar"] = DT_INT32;
  ::tensorflow::tfrt_forwarding_kernel_factories->RegisterFactory(
      "TestKernelFloatInt", reg);
  std::unique_ptr<TFRTOpKernel> op =
      tfrt_forwarding_kernel_factories->CreateKernel("TestKernelFloatInt",
                                                     &ctx);
  ASSERT_NE(op.get(), nullptr);
}

TEST(TFRTOpKernelTest, TestSecondKernelMatchesTypeConstraints) {
  tfrt::OpAttrs attrs;
  attrs.Set<tfrt::OpAttrType>("foo", tfrt::OpAttrType::I32);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);
  TFRTOpKernelReg reg1([](TFRTOpKernelConstruction* construction)
                           -> std::unique_ptr<TFRTOpKernel> {
    return std::make_unique<TestKernel>(construction);
  });
  TFRTOpKernelReg reg2([](TFRTOpKernelConstruction* construction)
                           -> std::unique_ptr<TFRTOpKernel> {
    return std::make_unique<TestKernel>(construction);
  });
  reg1.type_constraints["foo"] = DT_FLOAT;
  reg2.type_constraints["foo"] = DT_INT32;
  ::tensorflow::tfrt_forwarding_kernel_factories->RegisterFactory(
      "TestKernel2ndConstraint", reg1);
  ::tensorflow::tfrt_forwarding_kernel_factories->RegisterFactory(
      "TestKernel2ndConstraint", reg2);

  std::unique_ptr<TFRTOpKernel> op =
      tfrt_forwarding_kernel_factories->CreateKernel("TestKernel2ndConstraint",
                                                     &ctx);
  ASSERT_NE(op.get(), nullptr);
}

TEST(TFRTOpKernelTest, TestKernelDoesNotMatchTypeConstraints) {
  tfrt::OpAttrs attrs;
  attrs.Set<tfrt::OpAttrType>("foo", tfrt::OpAttrType::I32);
  attrs.Set<tfrt::OpAttrType>("bar", tfrt::OpAttrType::I32);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);
  TFRTOpKernelReg reg([](TFRTOpKernelConstruction* construction)
                          -> std::unique_ptr<TFRTOpKernel> {
    return std::make_unique<TestKernel>(construction);
  });
  reg.type_constraints["foo"] = DT_FLOAT;
  reg.type_constraints["bar"] = DT_INT32;
  ::tensorflow::tfrt_forwarding_kernel_factories->RegisterFactory(
      "TestKernelIntInt", reg);
  std::unique_ptr<TFRTOpKernel> op =
      tfrt_forwarding_kernel_factories->CreateKernel("TestKernelIntInt", &ctx);
  ASSERT_EQ(op.get(), nullptr);
}

TEST(TFRTOpKernelTest, TestAllocateTemp) {
  auto host_context = CreateTestHostContext(1);
  int num_outputs = 1;
  llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> inputs;
  TFRTOpMeta op_meta({DT_INT32});
  TFRTOpKernelContext ctx(inputs, num_outputs, &op_meta, host_context.get());

  Tensor out;
  ASSERT_EQ(out.AllocatedBytes(), 0);
  TF_EXPECT_OK(ctx.allocate_temp(DT_INT32, {}, &out));
  ASSERT_GT(out.AllocatedBytes(), 0);
  out.scalar<int32>()() = 123;
  ASSERT_EQ(out.dtype(), DT_INT32);
  ASSERT_EQ(out.shape().dims(), 0);
}

}  // namespace
}  // namespace tensorflow
