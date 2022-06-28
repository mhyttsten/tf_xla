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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewrite_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewrite_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewrite_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

class EagerOpRewriteTest : public ::testing::Test {
 public:
  EagerOpRewriteTest() : eager_ctx_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewrite_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite_test.cc", "EagerOpRewriteTest");
}
  ~EagerOpRewriteTest() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewrite_testDTcc mht_1(mht_1_v, 201, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite_test.cc", "~EagerOpRewriteTest");

    if (eager_ctx_) {
      eager_ctx_->Unref();
    }
  }

  // Creates a new op to be used as input to MKL eager rewrite.
  std::unique_ptr<tensorflow::EagerOperation> CreateOp(const string op_name) {
    std::unique_ptr<DeviceMgr> device_mgr =
        absl::make_unique<StaticDeviceMgr>(DeviceFactory::NewDevice(
            "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
    bool async = false;
    tensorflow::Rendezvous* rendezvous =
        new tensorflow::IntraProcessRendezvous(device_mgr.get());
    eager_ctx_ = new tensorflow::EagerContext(
        SessionOptions(),
        tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        async, device_mgr.get(), false, rendezvous);

    EagerExecutor executor_(false);
    std::unique_ptr<tensorflow::EagerOperation> op(
        new tensorflow::EagerOperation(eager_ctx_));
    EXPECT_EQ(Status::OK(),
              op.get()->Reset(op_name.c_str(), nullptr, false, &executor_));
    EXPECT_EQ(Status::OK(),
              op.get()->SetDeviceName(
                  "/job:localhost/replica:0/task:0/device:CPU:0"));
    return op;
  }

  // Validates the result of MKL eager rewrite.
  void CheckRewrite(EagerOperation* orig_op, string expected_op_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("expected_op_name: \"" + expected_op_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewrite_testDTcc mht_2(mht_2_v, 236, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite_test.cc", "CheckRewrite");

    std::unique_ptr<tensorflow::EagerOperation> out_op;
    EXPECT_EQ(Status::OK(),
              EagerOpRewriteRegistry::Global()->RunRewrite(
                  EagerOpRewriteRegistry::POST_PLACEMENT, orig_op, &out_op));

    // actual_op_name is same as original op name if rewrite didn't happen.
    string actual_op_name = orig_op->Name();
    if (out_op) {
      actual_op_name = out_op->Name();
    }

    EXPECT_EQ(actual_op_name, expected_op_name);
  }

 protected:
  tensorflow::EagerContext* eager_ctx_;
};

#define CONV_FORWARD_OPS "Conv2D", "Conv3D", "DepthwiseConv2dNative"

#define CONV_BACKWARD_OPS                                                  \
  "Conv2DBackpropInput", "Conv2DBackpropFilter", "Conv3DBackpropFilterV2", \
      "Conv3DBackpropInputV2", "DepthwiseConv2dNativeBackpropFilter",      \
      "DepthwiseConv2dNativeBackpropInput"

#define CONV_OPS CONV_FORWARD_OPS, CONV_BACKWARD_OPS

#define REGISTER_TEST(NAME, T, INPUT)                                 \
  TEST_F(EagerOpRewriteTest, NAME##_##T) {                            \
    std::vector<string> conv_ops = {CONV_OPS};                        \
    for (int i = 0; i < conv_ops.size(); ++i) {                       \
      auto orig_op = CreateOp(conv_ops[i]);                           \
      orig_op->MutableAttrs()->Set("T", T);                           \
      orig_op->MutableAttrs()->Set("padding", "VALID");               \
      CheckRewrite(orig_op.get(),                                     \
                   mkl_op_registry::GetMklNativeOpName(conv_ops[i])); \
    }                                                                 \
  }
REGISTER_TEST_ALL_TYPES(ConvOps_Positive);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                 \
  TEST_F(EagerOpRewriteTest, NAME##_##T) {                            \
    std::vector<string> conv_ops = {CONV_FORWARD_OPS};                \
    for (int i = 0; i < conv_ops.size(); ++i) {                       \
      auto orig_op = CreateOp(conv_ops[i]);                           \
      orig_op->MutableAttrs()->Set("T", T);                           \
      orig_op->MutableAttrs()->Set("padding", "EXPLICIT");            \
      CheckRewrite(orig_op.get(),                                     \
                   mkl_op_registry::GetMklNativeOpName(conv_ops[i])); \
    }                                                                 \
  }
REGISTER_TEST_ALL_TYPES(ConvOpsExplicitPadding_Positive);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                      \
  TEST_F(EagerOpRewriteTest, NAME##_##T) {                 \
    std::vector<string> conv_ops = {CONV_BACKWARD_OPS};    \
    for (int i = 0; i < conv_ops.size(); ++i) {            \
      auto orig_op = CreateOp(conv_ops[i]);                \
      orig_op->MutableAttrs()->Set("T", T);                \
      orig_op->MutableAttrs()->Set("padding", "EXPLICIT"); \
      CheckRewrite(orig_op.get(), conv_ops[i]);            \
    }                                                      \
  }
REGISTER_TEST_ALL_TYPES(ConvOpsExplicitPadding_Negative);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                            \
  TEST_F(EagerOpRewriteTest, NAME##_##T) {                       \
    std::vector<string> ops = {"AvgPool",                        \
                               "AvgPoolGrad",                    \
                               "AvgPool3D",                      \
                               "AvgPool3DGrad",                  \
                               "BatchMatMul",                    \
                               "Einsum",                         \
                               "FusedBatchNorm",                 \
                               "FusedBatchNormV2",               \
                               "FusedBatchNormV3",               \
                               "FusedBatchNormGrad",             \
                               "FusedBatchNormGradV2",           \
                               "FusedBatchNormGradV3",           \
                               "MatMul"};                        \
    for (int i = 0; i < ops.size(); ++i) {                       \
      auto orig_op = CreateOp(ops[i]);                           \
      orig_op->MutableAttrs()->Set("T", T);                      \
      CheckRewrite(orig_op.get(),                                \
                   mkl_op_registry::GetMklNativeOpName(ops[i])); \
    }                                                            \
  }
REGISTER_TEST_ALL_TYPES(MostOps_Positive);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                 \
  TEST_F(EagerOpRewriteTest, NAME##_##T) {                            \
    std::vector<string> Fused_BN_ops = {"FusedBatchNormV3",           \
                                        "FusedBatchNormGradV3"};      \
    for (int i = 0; i < Fused_BN_ops.size(); ++i) {                   \
      auto orig_op = CreateOp(Fused_BN_ops[i]);                       \
      orig_op->MutableAttrs()->Set("T", T);                           \
      orig_op->MutableAttrs()->Set("data_format", "" DATA_FORMAT ""); \
      CheckRewrite(orig_op.get(), Fused_BN_ops[i]);                   \
    }                                                                 \
  }
#define DATA_FORMAT "NCDHW"
REGISTER_TEST_ALL_TYPES(FusedBatchNormV3_5D_Negative_1);
#undef DATA_FORMAT

#define DATA_FORMAT "NDHWC"
REGISTER_TEST_ALL_TYPES(FusedBatchNormV3_5D_Negative_2);
#undef DATA_FORMAT

#undef REGISTER_TEST

}  // namespace tensorflow

#endif  // INTEL_MKL && ENABLE_MKL
