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
class MHTracer_DTPStensorflowPScPSkernelsPSsummary_op_testDTcc {
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
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSkernelsPSsummary_op_testDTcc() {
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

#include "tensorflow/c/kernels.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

class DummyDevice : public DeviceBase {
 public:
  explicit DummyDevice(Env* env) : DeviceBase(env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_op_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/c/kernels/summary_op_test.cc", "DummyDevice");
}
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_op_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/c/kernels/summary_op_test.cc", "GetAllocator");

    return cpu_allocator();
  }
};

// Helper for comparing output and expected output
void ExpectSummaryMatches(const Summary& actual, const string& expected_str) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("expected_str: \"" + expected_str + "\"");
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_op_testDTcc mht_2(mht_2_v, 224, "", "./tensorflow/c/kernels/summary_op_test.cc", "ExpectSummaryMatches");

  Summary expected;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(expected_str, &expected));
  EXPECT_EQ(expected.DebugString(), actual.DebugString());
}

void TestScalarSummaryOp(Tensor* tags, Tensor* values, string expected_output,
                         error::Code expected_code) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("expected_output: \"" + expected_output + "\"");
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_op_testDTcc mht_3(mht_3_v, 235, "", "./tensorflow/c/kernels/summary_op_test.cc", "TestScalarSummaryOp");

  // Initialize node used to fetch OpKernel
  Status status;
  NodeDef def;
  def.set_op("ScalarSummary");

  def.set_device(DEVICE_CPU);

  AttrValue valuesTypeAttr;
  SetAttrValue(values->dtype(), &valuesTypeAttr);
  (*def.mutable_attr())["T"] = valuesTypeAttr;

  def.add_input(strings::StrCat("input1: ", DataTypeString(tags->dtype())));
  def.add_input(strings::StrCat("input2: ", DataTypeString(values->dtype())));

  std::unique_ptr<OpKernel> kernel =
      CreateOpKernel(DeviceType(DEVICE_CPU), nullptr, nullptr, def, 1, &status);
  ASSERT_TRUE(status.ok()) << status.ToString();
  OpKernelContext::Params params;
  DummyDevice dummy_device(nullptr);
  params.device = &dummy_device;
  params.op_kernel = kernel.get();
  AllocatorAttributes alloc_attrs;
  params.output_attr_array = &alloc_attrs;
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.emplace_back(tags);
  inputs.emplace_back(values);
  params.inputs = &inputs;
  OpKernelContext ctx(&params, 1);
  kernel->Compute(&ctx);
  ASSERT_EQ(expected_code, ctx.status().code());
  if (expected_code == error::OK) {
    Summary summary;
    ASSERT_TRUE(ParseProtoUnlimited(
        &summary, ctx.mutable_output(0)->scalar<tstring>()()));
    ExpectSummaryMatches(summary, expected_output);
  } else {
    EXPECT_TRUE(absl::StrContains(ctx.status().ToString(), expected_output))
        << ctx.status();
  }
}

TEST(ScalarSummaryOpTest, SimpleFloat) {
  int vectorSize = 3;
  Tensor tags(DT_STRING, {vectorSize});
  Tensor values(DT_FLOAT, {vectorSize});
  tags.vec<tstring>()(0) = "tag1";
  tags.vec<tstring>()(1) = "tag2";
  tags.vec<tstring>()(2) = "tag3";
  values.vec<float>()(0) = 1.0f;
  values.vec<float>()(1) = -0.73f;
  values.vec<float>()(2) = 10000.0f;
  TestScalarSummaryOp(&tags, &values, R"(
                      value { tag: 'tag1' simple_value: 1.0 }
                      value { tag: 'tag2' simple_value: -0.73}
                      value { tag: 'tag3' simple_value: 10000.0})",
                      error::OK);
}

TEST(ScalarSummaryOpTest, SimpleDouble) {
  int vectorSize = 3;
  Tensor tags(DT_STRING, {vectorSize});
  Tensor values(DT_DOUBLE, {vectorSize});
  tags.vec<tstring>()(0) = "tag1";
  tags.vec<tstring>()(1) = "tag2";
  tags.vec<tstring>()(2) = "tag3";
  values.vec<double>()(0) = 1.0;
  values.vec<double>()(1) = -0.73;
  values.vec<double>()(2) = 10000.0;
  TestScalarSummaryOp(&tags, &values, R"(
                      value { tag: 'tag1' simple_value: 1.0 }
                      value { tag: 'tag2' simple_value: -0.73}
                      value { tag: 'tag3' simple_value: 10000.0})",
                      error::OK);
}

TEST(ScalarSummaryOpTest, SimpleHalf) {
  int vectorSize = 3;
  Tensor tags(DT_STRING, {vectorSize});
  Tensor values(DT_HALF, {vectorSize});
  tags.vec<tstring>()(0) = "tag1";
  tags.vec<tstring>()(1) = "tag2";
  tags.vec<tstring>()(2) = "tag3";
  values.vec<Eigen::half>()(0) = Eigen::half(1.0);
  values.vec<Eigen::half>()(1) = Eigen::half(-2.0);
  values.vec<Eigen::half>()(2) = Eigen::half(10000.0);
  TestScalarSummaryOp(&tags, &values, R"(
                      value { tag: 'tag1' simple_value: 1.0 }
                      value { tag: 'tag2' simple_value: -2.0}
                      value { tag: 'tag3' simple_value: 10000.0})",
                      error::OK);
}

TEST(ScalarSummaryOpTest, Error_WrongDimsTags) {
  Tensor tags(DT_STRING, {2, 1});
  Tensor values(DT_FLOAT, {2});
  tags.matrix<tstring>()(0, 0) = "tag1";
  tags.matrix<tstring>()(1, 0) = "tag2";
  values.vec<float>()(0) = 1.0f;
  values.vec<float>()(1) = -2.0f;
  TestScalarSummaryOp(&tags, &values, "tags and values are not the same shape",
                      error::INVALID_ARGUMENT);
}

TEST(ScalarSummaryOpTest, Error_WrongValuesTags) {
  Tensor tags(DT_STRING, {2});
  Tensor values(DT_FLOAT, {2, 1});
  tags.vec<tstring>()(0) = "tag1";
  tags.vec<tstring>()(1) = "tag2";
  values.matrix<float>()(0, 0) = 1.0f;
  values.matrix<float>()(1, 0) = -2.0f;
  TestScalarSummaryOp(&tags, &values, "tags and values are not the same shape",
                      error::INVALID_ARGUMENT);
}

TEST(ScalarSummaryOpTest, Error_WrongWithSingleTag) {
  Tensor tags(DT_STRING, {1});
  Tensor values(DT_FLOAT, {2, 1});
  tags.vec<tstring>()(0) = "tag1";
  values.matrix<float>()(0, 0) = 1.0f;
  values.matrix<float>()(1, 0) = -2.0f;
  TestScalarSummaryOp(&tags, &values, "tags and values are not the same shape",
                      error::INVALID_ARGUMENT);
}

TEST(ScalarSummaryOpTest, IsRegistered) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("ScalarSummary", &reg));
}

}  // namespace
}  // namespace tensorflow
