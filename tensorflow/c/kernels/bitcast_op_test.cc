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
class MHTracer_DTPStensorflowPScPSkernelsPSbitcast_op_testDTcc {
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
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSkernelsPSbitcast_op_testDTcc() {
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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class DummyDevice : public DeviceBase {
 public:
  explicit DummyDevice(Env* env) : DeviceBase(env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_op_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/c/kernels/bitcast_op_test.cc", "DummyDevice");
}
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_op_testDTcc mht_1(mht_1_v, 203, "", "./tensorflow/c/kernels/bitcast_op_test.cc", "GetAllocator");

    return cpu_allocator();
  }
};

void TestBitcastOp(Tensor* input_tensor, DataType out_type,
                   TensorShape expected_shape, error::Code expected_code) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_op_testDTcc mht_2(mht_2_v, 212, "", "./tensorflow/c/kernels/bitcast_op_test.cc", "TestBitcastOp");

  Status status;
  NodeDef def;
  def.set_op("Bitcast");
  def.set_device(DEVICE_CPU);

  AttrValue typeAttr;
  SetAttrValue(input_tensor->dtype(), &typeAttr);

  AttrValue outTypeAttr;
  SetAttrValue(out_type, &outTypeAttr);

  (*def.mutable_attr())["T"] = typeAttr;
  (*def.mutable_attr())["type"] = outTypeAttr;

  def.add_input(
      strings::StrCat("input1: ", DataTypeString(input_tensor->dtype())));

  std::unique_ptr<OpKernel> kernel =
      CreateOpKernel(DeviceType(DEVICE_CPU), nullptr, nullptr, def, 1, &status);
  ASSERT_TRUE(status.ok()) << status.ToString();

  OpKernelContext::Params params;
  DummyDevice dummy_device(nullptr);
  params.device = &dummy_device;
  params.op_kernel = kernel.get();
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.emplace_back(input_tensor);
  params.inputs = &inputs;

  OpKernelContext ctx(&params);
  kernel->Compute(&ctx);
  ASSERT_EQ(expected_code, ctx.status().code());
  if (expected_code == error::OK) {
    ASSERT_EQ(expected_shape, ctx.mutable_output(0)->shape())
        << ctx.mutable_output(0)->shape().DebugString();
  }
}

TEST(BitcastOpTest, TestUpcast) {
  Tensor int8_input(DT_UINT8, {8});
  for (int i = 0; i < 8; i++) {
    int8_input.vec<uint8>()(i) = static_cast<uint8>(1);
  }
  TestBitcastOp(&int8_input, DT_UINT64, TensorShape(), error::OK);
}

TEST(BitcastOpTest, TestDowncast) {
  Tensor int64_input(static_cast<uint64>(1));
  TestBitcastOp(&int64_input, DT_UINT8, TensorShape({8}), error::OK);
}

TEST(BitcastOpTest, TestCastToSameSize) {
  Tensor int32_input(DT_UINT32, {4, 6});
  TestBitcastOp(&int32_input, DT_UINT8, TensorShape({4, 6, 4}), error::OK);
}

TEST(BitcastOpTest, TestImpossibleCast) {
  Tensor int8_input(DT_UINT8, {1});
  TestBitcastOp(&int8_input, DT_UINT32, TensorShape(), error::INVALID_ARGUMENT);
}

PartialTensorShape S(std::initializer_list<int64_t> dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_op_testDTcc mht_3(mht_3_v, 277, "", "./tensorflow/c/kernels/bitcast_op_test.cc", "S");

  return PartialTensorShape(dims);
}

TEST(BitcastOpTest, TestShapeInference_LargerShape) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("Bitcast", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("type", DT_INT8)
                  .Attr("T", DT_INT64)
                  .Input(FakeInput(DT_INT64))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, def, op_def, {S({3, 4})}, {}, {}, {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(c.input("input", &input_shapes));
  ASSERT_EQ("[3,4]", c.DebugString(input_shapes[0]));
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[3,4,8]", c.DebugString(c.output(0)));
}

TEST(BitcastOpTest, TestShapeInference_SmallerShape) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("Bitcast", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("type", DT_INT64)
                  .Attr("T", DT_INT8)
                  .Input(FakeInput(DT_INT8))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, def, op_def, {S({3, 4, 8})}, {}, {},
                                      {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(c.input("input", &input_shapes));
  ASSERT_EQ("[3,4,8]", c.DebugString(input_shapes[0]));
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[3,4]", c.DebugString(c.output(0)));
}

TEST(BitcastOpTest, TestShapeInference_SameShape) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("Bitcast", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("type", DT_INT32)
                  .Attr("T", DT_FLOAT)
                  .Input(FakeInput(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, def, op_def, {S({3, 4})}, {}, {}, {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(c.input("input", &input_shapes));
  ASSERT_EQ("[3,4]", c.DebugString(input_shapes[0]));
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[3,4]", c.DebugString(c.output(0)));
}

}  // namespace
}  // namespace tensorflow
