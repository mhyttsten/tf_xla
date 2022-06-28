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
class MHTracer_DTPStensorflowPScPSops_testDTcc {
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
   MHTracer_DTPStensorflowPScPSops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSops_testDTcc() {
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

#include "tensorflow/c/ops.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(OpsTest, TestBasicOpRegistration) {
  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("SomeOp");
  TF_OpDefinitionBuilderAddAttr(builder, "attr1: string");
  TF_OpDefinitionBuilderAddInput(builder, "input1: uint8");
  TF_OpDefinitionBuilderAddInput(builder, "input2: uint16");
  TF_OpDefinitionBuilderAddOutput(builder, "output1: uint32");
  TF_Status* status = TF_NewStatus();
  TF_RegisterOpDefinition(builder, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_Buffer* op_list_buffer = TF_GetAllOpList();
  ::tensorflow::OpList op_list;
  op_list.ParseFromArray(op_list_buffer->data, op_list_buffer->length);
  bool found = false;
  for (const auto& op : op_list.op()) {
    if (op.name() == "SomeOp") {
      ASSERT_EQ(2, op.input_arg_size());
      ASSERT_EQ("input1", op.input_arg(0).name());
      ASSERT_EQ(::tensorflow::DT_UINT8, op.input_arg(0).type());
      ASSERT_EQ(1, op.attr_size());
      ASSERT_EQ("string", op.attr(0).type());
      found = true;
    }
  }
  EXPECT_TRUE(found);
  TF_DeleteStatus(status);
  TF_DeleteBuffer(op_list_buffer);
}

void identity_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSops_testDTcc mht_0(mht_0_v, 230, "", "./tensorflow/c/ops_test.cc", "identity_shape_fn");

  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status));
  TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
  TF_DeleteShapeHandle(handle);
}

TEST(OpsTest, TestShapeInference_IdentityFunction) {
  ShapeInferenceTestOp op("SomeTestOp");

  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("SomeTestOp");
  TF_OpDefinitionBuilderAddInput(builder, "input1: uint8");
  TF_OpDefinitionBuilderAddOutput(builder, "output1: uint8");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &identity_shape_fn);
  TF_Status* status = TF_NewStatus();
  TF_RegisterOpDefinition(builder, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_ASSERT_OK(
      shape_inference::ShapeInferenceTestutil::InferShapes(op, "[1,2]", "in0"));
  TF_DeleteStatus(status);
}

TEST(OpsTest, TestShapeInference_UnknownShape) {
  ShapeInferenceTestOp op("UnknownShapeOp");

  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("UnknownShapeOp");
  TF_OpDefinitionBuilderAddInput(builder, "input1: uint8");
  TF_OpDefinitionBuilderAddInput(builder, "input2: uint32");
  TF_OpDefinitionBuilderAddOutput(builder, "output1: uint8");
  TF_OpDefinitionBuilderAddOutput(builder, "output2: uint8");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(
      builder, &TF_ShapeInferenceContextSetUnknownShape);
  TF_Status* status = TF_NewStatus();
  TF_RegisterOpDefinition(builder, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_ASSERT_OK(shape_inference::ShapeInferenceTestutil::InferShapes(
      op, "[1,2];[3,4]", "?;?"));
  TF_DeleteStatus(status);
}

// Creates an output whose shape is a vector of length
// TF_ShapeInferenceContextRank.
void vectorize_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSops_testDTcc mht_1(mht_1_v, 278, "", "./tensorflow/c/ops_test.cc", "vectorize_shape_fn");

  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status));
  TF_ShapeHandle* new_shape = TF_ShapeInferenceContextVectorFromSize(
      ctx, TF_ShapeInferenceContextRank(ctx, handle));
  TF_ShapeInferenceContextSetOutput(ctx, 0, new_shape, status);
  TF_DeleteShapeHandle(handle);
  TF_DeleteShapeHandle(new_shape);
}

TEST(OpsTest, TestShapeInference_VectorizeFunction) {
  ShapeInferenceTestOp op("VectorizeTestOp");

  TF_OpDefinitionBuilder* builder =
      TF_NewOpDefinitionBuilder("VectorizeTestOp");
  TF_OpDefinitionBuilderAddInput(builder, "input1: uint8");
  TF_OpDefinitionBuilderAddOutput(builder, "output1: uint8");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &vectorize_shape_fn);
  TF_Status* status = TF_NewStatus();
  TF_RegisterOpDefinition(builder, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_ASSERT_OK(shape_inference::ShapeInferenceTestutil::InferShapes(
      op, "[4,5,9]", "[3]"));
  TF_DeleteStatus(status);
}

TEST(OpsTest, AttributeAccessors) {
  TF_OpDefinitionBuilder* builder =
      TF_NewOpDefinitionBuilder("AttributeAccessorsOp");
  TF_OpDefinitionBuilderAddAttr(builder, "foo1: int >= 2");
  TF_OpDefinitionBuilderAddAttr(builder, "foo2: string=\"my string\"");
  TF_OpDefinitionBuilderSetIsCommutative(builder, true);
  TF_OpDefinitionBuilderSetIsAggregate(builder, true);
  TF_OpDefinitionBuilderSetAllowsUninitializedInput(builder, true);
  std::string deprecation_msg = "use something else instead";
  TF_OpDefinitionBuilderDeprecated(builder, 4, deprecation_msg.c_str());

  TF_Status* status = TF_NewStatus();
  TF_RegisterOpDefinition(builder, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status));

  TF_Buffer* op_list_buffer = TF_GetAllOpList();
  ::tensorflow::OpList op_list;
  op_list.ParseFromArray(op_list_buffer->data, op_list_buffer->length);
  bool found = false;
  for (const auto& op : op_list.op()) {
    if (op.name() == "AttributeAccessorsOp") {
      ASSERT_TRUE(op.is_commutative());
      ASSERT_TRUE(op.is_aggregate());
      ASSERT_TRUE(op.allows_uninitialized_input());
      ASSERT_EQ(4, op.deprecation().version());
      ASSERT_EQ(deprecation_msg, op.deprecation().explanation());
      ASSERT_EQ(2, op.attr_size());
      ASSERT_EQ("int", op.attr(0).type());
      ASSERT_EQ(2, op.attr(0).minimum());
      ASSERT_EQ("string", op.attr(1).type());
      ASSERT_EQ("my string", op.attr(1).default_value().s());
      found = true;
    }
  }
  ASSERT_TRUE(found);
  TF_DeleteStatus(status);
  TF_DeleteBuffer(op_list_buffer);
}

#define C_CTX(x) reinterpret_cast<TF_ShapeInferenceContext*>(x)
#define C_SHP(x) reinterpret_cast<TF_ShapeHandle*>(x)

static OpDef MakeOpDef(int num_inputs, int num_outputs) {
  OpRegistrationData op_reg_data;
  OpDefBuilder b("dummy");
  for (int i = 0; i < num_inputs; ++i) {
    b.Input(strings::StrCat("i", i, ": float"));
  }
  for (int i = 0; i < num_outputs; ++i) {
    b.Output(strings::StrCat("o", i, ": float"));
  }
  CHECK(b.Attr("foo:string").Finalize(&op_reg_data).ok());
  return op_reg_data.op_def;
}

// Tests for shape inference

PartialTensorShape S(std::initializer_list<int64_t> dims) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSops_testDTcc mht_2(mht_2_v, 366, "", "./tensorflow/c/ops_test.cc", "S");

  return PartialTensorShape(dims);
}

PartialTensorShape Unknown() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSops_testDTcc mht_3(mht_3_v, 373, "", "./tensorflow/c/ops_test.cc", "Unknown");
 return PartialTensorShape(); }

TEST(OpsTest, ShapeInferenceWithRank) {
  NodeDef def;
  shape_inference::InferenceContext c(0, def, MakeOpDef(1, 0),
                                      {S({10, 20, 30})}, {}, {}, {});

  shape_inference::ShapeHandle in0 = c.input(0);
  shape_inference::ShapeHandle s1;

  TF_Status* status = TF_NewStatus();
  TF_ShapeInferenceContextWithRankAtMost(C_CTX(&c), C_SHP(&in0), 3, C_SHP(&s1),
                                         status);
  EXPECT_EQ("[10,20,30]", c.DebugString(s1));
  EXPECT_EQ(TF_OK, TF_GetCode(status));

  TF_ShapeInferenceContextWithRankAtLeast(C_CTX(&c), C_SHP(&in0), 3, C_SHP(&s1),
                                          status);
  EXPECT_EQ("[10,20,30]", c.DebugString(s1));
  EXPECT_EQ(TF_OK, TF_GetCode(status));

  TF_ShapeInferenceContextWithRankAtLeast(C_CTX(&c), C_SHP(&in0), 6, C_SHP(&s1),
                                          status);
  ASSERT_NE(TF_OK, TF_GetCode(status));

  TF_SetStatus(status, TF_OK, "");
  TF_ShapeInferenceContextWithRankAtMost(C_CTX(&c), C_SHP(&in0), 1, C_SHP(&s1),
                                         status);
  ASSERT_NE(TF_OK, TF_GetCode(status));

  TF_SetStatus(status, TF_OK, "");
  TF_ShapeInferenceContextWithRank(C_CTX(&c), C_SHP(&in0), 3, C_SHP(&s1),
                                   status);
  ASSERT_EQ(TF_OK, TF_GetCode(status));

  TF_ShapeInferenceContextWithRank(C_CTX(&c), C_SHP(&in0), 4, C_SHP(&s1),
                                   status);
  ASSERT_NE(TF_OK, TF_GetCode(status));

  TF_DeleteStatus(status);
}

TEST(OpsTest, ShapeInferenceWithRank_UnknownRank) {
  NodeDef def;
  shape_inference::InferenceContext c(0, def, MakeOpDef(2, 2),
                                      {Unknown(), S({1, -1, 3})}, {}, {}, {});

  shape_inference::ShapeHandle in0 = c.input(0);
  shape_inference::ShapeHandle s1;

  // WithRankAtMost and WithRankAtLeast on a shape with unknown dimensionality
  // always succeed.
  TF_Status* status = TF_NewStatus();
  TF_ShapeInferenceContextWithRankAtMost(C_CTX(&c), C_SHP(&in0), 1, C_SHP(&s1),
                                         status);
  EXPECT_EQ("?", c.DebugString(s1));
  EXPECT_EQ(TF_OK, TF_GetCode(status));

  TF_ShapeInferenceContextWithRankAtLeast(C_CTX(&c), C_SHP(&in0), 1, C_SHP(&s1),
                                          status);
  EXPECT_EQ("?", c.DebugString(s1));
  EXPECT_EQ(TF_OK, TF_GetCode(status));

  TF_DeleteStatus(status);
}

TEST(OpsTest, ShapeInferenceConcatenateShapes) {
  NodeDef def;
  shape_inference::InferenceContext c(0, def, MakeOpDef(2, 0),
                                      {S({1, 2}), S({3, 4})}, {}, {}, {});
  ASSERT_EQ(2, TF_ShapeInferenceContextNumInputs(C_CTX(&c)));
  shape_inference::ShapeHandle a = c.input(0);
  shape_inference::ShapeHandle b = c.input(1);
  TF_ShapeHandle* result = TF_NewShapeHandle();
  TF_Status* status = TF_NewStatus();
  TF_ShapeInferenceContextConcatenateShapes(C_CTX(&c), C_SHP(&a), C_SHP(&b),
                                            result, status);
  EXPECT_EQ(
      "[1,2,3,4]",
      c.DebugString(*reinterpret_cast<shape_inference::ShapeHandle*>(result)));
  EXPECT_EQ(TF_OK, TF_GetCode(status));
  TF_DeleteShapeHandle(result);
  TF_DeleteStatus(status);
}

TEST(OpsTest, DimensionHandleValueKnown) {
  NodeDef def;
  shape_inference::InferenceContext c(0, def, MakeOpDef(2, 0),
                                      {S({1, 2}), S({3, 4})}, {}, {}, {});
  TF_ShapeHandle* handle =
      TF_ShapeInferenceContextVectorFromSize(C_CTX(&c), 43);
  ASSERT_EQ(
      "[43]",
      c.DebugString(*reinterpret_cast<shape_inference::ShapeHandle*>(handle)));
  ASSERT_EQ(1, TF_ShapeInferenceContextRankKnown(C_CTX(&c), handle));
  ASSERT_EQ(1, TF_ShapeInferenceContextRank(C_CTX(&c), handle));

  TF_DimensionHandle* dim_handle = TF_NewDimensionHandle();
  TF_ShapeInferenceContextDim(C_CTX(&c), handle, 0, dim_handle);
  ASSERT_EQ(1, TF_DimensionHandleValueKnown(dim_handle));
  ASSERT_EQ(43, TF_DimensionHandleValue(dim_handle));
  TF_DeleteShapeHandle(handle);
  TF_DeleteDimensionHandle(dim_handle);
}

TEST(OpsTest, ShapeInferenceSubshape) {
  NodeDef def;
  shape_inference::InferenceContext c(0, def, MakeOpDef(1, 0),
                                      {S({10, 20, 30, 40, 50})}, {}, {}, {});
  ASSERT_EQ("[10,20,30,40,50]", c.DebugString(c.input(0)));

  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_Status* status = TF_NewStatus();
  TF_ShapeInferenceContextGetInput(C_CTX(&c), 0, handle, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status));
  TF_ShapeInferenceContextSubshape(C_CTX(&c), handle, 1, -1, handle, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status));
  ASSERT_EQ(
      "[20,30,40]",
      c.DebugString(*reinterpret_cast<shape_inference::ShapeHandle*>(handle)));
  TF_DeleteStatus(status);
  TF_DeleteShapeHandle(handle);
}

TEST(OpsTest, ShapeInferenceScalarShape) {
  NodeDef def;
  shape_inference::InferenceContext c(0, def, MakeOpDef(0, 0), {S({})}, {}, {},
                                      {});
  TF_ShapeHandle* TF_scalar_shape = TF_ShapeInferenceContextScalar(C_CTX(&c));
  shape_inference::ShapeHandle* scalar_shape =
      reinterpret_cast<shape_inference::ShapeHandle*>(TF_scalar_shape);
  ASSERT_EQ("[]", c.DebugString(*scalar_shape));
  TF_DeleteShapeHandle(TF_scalar_shape);
}

}  // namespace
}  // namespace tensorflow
