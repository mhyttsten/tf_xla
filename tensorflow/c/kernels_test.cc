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
class MHTracer_DTPStensorflowPScPSkernels_testDTcc {
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
   MHTracer_DTPStensorflowPScPSkernels_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSkernels_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "tensorflow/c/kernels.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

struct MyCustomKernel {
  bool created;
  bool compute_called;
};

static bool delete_called = false;

static void* MyCreateFunc(TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_0(mht_0_v, 233, "", "./tensorflow/c/kernels_test.cc", "MyCreateFunc");

  struct MyCustomKernel* s = new struct MyCustomKernel;
  s->created = true;
  s->compute_called = false;

  // Exercise attribute reads.
  TF_DataType type;
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrType(ctx, "SomeDataTypeAttr", &type, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));
  EXPECT_EQ(TF_FLOAT, type);
  TF_DeleteStatus(status);

  // Exercise kernel NodeDef name read
  TF_StringView name_string_view = TF_OpKernelConstruction_GetName(ctx);
  std::string node_name = "SomeNodeName";
  std::string candidate_node_name =
      std::string(name_string_view.data, name_string_view.len);
  EXPECT_EQ(node_name, candidate_node_name);
  return s;
}

static void MyComputeFunc(void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_1(mht_1_v, 258, "", "./tensorflow/c/kernels_test.cc", "MyComputeFunc");

  struct MyCustomKernel* s = static_cast<struct MyCustomKernel*>(kernel);
  s->compute_called = true;
  if (ctx != nullptr) {
    EXPECT_EQ(43, TF_StepId(ctx));
  }
}

static void MyDeleteFunc(void* kernel) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_2(mht_2_v, 269, "", "./tensorflow/c/kernels_test.cc", "MyDeleteFunc");

  struct MyCustomKernel* s = static_cast<struct MyCustomKernel*>(kernel);
  EXPECT_TRUE(s->created);
  EXPECT_TRUE(s->compute_called);
  delete_called = true;
  delete s;
}

namespace tensorflow {
Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);

static std::unique_ptr<OpKernel> GetFakeKernel(const char* device_name,
                                               const char* op_name,
                                               const char* node_name,
                                               Status* status) {
  NodeDef def;
  def.set_op(op_name);
  def.set_name(node_name);
  def.set_device(device_name);
  def.add_input("input1");
  def.add_input("input2");

  AttrValue v;
  v.set_type(DataType::DT_FLOAT);
  (*def.mutable_attr())["SomeDataTypeAttr"] = v;

  return CreateOpKernel(DeviceType(device_name), nullptr, nullptr, def, 1,
                        status);
}

// Tests registration of a single C kernel and checks that calls through the
// C/C++ boundary are being made.
TEST(TestKernel, TestRegisterKernelBuilder) {
  const char* node_name = "SomeNodeName";
  const char* op_name = "FooOp";
  const char* device_name = "FakeDeviceName1";

  REGISTER_OP(op_name)
      .Input("input1: double")
      .Input("input2: uint8")
      .Output("output1: uint8")
      .Attr("SomeDataTypeAttr: type");

  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, device_name, &MyCreateFunc, &MyComputeFunc, &MyDeleteFunc);

  {
    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(node_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_Buffer* buf = TF_GetRegisteredKernelsForOp(op_name, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    KernelList list;
    list.ParseFromArray(buf->data, buf->length);
    ASSERT_EQ(1, list.kernel_size());
    ASSERT_EQ(device_name, list.kernel(0).device_type());
    TF_DeleteBuffer(buf);
    TF_DeleteStatus(status);
  }

  {
    Status status;
    std::unique_ptr<OpKernel> kernel =
        GetFakeKernel(device_name, op_name, node_name, &status);
    TF_EXPECT_OK(status);
    ASSERT_NE(nullptr, kernel.get());
    kernel->Compute(nullptr);
  }

  ASSERT_TRUE(delete_called);
}

// REGISTER_OP for TF_OpKernelConstruction_GetAttr* test cases.
// Registers two ops, each with a single attribute called 'Attr'.
// The attribute in one op will have a type 'type', the other
// will have list(type).
#define ATTR_TEST_REGISTER_OP(name, type)                     \
  REGISTER_OP("TestKernelAttr" #name)                         \
      .Attr("Attr: " #type)                                   \
      .SetShapeFn(tensorflow::shape_inference::UnknownShape); \
  REGISTER_OP("TestKernelAttr" #name "List")                  \
      .Attr("Attr: list(" #type ")")                          \
      .SetShapeFn(tensorflow::shape_inference::UnknownShape)
ATTR_TEST_REGISTER_OP(String, string);
ATTR_TEST_REGISTER_OP(Int, int);
ATTR_TEST_REGISTER_OP(Float, float);
ATTR_TEST_REGISTER_OP(Bool, bool);
ATTR_TEST_REGISTER_OP(Type, type);
ATTR_TEST_REGISTER_OP(Tensor, tensor);
#undef ATTR_TEST_REGISTER_OP

// Helper macros for the TF_OpKernelConstruction_GetAttr* tests.
#define EXPECT_TF_SIZE(attr_name, expected_list_size, expected_total_size) \
  do {                                                                     \
    int32_t list_size, total_size;                                         \
    TF_OpKernelConstruction_GetAttrSize(ctx, attr_name, &list_size,        \
                                        &total_size, status);              \
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);            \
    EXPECT_EQ(expected_list_size, list_size);                              \
    EXPECT_EQ(expected_total_size, total_size);                            \
  } while (0)

typedef void* (*MyCreateFuncWithAttr)(TF_OpKernelConstruction*);
class TestKernelAttr : public ::testing::Test {
 public:
  TestKernelAttr() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_3(mht_3_v, 377, "", "./tensorflow/c/kernels_test.cc", "TestKernelAttr");
}
  ~TestKernelAttr() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_4(mht_4_v, 381, "", "./tensorflow/c/kernels_test.cc", "~TestKernelAttr");
}

  std::unique_ptr<OpKernel> GetFakeKernelWithAttr(const char* op_name,
                                                  AttrValue v, Status* status) {
    NodeDef def;
    def.set_op(op_name);
    def.set_name("FakeNode");
    def.set_device("FakeDevice");
    (*def.mutable_attr())["Attr"] = v;
    return CreateOpKernel(DeviceType("FakeDevice"), nullptr, nullptr, def, 1,
                          status);
  }

  void CreateAndCallKernelWithAttr(MyCreateFuncWithAttr MyCreateFuncAttr,
                                   const char* op_name, AttrValue& v) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_5(mht_5_v, 399, "", "./tensorflow/c/kernels_test.cc", "CreateAndCallKernelWithAttr");

    TF_KernelBuilder* builder = TF_NewKernelBuilder(
        op_name, "FakeDevice", MyCreateFuncAttr, &MyComputeFunc, &MyDeleteFunc);
    {
      TF_Status* status = TF_NewStatus();
      TF_RegisterKernelBuilder("FakeNode", builder, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status));
      TF_DeleteStatus(status);
    }
    Status status;
    std::unique_ptr<OpKernel> kernel =
        GetFakeKernelWithAttr(op_name, v, &status);
    TF_EXPECT_OK(status);
    ASSERT_NE(nullptr, kernel.get());
    kernel->Compute(nullptr);

    ASSERT_TRUE(delete_called);
  }
};

TEST_F(TestKernelAttr, String) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_6(mht_6_v, 423, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    std::unique_ptr<char[]> val(new char[5]);
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ 5);
    TF_OpKernelConstruction_GetAttrString(ctx, "Attr", val.get(),
                                          /*max_length*/ 5, status);

    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_EQ("bunny", string(static_cast<const char*>(val.get()), 5));
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  v.set_s("bunny");
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrString", v);
}

TEST_F(TestKernelAttr, StringList) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_7(mht_7_v, 450, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    std::vector<string> list = {"bugs", "bunny", "duck"};
    int list_total_size = 0;
    for (const auto& s : list) {
      list_total_size += s.size();
    }

    TF_Status* status = TF_NewStatus();
    std::unique_ptr<char*[]> values(new char*[list.size()]);
    std::unique_ptr<size_t[]> lens(new size_t[list.size()]);
    std::unique_ptr<char[]> storage(new char[list_total_size]);
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list.size(),
                   /*expected_total_size*/ list_total_size);
    TF_OpKernelConstruction_GetAttrStringList(
        ctx, "Attr", values.get(), lens.get(), list.size(), storage.get(),
        list_total_size, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    for (size_t i = 0; i < list.size(); ++i) {
      EXPECT_EQ(list[i].size(), lens[i]) << i;
      EXPECT_EQ(list[i], string(static_cast<const char*>(values[i]), lens[i]))
          << i;
    }
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  std::string attr_in[] = {"bugs", "bunny", "duck"};
  SetAttrValue(gtl::ArraySlice<std::string>(attr_in, 3), &v);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrStringList", v);
}

TEST_F(TestKernelAttr, Tensor) {
  struct TensorProtoHelpers {
   public:
    static ::tensorflow::TensorProto GenerateTensorProto() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_8(mht_8_v, 493, "", "./tensorflow/c/kernels_test.cc", "GenerateTensorProto");

      ::tensorflow::TensorProto tensor_proto;
      tensor_proto.mutable_tensor_shape()->add_dim()->set_size(2);
      tensor_proto.mutable_tensor_shape()->add_dim()->set_size(3);
      tensor_proto.set_dtype(DT_INT32);
      tensor_proto.add_int_val(1);
      tensor_proto.add_int_val(2);
      tensor_proto.add_int_val(3);
      tensor_proto.add_int_val(4);
      tensor_proto.add_int_val(5);
      tensor_proto.add_int_val(6);
      return tensor_proto;
    }
  };

  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_9(mht_9_v, 511, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    TF_Tensor* val;
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrTensor(ctx, "Attr", &val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    ::tensorflow::Tensor expected_tensor;
    EXPECT_TRUE(
        expected_tensor.FromProto(TensorProtoHelpers::GenerateTensorProto()));

    ::tensorflow::Tensor actual_tensor;
    EXPECT_TRUE(TF_TensorToTensor(val, &actual_tensor).ok());

    EXPECT_EQ(actual_tensor.tensor_data(), expected_tensor.tensor_data());
    EXPECT_EQ(actual_tensor.shape(), expected_tensor.shape());
    EXPECT_EQ(actual_tensor.dtype(), expected_tensor.dtype());

    TF_DeleteStatus(status);
    TF_DeleteTensor(val);
    return static_cast<void*>(s);
  };

  AttrValue v;
  ::tensorflow::TensorProto* tensor_proto = v.mutable_tensor();
  *tensor_proto = TensorProtoHelpers::GenerateTensorProto();

  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrTensor", v);
}

TEST_F(TestKernelAttr, TensorList) {
  struct TensorProtoHelpers {
   public:
    static ::tensorflow::TensorProto GenerateTensorProto1() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_10(mht_10_v, 552, "", "./tensorflow/c/kernels_test.cc", "GenerateTensorProto1");

      ::tensorflow::TensorProto tensor_proto;
      tensor_proto.mutable_tensor_shape()->add_dim()->set_size(2);
      tensor_proto.mutable_tensor_shape()->add_dim()->set_size(2);
      tensor_proto.set_dtype(DT_INT32);
      tensor_proto.add_int_val(1);
      tensor_proto.add_int_val(2);
      tensor_proto.add_int_val(3);
      tensor_proto.add_int_val(4);
      return tensor_proto;
    }

    static ::tensorflow::TensorProto GenerateTensorProto2() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_11(mht_11_v, 567, "", "./tensorflow/c/kernels_test.cc", "GenerateTensorProto2");

      ::tensorflow::TensorProto tensor_proto;
      tensor_proto.mutable_tensor_shape()->add_dim()->set_size(2);
      tensor_proto.mutable_tensor_shape()->add_dim()->set_size(3);
      tensor_proto.set_dtype(DT_FLOAT);
      tensor_proto.add_float_val(5.0f);
      tensor_proto.add_float_val(6.0f);
      tensor_proto.add_float_val(7.0f);
      tensor_proto.add_float_val(8.0f);
      tensor_proto.add_float_val(9.0f);
      tensor_proto.add_float_val(10.0f);
      return tensor_proto;
    }
  };

  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_12(mht_12_v, 585, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    const size_t list_size = 2;
    TF_Tensor* values[list_size];

    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list_size,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrTensorList(ctx, "Attr", values, list_size,
                                              status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    ::tensorflow::Tensor expected_tensor1;
    EXPECT_TRUE(
        expected_tensor1.FromProto(TensorProtoHelpers::GenerateTensorProto1()));

    ::tensorflow::Tensor actual_tensor1;
    EXPECT_TRUE(TF_TensorToTensor(values[0], &actual_tensor1).ok());

    EXPECT_EQ(actual_tensor1.tensor_data(), expected_tensor1.tensor_data());
    EXPECT_EQ(actual_tensor1.shape(), expected_tensor1.shape());
    EXPECT_EQ(actual_tensor1.dtype(), expected_tensor1.dtype());

    ::tensorflow::Tensor expected_tensor2;
    EXPECT_TRUE(
        expected_tensor2.FromProto(TensorProtoHelpers::GenerateTensorProto2()));

    ::tensorflow::Tensor actual_tensor2;
    EXPECT_TRUE(TF_TensorToTensor(values[1], &actual_tensor2).ok());

    EXPECT_EQ(actual_tensor2.tensor_data(), expected_tensor2.tensor_data());
    EXPECT_EQ(actual_tensor2.shape(), expected_tensor2.shape());
    EXPECT_EQ(actual_tensor2.dtype(), expected_tensor2.dtype());

    TF_DeleteStatus(status);
    TF_DeleteTensor(values[0]);
    TF_DeleteTensor(values[1]);
    return static_cast<void*>(s);
  };

  AttrValue v;
  ::tensorflow::TensorProto* tensor_proto1 = v.mutable_list()->add_tensor();
  *tensor_proto1 = TensorProtoHelpers::GenerateTensorProto1();

  ::tensorflow::TensorProto* tensor_proto2 = v.mutable_list()->add_tensor();
  *tensor_proto2 = TensorProtoHelpers::GenerateTensorProto2();

  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrTensorList", v);
}

TEST_F(TestKernelAttr, Int) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_13(mht_13_v, 642, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    int64_t val;
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrInt64(ctx, "Attr", &val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_EQ(1234, val);
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  v.set_i(1234);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrInt", v);
}

TEST_F(TestKernelAttr, IntList) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_14(mht_14_v, 667, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    const int64_t list[] = {1, 2, 3, 4};
    const size_t list_size = TF_ARRAYSIZE(list);
    int64_t values[list_size];

    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list_size,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrInt64List(ctx, "Attr", values, list_size,
                                             status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_TRUE(
        std::equal(std::begin(list), std::end(list), std::begin(values)));
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  int64_t attr_in[] = {1, 2, 3, 4};
  SetAttrValue(gtl::ArraySlice<int64_t>(attr_in, 4), &v);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrIntList", v);
}

TEST_F(TestKernelAttr, Float) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_15(mht_15_v, 698, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    float val;
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrFloat(ctx, "Attr", &val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_FLOAT_EQ(2.718, val);
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  v.set_f(2.718);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrFloat", v);
}

TEST_F(TestKernelAttr, FloatList) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_16(mht_16_v, 723, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    const float list[] = {1.414, 2.718, 3.1415};
    const size_t list_size = TF_ARRAYSIZE(list);
    float values[list_size];

    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list_size,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrFloatList(ctx, "Attr", values, list_size,
                                             status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_TRUE(
        std::equal(std::begin(list), std::end(list), std::begin(values)));
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  float attr_in[] = {1.414, 2.718, 3.1415};
  SetAttrValue(gtl::ArraySlice<float>(attr_in, 3), &v);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrFloatList", v);
}

TEST_F(TestKernelAttr, Bool) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_17(mht_17_v, 754, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    unsigned char val;
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrBool(ctx, "Attr", &val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_EQ(1, val);
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  v.set_b(true);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrBool", v);
}

TEST_F(TestKernelAttr, BoolList) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_18(mht_18_v, 779, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    const unsigned char list[] = {1, 0, 1, 0};
    const size_t list_size = TF_ARRAYSIZE(list);
    unsigned char values[list_size];

    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list_size,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrBoolList(ctx, "Attr", values, list_size,
                                            status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_TRUE(
        std::equal(std::begin(list), std::end(list), std::begin(values)));
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  bool attr_in[] = {true, false, true, false};
  SetAttrValue(gtl::ArraySlice<bool>(attr_in, 4), &v);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrBoolList", v);
}

TEST_F(TestKernelAttr, Type) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_19(mht_19_v, 810, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    TF_DataType val;
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrType(ctx, "Attr", &val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_EQ(TF_FLOAT, val);
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  v.set_type(DT_FLOAT);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrType", v);
}

TEST_F(TestKernelAttr, TypeList) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_20(mht_20_v, 835, "", "./tensorflow/c/kernels_test.cc", "lambda");

    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    const TF_DataType list[] = {TF_FLOAT, TF_DOUBLE, TF_HALF, TF_COMPLEX128};
    const size_t list_size = TF_ARRAYSIZE(list);
    TF_DataType values[list_size];

    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list_size,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrTypeList(ctx, "Attr", values, list_size,
                                            status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_TRUE(
        std::equal(std::begin(list), std::end(list), std::begin(values)));
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  DataType attr_in[] = {DT_FLOAT, DT_DOUBLE, DT_HALF, DT_COMPLEX128};
  SetAttrValue(gtl::ArraySlice<DataType>(attr_in, 4), &v);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrTypeList", v);
}
#undef EXPECT_TF_SIZE

class DummyDevice : public DeviceBase {
 public:
  explicit DummyDevice(Env* env) : DeviceBase(env) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_21(mht_21_v, 868, "", "./tensorflow/c/kernels_test.cc", "DummyDevice");
}
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_22(mht_22_v, 872, "", "./tensorflow/c/kernels_test.cc", "GetAllocator");

    return cpu_allocator();
  }
};

TEST(TestKernel, TestInputAndOutputCount) {
  const char* node_name = "InputOutputCounterKernel";
  const char* op_name = "BarOp";
  const char* device_name = "FakeDeviceName2";

  REGISTER_OP(op_name)
      .Input("input1: double")
      .Input("input2: uint8")
      .Output("output1: uint8")
      .Attr("SomeDataTypeAttr: type");

  static int num_inputs = 0;
  static int num_outputs = 0;

  // A kernel whose Compute function has a side-effect of updating num_inputs
  // and num_outputs. Various functions on TF_OpKernelContext are also
  // exercised.
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_23(mht_23_v, 897, "", "./tensorflow/c/kernels_test.cc", "lambda");

    num_inputs = TF_NumInputs(ctx);
    num_outputs = TF_NumOutputs(ctx);

    TF_Tensor* input = nullptr;
    TF_Status* s = TF_NewStatus();
    TF_GetInput(ctx, 0, &input, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s)) << "Failed to get input: " << TF_Message(s);
    EXPECT_EQ(123, *static_cast<tensorflow::uint8*>(TF_TensorData(input)));
    TF_GetInput(ctx, -1, &input, s);
    EXPECT_EQ(TF_OUT_OF_RANGE, TF_GetCode(s));
    TF_GetInput(ctx, 3, &input, s);
    EXPECT_EQ(TF_OUT_OF_RANGE, TF_GetCode(s));

    // Copy the input tensor to output.
    TF_SetOutput(ctx, 0, input, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));

    TF_SetOutput(ctx, 24, input, s);
    EXPECT_EQ(TF_OUT_OF_RANGE, TF_GetCode(s));

    EXPECT_EQ(TF_UINT8, TF_ExpectedOutputDataType(ctx, 0));

    EXPECT_DEATH({ TF_ExpectedOutputDataType(ctx, 1); },
                 "Check failed: i < cc_ctx->num_outputs");

    EXPECT_DEATH({ TF_ExpectedOutputDataType(ctx, -1); },
                 "Check failed: i >= 0");

    TF_DeleteStatus(s);
    if (input != nullptr) {
      TF_DeleteTensor(input);
    }
  };

  TF_KernelBuilder* builder = TF_NewKernelBuilder(op_name, device_name, nullptr,
                                                  my_compute_func, nullptr);

  {
    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(node_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_DeleteStatus(status);
  }

  {
    OpKernelContext::Params p;
    DummyDevice dummy_device(nullptr);
    p.device = &dummy_device;
    p.step_id = 43;

    Tensor t(tensorflow::uint8(123));

    gtl::InlinedVector<TensorValue, 4> inputs;
    // Simulate 2 inputs
    inputs.emplace_back(&t);
    inputs.emplace_back();
    p.inputs = &inputs;

    Status status;
    std::unique_ptr<OpKernel> kernel =
        GetFakeKernel(device_name, op_name, node_name, &status);
    TF_EXPECT_OK(status);
    ASSERT_NE(nullptr, kernel.get());

    p.op_kernel = kernel.get();
    OpKernelContext ctx(&p);
    kernel->Compute(&ctx);

    ASSERT_EQ(2, num_inputs);
    ASSERT_EQ(1, num_outputs);
    ASSERT_EQ(123, ctx.mutable_output(0)->scalar<tensorflow::uint8>()());
  }
}

TEST(TestKernel, DeleteKernelBuilderIsOkOnNull) {
  TF_DeleteKernelBuilder(nullptr);
}

std::string ExpectedString(const char* type) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("type: \"" + (type == nullptr ? std::string("nullptr") : std::string((char*)type)) + "\"");
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_24(mht_24_v, 980, "", "./tensorflow/c/kernels_test.cc", "ExpectedString");

  const auto format_str = R"str(kernel {
  op: "TypeOp%s"
  device_type: "FakeDeviceName1"
  constraint {
    name: "T"
    allowed_values {
      list {
        type: %s
      }
    }
  }
}
)str";
  return absl::StrFormat(format_str, type, type);
}

#define TEST_KERNEL_TYPE_CONSTRAINT(tf_type, dtype)                          \
  TEST(TestKernel, TestTypeConstraint##tf_type) {                            \
    const char* node_name = "SomeNodeName";                                  \
    const char* op_name = "TypeOp" #dtype;                                   \
    const char* device_name = "FakeDeviceName1";                             \
                                                                             \
    REGISTER_OP(op_name)                                                     \
        .Input("input1: double")                                             \
        .Input("input2: uint8")                                              \
        .Output("output1: uint8")                                            \
        .Attr("T: type");                                                    \
                                                                             \
    TF_KernelBuilder* builder = TF_NewKernelBuilder(                         \
        op_name, device_name, &MyCreateFunc, &MyComputeFunc, &MyDeleteFunc); \
    TF_Status* status = TF_NewStatus();                                      \
    TF_KernelBuilder_TypeConstraint(builder, "T", TF_DataType::tf_type,      \
                                    status);                                 \
    EXPECT_EQ(TF_OK, TF_GetCode(status));                                    \
    TF_RegisterKernelBuilder(node_name, builder, status);                    \
    EXPECT_EQ(TF_OK, TF_GetCode(status));                                    \
                                                                             \
    TF_Buffer* buf = TF_GetRegisteredKernelsForOp(op_name, status);          \
    EXPECT_EQ(TF_OK, TF_GetCode(status));                                    \
    KernelList list;                                                         \
    list.ParseFromArray(buf->data, buf->length);                             \
    KernelList expected_proto;                                               \
    protobuf::TextFormat::ParseFromString(ExpectedString(#dtype),            \
                                          &expected_proto);                  \
    ASSERT_EQ(expected_proto.DebugString(), list.DebugString());             \
                                                                             \
    TF_DeleteBuffer(buf);                                                    \
    TF_DeleteStatus(status);                                                 \
    TF_DeleteKernelBuilder(builder);                                         \
    ASSERT_TRUE(delete_called);                                              \
  }

TEST_KERNEL_TYPE_CONSTRAINT(TF_HALF, DT_HALF);
TEST_KERNEL_TYPE_CONSTRAINT(TF_BFLOAT16, DT_BFLOAT16);
TEST_KERNEL_TYPE_CONSTRAINT(TF_FLOAT, DT_FLOAT);
TEST_KERNEL_TYPE_CONSTRAINT(TF_DOUBLE, DT_DOUBLE);
TEST_KERNEL_TYPE_CONSTRAINT(TF_UINT64, DT_UINT64);
TEST_KERNEL_TYPE_CONSTRAINT(TF_UINT32, DT_UINT32);
TEST_KERNEL_TYPE_CONSTRAINT(TF_UINT16, DT_UINT16);
TEST_KERNEL_TYPE_CONSTRAINT(TF_UINT8, DT_UINT8);
TEST_KERNEL_TYPE_CONSTRAINT(TF_INT8, DT_INT8);
TEST_KERNEL_TYPE_CONSTRAINT(TF_INT32, DT_INT32);
TEST_KERNEL_TYPE_CONSTRAINT(TF_COMPLEX64, DT_COMPLEX64);
TEST_KERNEL_TYPE_CONSTRAINT(TF_COMPLEX128, DT_COMPLEX128);
TEST_KERNEL_TYPE_CONSTRAINT(TF_QINT8, DT_QINT8);
TEST_KERNEL_TYPE_CONSTRAINT(TF_QUINT8, DT_QUINT8);
TEST_KERNEL_TYPE_CONSTRAINT(TF_QINT32, DT_QINT32);
TEST_KERNEL_TYPE_CONSTRAINT(TF_QINT16, DT_QINT16);
TEST_KERNEL_TYPE_CONSTRAINT(TF_QUINT16, DT_QUINT16);

TEST(TestKernel, TestHostMemory) {
  const char* node_name = "SomeNodeName";
  const char* op_name = "HostMemoryOp";
  const char* device_name = "FakeDeviceName1";

  REGISTER_OP(op_name)
      .Input("input1: double")
      .Input("input2: uint8")
      .Output("output1: uint8")
      .Attr("T: type");

  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, device_name, &MyCreateFunc, &MyComputeFunc, &MyDeleteFunc);
  TF_KernelBuilder_HostMemory(builder, "input2");
  TF_KernelBuilder_HostMemory(builder, "output1");
  TF_Status* status = TF_NewStatus();
  TF_RegisterKernelBuilder(node_name, builder, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));

  TF_Buffer* buf = TF_GetRegisteredKernelsForOp(op_name, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));
  KernelList list;
  list.ParseFromArray(buf->data, buf->length);
  KernelList expected_proto;
  protobuf::TextFormat::ParseFromString(
      R"str(kernel {
  op: "HostMemoryOp"
  device_type: "FakeDeviceName1"
  host_memory_arg: "input2"
  host_memory_arg: "output1"
}
)str",
      &expected_proto);
  ASSERT_EQ(list.DebugString(), expected_proto.DebugString());

  TF_DeleteBuffer(buf);
  TF_DeleteStatus(status);
  TF_DeleteKernelBuilder(builder);
  ASSERT_TRUE(delete_called);
}

class DeviceKernelOpTest : public OpsTestBase {
 protected:
  void SetupOp(const char* op_name, const char* node_name,
               void (*compute_func)(void*, TF_OpKernelContext*)) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   mht_25_v.push_back("node_name: \"" + (node_name == nullptr ? std::string("nullptr") : std::string((char*)node_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_25(mht_25_v, 1100, "", "./tensorflow/c/kernels_test.cc", "SetupOp");

    TF_KernelBuilder* builder = TF_NewKernelBuilder(
        op_name, device_name_, nullptr, compute_func, nullptr);
    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(node_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_DeleteStatus(status);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice(device_name_, {}, "/job:a/replica:0/task:0"));
    OpsTestBase::SetDevice(DEVICE_GPU, std::move(device));
#endif
    TF_ASSERT_OK(NodeDefBuilder(op_name, op_name).Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  const char* device_name_ = tensorflow::DEVICE_GPU;
#else
  const char* device_name_ = tensorflow::DEVICE_CPU;
#endif
};

// Validates that the tensor has shape and type corresponding to
// dims and dtype.
void validate_tensor(TF_Tensor* tensor, int64_t* dims, int64_t num_dims,
                     TF_DataType dtype);

// Copies data of length tensor_size_bytes from values to tensor.
template <typename T>
void set_tensor_data(TF_Tensor* tensor, T* values, size_t tensor_size_bytes,
                     TF_OpKernelContext* ctx);

REGISTER_OP("StreamOp").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestStream) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_26(mht_26_v, 1140, "", "./tensorflow/c/kernels_test.cc", "lambda");

    TF_Status* s = TF_NewStatus();
    SP_Stream stream = TF_GetStream(ctx, s);
    // Stream is always null if device is not a pluggable device. More test
    // cases will be added when pluggable device mechanism is supported.
    EXPECT_EQ(stream, nullptr);
    EXPECT_NE(TF_OK, TF_GetCode(s));
    TF_DeleteStatus(s);
  };

  SetupOp("StreamOp", "StreamOp", my_compute_func);
  TF_ASSERT_OK(RunOpKernel());
}

REGISTER_OP("AllocateOutputOp1").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateOutputSizeOne) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_27(mht_27_v, 1160, "", "./tensorflow/c/kernels_test.cc", "lambda");

    // Allocate output
    TF_Status* s = TF_NewStatus();
    int64_t dim = 1;
    size_t tensor_size_bytes = TF_DataTypeSize(TF_FLOAT);
    TF_Tensor* output = TF_AllocateOutput(
        /*context=*/ctx, /*index=*/0, /*dtype=*/TF_FLOAT, /*dims=*/&dim,
        /*num_dims=*/1, /*len=*/tensor_size_bytes, s);
    validate_tensor(output, &dim, 1, TF_FLOAT);

    // Set output to 3
    float values[1] = {3.0f};
    set_tensor_data<float>(output, values, tensor_size_bytes, ctx);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateOutputOp1", "AllocateOutput1", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [1] values: 3>",
            output->DebugString(100));
}

REGISTER_OP("AllocateOutputOp0").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateEmptyOutput) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_28(mht_28_v, 1191, "", "./tensorflow/c/kernels_test.cc", "lambda");

    TF_Status* s = TF_NewStatus();
    // Allocate empty output
    int64_t dim = 0;
    TF_Tensor* output = TF_AllocateOutput(
        /*context=*/ctx, /*index=*/0, /*dtype=*/TF_FLOAT, /*dims=*/&dim,
        /*num_dims=*/1, /*len=*/0, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    validate_tensor(output, &dim, 1, TF_FLOAT);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateOutputOp0", "AllocateOutput0", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [0] values: >",
            output->DebugString(100));
}

REGISTER_OP("AllocateOutputOp2x3").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateOutputSize2x3) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_29(mht_29_v, 1218, "", "./tensorflow/c/kernels_test.cc", "lambda");

    TF_Status* s = TF_NewStatus();
    // Allocate 2x3 output
    int64_t dim[2] = {2, 3};
    size_t tensor_size_bytes = TF_DataTypeSize(TF_FLOAT) * 6;
    TF_Tensor* output = TF_AllocateOutput(
        /*context=*/ctx, /*index=*/0, /*dtype=*/TF_FLOAT, /*dims=*/dim,
        /*num_dims=*/2, /*len=*/tensor_size_bytes, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    validate_tensor(output, dim, 2, TF_FLOAT);

    // Set output to [1 2 3 4 5 6]
    float values[6] = {1, 2, 3, 4, 5, 6};
    set_tensor_data<float>(output, values, tensor_size_bytes, ctx);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateOutputOp2x3", "AllocateOutput2x3", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [2,3] values: [1 2 3][4 5 6]>",
            output->DebugString(100));
}

REGISTER_OP("AllocateTempOp1").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateTempSizeOne) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_30(mht_30_v, 1250, "", "./tensorflow/c/kernels_test.cc", "lambda");

    // Allocate scalar TF_Tensor
    TF_Status* s = TF_NewStatus();
    int64_t dim = 1;
    TF_AllocatorAttributes alloc_attrs;
    alloc_attrs.struct_size = TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    alloc_attrs.on_host = 0;
#else
    alloc_attrs.on_host = 1;
#endif
    TF_Tensor* output = TF_AllocateTemp(
        /*context=*/ctx, /*dtype=*/TF_FLOAT, /*dims=*/&dim,
        /*num_dims=*/1, /*allocator_attributes*/ &alloc_attrs, s);
    size_t tensor_size_bytes = TF_DataTypeSize(TF_FLOAT);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    validate_tensor(output, &dim, 1, TF_FLOAT);

    // Set TF_Tensor value to 3
    float values[1] = {3.0f};
    set_tensor_data<float>(output, values, tensor_size_bytes, ctx);
    TF_SetOutput(ctx, 0, output, s);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateTempOp1", "AllocateTemp1", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [1] values: 3>",
            output->DebugString(100));
}

REGISTER_OP("AllocateTempOp0").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateTempEmpty) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_31(mht_31_v, 1290, "", "./tensorflow/c/kernels_test.cc", "lambda");

    TF_Status* s = TF_NewStatus();
    // Allocate empty TF_Tensor
    int64_t dim = 0;
    TF_AllocatorAttributes alloc_attrs;
    alloc_attrs.struct_size = TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    alloc_attrs.on_host = 0;
#else
    alloc_attrs.on_host = 1;
#endif
    TF_Tensor* output = TF_AllocateTemp(
        /*context=*/ctx, /*dtype=*/TF_FLOAT, /*dims=*/&dim,
        /*num_dims=*/1, /*allocator_attributes*/ &alloc_attrs, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    validate_tensor(output, &dim, 1, TF_FLOAT);
    TF_SetOutput(ctx, 0, output, s);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateTempOp0", "AllocateTemp0", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [0] values: >",
            output->DebugString(100));
}

REGISTER_OP("AllocateTempOp2x3").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateTempSize2x3) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_32(mht_32_v, 1325, "", "./tensorflow/c/kernels_test.cc", "lambda");

    TF_Status* s = TF_NewStatus();
    size_t tensor_size_bytes = 6 * TF_DataTypeSize(TF_FLOAT);
    // Allocate 2x3 TF_Tensor
    int64_t dim[2] = {2, 3};
    TF_AllocatorAttributes alloc_attrs;
    alloc_attrs.struct_size = TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    alloc_attrs.on_host = 0;
#else
    alloc_attrs.on_host = 1;
#endif
    TF_Tensor* output = TF_AllocateTemp(
        /*context=*/ctx, /*dtype=*/TF_FLOAT, /*dims=*/dim,
        /*num_dims=*/2, /*allocator_attributes*/ &alloc_attrs, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    validate_tensor(output, dim, 2, TF_FLOAT);

    // Set TF_Tensor values to [1 2 3 4 5 6]
    float values[6] = {1, 2, 3, 4, 5, 6};
    set_tensor_data<float>(output, values, tensor_size_bytes, ctx);
    TF_SetOutput(ctx, 0, output, s);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateTempOp2x3", "AllocateTempOp2x3", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [2,3] values: [1 2 3][4 5 6]>",
            output->DebugString(100));
}

TEST_F(DeviceKernelOpTest, TestForwardInputOrAllocateOutput) {
  const char* node_name = "TestForwardInputOrAllocateOutputKernel";
  const char* op_name = "BazOp";
  const char* device_name = "FakeDeviceName";

  REGISTER_OP(op_name)
      .Input("input1: float")
      .Input("input2: float")
      .Output("output1: float")
      .Attr("SomeDataTypeAttr: type");

  // A kernel whose Compute function that forwards a scalar input to output
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_33(mht_33_v, 1374, "", "./tensorflow/c/kernels_test.cc", "lambda");

    TF_Status* s = TF_NewStatus();
    int candidate_input_indices[1] = {0};
    int forwarded_input;
    int64_t output_dims[1] = {};
    TF_Tensor* output = TF_ForwardInputOrAllocateOutput(
        /*context=*/ctx, candidate_input_indices,
        /*num_candidate_input_indices=*/1,
        /*output_index=*/0, output_dims, /*output_num_dims=*/0,
        &forwarded_input, /*status=*/s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    EXPECT_EQ(forwarded_input, 0);
    EXPECT_EQ(TF_FLOAT, TF_TensorType(output));
    EXPECT_EQ(0, TF_NumDims(output));
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  TF_KernelBuilder* builder = TF_NewKernelBuilder(op_name, device_name, nullptr,
                                                  my_compute_func, nullptr);

  {
    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(node_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_DeleteStatus(status);
  }

  {
    OpKernelContext::Params p;
    DummyDevice dummy_device(nullptr);
    p.device = &dummy_device;
    AllocatorAttributes alloc_attrs;
    p.output_attr_array = &alloc_attrs;

    Tensor t(123.0f);

    gtl::InlinedVector<TensorValue, 4> inputs;
    // GetFakeKernel requires a NodeDef with two inputs
    inputs.emplace_back(&t);
    inputs.emplace_back();
    p.inputs = &inputs;

    Status status;
    std::unique_ptr<OpKernel> kernel =
        GetFakeKernel(device_name, op_name, node_name, &status);
    TF_EXPECT_OK(status);
    ASSERT_NE(nullptr, kernel.get());

    p.op_kernel = kernel.get();
    OpKernelContext ctx(&p);
    kernel->Compute(&ctx);
    ASSERT_EQ(123, ctx.mutable_output(0)->scalar<float>()());
  }
}

void validate_tensor(TF_Tensor* tensor, int64_t* dims, int64_t num_dims,
                     TF_DataType dtype) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_34(mht_34_v, 1434, "", "./tensorflow/c/kernels_test.cc", "validate_tensor");

  EXPECT_EQ(TF_FLOAT, TF_TensorType(tensor));
  EXPECT_EQ(num_dims, TF_NumDims(tensor));
  for (int i = 0; i < num_dims; ++i) {
    EXPECT_EQ(dims[i], TF_Dim(tensor, i));
  }
}

template <typename T>
void set_tensor_data(TF_Tensor* tensor, T* values, size_t tensor_size_bytes,
                     TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScPSkernels_testDTcc mht_35(mht_35_v, 1447, "", "./tensorflow/c/kernels_test.cc", "set_tensor_data");

  T* data = reinterpret_cast<T*>(TF_TensorData(tensor));
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  OpKernelContext* cc_ctx = reinterpret_cast<OpKernelContext*>(ctx);
  cc_ctx->eigen_gpu_device().memcpyHostToDevice(data, values,
                                                tensor_size_bytes);
#else
  memcpy(data, values, tensor_size_bytes);
#endif
}
}  // namespace tensorflow
