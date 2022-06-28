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
class MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

namespace {

static int* GetCopyCPUToGPUCounter() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "GetCopyCPUToGPUCounter");

  static int* counter = new int(0);
  return counter;
}

static int* GetCopyGPUToCPUCounter() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "GetCopyGPUToCPUCounter");

  static int* counter = new int(0);
  return counter;
}

static int* GetCopyGPUToGPUCounter() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "GetCopyGPUToGPUCounter");

  static int* counter = new int(0);
  return counter;
}

struct StoredTensorValue {
  Tensor stored;
  string TypeName() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "TypeName");
 return "StoredTensorValue"; }
  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "Encode");
 data->tensors_ = {stored}; }
  bool Decode(const VariantTensorData& data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_5(mht_5_v, 248, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "Decode");

    CHECK_EQ(1, data.tensors_.size());
    stored = data.tensors_[0];
    return true;
  }
  static Status CopyCPUToGPU(
      const StoredTensorValue& from, StoredTensorValue* to,
      const std::function<Status(const Tensor&, Tensor*)>& copy) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_6(mht_6_v, 258, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "CopyCPUToGPU");

    ++*GetCopyCPUToGPUCounter();
    return copy(from.stored, &(to->stored));
  }
  static Status CopyGPUToCPU(
      const StoredTensorValue& from, StoredTensorValue* to,
      const std::function<Status(const Tensor&, Tensor*)>& copy) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_7(mht_7_v, 267, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "CopyGPUToCPU");

    ++*GetCopyGPUToCPUCounter();
    return copy(from.stored, &(to->stored));
  }
  static Status CopyGPUToGPU(
      const StoredTensorValue& from, StoredTensorValue* to,
      const std::function<Status(const Tensor&, Tensor*)>& copy) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_8(mht_8_v, 276, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "CopyGPUToGPU");

    ++*GetCopyGPUToGPUCounter();
    return copy(from.stored, &(to->stored));
  }
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(StoredTensorValue, "StoredTensorValue");

INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(
    StoredTensorValue, VariantDeviceCopyDirection::HOST_TO_DEVICE,
    StoredTensorValue::CopyCPUToGPU);

INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(
    StoredTensorValue, VariantDeviceCopyDirection::DEVICE_TO_HOST,
    StoredTensorValue::CopyGPUToCPU);

INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(
    StoredTensorValue, VariantDeviceCopyDirection::DEVICE_TO_DEVICE,
    StoredTensorValue::CopyGPUToGPU);

REGISTER_OP("CreateTestVariant")
    .Input("input: T")
    .Attr("T: type")
    .Output("output: variant")
    .SetShapeFn(shape_inference::UnknownShape);

class CreateTestVariantOp : public OpKernel {
 public:
  explicit CreateTestVariantOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_9(mht_9_v, 307, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "CreateTestVariantOp");
}
  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_10(mht_10_v, 311, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "Compute");

    // Take the scalar tensor fed as input, and emit a Tensor
    // containing 10 Variants (StoredTensorValues), both containing
    // the input tensor.
    const Tensor& stored_t = c->input(0);
    Tensor* out;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({10}), &out));
    StoredTensorValue store{stored_t};
    auto t = out->flat<Variant>();
    for (int i = 0; i < 10; ++i) {
      t(i) = store;
    }
    CHECK_EQ("StoredTensorValue", t(0).TypeName());
  }
};

REGISTER_KERNEL_BUILDER(Name("CreateTestVariant").Device(DEVICE_CPU),
                        CreateTestVariantOp);

class CreateTestVariant {
 public:
  explicit CreateTestVariant(const ::tensorflow::Scope& scope,
                             const Input& value) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_11(mht_11_v, 336, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "CreateTestVariant");

    if (!scope.ok()) return;
    auto _value = ops::AsNodeOut(scope, value);
    if (!scope.ok()) return;
    ::tensorflow::Node* ret;
    const auto unique_name = scope.GetUniqueNameForOp("CreateTestVariant");
    auto builder = ::tensorflow::NodeBuilder(unique_name, "CreateTestVariant")
                       .Input(_value);
    scope.UpdateBuilder(&builder);
    scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
    if (!scope.ok()) return;
    scope.UpdateStatus(scope.DoShapeInference(ret));
    if (!scope.ok()) return;
    this->output_ = Output(ret, 0);
  }

  // Intentionally not marked as explicit.
  // NOLINTNEXTLINE google-explicit-constructor
  operator ::tensorflow::Output() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_12(mht_12_v, 357, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "::tensorflow::Output");
 return output_; }
  // Intentionally not marked as explicit.
  // NOLINTNEXTLINE google-explicit-constructor
  operator ::tensorflow::Input() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_13(mht_13_v, 363, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "::tensorflow::Input");
 return output_; }

  ::tensorflow::Node* node() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_copy_testDTcc mht_14(mht_14_v, 368, "", "./tensorflow/core/framework/variant_op_copy_test.cc", "node");
 return output_.node(); }

  ::tensorflow::Output output_;
};

}  // end namespace

TEST(VariantOpCopyTest, CreateConstOnCPU) {
  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");

  // Create the input StoredTensorValue and serialize it.
  StoredTensorValue from;
  from.stored = Tensor(DT_INT64, TensorShape({}));
  from.stored.scalar<int64_t>()() = 0xdeadbeef;
  VariantTensorData data;
  data.set_type_name(from.TypeName());
  from.Encode(&data);

  TensorProto variant_proto;
  variant_proto.set_dtype(DT_VARIANT);
  TensorShape scalar_shape({});
  scalar_shape.AsProto(variant_proto.mutable_tensor_shape());
  data.ToProto(variant_proto.add_variant_val());

  Output create_const = ops::ConstFromProto(root, variant_proto);
  TF_ASSERT_OK(root.status());
  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_CHECK_OK(session.Run({create_const}, &outputs));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(DT_VARIANT, outputs[0].dtype());
  EXPECT_EQ(0, outputs[0].dims());
  const Variant& variant = outputs[0].scalar<Variant>()();
  EXPECT_EQ("StoredTensorValue", variant.TypeName());
  const StoredTensorValue* to = variant.get<StoredTensorValue>();
  EXPECT_EQ(to->stored.dtype(), DT_INT64);
  EXPECT_EQ(0xdeadbeef, to->stored.scalar<int64_t>()());
}

TEST(VariantOpCopyTest, CreateConstOnGPU) {
  if (!IsGoogleCudaEnabled()) return;

  Scope root = Scope::NewRootScope().WithDevice("/gpu:0");

  // Create the input StoredTensorValue and serialize it.
  StoredTensorValue from;
  from.stored = Tensor(DT_INT64, TensorShape({}));
  from.stored.scalar<int64_t>()() = 0xdeadbeef;
  VariantTensorData data;
  data.set_type_name(from.TypeName());
  from.Encode(&data);

  TensorProto variant_proto;
  variant_proto.set_dtype(DT_VARIANT);
  TensorShape scalar_shape({});
  scalar_shape.AsProto(variant_proto.mutable_tensor_shape());
  data.ToProto(variant_proto.add_variant_val());

  Output create_const = ops::ConstFromProto(root, variant_proto);
  TF_ASSERT_OK(root.status());
  ClientSession session(root);
  std::vector<Tensor> outputs;

  int copy_to_gpu_before = *GetCopyCPUToGPUCounter();
  int copy_to_cpu_before = *GetCopyGPUToCPUCounter();
  TF_CHECK_OK(session.Run({create_const}, &outputs));
  int copy_to_cpu_after = *GetCopyGPUToCPUCounter();
  int copy_to_gpu_after = *GetCopyCPUToGPUCounter();

  EXPECT_GT(copy_to_cpu_after - copy_to_cpu_before, 0);
  EXPECT_GT(copy_to_gpu_after - copy_to_gpu_before, 0);

  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(DT_VARIANT, outputs[0].dtype());
  EXPECT_EQ(0, outputs[0].dims());
  const Variant& variant = outputs[0].scalar<Variant>()();
  EXPECT_EQ("StoredTensorValue", variant.TypeName());
  const StoredTensorValue* to = variant.get<StoredTensorValue>();
  EXPECT_EQ(to->stored.dtype(), DT_INT64);
  EXPECT_EQ(0xdeadbeef, to->stored.scalar<int64_t>()());
}

TEST(VariantOpCopyTest, CreateConstOnGPUFailsGracefully) {
  if (!IsGoogleCudaEnabled()) return;

  Scope root = Scope::NewRootScope().WithDevice("/gpu:0");

  // Create the input StoredTensorValue and serialize it.
  StoredTensorValue from;
  from.stored = Tensor(DT_STRING, TensorShape({}));
  from.stored.scalar<tstring>()() = "hi";
  VariantTensorData data;
  data.set_type_name(from.TypeName());
  from.Encode(&data);

  TensorProto variant_proto;
  variant_proto.set_dtype(DT_VARIANT);
  TensorShape scalar_shape({});
  scalar_shape.AsProto(variant_proto.mutable_tensor_shape());
  data.ToProto(variant_proto.add_variant_val());

  Output create_const = ops::ConstFromProto(root, variant_proto);
  TF_ASSERT_OK(root.status());
  ClientSession session(root);
  std::vector<Tensor> outputs;
  Status s = session.Run({create_const}, &outputs);
  EXPECT_TRUE(absl::StrContains(s.error_message(),
                                "GPU copy from non-DMA string tensor"))
      << s.ToString();
}

TEST(VariantOpCopyTest, CreateCopyCPUToCPU) {
  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");
  Tensor t_42(DT_INT32, TensorShape({}));
  t_42.flat<int32>()(0) = 42;
  Output create_op = CreateTestVariant(root, t_42);
  Output identity = ops::Identity(root, create_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_CHECK_OK(session.Run({create_op, identity}, &outputs));
  EXPECT_EQ(2, outputs.size());
  EXPECT_EQ(10, outputs[1].dim_size(0));
  auto output = outputs[1].flat<Variant>();
  for (int i = 0; i < 10; ++i) {
    const Variant& r1 = output(i);
    EXPECT_EQ("StoredTensorValue", r1.TypeName());
    const StoredTensorValue* v1 = r1.get<StoredTensorValue>();
    EXPECT_NE(v1, nullptr);
    EXPECT_EQ(42, v1->stored.scalar<int32>()());
  }
}

TEST(VariantOpCopyTest, CreateCopyCPUToCPUString) {
  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");
  Tensor t_str(DT_STRING, TensorShape({}));
  t_str.scalar<tstring>()() = "hi";
  Output create_op = CreateTestVariant(root, t_str);
  Output identity = ops::Identity(root, create_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_CHECK_OK(session.Run({create_op, identity}, &outputs));
  EXPECT_EQ(2, outputs.size());
  EXPECT_EQ(10, outputs[1].dim_size(0));
  auto output = outputs[1].flat<Variant>();
  for (int i = 0; i < 10; ++i) {
    const Variant& r1 = output(i);
    EXPECT_EQ("StoredTensorValue", r1.TypeName());
    const StoredTensorValue* v1 = r1.get<StoredTensorValue>();
    EXPECT_NE(v1, nullptr);
    EXPECT_EQ("hi", v1->stored.scalar<tstring>()());
  }
}

TEST(VariantOpCopyTest, CreateCopyCPUToGPU) {
  if (!IsGoogleCudaEnabled()) return;

  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");
  Scope with_gpu = root.WithDevice("/gpu:0");
  Tensor t_42(DT_INT32, TensorShape({}));
  t_42.scalar<int32>()() = 42;
  Output create_op = CreateTestVariant(root, t_42);
  Output identity = ops::Identity(with_gpu, create_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  int copy_to_gpu_before = *GetCopyCPUToGPUCounter();
  int copy_to_cpu_before = *GetCopyGPUToCPUCounter();
  // Force the identity to run on GPU, and then the data to be copied
  // back to CPU for the final output.
  TF_CHECK_OK(session.Run({create_op, identity}, &outputs));
  int copy_to_cpu_after = *GetCopyGPUToCPUCounter();
  int copy_to_gpu_after = *GetCopyCPUToGPUCounter();

  EXPECT_GT(copy_to_cpu_after - copy_to_cpu_before, 0);
  EXPECT_GT(copy_to_gpu_after - copy_to_gpu_before, 0);

  EXPECT_EQ(2, outputs.size());
  EXPECT_EQ(10, outputs[1].dim_size(0));
  auto output = outputs[1].flat<Variant>();
  for (int i = 0; i < 10; ++i) {
    const Variant& r1 = output(i);
    EXPECT_EQ("StoredTensorValue", r1.TypeName());
    const StoredTensorValue* v1 = r1.get<StoredTensorValue>();
    EXPECT_NE(v1, nullptr);
    EXPECT_EQ(42, v1->stored.scalar<int32>()());
  }
}

TEST(VariantOpCopyTest, CreateCopyCPUToGPUStringFailsSafely) {
  if (!IsGoogleCudaEnabled()) return;

  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");
  Scope with_gpu = root.WithDevice("/gpu:0");
  Tensor t_str(DT_STRING, TensorShape({}));
  t_str.scalar<tstring>()() = "hi";
  Output create_op = CreateTestVariant(root, t_str);
  Output identity = ops::Identity(with_gpu, create_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  Status err = session.Run({create_op, identity}, &outputs);
  EXPECT_EQ(err.code(), errors::Code::INVALID_ARGUMENT);
  EXPECT_TRUE(
      absl::StrContains(err.error_message(),
                        "During Variant Host->Device Copy: non-DMA-copy "
                        "attempted of tensor type: string"))
      << err.error_message();
}

// TODO(ebrevdo): Identify a way to create two virtual GPUs within a
// single session, so that we can test the Device <-> Device copy
// branch.

}  // end namespace tensorflow
