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
class MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registry_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registry_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registry_testDTcc() {
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

#include <memory>
#include "tensorflow/core/lib/strings/str_util.h"

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/variant_op_registry.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {

struct VariantValue {
  string TypeName() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registry_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/framework/variant_op_registry_test.cc", "TypeName");
 return "TEST VariantValue"; }
  static Status CPUZerosLikeFn(OpKernelContext* ctx, const VariantValue& v,
                               VariantValue* v_out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registry_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/framework/variant_op_registry_test.cc", "CPUZerosLikeFn");

    if (v.early_exit) {
      return errors::InvalidArgument("early exit zeros_like!");
    }
    v_out->value = 1;  // CPU
    return Status::OK();
  }
  static Status GPUZerosLikeFn(OpKernelContext* ctx, const VariantValue& v,
                               VariantValue* v_out) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registry_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/framework/variant_op_registry_test.cc", "GPUZerosLikeFn");

    if (v.early_exit) {
      return errors::InvalidArgument("early exit zeros_like!");
    }
    v_out->value = 2;  // GPU
    return Status::OK();
  }
  static Status CPUAddFn(OpKernelContext* ctx, const VariantValue& a,
                         const VariantValue& b, VariantValue* out) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registry_testDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/framework/variant_op_registry_test.cc", "CPUAddFn");

    if (a.early_exit) {
      return errors::InvalidArgument("early exit add!");
    }
    out->value = a.value + b.value;  // CPU
    return Status::OK();
  }
  static Status GPUAddFn(OpKernelContext* ctx, const VariantValue& a,
                         const VariantValue& b, VariantValue* out) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registry_testDTcc mht_4(mht_4_v, 248, "", "./tensorflow/core/framework/variant_op_registry_test.cc", "GPUAddFn");

    if (a.early_exit) {
      return errors::InvalidArgument("early exit add!");
    }
    out->value = -(a.value + b.value);  // GPU
    return Status::OK();
  }
  static Status CPUToGPUCopyFn(
      const VariantValue& from, VariantValue* to,
      const std::function<Status(const Tensor&, Tensor*)>& copier) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registry_testDTcc mht_5(mht_5_v, 260, "", "./tensorflow/core/framework/variant_op_registry_test.cc", "CPUToGPUCopyFn");

    TF_RETURN_IF_ERROR(copier(Tensor(), nullptr));
    to->value = 0xdeadbeef;
    return Status::OK();
  }
  bool early_exit;
  int value;
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(VariantValue, "TEST VariantValue");

INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(
    VariantValue, VariantDeviceCopyDirection::HOST_TO_DEVICE,
    VariantValue::CPUToGPUCopyFn);

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP,
                                         DEVICE_CPU, VariantValue,
                                         VariantValue::CPUZerosLikeFn);

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP,
                                         DEVICE_GPU, VariantValue,
                                         VariantValue::GPUZerosLikeFn);

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_CPU,
                                          VariantValue, VariantValue::CPUAddFn);

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_GPU,
                                          VariantValue, VariantValue::GPUAddFn);

}  // namespace

TEST(VariantOpDecodeRegistryTest, TestBasic) {
  EXPECT_EQ(UnaryVariantOpRegistry::Global()->GetDecodeFn("YOU SHALL NOT PASS"),
            nullptr);

  auto* decode_fn =
      UnaryVariantOpRegistry::Global()->GetDecodeFn("TEST VariantValue");
  EXPECT_NE(decode_fn, nullptr);

  VariantValue vv{true /* early_exit */};
  Variant v = vv;
  VariantTensorData data;
  v.Encode(&data);
  VariantTensorDataProto proto;
  data.ToProto(&proto);
  Variant encoded = std::move(proto);
  EXPECT_TRUE((*decode_fn)(&encoded));
  VariantValue* decoded = encoded.get<VariantValue>();
  EXPECT_NE(decoded, nullptr);
  EXPECT_EQ(decoded->early_exit, true);
}

TEST(VariantOpDecodeRegistryTest, TestEmpty) {
  VariantTensorDataProto empty_proto;
  Variant empty_encoded = std::move(empty_proto);
  EXPECT_TRUE(DecodeUnaryVariant(&empty_encoded));
  EXPECT_TRUE(empty_encoded.is_empty());

  VariantTensorData data;
  Variant number = 3.0f;
  number.Encode(&data);
  VariantTensorDataProto proto;
  data.ToProto(&proto);
  proto.set_type_name("");
  Variant encoded = std::move(proto);
  // Failure when type name is empty but there's data in the proto.
  EXPECT_FALSE(DecodeUnaryVariant(&encoded));
}

TEST(VariantOpDecodeRegistryTest, TestDuplicate) {
  UnaryVariantOpRegistry registry;
  UnaryVariantOpRegistry::VariantDecodeFn f;
  string kTypeName = "fjfjfj";
  registry.RegisterDecodeFn(kTypeName, f);
  EXPECT_DEATH(registry.RegisterDecodeFn(kTypeName, f),
               "fjfjfj already registered");
}

TEST(VariantOpCopyToGPURegistryTest, TestBasic) {
  // No registered copy fn for GPU<->GPU.
  EXPECT_EQ(UnaryVariantOpRegistry::Global()->GetDeviceCopyFn(
                VariantDeviceCopyDirection::DEVICE_TO_DEVICE,
                TypeIndex::Make<VariantValue>()),
            nullptr);

  auto* copy_to_gpu_fn = UnaryVariantOpRegistry::Global()->GetDeviceCopyFn(
      VariantDeviceCopyDirection::HOST_TO_DEVICE,
      TypeIndex::Make<VariantValue>());
  EXPECT_NE(copy_to_gpu_fn, nullptr);

  VariantValue vv{true /* early_exit */};
  Variant v = vv;
  Variant v_out;
  bool dummy_executed = false;
  auto dummy_copy_fn = [&dummy_executed](const Tensor& from,
                                         Tensor* to) -> Status {
    dummy_executed = true;
    return Status::OK();
  };
  TF_EXPECT_OK((*copy_to_gpu_fn)(v, &v_out, dummy_copy_fn));
  EXPECT_TRUE(dummy_executed);
  VariantValue* copied_value = v_out.get<VariantValue>();
  EXPECT_NE(copied_value, nullptr);
  EXPECT_EQ(copied_value->value, 0xdeadbeef);
}

TEST(VariantOpCopyToGPURegistryTest, TestDuplicate) {
  UnaryVariantOpRegistry registry;
  UnaryVariantOpRegistry::AsyncVariantDeviceCopyFn f;
  class FjFjFj {};
  const auto kTypeIndex = TypeIndex::Make<FjFjFj>();
  registry.RegisterDeviceCopyFn(VariantDeviceCopyDirection::HOST_TO_DEVICE,
                                kTypeIndex, f);
  EXPECT_DEATH(registry.RegisterDeviceCopyFn(
                   VariantDeviceCopyDirection::HOST_TO_DEVICE, kTypeIndex, f),
               "FjFjFj already registered");
}

TEST(VariantOpZerosLikeRegistryTest, TestBasicCPU) {
  class Blah {};
  EXPECT_EQ(
      UnaryVariantOpRegistry::Global()->GetUnaryOpFn(
          ZEROS_LIKE_VARIANT_UNARY_OP, DEVICE_CPU, TypeIndex::Make<Blah>()),
      nullptr);

  VariantValue vv_early_exit{true /* early_exit */, 0 /* value */};
  Variant v = vv_early_exit;
  Variant v_out = VariantValue();

  OpKernelContext* null_context_pointer = nullptr;
  Status s0 = UnaryOpVariant<CPUDevice>(null_context_pointer,
                                        ZEROS_LIKE_VARIANT_UNARY_OP, v, &v_out);
  EXPECT_FALSE(s0.ok());
  EXPECT_TRUE(absl::StrContains(s0.error_message(), "early exit zeros_like"));

  VariantValue vv_ok{false /* early_exit */, 0 /* value */};
  v = vv_ok;
  TF_EXPECT_OK(UnaryOpVariant<CPUDevice>(
      null_context_pointer, ZEROS_LIKE_VARIANT_UNARY_OP, v, &v_out));
  VariantValue* vv_out = CHECK_NOTNULL(v_out.get<VariantValue>());
  EXPECT_EQ(vv_out->value, 1);  // CPU
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TEST(VariantOpUnaryOpRegistryTest, TestBasicGPU) {
  class Blah {};
  EXPECT_EQ(
      UnaryVariantOpRegistry::Global()->GetUnaryOpFn(
          ZEROS_LIKE_VARIANT_UNARY_OP, DEVICE_GPU, TypeIndex::Make<Blah>()),
      nullptr);

  VariantValue vv_early_exit{true /* early_exit */, 0 /* value */};
  Variant v = vv_early_exit;
  Variant v_out = VariantValue();

  OpKernelContext* null_context_pointer = nullptr;
  Status s0 = UnaryOpVariant<GPUDevice>(null_context_pointer,
                                        ZEROS_LIKE_VARIANT_UNARY_OP, v, &v_out);
  EXPECT_FALSE(s0.ok());
  EXPECT_TRUE(absl::StrContains(s0.error_message(), "early exit zeros_like"));

  VariantValue vv_ok{false /* early_exit */, 0 /* value */};
  v = vv_ok;
  TF_EXPECT_OK(UnaryOpVariant<GPUDevice>(
      null_context_pointer, ZEROS_LIKE_VARIANT_UNARY_OP, v, &v_out));
  VariantValue* vv_out = CHECK_NOTNULL(v_out.get<VariantValue>());
  EXPECT_EQ(vv_out->value, 2);  // GPU
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TEST(VariantOpUnaryOpRegistryTest, TestDuplicate) {
  UnaryVariantOpRegistry registry;
  UnaryVariantOpRegistry::VariantUnaryOpFn f;
  class FjFjFj {};
  const auto kTypeIndex = TypeIndex::Make<FjFjFj>();

  registry.RegisterUnaryOpFn(ZEROS_LIKE_VARIANT_UNARY_OP, DEVICE_CPU,
                             kTypeIndex, f);
  EXPECT_DEATH(registry.RegisterUnaryOpFn(ZEROS_LIKE_VARIANT_UNARY_OP,
                                          DEVICE_CPU, kTypeIndex, f),
               "FjFjFj already registered");

  registry.RegisterUnaryOpFn(ZEROS_LIKE_VARIANT_UNARY_OP, DEVICE_GPU,
                             kTypeIndex, f);
  EXPECT_DEATH(registry.RegisterUnaryOpFn(ZEROS_LIKE_VARIANT_UNARY_OP,
                                          DEVICE_GPU, kTypeIndex, f),
               "FjFjFj already registered");
}

TEST(VariantOpAddRegistryTest, TestBasicCPU) {
  class Blah {};
  EXPECT_EQ(UnaryVariantOpRegistry::Global()->GetBinaryOpFn(
                ADD_VARIANT_BINARY_OP, DEVICE_CPU, TypeIndex::Make<Blah>()),
            nullptr);

  VariantValue vv_early_exit{true /* early_exit */, 3 /* value */};
  VariantValue vv_other{true /* early_exit */, 4 /* value */};
  Variant v_a = vv_early_exit;
  Variant v_b = vv_other;
  Variant v_out = VariantValue();

  OpKernelContext* null_context_pointer = nullptr;
  Status s0 = BinaryOpVariants<CPUDevice>(
      null_context_pointer, ADD_VARIANT_BINARY_OP, v_a, v_b, &v_out);
  EXPECT_FALSE(s0.ok());
  EXPECT_TRUE(absl::StrContains(s0.error_message(), "early exit add"));

  VariantValue vv_ok{false /* early_exit */, 3 /* value */};
  v_a = vv_ok;
  TF_EXPECT_OK(BinaryOpVariants<CPUDevice>(
      null_context_pointer, ADD_VARIANT_BINARY_OP, v_a, v_b, &v_out));
  VariantValue* vv_out = CHECK_NOTNULL(v_out.get<VariantValue>());
  EXPECT_EQ(vv_out->value, 7);  // CPU
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TEST(VariantOpAddRegistryTest, TestBasicGPU) {
  class Blah {};
  EXPECT_EQ(UnaryVariantOpRegistry::Global()->GetBinaryOpFn(
                ADD_VARIANT_BINARY_OP, DEVICE_GPU, TypeIndex::Make<Blah>()),
            nullptr);

  VariantValue vv_early_exit{true /* early_exit */, 3 /* value */};
  VariantValue vv_other{true /* early_exit */, 4 /* value */};
  Variant v_a = vv_early_exit;
  Variant v_b = vv_other;
  Variant v_out = VariantValue();

  OpKernelContext* null_context_pointer = nullptr;
  Status s0 = BinaryOpVariants<GPUDevice>(
      null_context_pointer, ADD_VARIANT_BINARY_OP, v_a, v_b, &v_out);
  EXPECT_FALSE(s0.ok());
  EXPECT_TRUE(absl::StrContains(s0.error_message(), "early exit add"));

  VariantValue vv_ok{false /* early_exit */, 3 /* value */};
  v_a = vv_ok;
  TF_EXPECT_OK(BinaryOpVariants<GPUDevice>(
      null_context_pointer, ADD_VARIANT_BINARY_OP, v_a, v_b, &v_out));
  VariantValue* vv_out = CHECK_NOTNULL(v_out.get<VariantValue>());
  EXPECT_EQ(vv_out->value, -7);  // GPU
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TEST(VariantOpAddRegistryTest, TestDuplicate) {
  UnaryVariantOpRegistry registry;
  UnaryVariantOpRegistry::VariantBinaryOpFn f;
  class FjFjFj {};
  const auto kTypeIndex = TypeIndex::Make<FjFjFj>();

  registry.RegisterBinaryOpFn(ADD_VARIANT_BINARY_OP, DEVICE_CPU, kTypeIndex, f);
  EXPECT_DEATH(registry.RegisterBinaryOpFn(ADD_VARIANT_BINARY_OP, DEVICE_CPU,
                                           kTypeIndex, f),
               "FjFjFj already registered");

  registry.RegisterBinaryOpFn(ADD_VARIANT_BINARY_OP, DEVICE_GPU, kTypeIndex, f);
  EXPECT_DEATH(registry.RegisterBinaryOpFn(ADD_VARIANT_BINARY_OP, DEVICE_GPU,
                                           kTypeIndex, f),
               "FjFjFj already registered");
}

}  // namespace tensorflow
