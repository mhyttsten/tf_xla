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
class MHTracer_DTPStensorflowPScorePSkernelsPSfingerprint_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfingerprint_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfingerprint_op_testDTcc() {
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
#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
Status MakeNodeDef(DataType dtype, NodeDef* node_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfingerprint_op_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/fingerprint_op_test.cc", "MakeNodeDef");

  return NodeDefBuilder("fingerprint", "Fingerprint")
      .Input(FakeInput(dtype))
      .Input(FakeInput(DT_STRING))
      .Finalize(node_def);
}

class FingerprintOpTest : public OpsTestBase {
 protected:
  Status MakeFingerprintOp(Tensor* tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfingerprint_op_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/fingerprint_op_test.cc", "MakeFingerprintOp");

    return MakeFingerprintOp(tensor, "farmhash64");
  }

  Status MakeFingerprintOp(Tensor* data, const string& method) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("method: \"" + method + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSfingerprint_op_testDTcc mht_2(mht_2_v, 223, "", "./tensorflow/core/kernels/fingerprint_op_test.cc", "MakeFingerprintOp");

    TF_RETURN_IF_ERROR(MakeNodeDef(data->dtype(), node_def()));
    TF_RETURN_IF_ERROR(InitOp());

    inputs_.clear();
    inputs_.push_back(TensorValue(data));

    method_ = Tensor(DT_STRING, TensorShape{});
    method_.scalar<tstring>()() = method;
    inputs_.push_back(TensorValue(&method_));
    return Status::OK();
  }

  Tensor batch_dims_;
  Tensor method_;
};

TEST_F(FingerprintOpTest, Empty) {
  Tensor tensor(DT_UINT8, {0});

  TF_ASSERT_OK(MakeFingerprintOp(&tensor));
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->shape(), (TensorShape{0, 8}));
  EXPECT_EQ(GetOutput(0)->tensor_data(), "");
}

// This test detects changes in fingerprint method.
TEST_F(FingerprintOpTest, GoldenValue) {
  Tensor tensor(DT_UINT8, {1, 3, 4, 5, 6, 7});
  auto buffer = tensor.flat<uint8>();
  std::iota(buffer.data(), buffer.data() + buffer.size(),
            static_cast<uint8>(47));

  TF_ASSERT_OK(MakeFingerprintOp(&tensor));
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->shape(), (TensorShape{1, 8}));
  EXPECT_EQ(GetOutput(0)->tensor_data(), "\x2d\x90\xdf\x03\x79\x36\x3c\x43");
}

// String types have a different compute path. This test detects changes in this
// special-case handling.
TEST_F(FingerprintOpTest, StringGoldenValue) {
  Tensor data(DT_STRING, {1, 2, 2});
  auto buffer = data.flat<tstring>();
  buffer(0).resize(10);
  buffer(1).resize(7);
  buffer(2).resize(0);
  buffer(3).resize(19);
  std::iota(&buffer(0)[0], &buffer(0)[0] + buffer(0).size(), 0);
  std::iota(&buffer(1)[0], &buffer(1)[0] + buffer(1).size(), 7);
  std::iota(&buffer(2)[0], &buffer(2)[0] + buffer(2).size(), 71);
  std::iota(&buffer(3)[0], &buffer(3)[0] + buffer(3).size(), 41);

  TF_ASSERT_OK(MakeFingerprintOp(&data));
  TF_ASSERT_OK(RunOpKernel());
  ASSERT_EQ(GetOutput(0)->shape(), (TensorShape{1, 8}));
  EXPECT_EQ(GetOutput(0)->tensor_data(), "\x92\x43\x28\x52\xa3\x7c\x48\x18");

  // When each batch item has exactly one string, Fingerprint op avoids
  // double-fingerprint. Adding a test to detect any change in this logic.
  ASSERT_TRUE(data.CopyFrom(data, TensorShape{4}));
  TF_ASSERT_OK(MakeFingerprintOp(&data));
  TF_ASSERT_OK(RunOpKernel());
  ASSERT_EQ(GetOutput(0)->shape(), (TensorShape{4, 8}));
  EXPECT_EQ(GetOutput(0)->tensor_data(),
            "\xea\xff\xd6\xb2\xb2\x4d\x70\x9b"
            "\x6e\x9d\xed\x21\xc6\x4a\x61\x52"
            "\x4f\x40\x90\x2f\x3b\x6a\xe1\x9a"
            "\x0d\x9b\x7f\x63\x23\x14\x1c\xb8");
}

TEST_F(FingerprintOpTest, Collision) {
  const TensorShape shape = {1, 2, 4, 6};
  for (DataType dtype : kRealNumberTypes) {
    const int64_t size = shape.num_elements() * DataTypeSize(dtype);

    Tensor tensor(dtype, shape);
    auto buffer = tensor.bit_casted_shaped<uint8, 1>({size});
    buffer.setRandom();

    TF_ASSERT_OK(MakeFingerprintOp(&tensor));
    TF_ASSERT_OK(RunOpKernel());
    const Tensor fingerprint0 = *GetOutput(0);

    // Alter a byte value in the buffer.
    const int offset = buffer(0) % buffer.size();
    buffer(offset) = ~buffer(offset);

    TF_ASSERT_OK(MakeFingerprintOp(&tensor));
    TF_ASSERT_OK(RunOpKernel());
    const Tensor fingerprint1 = *GetOutput(0);

    EXPECT_NE(fingerprint0.tensor_data(), fingerprint1.tensor_data());
  }
}

TEST_F(FingerprintOpTest, CollisionString) {
  constexpr int64_t size = 256;

  Tensor tensor(DT_STRING, {1});
  auto& input = tensor.vec<tstring>()(0);
  input.resize(size);

  TTypes<uint8>::UnalignedFlat buffer(reinterpret_cast<uint8*>(&input[0]),
                                      input.size());
  buffer.setRandom();

  TF_ASSERT_OK(MakeFingerprintOp(&tensor));
  TF_ASSERT_OK(RunOpKernel());
  const Tensor fingerprint0 = *GetOutput(0);

  // Alter a byte value in the buffer.
  const int offset = buffer(0) % buffer.size();
  buffer(offset) = ~buffer(offset);

  TF_ASSERT_OK(MakeFingerprintOp(&tensor));
  TF_ASSERT_OK(RunOpKernel());
  const Tensor fingerprint1 = *GetOutput(0);

  EXPECT_NE(fingerprint0.tensor_data(), fingerprint1.tensor_data());
}

TEST_F(FingerprintOpTest, CompareBytesAndString) {
  Tensor pods_tensor(DT_FLOAT, {4, 64});
  Tensor strings_tensor(DT_STRING, {4});

  auto pods = pods_tensor.matrix<float>();
  pods.setRandom();

  auto strings = strings_tensor.vec<tstring>();
  for (int64_t i = 0; i < strings.size(); ++i) {
    strings(i).assign(reinterpret_cast<const char*>(&pods(i, 0)),
                      pods.dimension(1) * sizeof(pods(i, 0)));
  }

  TF_ASSERT_OK(MakeFingerprintOp(&pods_tensor));
  TF_ASSERT_OK(RunOpKernel());
  Tensor pods_fingerprints = *GetOutput(0);

  TF_ASSERT_OK(MakeFingerprintOp(&strings_tensor));
  TF_ASSERT_OK(RunOpKernel());
  Tensor strings_fingerprints = *GetOutput(0);

  EXPECT_EQ(pods_fingerprints.tensor_data(),
            strings_fingerprints.tensor_data());
}

TEST_F(FingerprintOpTest, SupportedMethods) {
  Tensor tensor(DT_STRING, TensorShape{1});
  TF_ASSERT_OK(MakeFingerprintOp(&tensor, "unsupported_method"));

  const Status status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.error_message().find("unsupported_method"), string::npos);
}

TEST_F(FingerprintOpTest, SupportedTypes) {
  Tensor input(DT_RESOURCE, TensorShape{1});
  EXPECT_FALSE(MakeFingerprintOp(&input).ok());
}

TEST(FingerprintOpShapeFnTest, MethodKnownStatically) {
  ShapeInferenceTestOp op("Fingerprint");

  Tensor method(DT_STRING, TensorShape{});
  method.scalar<tstring>()() = "farmhash64";
  op.input_tensors.assign({nullptr, &method});

  TF_ASSERT_OK(MakeNodeDef(DT_UINT8, &op.node_def));
  INFER_OK(op, "?;?", "[?,8]");
  INFER_ERROR("must be at least rank 1", op, "[];?");
  INFER_OK(op, "[?];?", "[d0_0,8]");
  INFER_OK(op, "[1,?];?", "[d0_0,8]");
  INFER_OK(op, "[?,2,3];?", "[d0_0,8]");
}

TEST(FingerprintOpShapeFnTest, MethodUnknownStatically) {
  ShapeInferenceTestOp op("Fingerprint");

  TF_ASSERT_OK(MakeNodeDef(DT_FLOAT, &op.node_def));
  INFER_OK(op, "?;?", "[?,?]");
  INFER_ERROR("must be at least rank 1", op, "[];?");
  INFER_OK(op, "[?];?", "[d0_0,?]");
  INFER_OK(op, "[1,?];?", "[d0_0,?]");
  INFER_OK(op, "[?,2,3];?", "[d0_0,?]");
}

TEST(FingerprintOpShapeFnTest, InvalidMethod) {
  ShapeInferenceTestOp op("Fingerprint");

  // When `method` shape is known statically.
  INFER_ERROR("must be rank 0", op, "[1];[1]");

  // When `method` shape is unknown statically.
  Tensor method(DT_STRING, TensorShape{1});
  method.vec<tstring>()(0) = "farmhash64";
  op.input_tensors.assign({nullptr, &method});
  INFER_ERROR("must be rank 0", op, "?;?");

  method = Tensor(DT_STRING, TensorShape{});
  method.scalar<tstring>()() = "unsupported_method";
  op.input_tensors.assign({nullptr, &method});
  INFER_ERROR("unsupported_method", op, "?;?");
}
}  // namespace
}  // namespace tensorflow
