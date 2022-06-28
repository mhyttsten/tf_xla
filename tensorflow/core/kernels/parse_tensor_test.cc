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
class MHTracer_DTPStensorflowPScorePSkernelsPSparse_tensor_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSparse_tensor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSparse_tensor_testDTcc() {
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
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

class SerializeTensorOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp(const TensorShape& input_shape, std::function<T(int)> functor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSparse_tensor_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/parse_tensor_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "SerializeTensor")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    AddInput<T>(input_shape, functor);
  }
  void ParseSerializedWithNodeDef(const NodeDef& parse_node_def,
                                  Tensor* serialized, Tensor* parse_output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSparse_tensor_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/parse_tensor_test.cc", "ParseSerializedWithNodeDef");

    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));
    gtl::InlinedVector<TensorValue, 4> inputs;
    inputs.push_back({nullptr, serialized});
    Status status;
    std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                                cpu_allocator(), parse_node_def,
                                                TF_GRAPH_DEF_VERSION, &status));
    TF_EXPECT_OK(status);
    OpKernelContext::Params params;
    params.device = device.get();
    params.inputs = &inputs;
    params.frame_iter = FrameAndIter(0, 0);
    params.op_kernel = op.get();
    std::vector<AllocatorAttributes> attrs;
    test::SetOutputAttrs(&params, &attrs);
    OpKernelContext ctx(&params);
    op->Compute(&ctx);
    TF_EXPECT_OK(status);
    *parse_output = *ctx.mutable_output(0);
  }
  template <typename T>
  void ParseSerializedOutput(Tensor* serialized, Tensor* parse_output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSparse_tensor_testDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/kernels/parse_tensor_test.cc", "ParseSerializedOutput");

    NodeDef parse;
    TF_ASSERT_OK(NodeDefBuilder("parse", "ParseTensor")
                     .Input(FakeInput(DT_STRING))
                     .Attr("out_type", DataTypeToEnum<T>::value)
                     .Finalize(&parse));
    ParseSerializedWithNodeDef(parse, serialized, parse_output);
  }
};

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_half) {
  MakeOp<Eigen::half>(TensorShape({10}), [](int x) -> Eigen::half {
    return static_cast<Eigen::half>(x / 10.);
  });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<Eigen::half>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<Eigen::half>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_float) {
  MakeOp<float>(TensorShape({1, 10}),
                [](int x) -> float { return static_cast<float>(x / 10.); });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<float>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<float>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_double) {
  MakeOp<double>(TensorShape({5, 5}),
                 [](int x) -> double { return static_cast<double>(x / 10.); });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<double>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<double>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_int64) {
  MakeOp<int64_t>(TensorShape({2, 3, 4}),
                  [](int x) -> int64 { return static_cast<int64_t>(x - 10); });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<int64_t>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<int64_t>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_int32) {
  MakeOp<int32>(TensorShape({4, 2}),
                [](int x) -> int32 { return static_cast<int32>(x + 7); });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<int32>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<int32>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_int16) {
  MakeOp<int16>(TensorShape({8}),
                [](int x) -> int16 { return static_cast<int16>(x + 18); });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<int16>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<int16>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_int8) {
  MakeOp<int8>(TensorShape({2}),
               [](int x) -> int8 { return static_cast<int8>(x + 8); });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<int8>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<int8>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_uint16) {
  MakeOp<uint16>(TensorShape({1, 3}),
                 [](int x) -> uint16 { return static_cast<uint16>(x + 2); });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<uint16>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<uint16>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_uint8) {
  MakeOp<uint8>(TensorShape({2, 1, 1}),
                [](int x) -> uint8 { return static_cast<uint8>(x + 1); });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<uint8>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<uint8>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_complex64) {
  MakeOp<complex64>(TensorShape({}), [](int x) -> complex64 {
    return complex64{static_cast<float>(x / 8.), static_cast<float>(x / 2.)};
  });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<complex64>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<complex64>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_complex128) {
  MakeOp<complex128>(TensorShape({3}), [](int x) -> complex128 {
    return complex128{x / 3., x / 2.};
  });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<complex128>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<complex128>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_bool) {
  MakeOp<bool>(TensorShape({1}),
               [](int x) -> bool { return static_cast<bool>(x % 2); });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<bool>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<bool>(parse_output, GetInput(0));
}

TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_string) {
  MakeOp<tstring>(TensorShape({10}),
                  [](int x) -> tstring { return std::to_string(x / 10.); });
  TF_ASSERT_OK(RunOpKernel());
  Tensor parse_output;
  ParseSerializedOutput<tstring>(GetOutput(0), &parse_output);
  test::ExpectTensorEqual<tstring>(parse_output, GetInput(0));
}

}  // namespace
}  // namespace tensorflow
