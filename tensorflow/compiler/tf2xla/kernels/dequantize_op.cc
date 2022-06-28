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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdequantize_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdequantize_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdequantize_opDTcc() {
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

#include <array>

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

// TODO(mingyao|ylc): Support 16bits and 32 bits.
constexpr std::array<DataType, 2> kQuantizedType = {{DT_QINT8, DT_QUINT8}};

template <typename T>
float get_fullrange() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdequantize_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/tf2xla/kernels/dequantize_op.cc", "get_fullrange");

  return static_cast<float>(std::numeric_limits<T>::max()) -
         std::numeric_limits<T>::min();
}

class DequantizeOp : public XlaOpKernel {
 public:
  explicit DequantizeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdequantize_opDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/tf2xla/kernels/dequantize_op.cc", "DequantizeOp");

    string mode_string;
    int axis;
    bool narrow_range;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(
        ctx, (mode_string == "MIN_COMBINED"),
        errors::InvalidArgument("Mode string must be 'MIN_COMBINED' is " +
                                mode_string + "'"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range));
    OP_REQUIRES(ctx, narrow_range == false,
                errors::InvalidArgument("narrow_range must be false"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis));
    OP_REQUIRES(ctx, axis == -1,
                errors::InvalidArgument("axis must be -1' is ", axis));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  ~DequantizeOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdequantize_opDTcc mht_2(mht_2_v, 238, "", "./tensorflow/compiler/tf2xla/kernels/dequantize_op.cc", "Compile");

    DataType input_type = ctx->input_type(0);

    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp output = xla::ConvertElementType(input, xla::F32);
    xla::XlaOp min_range = xla::ConvertElementType(ctx->Input(1), xla::F32);
    xla::XlaOp max_range = xla::ConvertElementType(ctx->Input(2), xla::F32);
    xla::XlaOp full_range;
    xla::XlaOp half_range;
    if (input_type == DT_QINT8) {
      full_range = ScalarLike(output, get_fullrange<qint8>());
      half_range =
          (full_range + ScalarLike(output, 1.0f)) / ScalarLike(output, 2.0f);
    } else {
      OP_REQUIRES(ctx, input_type == DT_QUINT8,
                  errors::InvalidArgument(
                      "Only support DT_QINT8 or DT_QUINT8, got ", input_type));
      full_range = ScalarLike(output, get_fullrange<quint8>());
      half_range = ScalarLike(output, 0.0f);
    }

    xla::XlaOp scale = (max_range - min_range) / full_range;

    output = xla::Add(xla::Mul(xla::Add(output, half_range), scale), min_range);

    if (dtype_ == DT_BFLOAT16) {
      output = xla::ConvertElementType(output, xla::BF16);
    }
    ctx->SetOutput(0, output);
  }

 private:
  DataType dtype_;
};

REGISTER_XLA_OP(Name("Dequantize").TypeConstraint("T", kQuantizedType),
                DequantizeOp);

}  // namespace
}  // namespace tensorflow
