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
class MHTracer_DTPStensorflowPScorePSkernelsPSas_string_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSas_string_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSas_string_opDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/string_ops.cc.

#include <string>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {

class AsStringOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit AsStringOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSas_string_opDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/as_string_op.cc", "AsStringOp");

    int32_t precision;
    bool scientific;
    bool shortest;
    int32_t width;
    string fill_string;
    DataType dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("precision", &precision));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scientific", &scientific));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shortest", &shortest));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("width", &width));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill", &fill_string));
    switch (dtype) {
      case DT_HALF:
      case DT_BFLOAT16:
      case DT_FLOAT:
      case DT_DOUBLE:
      case DT_COMPLEX64:
      case DT_COMPLEX128:
        break;
      default:
        OP_REQUIRES(ctx, !(scientific || shortest),
                    errors::InvalidArgument("scientific and shortest format "
                                            "not supported for datatype ",
                                            DataTypeString(dtype)));
        OP_REQUIRES(ctx, precision < 0,
                    errors::InvalidArgument("precision not supported "
                                            "for datatype ",
                                            DataTypeString(dtype)));
    }
    OP_REQUIRES(
        ctx, fill_string.size() <= 1,
        errors::InvalidArgument("Fill string must be one or fewer characters"));
    OP_REQUIRES(ctx, !(scientific && shortest),
                errors::InvalidArgument(
                    "Cannot select both scientific and shortest notation"));

    format_ = "%";
    if (!fill_string.empty()) {
      switch (fill_string[0]) {
        case ' ':
        case '+':
        case '-':
        case '0':
        case '#':
          strings::Appendf(&format_, "%s", fill_string.c_str());
          break;
        default:
          bool fill_not_supported = true;
          OP_REQUIRES(ctx, !fill_not_supported,
                      errors::InvalidArgument("Fill argument not supported: \"",
                                              fill_string, "\""));
      }
    }
    if (width > -1) {
      strings::Appendf(&format_, "%d", width);
    }
    if (precision > -1) {
      strings::Appendf(&format_, ".%d", precision);
    }
    switch (dtype) {
      case DT_UINT8:
      case DT_UINT16:
      case DT_UINT32:
        strings::Appendf(&format_, "u");
        break;
      case DT_UINT64:
        strings::Appendf(&format_, "llu");
        break;
      case DT_INT8:
      case DT_INT16:
      case DT_INT32:
        strings::Appendf(&format_, "d");
        break;
      case DT_INT64:
        strings::Appendf(&format_, "lld");
        break;
      case DT_HALF:
      case DT_BFLOAT16:
      case DT_FLOAT:
      case DT_DOUBLE:
      case DT_COMPLEX64:
      case DT_COMPLEX128:
        if (shortest) {
          strings::Appendf(&format_, "g");
        } else if (scientific) {
          strings::Appendf(&format_, "e");
        } else {
          strings::Appendf(&format_, "f");
        }
        break;
      case DT_BOOL:
        break;
      case DT_VARIANT:
        break;
      default:
        bool type_not_supported = true;
        OP_REQUIRES(ctx, !type_not_supported,
                    errors::InvalidArgument("Type not supported: ",
                                            DataTypeString(dtype)));
    }

    if (dtype == DT_COMPLEX64 || dtype == DT_COMPLEX128) {
      format_ = strings::Printf("(%s,%s)", format_.c_str(), format_.c_str());
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSas_string_opDTcc mht_1(mht_1_v, 316, "", "./tensorflow/core/kernels/as_string_op.cc", "Compute");

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const DataType& dtype = input_tensor->dtype();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

#define ENCODE_TYPE(type, T, enc_str)                                     \
  case (type): {                                                          \
    const auto& input_flat = input_tensor->flat<T>();                     \
    for (int i = 0; i < input_flat.size(); ++i) {                         \
      output_flat(i) = strings::Printf((enc_str.c_str()), input_flat(i)); \
    }                                                                     \
  } break

    switch (dtype) {
      ENCODE_TYPE(DT_UINT8, uint8, format_);
      ENCODE_TYPE(DT_UINT16, uint16, format_);
      ENCODE_TYPE(DT_UINT32, uint32, format_);
      ENCODE_TYPE(DT_UINT64, uint64, format_);
      ENCODE_TYPE(DT_INT8, int8, format_);
      ENCODE_TYPE(DT_INT16, int16, format_);
      ENCODE_TYPE(DT_INT32, int32, format_);
      ENCODE_TYPE(DT_INT64, int64_t, format_);
      ENCODE_TYPE(DT_FLOAT, float, format_);
      ENCODE_TYPE(DT_DOUBLE, double, format_);
      case (DT_BOOL): {
        const auto& input_flat = input_tensor->flat<bool>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) = (input_flat(i)) ? "true" : "false";
        }
      } break;
      case (DT_VARIANT): {
        const auto& input_flat = input_tensor->flat<Variant>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) = input_flat(i).DebugString();
        }
      } break;
      case (DT_HALF): {
        const auto& input_flat = input_tensor->flat<Eigen::half>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) = strings::Printf(format_.c_str(),
                                           static_cast<float>(input_flat(i)));
        }
      } break;
      case (DT_BFLOAT16): {
        const auto& input_flat = input_tensor->flat<bfloat16>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) = strings::Printf(format_.c_str(),
                                           static_cast<float>(input_flat(i)));
        }
      } break;
      case (DT_COMPLEX64): {
        const auto& input_flat = input_tensor->flat<complex64>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) = strings::Printf(
              format_.c_str(), input_flat(i).real(), input_flat(i).imag());
        }
      } break;
      case (DT_COMPLEX128): {
        const auto& input_flat = input_tensor->flat<complex128>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) = strings::Printf(
              format_.c_str(), input_flat(i).real(), input_flat(i).imag());
        }
      } break;
      default:
        bool can_encode_type = false;
        OP_REQUIRES(context, can_encode_type,
                    errors::InvalidArgument("Cannot encode input of type ",
                                            DataTypeString(dtype)));
    }

#undef ENCODE_TYPE
  }

 private:
  string format_;
};

REGISTER_KERNEL_BUILDER(Name("AsString").Device(DEVICE_CPU), AsStringOp);

}  // namespace tensorflow
