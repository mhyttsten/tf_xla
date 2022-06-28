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
class MHTracer_DTPStensorflowPScorePSkernelsPSdecode_raw_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_raw_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdecode_raw_opDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/parse_ops.cc.

#include <algorithm>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/byte_order.h"

namespace tensorflow {

template <typename T>
class DecodeRawOp : public OpKernel {
 public:
  explicit DecodeRawOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_raw_opDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/decode_raw_op.cc", "DecodeRawOp");

    OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));

    const bool host_is_little_endian = port::kLittleEndian;
    bool data_is_little_endian;
    OP_REQUIRES_OK(context,
                   context->GetAttr("little_endian", &data_is_little_endian));
    convert_data_endianness_ = host_is_little_endian != data_is_little_endian;
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_raw_opDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/decode_raw_op.cc", "Compute");

    const auto& input = context->input(0);
    int64_t str_size = -1;
    auto flat_in = input.flat<tstring>();
    for (int64_t i = 0; i < flat_in.size(); ++i) {
      const tstring& in_str = flat_in(i);
      if (str_size == -1) {
        str_size = in_str.size();
      } else {
        OP_REQUIRES(context, str_size == in_str.size(),
                    errors::InvalidArgument(
                        "DecodeRaw requires input strings to all be the same "
                        "size, but element ",
                        i, " has size ", str_size, " != ", in_str.size()));
      }
    }
    TensorShape out_shape = input.shape();
    if (str_size == -1 || str_size == 0) {  // Empty input
      out_shape.AddDim(0);
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output("output", out_shape,
                                                       &output_tensor));
      return;
    }
    OP_REQUIRES(
        context, str_size % sizeof(T) == 0,
        errors::InvalidArgument("Input to DecodeRaw has length ", str_size,
                                " that is not a multiple of ", sizeof(T),
                                ", the size of ", DataTypeString(out_type_)));
    const int64_t added_dim = str_size / sizeof(T);
    out_shape.AddDim(added_dim);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("output", out_shape, &output_tensor));
    auto out = output_tensor->flat_inner_dims<T>();
    DCHECK_EQ(flat_in.size(), out.dimensions()[0]);
    T* out_data = out.data();

    // If the data is already in the host's byte order, or if the width of the
    // output type is a single byte, we can copy the memory directly.
    if (!convert_data_endianness_ || sizeof(T) == 1) {
      for (int64_t i = 0; i < flat_in.size(); ++i) {
        const T* in_data = reinterpret_cast<const T*>(flat_in(i).data());
        memcpy(out_data, in_data, str_size);
        out_data += added_dim;
      }
    } else {
      // Otherwise, the data is not in the host's byte order, and rather than a
      // direct copy, we need to reverse the byte ordering of each element.
      int64_t element_size;
      if (out_type_ == DT_COMPLEX64 || out_type_ == DT_COMPLEX128) {
        // For Complex data type, real and imaginary parts need to be swapped
        // separately
        element_size = sizeof(T) / 2;
      } else {
        element_size = sizeof(T);
      }
      for (int64_t i = 0; i < flat_in.size(); ++i) {
        const char* in_data_bytes =
            reinterpret_cast<const char*>(flat_in(i).data());
        char* out_data_bytes = reinterpret_cast<char*>(out_data);
        const char* p = in_data_bytes;
        char* q = out_data_bytes;
        for (; p < in_data_bytes + str_size;
             p += element_size, q += element_size) {
          std::reverse_copy(p, p + element_size, q);
        }
        out_data += added_dim;
      }
    }
  }

 private:
  // True if the endianness of the data and the endianness of the host are
  // different, and the data needs conversion.
  bool convert_data_endianness_;

  // True if the input data is in little endian format.
  bool data_is_little_endian_;
  DataType out_type_;
};

#define REGISTER(type)                                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("DecodeRaw").Device(DEVICE_CPU).TypeConstraint<type>("out_type"), \
      DecodeRawOp<type>)

REGISTER(Eigen::half);
REGISTER(float);
REGISTER(double);
REGISTER(int32);
REGISTER(uint16);
REGISTER(uint8);
REGISTER(int16);
REGISTER(int8);
REGISTER(int64_t);
REGISTER(bool);
REGISTER(complex64);
REGISTER(complex128);
REGISTER(bfloat16);

#undef REGISTER

}  // namespace tensorflow
