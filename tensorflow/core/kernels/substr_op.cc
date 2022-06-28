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
class MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_opDTcc() {
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

#include <cstddef>
#include <cstdlib>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/string_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

// Position/length can be 32 or 64-bit integers
template <typename T>
class SubstrOp : public OpKernel {
 public:
  explicit SubstrOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_opDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/substr_op.cc", "SubstrOp");

    string unit;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unit", &unit));
    OP_REQUIRES_OK(ctx, ParseCharUnit(unit, &unit_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_opDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/substr_op.cc", "Compute");

    // Get inputs
    const Tensor& input_tensor = context->input(0);
    const Tensor& pos_tensor = context->input(1);
    const Tensor& len_tensor = context->input(2);
    const TensorShape& input_shape = input_tensor.shape();
    const TensorShape& pos_shape = pos_tensor.shape();
    const TensorShape& len_shape = len_tensor.shape();
    OP_REQUIRES(context, (pos_shape == len_shape),
                errors::InvalidArgument(
                    "pos and len should have the same shape, got: ",
                    pos_shape.DebugString(), " vs. ", len_shape.DebugString()));

    bool is_scalar = TensorShapeUtils::IsScalar(pos_shape);

    if (is_scalar || input_shape == pos_shape) {
      // pos/len are either scalar or match the shape of input_tensor
      // Do not need to do broadcasting

      // Reshape input
      auto input = input_tensor.flat<tstring>();
      // Allocate output
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output("output", input_tensor.shape(),
                                              &output_tensor));
      auto output = output_tensor->flat<tstring>();
      if (is_scalar) {
        // Perform Op with scalar pos/len
        const T pos =
            tensorflow::internal::SubtleMustCopy(pos_tensor.scalar<T>()());
        const T len =
            tensorflow::internal::SubtleMustCopy(len_tensor.scalar<T>()());
        for (size_t i = 0; i < input_tensor.NumElements(); ++i) {
          StringPiece in(input(i));
          T byte_pos = pos;
          T byte_len = len;
          switch (unit_) {
            case CharUnit::UTF8_CHAR:
              OP_REQUIRES(
                  context, UpdatePosAndLenForUtf8(in, &byte_pos, &byte_len),
                  errors::InvalidArgument("pos ", pos, " out of range for ",
                                          "string at index ", i));
              break;
            case CharUnit::BYTE:
              byte_pos = AdjustedPosIndex(byte_pos, in);
              OP_REQUIRES(
                  context, FastBoundsCheck(byte_pos, in.size() + 1),
                  errors::InvalidArgument("pos ", pos, " out of range for ",
                                          "string b'", in, "' at index ", i));
          }
          StringPiece sub_in = in.substr(byte_pos, byte_len);
          output(i).assign(sub_in.data(), sub_in.size());
        }
      } else {
        // Perform Op element-wise with tensor pos/len
        auto pos_flat = pos_tensor.flat<T>();
        auto len_flat = len_tensor.flat<T>();
        for (size_t i = 0; i < input_tensor.NumElements(); ++i) {
          StringPiece in(input(i));
          const T pos = tensorflow::internal::SubtleMustCopy(pos_flat(i));
          const T len = tensorflow::internal::SubtleMustCopy(len_flat(i));
          T byte_pos = pos;
          T byte_len = len;
          switch (unit_) {
            case CharUnit::UTF8_CHAR:
              OP_REQUIRES(
                  context, UpdatePosAndLenForUtf8(in, &byte_pos, &byte_len),
                  errors::InvalidArgument("pos ", pos, " out of range for ",
                                          "string at index ", i));
              break;
            case CharUnit::BYTE:
              byte_pos = AdjustedPosIndex(byte_pos, in);
              OP_REQUIRES(
                  context, FastBoundsCheck(byte_pos, in.size() + 1),
                  errors::InvalidArgument("pos ", pos, " out of range for ",
                                          "string b'", in, "' at index ", i));
          }
          StringPiece sub_in = in.substr(byte_pos, byte_len);
          output(i).assign(sub_in.data(), sub_in.size());
        }
      }
    } else {
      // Perform op with broadcasting
      // TODO: Use ternary broadcasting for once available in Eigen. Current
      //       implementation iterates through broadcasted ops element-wise;
      //       this should be parallelized.

      // Create BCast helper with shape of input and pos/len
      BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(pos_shape));
      OP_REQUIRES(context, bcast.IsValid(),
                  errors::InvalidArgument(
                      "Incompatible shapes: ", input_shape.DebugString(),
                      " vs. ", pos_shape.DebugString()));
      TensorShape output_shape = BCast::ToShape(bcast.result_shape());
      int ndims = output_shape.dims();
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output("output", output_shape,
                                                       &output_tensor));
      switch (ndims) {
        case 1: {
          // Reshape tensors according to BCast results
          auto input = input_tensor.shaped<tstring, 1>(bcast.x_reshape());
          auto output = output_tensor->shaped<tstring, 1>(bcast.result_shape());
          auto pos_shaped = pos_tensor.shaped<T, 1>(bcast.y_reshape());
          auto len_shaped = len_tensor.shaped<T, 1>(bcast.y_reshape());

          // Allocate temporary buffer for broadcasted position tensor
          Tensor pos_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &pos_buffer));
          typename TTypes<T, 1>::Tensor pos_bcast(
              pos_buffer.shaped<T, 1>(bcast.result_shape()));
          pos_bcast =
              pos_shaped.broadcast(BCast::ToIndexArray<1>(bcast.y_bcast()));

          // Allocate temporary buffer for broadcasted length tensor
          Tensor len_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &len_buffer));
          typename TTypes<T, 1>::Tensor len_bcast(
              len_buffer.shaped<T, 1>(bcast.result_shape()));
          len_bcast =
              len_shaped.broadcast(BCast::ToIndexArray<1>(bcast.y_bcast()));

          // Iterate through broadcasted tensors and perform substr
          for (int i = 0; i < output_shape.dim_size(0); ++i) {
            StringPiece in(input(input.dimension(0) > 1 ? i : 0));
            const T pos = tensorflow::internal::SubtleMustCopy(pos_bcast(i));
            const T len = tensorflow::internal::SubtleMustCopy(len_bcast(i));
            T byte_pos = pos;
            T byte_len = len;
            switch (unit_) {
              case CharUnit::UTF8_CHAR:
                OP_REQUIRES(
                    context, UpdatePosAndLenForUtf8(in, &byte_pos, &byte_len),
                    errors::InvalidArgument("pos ", pos, " out of range for ",
                                            "string at index ", i));
                break;
              case CharUnit::BYTE:
                byte_pos = AdjustedPosIndex(byte_pos, in);
                OP_REQUIRES(
                    context, FastBoundsCheck(byte_pos, in.size() + 1),
                    errors::InvalidArgument("pos ", pos, " out of range for ",
                                            "string b'", in, "' at index ", i));
            }
            StringPiece sub_in = in.substr(byte_pos, byte_len);
            output(i).assign(sub_in.data(), sub_in.size());
          }
          break;
        }
        case 2: {
          // Reshape tensors according to BCast results
          auto input = input_tensor.shaped<tstring, 2>(bcast.x_reshape());
          auto output = output_tensor->shaped<tstring, 2>(bcast.result_shape());
          auto pos_shaped = pos_tensor.shaped<T, 2>(bcast.y_reshape());
          auto len_shaped = len_tensor.shaped<T, 2>(bcast.y_reshape());

          // Allocate temporary buffer for broadcasted position tensor
          Tensor pos_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &pos_buffer));
          typename TTypes<T, 2>::Tensor pos_bcast(
              pos_buffer.shaped<T, 2>(bcast.result_shape()));
          pos_bcast =
              pos_shaped.broadcast(BCast::ToIndexArray<2>(bcast.y_bcast()));

          // Allocate temporary buffer for broadcasted length tensor
          Tensor len_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &len_buffer));
          typename TTypes<T, 2>::Tensor len_bcast(
              len_buffer.shaped<T, 2>(bcast.result_shape()));
          len_bcast =
              len_shaped.broadcast(BCast::ToIndexArray<2>(bcast.y_bcast()));

          // Iterate through broadcasted tensors and perform substr
          for (int i = 0; i < output_shape.dim_size(0); ++i) {
            for (int j = 0; j < output_shape.dim_size(1); ++j) {
              StringPiece in(input(input.dimension(0) > 1 ? i : 0,
                                   input.dimension(1) > 1 ? j : 0));
              const T pos =
                  tensorflow::internal::SubtleMustCopy(pos_bcast(i, j));
              const T len =
                  tensorflow::internal::SubtleMustCopy(len_bcast(i, j));
              T byte_pos = pos;
              T byte_len = len;
              switch (unit_) {
                case CharUnit::UTF8_CHAR:
                  OP_REQUIRES(
                      context, UpdatePosAndLenForUtf8(in, &byte_pos, &byte_len),
                      errors::InvalidArgument("pos ", pos, " out of range for ",
                                              "string at index ", i));
                  break;
                case CharUnit::BYTE:
                  byte_pos = AdjustedPosIndex(byte_pos, in);
                  OP_REQUIRES(
                      context, FastBoundsCheck(byte_pos, in.size() + 1),
                      errors::InvalidArgument("pos ", pos, " out of range for ",
                                              "string b'", in, "' at index (",
                                              i, ", ", j, ")"));
              }
              StringPiece sub_in = in.substr(byte_pos, byte_len);
              output(i, j).assign(sub_in.data(), sub_in.size());
            }
          }
          break;
        }
        default: {
          context->SetStatus(errors::Unimplemented(
              "Substr broadcast not implemented for ", ndims, " dimensions"));
        }
      }
    }
  }

 private:
  // This adjusts the requested position. Note it does not perform any bound
  // checks.
  static inline T AdjustedPosIndex(const T pos_requested, const StringPiece s) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_opDTcc mht_2(mht_2_v, 445, "", "./tensorflow/core/kernels/substr_op.cc", "AdjustedPosIndex");

    if (pos_requested < 0) {
      return s.size() + pos_requested;
    }
    return pos_requested;
  }

  // Return true if successful; otherwise, return false if the `pos` argument
  // is out of range in the string.
  static inline bool UpdatePosAndLenForUtf8(const StringPiece in, T* pos,
                                            T* len) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_opDTcc mht_3(mht_3_v, 458, "", "./tensorflow/core/kernels/substr_op.cc", "UpdatePosAndLenForUtf8");

    if (*pos >= 0) {
      return UpdatePositivePosAndLenForUtf8(in, *pos, *len, pos, len);
    } else {
      return UpdateNegativePosAndLenForUtf8(in, *pos, *len, pos, len);
    }
  }

  static bool UpdatePositivePosAndLenForUtf8(const StringPiece in, const T pos,
                                             const T len, T* char_pos,
                                             T* char_len) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_opDTcc mht_4(mht_4_v, 471, "", "./tensorflow/core/kernels/substr_op.cc", "UpdatePositivePosAndLenForUtf8");

    *char_pos = 0;
    // Determine byte position of the substring start.
    if (!ForwardNUTF8CharPositions(in, pos, char_pos)) {
      return false;
    }
    // Determine position of the end of the substring.
    // The length will be capped at the end of the string, and we ignore whether
    // the string had enough characters to handle it or not.
    *char_len = *char_pos;
    ForwardNUTF8CharPositions(in, len, char_len);
    // The length in bytes is the position end of the substring less the start.
    *char_len = *char_len - *char_pos;
    return true;
  }

  // This function expects a negative position relative to the end of the
  // string, but will update the character position to a positive number
  // relative to the beginning of the string.
  static bool UpdateNegativePosAndLenForUtf8(const StringPiece in, const T pos,
                                             const T len, T* char_pos,
                                             T* char_len) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsubstr_opDTcc mht_5(mht_5_v, 495, "", "./tensorflow/core/kernels/substr_op.cc", "UpdateNegativePosAndLenForUtf8");

    // Initially treat the length as position of the end of the substring.
    *char_len = in.size();
    // This is the number of character to skip from the end of the string to
    // arrive at the position where the substring should end.
    T utf8_chars_to_skip = -pos - len;
    if (utf8_chars_to_skip < 0) {
      utf8_chars_to_skip = 0;
    }
    // Find the byte position where the substring should end using the computed
    // number of characters to skip.
    if (!BackNUTF8CharPositions(in, utf8_chars_to_skip, char_len)) {
      return false;
    }
    // Next, determine where the substring should begin. The number of chars to
    // skip is the requested position minus the chars we've previously skipped.
    *char_pos = *char_len;
    if (!BackNUTF8CharPositions(in, -pos - utf8_chars_to_skip, char_pos)) {
      return false;
    }
    // The length in bytes is the position end of the substring less the start.
    *char_len = *char_len - *char_pos;
    return true;
  }

  CharUnit unit_ = CharUnit::BYTE;
};

#define REGISTER_SUBSTR(type)                                      \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Substr").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SubstrOp<type>);
REGISTER_SUBSTR(int32);
REGISTER_SUBSTR(int64_t);
}  // namespace tensorflow
