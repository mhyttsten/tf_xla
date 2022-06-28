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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_sequence_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_sequence_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_sequence_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class ReverseSequenceOp : public XlaOpKernel {
 public:
  explicit ReverseSequenceOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_sequence_opDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/tf2xla/kernels/reverse_sequence_op.cc", "ReverseSequenceOp");

    OP_REQUIRES_OK(context, context->GetAttr("batch_dim", &batch_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("seq_dim", &seq_dim_));
  }

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_sequence_opDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/tf2xla/kernels/reverse_sequence_op.cc", "Compile");

    const TensorShape input_shape = context->InputShape(0);
    const TensorShape seq_lens_shape = context->InputShape(1);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(seq_lens_shape),
                errors::InvalidArgument("seq_lengths must be 1-dim, not ",
                                        seq_lens_shape.dims()));
    OP_REQUIRES(context, batch_dim_ != seq_dim_,
                errors::InvalidArgument("batch_dim == seq_dim == ", seq_dim_));
    OP_REQUIRES(
        context, seq_dim_ < input_shape.dims(),
        errors::InvalidArgument("seq_dim must be < input rank", " ( ", seq_dim_,
                                " vs. ", input_shape.dims(), ")"));
    OP_REQUIRES(
        context, batch_dim_ < input_shape.dims(),
        errors::InvalidArgument("batch_dim must be < input rank", " ( ",
                                batch_dim_, " vs. ", input_shape.dims(), ")"));
    OP_REQUIRES(
        context,
        seq_lens_shape.num_elements() == input_shape.dim_size(batch_dim_),
        errors::InvalidArgument("Length of seq_lengths != input.dims(",
                                batch_dim_, "), ", "(",
                                seq_lens_shape.num_elements(), " vs. ",
                                input_shape.dim_size(batch_dim_), ")"));

    xla::XlaBuilder* builder = context->builder();
    const auto input = context->Input(0);
    const auto seq_lens = context->Input(1);

    const int64_t batch_size = input_shape.dim_size(batch_dim_);
    if (batch_size == 0) {
      context->SetOutput(0, input);
      return;
    }

    const xla::PrimitiveType seq_lens_type = context->input_xla_type(1);
    const int64_t max_seq_len = input_shape.dim_size(seq_dim_);

    // Create [batch, sequence, 2] tensor that contains the indices where the
    // real data belongs
    xla::XlaOp back = xla::Sub(seq_lens, xla::ScalarLike(seq_lens, 1));
    xla::XlaOp batch_idx = xla::Iota(
        builder,
        xla::ShapeUtil::MakeShape(seq_lens_type, {batch_size, max_seq_len, 1}),
        /*iota_dimension=*/0);
    xla::XlaOp forward_idx = xla::Iota(
        builder,
        xla::ShapeUtil::MakeShape(seq_lens_type, {batch_size, max_seq_len, 1}),
        /*iota_dimension=*/1);
    xla::XlaOp reverse_idx = xla::Sub(back, forward_idx, {0});
    reverse_idx = xla::Select(xla::Lt(reverse_idx, xla::ZerosLike(reverse_idx)),
                              forward_idx, reverse_idx);
    if (batch_dim_ > seq_dim_) {
      // The output of the XLA gather op keeps indices dimensions in the same
      // order as they appear in the input. If the batch_dim_ needs to be after
      // the seq_dim_ in the output, it also needs to be that way in the input
      // so we transpose.
      batch_idx = xla::Transpose(batch_idx, {1, 0, 2});
      forward_idx = xla::Transpose(forward_idx, {1, 0, 2});
      reverse_idx = xla::Transpose(reverse_idx, {1, 0, 2});
    }
    xla::XlaOp start_indices =
        xla::ConcatInDim(builder, {batch_idx, reverse_idx},
                         /*dimension=*/2);

    xla::GatherDimensionNumbers dnums;
    dnums.set_index_vector_dim(2);
    // The first and second element in the third dimension of reverse_idx are
    // the batch_dim_ offset and the seq_dim_ offset respectively.
    dnums.add_start_index_map(batch_dim_);
    dnums.add_start_index_map(seq_dim_);

    // batch_dim_ and seq_dim_ are collapsed and the other dimensions are kept
    // in the gather.
    for (int i = 0; i < input_shape.dims(); ++i) {
      if (i != batch_dim_ && i != seq_dim_) {
        dnums.add_offset_dims(i);
      } else {
        dnums.add_collapsed_slice_dims(i);
      }
    }

    auto slice_sizes = input_shape.dim_sizes();
    slice_sizes[batch_dim_] = 1;
    slice_sizes[seq_dim_] = 1;

    context->SetOutput(0,
                       xla::Gather(input, start_indices, dnums, slice_sizes));
  }

 private:
  int32 batch_dim_;
  int32 seq_dim_;
};

REGISTER_XLA_OP(Name("ReverseSequence"), ReverseSequenceOp);

}  // namespace
}  // namespace tensorflow
