/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_OUTFEED_OPS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_OUTFEED_OPS_H_
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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPSoutfeed_opsDTh {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSoutfeed_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPSoutfeed_opsDTh() {
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


#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tpu/kernels/transfer_ops.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tensorflow {

// The OutfeedDequeue op is used to retrieve a single tensor from the device
// outfeed queue.
template <class T>
class TpuOutfeedDequeueOp : public T {
 public:
  explicit TpuOutfeedDequeueOp(
      OpKernelConstruction* ctx,
      std::unique_ptr<TpuTransferOpInterface> transfer_op)
      : T(ctx, "outfeed_dequeue", 1, std::move(transfer_op)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape_, &xla_shape_));
  }

  Status DoWork(OpKernelContext* ctx, int device_ordinal) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSoutfeed_opsDTh mht_0(mht_0_v, 219, "", "./tensorflow/core/tpu/kernels/outfeed_ops.h", "DoWork");

    Tensor* output;
    TF_RETURN_IF_ERROR(ctx->allocate_output(0, shape_, &output));

    // Transfer from the outfeed interface of the device.
    xla::MutableBorrowingLiteral literal;
    TF_RETURN_IF_ERROR(
        HostTensorToMutableBorrowingLiteral(xla_shape_, output, &literal));

    VLOG(1) << "TransferLiteralFromOutfeed "
            << xla::ShapeUtil::HumanStringWithLayout(xla_shape_);

    TF_RETURN_IF_ERROR(
        T::transfer_op_->TransferLiteralFromOutfeed(device_ordinal, literal));

    VLOG(1) << "TransferLiteralFromOutfeed complete.";

    return Status::OK();
  }

 private:
  TensorShape shape_;
  DataType dtype_;
  xla::Shape xla_shape_;

  // OutfeedDequeueOp is neither copyable nor movable.
  TpuOutfeedDequeueOp(const TpuOutfeedDequeueOp&) = delete;
  TpuOutfeedDequeueOp& operator=(const TpuOutfeedDequeueOp&) = delete;
};

// The OutfeedDequeueTuple op is used to retrieve multiple tensors from the
// device outfeed queue.
template <class T>
class TpuOutfeedDequeueTupleOp : public T {
 public:
  explicit TpuOutfeedDequeueTupleOp(
      OpKernelConstruction* ctx,
      std::unique_ptr<TpuTransferOpInterface> transfer_op)
      : T(ctx, "outfeed_dequeue", 1, std::move(transfer_op)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
    OP_REQUIRES(
        ctx, shapes_.size() == dtypes_.size(),
        errors::InvalidArgument("shapes and dtypes must be the same length."));
    // The `dtypes` list is inferred from the supplied inputs, so it
    // is always the correct length.
    for (int i = 0; i < shapes_.size(); i++) {
      xla::Shape xla_shape;
      OP_REQUIRES_OK(ctx,
                     TensorShapeToXLAShape(dtypes_[i], shapes_[i], &xla_shape));
      xla_shapes_.push_back(xla_shape);
    }
    tuple_shape_ = xla::ShapeUtil::MakeTupleShape(xla_shapes_);
  }

  Status DoWork(OpKernelContext* ctx, int device_ordinal) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSoutfeed_opsDTh mht_1(mht_1_v, 277, "", "./tensorflow/core/tpu/kernels/outfeed_ops.h", "DoWork");

    VLOG(1) << "TransferLiteralFromOutfeed "
            << xla::ShapeUtil::HumanStringWithLayout(tuple_shape_);

    for (int i = 0; i < shapes_.size(); ++i) {
      Tensor* output;
      TF_RETURN_IF_ERROR(ctx->allocate_output(i, shapes_[i], &output));

      xla::MutableBorrowingLiteral literal;
      TF_RETURN_IF_ERROR(HostTensorToMutableBorrowingLiteral(xla_shapes_[i],
                                                             output, &literal));
      TF_RETURN_IF_ERROR(
          T::transfer_op_->TransferLiteralFromOutfeed(device_ordinal, literal));
    }
    return Status::OK();
  }

 private:
  std::vector<TensorShape> shapes_;
  DataTypeVector dtypes_;
  std::vector<xla::Shape> xla_shapes_;
  xla::Shape tuple_shape_;

  // OutfeedDequeueTupleOp is neither copyable nor movable.
  TpuOutfeedDequeueTupleOp(const TpuOutfeedDequeueTupleOp&) = delete;
  TpuOutfeedDequeueTupleOp& operator=(const TpuOutfeedDequeueTupleOp&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_OUTFEED_OPS_H_
