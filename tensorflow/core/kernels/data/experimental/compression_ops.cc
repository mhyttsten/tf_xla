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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompression_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompression_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompression_opsDTcc() {
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

#include "tensorflow/core/kernels/data/experimental/compression_ops.h"

#include "tensorflow/core/data/compression_utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace experimental {

CompressElementOp::CompressElementOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompression_opsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/data/experimental/compression_ops.cc", "CompressElementOp::CompressElementOp");
}

void CompressElementOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompression_opsDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/kernels/data/experimental/compression_ops.cc", "CompressElementOp::Compute");

  std::vector<Tensor> components;
  for (size_t i = 0; i < ctx->num_inputs(); ++i) {
    components.push_back(ctx->input(i));
  }
  CompressedElement compressed;
  OP_REQUIRES_OK(ctx, CompressElement(components, &compressed));

  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
  output->scalar<Variant>()() = std::move(compressed);
}

UncompressElementOp::UncompressElementOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompression_opsDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/kernels/data/experimental/compression_ops.cc", "UncompressElementOp::UncompressElementOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void UncompressElementOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompression_opsDTcc mht_3(mht_3_v, 227, "", "./tensorflow/core/kernels/data/experimental/compression_ops.cc", "UncompressElementOp::Compute");

  Tensor tensor = ctx->input(0);
  OP_REQUIRES(
      ctx, tensor.dims() == 0,
      errors::InvalidArgument("UncompressElement requires its input to be a "
                              "scalar, but encountered an input with ",
                              tensor.dims(), " dimensions."));
  OP_REQUIRES(
      ctx, tensor.dtype() == DT_VARIANT,
      errors::InvalidArgument("UncompressElement requires its input to be a "
                              "variant, but encountered an input with dtype ",
                              DataTypeString(tensor.dtype())));
  const Variant& variant = tensor.scalar<Variant>()();
  const CompressedElement* compressed = variant.get<CompressedElement>();
  OP_REQUIRES(
      ctx, compressed != nullptr,
      errors::InvalidArgument(
          "Input does not contain a compressed element. Instead got tensor ",
          tensor.DebugString()));

  std::vector<Tensor> components;
  OP_REQUIRES_OK(ctx, UncompressElement(*compressed, &components));
  OP_REQUIRES(ctx, components.size() == output_types_.size(),
              errors::FailedPrecondition("Expected ", output_types_.size(),
                                         " outputs from uncompress, but got ",
                                         components.size()));
  for (int i = 0; i < components.size(); ++i) {
    OP_REQUIRES(
        ctx, components[i].dtype() == output_types_[i],
        errors::FailedPrecondition("Expected a tensor of type ",
                                   DataTypeString(output_types_[i]),
                                   " but got a tensor of type ",
                                   DataTypeString(components[i].dtype())));
    ctx->set_output(i, components[i]);
  }
}

REGISTER_KERNEL_BUILDER(Name("CompressElement").Device(DEVICE_CPU),
                        CompressElementOp);
REGISTER_KERNEL_BUILDER(Name("UncompressElement").Device(DEVICE_CPU),
                        UncompressElementOp);

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
