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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSencode_jpeg_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSencode_jpeg_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSencode_jpeg_opDTcc() {
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

// See docs in ../ops/image_ops.cc

#include <memory>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Encode an image to a JPEG stream
class EncodeJpegOp : public OpKernel {
 public:
  explicit EncodeJpegOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSencode_jpeg_opDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/image/encode_jpeg_op.cc", "EncodeJpegOp");

    OP_REQUIRES_OK(context, context->GetAttr("format", &format_));
    if (format_.empty()) {
      flags_.format = static_cast<jpeg::Format>(0);
    } else if (format_ == "grayscale") {
      flags_.format = jpeg::FORMAT_GRAYSCALE;
    } else if (format_ == "rgb") {
      flags_.format = jpeg::FORMAT_RGB;
    } else {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument(
                      "format must be '', grayscale or rgb, got ", format_));
    }

    OP_REQUIRES_OK(context, context->GetAttr("quality", &flags_.quality));
    OP_REQUIRES(context, 0 <= flags_.quality && flags_.quality <= 100,
                errors::InvalidArgument("quality must be in [0,100], got ",
                                        flags_.quality));
    OP_REQUIRES_OK(context,
                   context->GetAttr("progressive", &flags_.progressive));
    OP_REQUIRES_OK(
        context, context->GetAttr("optimize_size", &flags_.optimize_jpeg_size));
    OP_REQUIRES_OK(context, context->GetAttr("chroma_downsampling",
                                             &flags_.chroma_downsampling));

    string density_unit;
    OP_REQUIRES_OK(context, context->GetAttr("density_unit", &density_unit));
    if (density_unit == "in") {
      flags_.density_unit = 1;
    } else if (density_unit == "cm") {
      flags_.density_unit = 2;
    } else {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("density_unit must be 'in' or 'cm'",
                                          density_unit));
    }

    OP_REQUIRES_OK(context, context->GetAttr("x_density", &flags_.x_density));
    OP_REQUIRES_OK(context, context->GetAttr("y_density", &flags_.y_density));
    OP_REQUIRES_OK(context, context->GetAttr("xmp_metadata", &xmp_metadata_));
    flags_.xmp_metadata = xmp_metadata_;  // StringPiece doesn't own data
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSencode_jpeg_opDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/kernels/image/encode_jpeg_op.cc", "Compute");

    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 3,
                errors::InvalidArgument("image must be 3-dimensional",
                                        image.shape().DebugString()));

    OP_REQUIRES(
        context,
        FastBoundsCheck(image.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument(
            "Cannot encode images with >= max int32 elements"));

    const int32_t dim_size0 = static_cast<int32>(image.dim_size(0));
    const int32_t dim_size1 = static_cast<int32>(image.dim_size(1));
    const int32_t dim_size2 = static_cast<int32>(image.dim_size(2));

    // Autodetect format if desired, otherwise make sure format and
    // image channels are consistent.
    int channels;
    jpeg::CompressFlags adjusted_flags = flags_;
    if (flags_.format == 0) {
      channels = dim_size2;
      if (channels == 1) {
        adjusted_flags.format = jpeg::FORMAT_GRAYSCALE;
      } else if (channels == 3) {
        adjusted_flags.format = jpeg::FORMAT_RGB;
      } else {
        OP_REQUIRES(
            context, false,
            errors::InvalidArgument("image must have 1 or 3 channels, got ",
                                    image.shape().DebugString()));
      }
    } else {
      if (flags_.format == jpeg::FORMAT_GRAYSCALE) {
        channels = 1;
      } else {  // RGB
        channels = 3;
      }
      OP_REQUIRES(context, channels == dim_size2,
                  errors::InvalidArgument("format ", format_, " expects ",
                                          channels, " channels, got ",
                                          image.shape().DebugString()));
    }

    // Encode image to jpeg string
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES(context,
                jpeg::Compress(image.flat<uint8>().data(), dim_size1, dim_size0,
                               adjusted_flags, &output->scalar<tstring>()()),
                errors::Internal("JPEG encoding failed"));
  }

 private:
  string format_;
  string xmp_metadata_;  // Owns data referenced by flags_
  jpeg::CompressFlags flags_;
};
REGISTER_KERNEL_BUILDER(Name("EncodeJpeg").Device(DEVICE_CPU), EncodeJpegOp);

class EncodeJpegVariableQualityOp : public OpKernel {
 public:
  explicit EncodeJpegVariableQualityOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSencode_jpeg_opDTcc mht_2(mht_2_v, 316, "", "./tensorflow/core/kernels/image/encode_jpeg_op.cc", "EncodeJpegVariableQualityOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSencode_jpeg_opDTcc mht_3(mht_3_v, 321, "", "./tensorflow/core/kernels/image/encode_jpeg_op.cc", "Compute");

    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 3,
                errors::InvalidArgument("image must be 3-dimensional",
                                        image.shape().DebugString()));

    OP_REQUIRES(
        context,
        FastBoundsCheck(image.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument(
            "Cannot encode images with >= max int32 elements"));

    const int32_t dim_size0 = static_cast<int32>(image.dim_size(0));
    const int32_t dim_size1 = static_cast<int32>(image.dim_size(1));
    const int32_t dim_size2 = static_cast<int32>(image.dim_size(2));

    // Use default jpeg compression flags except for format and quality.
    jpeg::CompressFlags adjusted_flags;

    // Get jpeg encoding quality.
    const Tensor& quality = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(quality.shape()),
                errors::InvalidArgument("quality must be scalar: ",
                                        quality.shape().DebugString()));
    adjusted_flags.quality = quality.scalar<int>()();
    OP_REQUIRES(context,
                0 <= adjusted_flags.quality && adjusted_flags.quality <= 100,
                errors::InvalidArgument("quality must be in [0,100], got ",
                                        adjusted_flags.quality));

    // Autodetect format.
    int channels;
    channels = dim_size2;
    if (channels == 1) {
      adjusted_flags.format = jpeg::FORMAT_GRAYSCALE;
    } else if (channels == 3) {
      adjusted_flags.format = jpeg::FORMAT_RGB;
    } else {
      OP_REQUIRES(
          context, false,
          errors::InvalidArgument("image must have 1 or 3 channels, got ",
                                  image.shape().DebugString()));
    }

    // Encode image to jpeg string
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES(context,
                jpeg::Compress(image.flat<uint8>().data(), dim_size1, dim_size0,
                               adjusted_flags, &output->scalar<tstring>()()),
                errors::Internal("JPEG encoding failed"));
  }
};
REGISTER_KERNEL_BUILDER(Name("EncodeJpegVariableQuality").Device(DEVICE_CPU),
                        EncodeJpegVariableQualityOp);

}  // namespace tensorflow
