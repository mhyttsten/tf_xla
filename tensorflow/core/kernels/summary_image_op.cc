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
class MHTracer_DTPStensorflowPScorePSkernelsPSsummary_image_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_image_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsummary_image_opDTcc() {
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

// Operators that deal with SummaryProtos (encoded as DT_STRING tensors) as
// inputs or outputs in various ways.

// See docs in ../ops/summary_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class SummaryImageOp : public OpKernel {
 public:
  typedef Eigen::Tensor<uint8, 2, Eigen::RowMajor> Uint8Image;

  explicit SummaryImageOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_image_opDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/summary_image_op.cc", "SummaryImageOp");

    int64_t max_images_tmp;
    OP_REQUIRES_OK(context, context->GetAttr("max_images", &max_images_tmp));
    OP_REQUIRES(context, max_images_tmp < (1LL << 31),
                errors::InvalidArgument("max_images must be < 2^31"));
    max_images_ = static_cast<int32>(max_images_tmp);
    const TensorProto* proto;
    OP_REQUIRES_OK(context, context->GetAttr("bad_color", &proto));
    OP_REQUIRES_OK(context, context->device()->MakeTensorFromProto(
                                *proto, AllocatorAttributes(), &bad_color_));
    OP_REQUIRES(context, bad_color_.dtype() == DT_UINT8,
                errors::InvalidArgument("bad_color must be uint8, got ",
                                        DataTypeString(bad_color_.dtype())));
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(bad_color_.shape()),
        errors::InvalidArgument("bad_color must be a vector, got shape ",
                                bad_color_.shape().DebugString()));
  }

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_image_opDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/kernels/summary_image_op.cc", "Compute");

    const Tensor& tags = c->input(0);
    const Tensor& tensor = c->input(1);
    OP_REQUIRES(c, TensorShapeUtils::IsScalar(tags.shape()),
                errors::InvalidArgument("Tags must be a scalar"));
    OP_REQUIRES(c,
                tensor.dims() == 4 &&
                    (tensor.dim_size(3) == 1 || tensor.dim_size(3) == 3 ||
                     tensor.dim_size(3) == 4),
                errors::InvalidArgument(
                    "Tensor must be 4-D with last dim 1, 3, or 4, not ",
                    tensor.shape().DebugString()));
    const string& base_tag = tags.scalar<tstring>()();

    OP_REQUIRES(c,
                tensor.dim_size(0) < (1LL << 31) &&
                    tensor.dim_size(1) < (1LL << 31) &&
                    tensor.dim_size(2) < (1LL << 31) &&
                    (tensor.dim_size(1) * tensor.dim_size(2)) < (1LL << 29),
                errors::InvalidArgument("Tensor too large for summary ",
                                        tensor.shape().DebugString()));

    // The casts and h * w cannot overflow because of the limits above.
    const int batch_size = static_cast<int>(tensor.dim_size(0));
    const int h = static_cast<int>(tensor.dim_size(1));
    const int w = static_cast<int>(tensor.dim_size(2));
    const int hw = h * w;  // Compact these two dims for simplicity
    const int depth = static_cast<int>(tensor.dim_size(3));

    OP_REQUIRES(c, hw > 0 && depth > 0,
                errors::InvalidArgument(
                    "input tensor must have non-zero dims. Found: [",
                    batch_size, ", ", h, ", ", w, ", ", depth, "]."));

    Summary s;
    if (tensor.dtype() == DT_UINT8) {
      // For uint8 input, no normalization is necessary
      auto ith_image = [&tensor, batch_size, hw, depth](int i) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_image_opDTcc mht_2(mht_2_v, 264, "", "./tensorflow/core/kernels/summary_image_op.cc", "lambda");

        auto values = tensor.shaped<uint8, 3>({batch_size, hw, depth});
        return typename TTypes<uint8>::ConstMatrix(
            &values(i, 0, 0), Eigen::DSizes<Eigen::DenseIndex, 2>(hw, depth));
      };
      OP_REQUIRES_OK(
          c, AddImages(base_tag, batch_size, w, h, depth, ith_image, &s));
    } else if (tensor.dtype() == DT_HALF) {
      NormalizeAndAddImages<Eigen::half>(c, tensor, h, w, hw, depth, batch_size,
                                         base_tag, &s);
    } else if (tensor.dtype() == DT_FLOAT) {
      NormalizeAndAddImages<float>(c, tensor, h, w, hw, depth, batch_size,
                                   base_tag, &s);
    } else {  // tensor.dtype() = DT_DOUBLE
      NormalizeAndAddImages<double>(c, tensor, h, w, hw, depth, batch_size,
                                    base_tag, &s);
    }

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }

  template <class T>
  void NormalizeAndAddImages(OpKernelContext* c, const Tensor& tensor, int h,
                             int w, int hw, int depth, int batch_size,
                             const string& base_tag, Summary* s) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("base_tag: \"" + base_tag + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_image_opDTcc mht_3(mht_3_v, 294, "", "./tensorflow/core/kernels/summary_image_op.cc", "NormalizeAndAddImages");

    // For float and half images, nans and infs are replaced with bad_color.
    OP_REQUIRES(c, bad_color_.dim_size(0) >= depth,
                errors::InvalidArgument(
                    "expected depth <= bad_color.size, got depth = ", depth,
                    ", bad_color.size = ", bad_color_.dim_size(0)));
    auto bad_color_full = bad_color_.vec<uint8>();
    typename TTypes<uint8>::ConstVec bad_color(bad_color_full.data(), depth);

    // Float images must be scaled and translated.
    Uint8Image image(hw, depth);
    auto ith_image = [&tensor, &image, bad_color, batch_size, hw,
                      depth](int i) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_image_opDTcc mht_4(mht_4_v, 309, "", "./tensorflow/core/kernels/summary_image_op.cc", "lambda");

      auto tensor_eigen = tensor.template shaped<T, 3>({batch_size, hw, depth});
      typename TTypes<T>::ConstMatrix values(
          &tensor_eigen(i, 0, 0),
          Eigen::DSizes<Eigen::DenseIndex, 2>(hw, depth));
      NormalizeFloatImage<T>(hw, depth, values, bad_color, &image);
      return image;
    };
    OP_REQUIRES_OK(c,
                   AddImages(base_tag, batch_size, w, h, depth, ith_image, s));
  }

  // Add the sequence of images specified by ith_image to the summary.
  //
  // Factoring this loop out into a helper function lets ith_image behave
  // differently in the float and uint8 cases: the float case needs a temporary
  // buffer which can be shared across calls to ith_image, but the uint8 case
  // does not.
  Status AddImages(const string& tag, int batch_size, int w, int h, int depth,
                   const std::function<Uint8Image(int)>& ith_image,
                   Summary* s) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_image_opDTcc mht_5(mht_5_v, 333, "", "./tensorflow/core/kernels/summary_image_op.cc", "AddImages");

    const int N = std::min<int>(max_images_, batch_size);
    for (int i = 0; i < N; ++i) {
      Summary::Value* v = s->add_value();
      // The tag depends on the number of requested images (not the number
      // produced.)
      //
      // Note that later on avisu uses "/" to figure out a consistent naming
      // convention for display, so we append "/image" to guarantee that the
      // image(s) won't be displayed in the global scope with no name.
      if (max_images_ > 1) {
        v->set_tag(strings::StrCat(tag, "/image/", i));
      } else {
        v->set_tag(strings::StrCat(tag, "/image"));
      }

      auto image = ith_image(i);
      Summary::Image* si = v->mutable_image();
      si->set_height(h);
      si->set_width(w);
      si->set_colorspace(depth);
      const int channel_bits = 8;
      const int compression = -1;  // Use zlib default
      if (!png::WriteImageToBuffer(
              image.data(), w, h, w * depth, depth, channel_bits, compression,
              si->mutable_encoded_image_string(), nullptr)) {
        return errors::Internal("PNG encoding failed");
      }
    }
    return Status::OK();
  }

  template <class T>
  static void NormalizeFloatImage(int hw, int depth,
                                  typename TTypes<T>::ConstMatrix values,
                                  typename TTypes<uint8>::ConstVec bad_color,
                                  Uint8Image* image) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_image_opDTcc mht_6(mht_6_v, 372, "", "./tensorflow/core/kernels/summary_image_op.cc", "NormalizeFloatImage");

    if (!image->size()) return;  // Nothing to do for empty images

    // Rescale the image to uint8 range.
    //
    // We are trying to generate an RGB image from a float/half tensor.  We do
    // not have any info about the expected range of values in the tensor
    // but the generated image needs to have all RGB values within [0, 255].
    //
    // We use two different algorithms to generate these values.  If the
    // tensor has only positive values we scale them all by 255/max(values).
    // If the tensor has both negative and positive values we scale them by
    // the max of their absolute values and center them around 127.
    //
    // This works for most cases, but does not respect the relative dynamic
    // range across different instances of the tensor.

    // Compute min and max ignoring nonfinite pixels
    float image_min = std::numeric_limits<float>::infinity();
    float image_max = -image_min;
    for (int i = 0; i < hw; i++) {
      bool finite = true;
      for (int j = 0; j < depth; j++) {
        if (!Eigen::numext::isfinite(values(i, j))) {
          finite = false;
          break;
        }
      }
      if (finite) {
        for (int j = 0; j < depth; j++) {
          float value(values(i, j));
          image_min = std::min(image_min, value);
          image_max = std::max(image_max, value);
        }
      }
    }

    // Pick an affine transform into uint8
    const float kZeroThreshold = 1e-6;
    T scale, offset;
    if (image_min < 0) {
      float max_val = std::max(std::abs(image_min), std::abs(image_max));
      scale = T(max_val < kZeroThreshold ? 0.0f : 127.0f / max_val);
      offset = T(128.0f);
    } else {
      scale = T(image_max < kZeroThreshold ? 0.0f : 255.0f / image_max);
      offset = T(0.0f);
    }

    // Transform image, turning nonfinite values to bad_color
    for (int i = 0; i < hw; i++) {
      bool finite = true;
      for (int j = 0; j < depth; j++) {
        if (!Eigen::numext::isfinite(values(i, j))) {
          finite = false;
          break;
        }
      }
      if (finite) {
        image->chip<0>(i) = (values.template chip<0>(i) * scale + offset)
                                .template cast<uint8>();
      } else {
        image->chip<0>(i) = bad_color;
      }
    }
  }

 private:
  int32 max_images_;
  Tensor bad_color_;
};

REGISTER_KERNEL_BUILDER(Name("ImageSummary").Device(DEVICE_CPU),
                        SummaryImageOp);

}  // namespace tensorflow
