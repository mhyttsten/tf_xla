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
class MHTracer_DTPStensorflowPScorePSopsPSspectral_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSspectral_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSspectral_opsDTcc() {
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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("FFT")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

REGISTER_OP("IFFT")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

REGISTER_OP("FFT2D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    });

REGISTER_OP("IFFT2D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    });

REGISTER_OP("FFT3D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

REGISTER_OP("IFFT3D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

Status RFFTShape(InferenceContext* c, const bool forward, const int rank) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSspectral_opsDTcc mht_0(mht_0_v, 244, "", "./tensorflow/core/ops/spectral_ops.cc", "RFFTShape");

  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), rank, &out));

  // Check that fft_length has shape [rank].
  ShapeHandle unused_shape;
  DimensionHandle unused_dim;
  ShapeHandle fft_length_input = c->input(1);
  TF_RETURN_IF_ERROR(c->WithRank(fft_length_input, 1, &unused_shape));
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(fft_length_input, 0), rank, &unused_dim));
  const Tensor* fft_length_tensor = c->input_tensor(1);

  // If fft_length is unknown at graph creation time, we can't predict the
  // output size.
  if (fft_length_tensor == nullptr) {
    // We can't know the dimension of any of the rank inner dimensions of the
    // output without knowing fft_length.
    for (int i = 0; i < rank; ++i) {
      TF_RETURN_IF_ERROR(c->ReplaceDim(out, -rank + i, c->UnknownDim(), &out));
    }
  } else {
    auto fft_length_as_vec = fft_length_tensor->vec<int32>();
    for (int i = 0; i < rank; ++i) {
      // For RFFT, replace the last dimension with fft_length/2 + 1.
      auto dim = forward && i == rank - 1 && fft_length_as_vec(i) != 0
                     ? fft_length_as_vec(i) / 2 + 1
                     : fft_length_as_vec(i);
      TF_RETURN_IF_ERROR(c->ReplaceDim(out, -rank + i, c->MakeDim(dim), &out));
    }
  }

  c->set_output(0, out);
  return Status::OK();
}

REGISTER_OP("RFFT")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 1); });

REGISTER_OP("IRFFT")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 1); });

REGISTER_OP("RFFT2D")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 2); });

REGISTER_OP("IRFFT2D")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 2); });

REGISTER_OP("RFFT3D")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 3); });

REGISTER_OP("IRFFT3D")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 3); });

// Deprecated ops:
REGISTER_OP("BatchFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use FFT");
REGISTER_OP("BatchIFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use IFFT");
REGISTER_OP("BatchFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use FFT2D");
REGISTER_OP("BatchIFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use IFFT2D");
REGISTER_OP("BatchFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use FFT3D");
REGISTER_OP("BatchIFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use IFFT3D");

}  // namespace tensorflow
