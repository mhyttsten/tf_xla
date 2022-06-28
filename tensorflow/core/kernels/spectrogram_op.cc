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
class MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_opDTcc() {
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

// See docs in ../ops/audio_ops.cc

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/spectrogram.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Create a spectrogram frequency visualization from audio data.
class AudioSpectrogramOp : public OpKernel {
 public:
  explicit AudioSpectrogramOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_opDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/spectrogram_op.cc", "AudioSpectrogramOp");

    OP_REQUIRES_OK(context, context->GetAttr("window_size", &window_size_));
    OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("magnitude_squared", &magnitude_squared_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_opDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/kernels/spectrogram_op.cc", "Compute");

    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 2,
                errors::InvalidArgument("input must be 2-dimensional",
                                        input.shape().DebugString()));
    Spectrogram spectrogram;
    OP_REQUIRES(context, spectrogram.Initialize(window_size_, stride_),
                errors::InvalidArgument(
                    "Spectrogram initialization failed for window size ",
                    window_size_, " and stride ", stride_));

    const auto input_as_matrix = input.matrix<float>();

    const int64_t sample_count = input.dim_size(0);
    const int64_t channel_count = input.dim_size(1);

    const int64_t output_width = spectrogram.output_frequency_channels();
    const int64_t length_minus_window = (sample_count - window_size_);
    int64_t output_height;
    if (length_minus_window < 0) {
      output_height = 0;
    } else {
      output_height = 1 + (length_minus_window / stride_);
    }
    const int64_t output_slices = channel_count;

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({output_slices, output_height, output_width}),
            &output_tensor));
    auto output_flat = output_tensor->flat<float>().data();

    std::vector<float> input_for_channel(sample_count);
    for (int64_t channel = 0; channel < channel_count; ++channel) {
      OP_REQUIRES(context, spectrogram.Reset(),
                  errors::InvalidArgument("Failed to Reset()"));

      float* output_slice =
          output_flat + (channel * output_height * output_width);
      for (int i = 0; i < sample_count; ++i) {
        input_for_channel[i] = input_as_matrix(i, channel);
      }
      std::vector<std::vector<float>> spectrogram_output;
      OP_REQUIRES(context,
                  spectrogram.ComputeSquaredMagnitudeSpectrogram(
                      input_for_channel, &spectrogram_output),
                  errors::InvalidArgument("Spectrogram compute failed"));
      OP_REQUIRES(context, (spectrogram_output.size() == output_height),
                  errors::InvalidArgument(
                      "Spectrogram size calculation failed: Expected height ",
                      output_height, " but got ", spectrogram_output.size()));
      OP_REQUIRES(context,
                  spectrogram_output.empty() ||
                      (spectrogram_output[0].size() == output_width),
                  errors::InvalidArgument(
                      "Spectrogram size calculation failed: Expected width ",
                      output_width, " but got ", spectrogram_output[0].size()));
      for (int row_index = 0; row_index < output_height; ++row_index) {
        const std::vector<float>& spectrogram_row =
            spectrogram_output[row_index];
        DCHECK_EQ(spectrogram_row.size(), output_width);
        float* output_row = output_slice + (row_index * output_width);
        if (magnitude_squared_) {
          for (int i = 0; i < output_width; ++i) {
            output_row[i] = spectrogram_row[i];
          }
        } else {
          for (int i = 0; i < output_width; ++i) {
            output_row[i] = sqrtf(spectrogram_row[i]);
          }
        }
      }
    }
  }

 private:
  int32 window_size_;
  int32 stride_;
  bool magnitude_squared_;
};
REGISTER_KERNEL_BUILDER(Name("AudioSpectrogram").Device(DEVICE_CPU),
                        AudioSpectrogramOp);

}  // namespace tensorflow
