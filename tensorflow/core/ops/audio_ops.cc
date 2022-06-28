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
class MHTracer_DTPStensorflowPScorePSopsPSaudio_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSaudio_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSaudio_opsDTcc() {
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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/bits.h"

namespace tensorflow {

namespace {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status DecodeWavShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSaudio_opsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/ops/audio_ops.cc", "DecodeWavShapeFn");

  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

  DimensionHandle channels_dim;
  int32_t desired_channels;
  TF_RETURN_IF_ERROR(c->GetAttr("desired_channels", &desired_channels));
  if (desired_channels == -1) {
    channels_dim = c->UnknownDim();
  } else {
    if (desired_channels < 0) {
      return errors::InvalidArgument("channels must be non-negative, got ",
                                     desired_channels);
    }
    channels_dim = c->MakeDim(desired_channels);
  }
  DimensionHandle samples_dim;
  int32_t desired_samples;
  TF_RETURN_IF_ERROR(c->GetAttr("desired_samples", &desired_samples));
  if (desired_samples == -1) {
    samples_dim = c->UnknownDim();
  } else {
    if (desired_samples < 0) {
      return errors::InvalidArgument("samples must be non-negative, got ",
                                     desired_samples);
    }
    samples_dim = c->MakeDim(desired_samples);
  }
  c->set_output(0, c->MakeShape({samples_dim, channels_dim}));
  c->set_output(1, c->Scalar());
  return Status::OK();
}

Status EncodeWavShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSaudio_opsDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/ops/audio_ops.cc", "EncodeWavShapeFn");

  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  c->set_output(0, c->Scalar());
  return Status::OK();
}

Status SpectrogramShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSaudio_opsDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/ops/audio_ops.cc", "SpectrogramShapeFn");

  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
  int32_t window_size;
  TF_RETURN_IF_ERROR(c->GetAttr("window_size", &window_size));
  int32_t stride;
  TF_RETURN_IF_ERROR(c->GetAttr("stride", &stride));

  DimensionHandle input_length = c->Dim(input, 0);
  DimensionHandle input_channels = c->Dim(input, 1);

  DimensionHandle output_length;
  if (!c->ValueKnown(input_length)) {
    output_length = c->UnknownDim();
  } else {
    const int64_t input_length_value = c->Value(input_length);
    const int64_t length_minus_window = (input_length_value - window_size);
    int64_t output_length_value;
    if (length_minus_window < 0) {
      output_length_value = 0;
    } else {
      output_length_value = 1 + (length_minus_window / stride);
    }
    output_length = c->MakeDim(output_length_value);
  }

  DimensionHandle output_channels =
      c->MakeDim(1 + NextPowerOfTwo(window_size) / 2);
  c->set_output(0,
                c->MakeShape({input_channels, output_length, output_channels}));
  return Status::OK();
}

Status MfccShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSaudio_opsDTcc mht_3(mht_3_v, 281, "", "./tensorflow/core/ops/audio_ops.cc", "MfccShapeFn");

  ShapeHandle spectrogram;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &spectrogram));
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

  int32_t dct_coefficient_count;
  TF_RETURN_IF_ERROR(
      c->GetAttr("dct_coefficient_count", &dct_coefficient_count));

  DimensionHandle spectrogram_channels = c->Dim(spectrogram, 0);
  DimensionHandle spectrogram_length = c->Dim(spectrogram, 1);

  DimensionHandle output_channels = c->MakeDim(dct_coefficient_count);

  c->set_output(0, c->MakeShape({spectrogram_channels, spectrogram_length,
                                 output_channels}));
  return Status::OK();
}

}  // namespace

REGISTER_OP("DecodeWav")
    .Input("contents: string")
    .Attr("desired_channels: int = -1")
    .Attr("desired_samples: int = -1")
    .Output("audio: float")
    .Output("sample_rate: int32")
    .SetShapeFn(DecodeWavShapeFn);

REGISTER_OP("EncodeWav")
    .Input("audio: float")
    .Input("sample_rate: int32")
    .Output("contents: string")
    .SetShapeFn(EncodeWavShapeFn);

REGISTER_OP("AudioSpectrogram")
    .Input("input: float")
    .Attr("window_size: int")
    .Attr("stride: int")
    .Attr("magnitude_squared: bool = false")
    .Output("spectrogram: float")
    .SetShapeFn(SpectrogramShapeFn);

REGISTER_OP("Mfcc")
    .Input("spectrogram: float")
    .Input("sample_rate: int32")
    .Attr("upper_frequency_limit: float = 4000")
    .Attr("lower_frequency_limit: float = 20")
    .Attr("filterbank_channel_count: int = 40")
    .Attr("dct_coefficient_count: int = 13")
    .Output("output: float")
    .SetShapeFn(MfccShapeFn);

}  // namespace tensorflow
