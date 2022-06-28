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
class MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <complex>

#include "third_party/fft2d/fft2d.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace irfft2d {

using std::complex;

constexpr int kInputTensor = 0;
constexpr int kFftLengthTensor = 1;
constexpr int kOutputTensor = 0;
constexpr int kFftIntegerWorkingAreaTensor = 0;
constexpr int kFftDoubleWorkingAreaTensor = 1;
constexpr int kTensorNotAllocated = -1;

struct OpData {
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int fft_integer_working_area_id = kTensorNotAllocated;
  int fft_double_working_area_id = kTensorNotAllocated;
};

bool IsPowerOfTwo(uint32_t v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_0(mht_0_v, 222, "", "./tensorflow/lite/kernels/irfft2d.cc", "IsPowerOfTwo");
 return v && !(v & (v - 1)); }

static TfLiteStatus InitTemporaryTensors(TfLiteContext* context,
                                         TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_1(mht_1_v, 228, "", "./tensorflow/lite/kernels/irfft2d.cc", "InitTemporaryTensors");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  // The prepare function may be executed multiple times. But temporary tensors
  // only need to be initiated once.
  if (data->fft_integer_working_area_id != kTensorNotAllocated &&
      data->fft_double_working_area_id != kTensorNotAllocated) {
    return kTfLiteOk;
  }

  TfLiteIntArrayFree(node->temporaries);
  // Create two temporary tensors.
  node->temporaries = TfLiteIntArrayCreate(2);
  int first_new_index;
  TF_LITE_ENSURE_STATUS(context->AddTensors(context, 2, &first_new_index));
  node->temporaries->data[kFftIntegerWorkingAreaTensor] = first_new_index;
  data->fft_integer_working_area_id = first_new_index;
  node->temporaries->data[kFftDoubleWorkingAreaTensor] = first_new_index + 1;
  data->fft_double_working_area_id = first_new_index + 1;

  // Set up FFT integer working area buffer.
  TfLiteTensor* fft_integer_working_area;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, kFftIntegerWorkingAreaTensor,
                                &fft_integer_working_area));
  fft_integer_working_area->type = kTfLiteInt32;
  // If fft_length is not a constant tensor, fft_integer_working_area will be
  // set to dynamic later in Prepare.
  fft_integer_working_area->allocation_type = kTfLiteArenaRw;

  // Set up FFT double working area buffer.
  TfLiteTensor* fft_double_working_area;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, kFftDoubleWorkingAreaTensor,
                                     &fft_double_working_area));
  // fft_double_working_area is a double tensor. Ideally, double should be
  // added into tflite data types. However, since fft_double_working_area is a
  // temporary tensor, and there are no ops having double input/output tensors
  // in tflite at this point, adding double as a tflite data type may confuse
  // users that double is supported. As a results, kTfLiteInt64 is used here
  // for memory allocation. And it will be cast into double in Eval when being
  // used.
  fft_double_working_area->type = kTfLiteInt64;
  // If fft_length is not a constant tensor, fft_double_working_area will be
  // set to dynamic later in Prepare.
  fft_double_working_area->allocation_type = kTfLiteArenaRw;

  return kTfLiteOk;
}

TfLiteStatus ResizeOutputandTemporaryTensors(TfLiteContext* context,
                                             TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_2(mht_2_v, 281, "", "./tensorflow/lite/kernels/irfft2d.cc", "ResizeOutputandTemporaryTensors");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const int num_dims = NumDimensions(input);
  TF_LITE_ENSURE(context, num_dims >= 2);
  const TfLiteTensor* fft_length;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFftLengthTensor, &fft_length));
  const int32_t* fft_length_data = GetTensorData<int32_t>(fft_length);
  // The lib, fft2d, can only handle fft_lengths of power of 2.
  TF_LITE_ENSURE(context, IsPowerOfTwo(fft_length_data[0]));
  TF_LITE_ENSURE(context, IsPowerOfTwo(fft_length_data[1]));

  int fft_height, fft_width;
  fft_height = fft_length_data[0];
  fft_width = fft_length_data[1];
  int fft_working_length = std::max(fft_height, fft_width / 2);
  int half_fft_working_length = fft_working_length / 2;

  // Resize output tensor.
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  output_shape->data[num_dims - 2] = fft_length_data[0];
  output_shape->data[num_dims - 1] = fft_length_data[1];
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_shape));

  // Resize temporary tensors, fft_integer_working_area.
  TfLiteTensor* fft_integer_working_area;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, kFftIntegerWorkingAreaTensor,
                                &fft_integer_working_area));
  TfLiteIntArray* fft_integer_working_area_shape = TfLiteIntArrayCreate(1);
  fft_integer_working_area_shape->data[0] =
      2 + static_cast<int>(sqrt(fft_working_length));
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, fft_integer_working_area,
                                              fft_integer_working_area_shape));

  // Resize temporary tensors, fft_double_working_area.
  TfLiteTensor* fft_double_working_area;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, kFftDoubleWorkingAreaTensor,
                                     &fft_double_working_area));
  TfLiteIntArray* fft_double_working_area_shape = TfLiteIntArrayCreate(1);
  fft_double_working_area_shape->data[0] =
      half_fft_working_length + fft_width / 4;
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, fft_double_working_area,
                                              fft_double_working_area_shape));

  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_3(mht_3_v, 338, "", "./tensorflow/lite/kernels/irfft2d.cc", "Init");

  auto* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_4(mht_4_v, 346, "", "./tensorflow/lite/kernels/irfft2d.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_5(mht_5_v, 353, "", "./tensorflow/lite/kernels/irfft2d.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Check type and shape of the input tensor
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TF_LITE_ENSURE(context, NumDimensions(input) >= 2);
  if (input->type != kTfLiteComplex64) {
    TF_LITE_KERNEL_LOG(context,
                       "Type '%s' for input is not supported by irfft2.",
                       TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  // Check type and shape of the fft_length tensor
  const TfLiteTensor* fft_length;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFftLengthTensor, &fft_length));
  const RuntimeShape fft_length_shape = GetTensorShape(fft_length);

  TF_LITE_ENSURE_EQ(context, NumDimensions(fft_length), 1);
  TF_LITE_ENSURE_EQ(context, fft_length_shape.Dims(0), 2);
  if (fft_length->type != kTfLiteInt32) {
    TF_LITE_KERNEL_LOG(context,
                       "Type '%s' for fft_length is not supported by irfft2.",
                       TfLiteTypeGetName(fft_length->type));
    return kTfLiteError;
  }

  // Setup temporary tensors for fft computation.
  TF_LITE_ENSURE_STATUS(InitTemporaryTensors(context, node));

  // Set output type
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = kTfLiteFloat32;

  // Exit early if fft_length is a non-const tensor. Set output tensor and
  // temporary tensors to dynamic, so that their tensor sizes can be determined
  // in Eval.
  if (!IsConstantTensor(fft_length)) {
    TfLiteTensor* fft_integer_working_area;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kFftIntegerWorkingAreaTensor,
                                  &fft_integer_working_area));
    TfLiteTensor* fft_double_working_area;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kFftDoubleWorkingAreaTensor,
                                  &fft_double_working_area));
    SetTensorToDynamic(fft_integer_working_area);
    SetTensorToDynamic(fft_double_working_area);
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }

  TF_LITE_ENSURE_STATUS(ResizeOutputandTemporaryTensors(context, node));
  return kTfLiteOk;
}

// TODO(b/187490449) Make the behavior matches TensorFlow kernel.
void Irfft2dReorder(int fft_height, int fft_width, double** fft_input_output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_6(mht_6_v, 418, "", "./tensorflow/lite/kernels/irfft2d.cc", "Irfft2dReorder");

  ruy::profiler::ScopeLabel label("Irfft2dReorder");
  // Take complex conjugate of frequency matrix
  for (int i = 0; i < fft_height; ++i) {
    for (int j = 1; j < fft_width + 2; j += 2) {
      fft_input_output[i][j] = -fft_input_output[i][j];
    }
  }

  // Arrange matrix into correct format for rdft2d function
  const int kBackwardFft = -1;
  rdft2dsort(fft_height, fft_width, kBackwardFft, fft_input_output);
}

void Irfft2dImpl(int fft_height, int fft_width, double** fft_input_output,
                 int* fft_integer_working_area_data,
                 double* fft_double_working_area_data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_7(mht_7_v, 437, "", "./tensorflow/lite/kernels/irfft2d.cc", "Irfft2dImpl");

  ruy::profiler::ScopeLabel label("Irfft2dImpl");
  Irfft2dReorder(fft_height, fft_width, fft_input_output);

  // Working data areas for the FFT routines.
  double* fft_dynamic_working_area = nullptr;
  const int kBackwardFft = -1;
  rdft2d(fft_height, fft_width, kBackwardFft, fft_input_output,
         fft_dynamic_working_area, fft_integer_working_area_data,
         fft_double_working_area_data);
}

void PrepareInputBuffer(const complex<float>* input_data, int input_height,
                        int input_width, int fft_height, int fft_width,
                        double** fft_input_output) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_8(mht_8_v, 454, "", "./tensorflow/lite/kernels/irfft2d.cc", "PrepareInputBuffer");

  int valid_input_height = std::min(input_height, fft_height);
  int valid_input_width = std::min(input_width, fft_width / 2 + 1);
  for (int i = 0; i < valid_input_height; ++i) {
    int in_pos = i * input_width;
    for (int j = 0; j < valid_input_width; ++j) {
      fft_input_output[i][2 * j] = input_data[in_pos].real();
      fft_input_output[i][2 * j + 1] = input_data[in_pos].imag();
      ++in_pos;
    }
    // Zero-pad the rest of the input buffer
    for (int j = valid_input_width; j < fft_width / 2 + 1; ++j) {
      fft_input_output[i][2 * j] = 0;
      fft_input_output[i][2 * j + 1] = 0;
    }
  }

  // Zero-pad input buffer, if fft_height is greater than valid_input_height.
  for (int i = valid_input_height; i < fft_height; ++i) {
    for (int j = 0; j < fft_width / 2 + 1; ++j) {
      fft_input_output[i][2 * j] = 0;
      fft_input_output[i][2 * j + 1] = 0;
    }
  }
}

void PrepareOutputBuffer(float* output_data, int fft_height, int fft_width,
                         double** fft_input_output) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_9(mht_9_v, 484, "", "./tensorflow/lite/kernels/irfft2d.cc", "PrepareOutputBuffer");

  int cnt = 0;
  float norm = 2.0 / static_cast<float>(fft_height * fft_width);
  for (int i = 0; i < fft_height; ++i) {
    for (int j = 0; j < fft_width; ++j) {
      output_data[cnt++] = fft_input_output[i][j] * norm;
    }
  }
}

TfLiteStatus Irfft2dHelper(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_10(mht_10_v, 497, "", "./tensorflow/lite/kernels/irfft2d.cc", "Irfft2dHelper");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const complex<float>* input_data = GetTensorData<complex<float>>(input);
  const TfLiteTensor* fft_length;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFftLengthTensor, &fft_length));
  const int32_t* fft_length_data = GetTensorData<int32_t>(fft_length);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  float* output_data = GetTensorData<float>(output);

  int fft_height, fft_width;
  fft_height = fft_length_data[0];
  fft_width = fft_length_data[1];

  // FFT is processed for every slice on the inner most 2 dimensions.
  // Count the number of slices in the input tensor.
  const RuntimeShape input_shape = GetTensorShape(input);
  const int input_dims_count = input_shape.DimensionsCount();
  const auto* input_dims_data = input_shape.DimsData();
  int num_slices = 1;
  for (int i = 0; i < input_dims_count - 2; ++i) {
    num_slices *= input_dims_data[i];
  }

  int input_height = input_dims_data[input_dims_count - 2];
  int input_width = input_dims_data[input_dims_count - 1];
  int input_slice_size = input_height * input_width;
  int output_slice_size = fft_height * fft_width;

  // Create input/output buffer for FFT
  double** fft_input_output = new double*[fft_height];
  for (int i = 0; i < fft_height; ++i) {
    fft_input_output[i] = new double[fft_width + 2];
  }

  // Get buffer for integer working area.
  TfLiteTensor* fft_integer_working_area;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, kFftIntegerWorkingAreaTensor,
                                &fft_integer_working_area));
  int* fft_integer_working_area_data =
      GetTensorData<int>(fft_integer_working_area);

  // Get buffer for double working area.
  TfLiteTensor* fft_double_working_area;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, kFftDoubleWorkingAreaTensor,
                                     &fft_double_working_area));
  // Get double value out of the memory of fft_double_working_area_data.
  double* fft_double_working_area_data = reinterpret_cast<double*>(
      GetTensorData<int64_t>(fft_double_working_area));

  // Process every slice in the input buffer
  for (int i = 0; i < num_slices; ++i) {
    PrepareInputBuffer(input_data, input_height, input_width, fft_height,
                       fft_width, fft_input_output);
    memset(fft_integer_working_area_data, 0, fft_integer_working_area->bytes);
    memset(fft_double_working_area_data, 0, fft_double_working_area->bytes);
    Irfft2dImpl(fft_height, fft_width, fft_input_output,
                fft_integer_working_area_data, fft_double_working_area_data);
    PrepareOutputBuffer(output_data, fft_height, fft_width, fft_input_output);
    input_data += input_slice_size;
    output_data += output_slice_size;
  }

  // Delete the input buffer
  for (int i = 0; i < fft_height; ++i) {
    delete[] fft_input_output[i];
  }
  delete[] fft_input_output;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_11(mht_11_v, 577, "", "./tensorflow/lite/kernels/irfft2d.cc", "Eval");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* fft_length;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFftLengthTensor, &fft_length));
  const int32_t* fft_length_data = GetTensorData<int32_t>(fft_length);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (output->type != kTfLiteFloat32) {
    TF_LITE_KERNEL_LOG(context,
                       "Type '%s' for output is not supported by irfft2.",
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  // Resize the output tensor if the fft_length tensor is not constant.
  // Otherwise, check if the output shape is correct.
  if (!IsConstantTensor(fft_length)) {
    TF_LITE_ENSURE_STATUS(ResizeOutputandTemporaryTensors(context, node));
  } else {
    int num_dims_output = NumDimensions(output);
    const RuntimeShape output_shape = GetTensorShape(output);
    TF_LITE_ENSURE_EQ(context, num_dims_output, NumDimensions(input));
    TF_LITE_ENSURE(context, num_dims_output >= 2);
    TF_LITE_ENSURE_EQ(context, output_shape.Dims(num_dims_output - 2),
                      fft_length_data[0]);
    TF_LITE_ENSURE_EQ(context, output_shape.Dims(num_dims_output - 1),
                      fft_length_data[1]);
  }

  return Irfft2dHelper(context, node);
}

}  // namespace irfft2d

TfLiteRegistration* Register_IRFFT2D() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2dDTcc mht_12(mht_12_v, 618, "", "./tensorflow/lite/kernels/irfft2d.cc", "Register_IRFFT2D");

  static TfLiteRegistration r = {irfft2d::Init, irfft2d::Free, irfft2d::Prepare,
                                 irfft2d::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
