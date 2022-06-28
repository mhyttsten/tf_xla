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
class MHTracer_DTPStensorflowPSlitePSkernelsPSrollDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSrollDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSrollDTcc() {
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
#include <stdint.h>
#include <stdlib.h>

#include <cstring>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace roll {
namespace {

// A helper function to extract int32_t or int64_t tensor data.
std::vector<int32_t> ExtractIntegerVector(const TfLiteTensor* t) {
  TFLITE_DCHECK(t->type == kTfLiteInt32 || t->type == kTfLiteInt64);
  const RuntimeShape& shape = GetTensorShape(t);
  std::vector<int32_t> result(shape.FlatSize());
  if (t->type == kTfLiteInt32) {
    memcpy(result.data(), t->data.raw_const, t->bytes);
  } else {
    const int64_t* data = GetTensorData<int64_t>(t);
    for (int i = 0; i < result.size(); ++i) {
      result[i] = static_cast<int32_t>(data[i]);
    }
  }
  return result;
}

template <typename T>
inline void Pool(const std::vector<int32_t>& shift_map,
                 const RuntimeShape& shape, const TfLiteTensor* input,
                 TfLiteTensor* cache, TfLiteTensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrollDTcc mht_0(mht_0_v, 223, "", "./tensorflow/lite/kernels/roll.cc", "Pool");

  int stride = 1, outer_size, next_stride;
  bool in_place_rolling = false;
  for (int i = shift_map.size() - 1; i >= 0; --i, stride = next_stride) {
    next_stride = stride * shape.Dims(i);
    if (shift_map[i] == 0) continue;

    TFLITE_DCHECK_EQ(shape.FlatSize() % next_stride, 0);
    outer_size = shape.FlatSize() / next_stride;
    const TfLiteTensor* source = input;
    if (in_place_rolling) {
      SequentialTensorWriter<T> writer(output, cache);
      writer.WriteN(0, shape.FlatSize());
      source = cache;
    }
    SequentialTensorWriter<T> writer(source, output);
    for (int j = 0; j < outer_size; ++j) {
      // Copies the first stride.
      const int begin_1 =
          j * next_stride + (shape.Dims(i) - shift_map[i]) * stride;
      const int size_1 = shift_map[i] * stride;
      writer.WriteN(begin_1, size_1);
      // Copies the second stride.
      const int begin_2 = j * next_stride;
      const int size_2 = (shape.Dims(i) - shift_map[i]) * stride;
      writer.WriteN(begin_2, size_2);
    }
    in_place_rolling = true;
  }

  // Copies input to output if no rolling is needed.
  if (!in_place_rolling) {
    SequentialTensorWriter<T> writer(input, output);
    writer.WriteN(0, shape.FlatSize());
    return;
  }
}

}  // namespace

constexpr int kInputTensor = 0;
constexpr int kShiftTensor = 1;
constexpr int kAxisTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kTensorNotAllocated = -1;

struct OpData {
  // A temporary tensor to store intermediate output data when doing in-place
  // rolling.
  int cache_tensor_id = kTensorNotAllocated;
  int32_t cache_index = kTensorNotAllocated;
  bool need_cache = false;
};

TfLiteStatus AllocateTemporaryTensorsIfRequired(TfLiteContext* context,
                                                TfLiteNode* node,
                                                OpData* opdata,
                                                const TfLiteTensor* input,
                                                const TfLiteTensor* shift) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrollDTcc mht_1(mht_1_v, 284, "", "./tensorflow/lite/kernels/roll.cc", "AllocateTemporaryTensorsIfRequired");

  int temporaries_count = 0;
  opdata->need_cache = (NumElements(shift) > 1);
  if (opdata->need_cache) {
    if (opdata->cache_tensor_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &opdata->cache_tensor_id));
    }
    opdata->cache_index = temporaries_count++;
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(temporaries_count);

  if (opdata->need_cache) {
    node->temporaries->data[opdata->cache_index] = opdata->cache_tensor_id;
    TfLiteTensor* cache;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, opdata->cache_index, &cache));
    cache->type = input->type;
    cache->allocation_type = kTfLiteArenaRw;
    TfLiteIntArray* cache_shape = TfLiteIntArrayCopy(input->dims);
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, cache, cache_shape));
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSrollDTcc mht_2(mht_2_v, 316, "", "./tensorflow/lite/kernels/roll.cc", "Init");

  auto* opdata = new OpData;
  return opdata;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrollDTcc mht_3(mht_3_v, 324, "", "./tensorflow/lite/kernels/roll.cc", "Free");

  delete static_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrollDTcc mht_4(mht_4_v, 331, "", "./tensorflow/lite/kernels/roll.cc", "Prepare");

  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* shift;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kShiftTensor, &shift));
  const TfLiteTensor* axis;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxisTensor, &axis));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Check tensor type.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(
      context, (shift->type == kTfLiteInt32) || (shift->type == kTfLiteInt64));
  TF_LITE_ENSURE(context,
                 (axis->type == kTfLiteInt32) || (axis->type == kTfLiteInt64));

  // Make sure shift and axis are scalars or 1-D tensors.
  TF_LITE_ENSURE(context,
                 (NumDimensions(shift) == 0) || (NumDimensions(shift) == 1));
  TF_LITE_ENSURE(context,
                 (NumDimensions(shift) == 0) || (NumDimensions(shift) == 1));
  TF_LITE_ENSURE_EQ(context, NumElements(shift), NumElements(axis));

  TF_LITE_ENSURE_OK(context, AllocateTemporaryTensorsIfRequired(
                                 context, node, opdata, input, shift));

  // Output shape always equals to input shape.
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrollDTcc mht_5(mht_5_v, 371, "", "./tensorflow/lite/kernels/roll.cc", "Eval");

  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* shift;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kShiftTensor, &shift));
  const TfLiteTensor* axis;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxisTensor, &axis));

  TfLiteTensor* cache = GetTemporary(context, node, opdata->cache_index);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Extract the shift and axis information.
  std::vector<int32_t> shift_data = ExtractIntegerVector(shift);
  std::vector<int32_t> axis_data = ExtractIntegerVector(axis);

  // Maps from index as axis to its corresponding shift value.
  const int input_rank = NumDimensions(input);
  std::vector<int32_t> shift_map(input_rank, 0);

  // Make sure axis is in range [0, rank(input)).
  for (int i = 0; i < axis_data.size(); ++i) {
    int32_t axis_i = axis_data[i];
    if (axis_i < 0) axis_i += input_rank;
    shift_map[axis_i] += shift_data[i];
  }

  // Make sure shift is range [0, rank(input)).
  for (int i = 0; i < input_rank; ++i) {
    const int32_t input_dims_i = SizeOfDimension(input, i);
    int32_t shift_i = shift_map[i] % input_dims_i;
    if (shift_i < 0) shift_i += input_dims_i;
    shift_map[i] = shift_i;
  }

#define TF_LITE_ROLL(type) \
  Pool<type>(shift_map, GetTensorShape(input), input, cache, output);

  // The type itself doesn't matter, we only care about type size.
  switch (input->type) {
    case kTfLiteFloat32:
      TF_LITE_ROLL(float);
      break;
    case kTfLiteInt32:
      TF_LITE_ROLL(int32_t);
      break;
    case kTfLiteInt64:
      TF_LITE_ROLL(int64_t);
      break;
    case kTfLiteInt8:
      TF_LITE_ROLL(int8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_ROLL(int16_t);
      break;
    case kTfLiteUInt8:
      TF_LITE_ROLL(uint8_t);
      break;
    case kTfLiteBool:
      TF_LITE_ROLL(bool);
      break;
    case kTfLiteString:
      TF_LITE_ROLL(string);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Type %d is currently not supported by Slice.", input->type);
      return kTfLiteError;
  }
#undef TF_LITE_ROLL
  return kTfLiteOk;
}
}  // namespace roll

TfLiteRegistration* Register_ROLL() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrollDTcc mht_6(mht_6_v, 450, "", "./tensorflow/lite/kernels/roll.cc", "Register_ROLL");

  static TfLiteRegistration r = {roll::Init, roll::Free, roll::Prepare,
                                 roll::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
