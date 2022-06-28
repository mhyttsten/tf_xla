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
class MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projectionDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projectionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projectionDTcc() {
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

// LSH Projection projects an input to a bit vector via locality sensitive
// hashing.
//
// Options:
//   Sparse:
//     Computed bit vector is considered to be sparse.
//     Each output element is an int32 made up by multiple bits computed from
// hash functions.
//
//   Dense:
//     Computed bit vector is considered to be dense. Each output element is
// either 0 or 1 that represents a bit.
//
// Input:
//   Tensor[0]: Hash functions. Dim.size == 2, DataType: Float.
//              Tensor[0].Dim[0]: Num of hash functions. Must be at least 1.
//              Tensor[0].Dim[1]: Num of projected output bits generated by
//                                each hash function.
//   In sparse case, Tensor[0].Dim[1] + ceil( log2(Tensor[0].Dim[0] )) <= 32.
//
//   Tensor[1]: Input. Dim.size >= 1, No restriction on DataType.
//   Tensor[2]: Optional, Weight. Dim.size == 1, DataType: Float.
//              If not set, each element of input is considered to have same
// weight of 1.0 Tensor[1].Dim[0] == Tensor[2].Dim[0]
//
// Output:
//   Sparse:
//     Output.Dim == { Tensor[0].Dim[0] }
//     A tensor of int32 that represents hash signatures,
//
//     NOTE: To avoid collisions across hash functions, an offset value of
//     k * (1 << Tensor[0].Dim[1]) will be added to each signature,
//     k is the index of the hash function.
//   Dense:
//     Output.Dim == { Tensor[0].Dim[0] * Tensor[0].Dim[1] }
//     A flattened tensor represents projected bit vectors.

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <memory>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include <farmhash.h>

namespace tflite {
namespace ops {
namespace builtin {
namespace lsh_projection {

TfLiteStatus Resize(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projectionDTcc mht_0(mht_0_v, 239, "", "./tensorflow/lite/kernels/lsh_projection.cc", "Resize");

  auto* params =
      reinterpret_cast<TfLiteLSHProjectionParams*>(node->builtin_data);
  TF_LITE_ENSURE(context, NumInputs(node) == 2 || NumInputs(node) == 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* hash;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &hash));
  TF_LITE_ENSURE_EQ(context, NumDimensions(hash), 2);
  // Support up to 32 bits.
  TF_LITE_ENSURE(context, SizeOfDimension(hash, 1) <= 32);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &input));
  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);
  TF_LITE_ENSURE(context, SizeOfDimension(input, 0) >= 1);

  if (NumInputs(node) == 3) {
    const TfLiteTensor* weight;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &weight));
    TF_LITE_ENSURE_EQ(context, NumDimensions(weight), 1);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(weight, 0),
                      SizeOfDimension(input, 0));
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
  switch (params->type) {
    case kTfLiteLshProjectionSparse:
      outputSize->data[0] = SizeOfDimension(hash, 0);
      break;
    case kTfLiteLshProjectionDense:
      outputSize->data[0] = SizeOfDimension(hash, 0) * SizeOfDimension(hash, 1);
      break;
    default:
      return kTfLiteError;
  }
  return context->ResizeTensor(context, output, outputSize);
}

// Compute sign bit of dot product of hash(seed, input) and weight.
// NOTE: use float as seed, and convert it to double as a temporary solution
//       to match the trained model. This is going to be changed once the new
//       model is trained in an optimized method.
//
int RunningSignBit(const TfLiteTensor* input, const TfLiteTensor* weight,
                   float seed) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projectionDTcc mht_1(mht_1_v, 289, "", "./tensorflow/lite/kernels/lsh_projection.cc", "RunningSignBit");

  double score = 0.0;
  int input_item_bytes = input->bytes / SizeOfDimension(input, 0);
  char* input_ptr = input->data.raw;

  const size_t seed_size = sizeof(float);
  const size_t key_bytes = sizeof(float) + input_item_bytes;
  std::unique_ptr<char[]> key(new char[key_bytes]);

  const float* weight_ptr = GetTensorData<float>(weight);

  for (int i = 0; i < SizeOfDimension(input, 0); ++i) {
    // Create running hash id and value for current dimension.
    memcpy(key.get(), &seed, seed_size);
    memcpy(key.get() + seed_size, input_ptr, input_item_bytes);

    int64_t hash_signature = ::util::Fingerprint64(key.get(), key_bytes);
    double running_value = static_cast<double>(hash_signature);
    input_ptr += input_item_bytes;
    if (weight_ptr == nullptr) {
      score += running_value;
    } else {
      score += weight_ptr[i] * running_value;
    }
  }

  return (score > 0) ? 1 : 0;
}

void SparseLshProjection(const TfLiteTensor* hash, const TfLiteTensor* input,
                         const TfLiteTensor* weight, int32_t* out_buf) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projectionDTcc mht_2(mht_2_v, 322, "", "./tensorflow/lite/kernels/lsh_projection.cc", "SparseLshProjection");

  int num_hash = SizeOfDimension(hash, 0);
  int num_bits = SizeOfDimension(hash, 1);
  for (int i = 0; i < num_hash; i++) {
    int32_t hash_signature = 0;
    for (int j = 0; j < num_bits; j++) {
      float seed = GetTensorData<float>(hash)[i * num_bits + j];
      int bit = RunningSignBit(input, weight, seed);
      hash_signature = (hash_signature << 1) | bit;
    }
    *out_buf++ = hash_signature + i * (1 << num_bits);
  }
}

void DenseLshProjection(const TfLiteTensor* hash, const TfLiteTensor* input,
                        const TfLiteTensor* weight, int32_t* out_buf) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projectionDTcc mht_3(mht_3_v, 340, "", "./tensorflow/lite/kernels/lsh_projection.cc", "DenseLshProjection");

  int num_hash = SizeOfDimension(hash, 0);
  int num_bits = SizeOfDimension(hash, 1);
  for (int i = 0; i < num_hash; i++) {
    for (int j = 0; j < num_bits; j++) {
      float seed = GetTensorData<float>(hash)[i * num_bits + j];
      int bit = RunningSignBit(input, weight, seed);
      *out_buf++ = bit;
    }
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projectionDTcc mht_4(mht_4_v, 355, "", "./tensorflow/lite/kernels/lsh_projection.cc", "Eval");

  auto* params =
      reinterpret_cast<TfLiteLSHProjectionParams*>(node->builtin_data);

  TfLiteTensor* out_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &out_tensor));
  int32_t* out_buf = out_tensor->data.i32;
  const TfLiteTensor* hash;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &hash));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &input));
  const TfLiteTensor* weight =
      NumInputs(node) == 2 ? nullptr : GetInput(context, node, 2);

  switch (params->type) {
    case kTfLiteLshProjectionDense:
      DenseLshProjection(hash, input, weight, out_buf);
      break;
    case kTfLiteLshProjectionSparse:
      SparseLshProjection(hash, input, weight, out_buf);
      break;
    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace lsh_projection

TfLiteRegistration* Register_LSH_PROJECTION() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlsh_projectionDTcc mht_5(mht_5_v, 387, "", "./tensorflow/lite/kernels/lsh_projection.cc", "Register_LSH_PROJECTION");

  static TfLiteRegistration r = {nullptr, nullptr, lsh_projection::Resize,
                                 lsh_projection::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
