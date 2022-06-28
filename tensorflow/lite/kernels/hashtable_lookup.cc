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
class MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookupDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookupDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookupDTcc() {
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

// Op that looks up items from hashtable.
//
// Input:
//     Tensor[0]: Hash key to lookup, dim.size == 1, int32
//     Tensor[1]: Key of hashtable, dim.size == 1, int32
//                *MUST* be sorted in ascending order.
//     Tensor[2]: Value of hashtable, dim.size >= 1
//                Tensor[1].Dim[0] == Tensor[2].Dim[0]
//
// Output:
//   Output[0].dim[0] == Tensor[0].dim[0], num of lookups
//   Each item in output is a raw bytes copy of corresponding item in input.
//   When key does not exist in hashtable, the returned bytes are all 0s.
//
//   Output[1].dim = { Tensor[0].dim[0] }, num of lookups
//   Each item indicates whether the corresponding lookup has a returned value.
//   0 for missing key, 1 for found key.

#include <stdint.h>

#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace {

int greater(const void* a, const void* b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookupDTcc mht_0(mht_0_v, 219, "", "./tensorflow/lite/kernels/hashtable_lookup.cc", "greater");

  return *static_cast<const int*>(a) - *static_cast<const int*>(b);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookupDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/kernels/hashtable_lookup.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  const TfLiteTensor* lookup;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &lookup));
  TF_LITE_ENSURE_EQ(context, NumDimensions(lookup), 1);
  TF_LITE_ENSURE_EQ(context, lookup->type, kTfLiteInt32);

  const TfLiteTensor* key;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &key));
  TF_LITE_ENSURE_EQ(context, NumDimensions(key), 1);
  TF_LITE_ENSURE_EQ(context, key->type, kTfLiteInt32);

  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &value));
  TF_LITE_ENSURE(context, NumDimensions(value) >= 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(key, 0),
                    SizeOfDimension(value, 0));
  if (value->type == kTfLiteString) {
    TF_LITE_ENSURE_EQ(context, NumDimensions(value), 1);
  }

  TfLiteTensor* hits;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &hits));
  TF_LITE_ENSURE_EQ(context, hits->type, kTfLiteUInt8);
  TfLiteIntArray* hitSize = TfLiteIntArrayCreate(1);
  hitSize->data[0] = SizeOfDimension(lookup, 0);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_EQ(context, value->type, output->type);

  TfLiteStatus status = kTfLiteOk;
  if (output->type != kTfLiteString) {
    TfLiteIntArray* outputSize = TfLiteIntArrayCreate(NumDimensions(value));
    outputSize->data[0] = SizeOfDimension(lookup, 0);
    for (int i = 1; i < NumDimensions(value); i++) {
      outputSize->data[i] = SizeOfDimension(value, i);
    }
    status = context->ResizeTensor(context, output, outputSize);
  }
  if (context->ResizeTensor(context, hits, hitSize) != kTfLiteOk) {
    status = kTfLiteError;
  }
  return status;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookupDTcc mht_2(mht_2_v, 277, "", "./tensorflow/lite/kernels/hashtable_lookup.cc", "Eval");

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TfLiteTensor* hits;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &hits));
  const TfLiteTensor* lookup;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &lookup));
  const TfLiteTensor* key;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &key));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &value));

  const int num_rows = SizeOfDimension(value, 0);
  TF_LITE_ENSURE(context, num_rows != 0);
  const int row_bytes = value->bytes / num_rows;
  void* pointer = nullptr;
  DynamicBuffer buf;

  for (int i = 0; i < SizeOfDimension(lookup, 0); i++) {
    int idx = -1;
    pointer = bsearch(&(lookup->data.i32[i]), key->data.i32, num_rows,
                      sizeof(int32_t), greater);
    if (pointer != nullptr) {
      idx = (reinterpret_cast<char*>(pointer) - (key->data.raw)) /
            sizeof(int32_t);
    }

    if (idx >= num_rows || idx < 0) {
      if (output->type == kTfLiteString) {
        buf.AddString(nullptr, 0);
      } else {
        memset(output->data.raw + i * row_bytes, 0, row_bytes);
      }
      hits->data.uint8[i] = 0;
    } else {
      if (output->type == kTfLiteString) {
        buf.AddString(GetString(value, idx));
      } else {
        memcpy(output->data.raw + i * row_bytes,
               value->data.raw + idx * row_bytes, row_bytes);
      }
      hits->data.uint8[i] = 1;
    }
  }
  if (output->type == kTfLiteString) {
    buf.WriteToTensorAsVector(output);
  }

  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration* Register_HASHTABLE_LOOKUP() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookupDTcc mht_3(mht_3_v, 332, "", "./tensorflow/lite/kernels/hashtable_lookup.cc", "Register_HASHTABLE_LOOKUP");

  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
