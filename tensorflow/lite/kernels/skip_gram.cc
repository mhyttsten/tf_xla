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
class MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gramDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gramDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gramDTcc() {
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

// Generate a list of skip grams from an input.
//
// Options:
//   ngram_size: num of words for each output item.
//   max_skip_size: max num of words to skip.
//                  The op generates ngrams when it is 0.
//   include_all_ngrams: include all ngrams with size up to ngram_size.
//
// Input:
//   A string tensor to generate n-grams.
//   Dim = {1}
//
// Output:
//   A list of strings, each of which contains ngram_size words.
//   Dim = {num_ngram}

#include <ctype.h>

#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gramDTcc mht_0(mht_0_v, 217, "", "./tensorflow/lite/kernels/skip_gram.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input_tensor));
  TF_LITE_ENSURE_TYPES_EQ(context, input_tensor->type, kTfLiteString);
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output_tensor));
  TF_LITE_ENSURE_TYPES_EQ(context, output_tensor->type, kTfLiteString);
  return kTfLiteOk;
}

bool ShouldIncludeCurrentNgram(const TfLiteSkipGramParams* params, int size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gramDTcc mht_1(mht_1_v, 233, "", "./tensorflow/lite/kernels/skip_gram.cc", "ShouldIncludeCurrentNgram");

  if (size <= 0) {
    return false;
  }
  if (params->include_all_ngrams) {
    return size <= params->ngram_size;
  } else {
    return size == params->ngram_size;
  }
}

bool ShouldStepInRecursion(const TfLiteSkipGramParams* params,
                           const std::vector<int>& stack, int stack_idx,
                           int num_words) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gramDTcc mht_2(mht_2_v, 249, "", "./tensorflow/lite/kernels/skip_gram.cc", "ShouldStepInRecursion");

  // If current stack size and next word enumeration are within valid range.
  if (stack_idx < params->ngram_size && stack[stack_idx] + 1 < num_words) {
    // If this stack is empty, step in for first word enumeration.
    if (stack_idx == 0) {
      return true;
    }
    // If next word enumeration are within the range of max_skip_size.
    // NOTE: equivalent to
    //   next_word_idx = stack[stack_idx] + 1
    //   next_word_idx - stack[stack_idx-1] <= max_skip_size + 1
    if (stack[stack_idx] - stack[stack_idx - 1] <= params->max_skip_size) {
      return true;
    }
  }
  return false;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gramDTcc mht_3(mht_3_v, 270, "", "./tensorflow/lite/kernels/skip_gram.cc", "Eval");

  auto* params = reinterpret_cast<TfLiteSkipGramParams*>(node->builtin_data);

  // Split sentence to words.
  std::vector<StringRef> words;
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  tflite::StringRef strref = tflite::GetString(input, 0);
  int prev_idx = 0;
  for (int i = 1; i < strref.len; i++) {
    if (isspace(*(strref.str + i))) {
      if (i > prev_idx && !isspace(*(strref.str + prev_idx))) {
        words.push_back({strref.str + prev_idx, i - prev_idx});
      }
      prev_idx = i + 1;
    }
  }
  if (strref.len > prev_idx) {
    words.push_back({strref.str + prev_idx, strref.len - prev_idx});
  }

  // Generate n-grams recursively.
  tflite::DynamicBuffer buf;
  if (words.size() < params->ngram_size) {
    buf.WriteToTensorAsVector(GetOutput(context, node, 0));
    return kTfLiteOk;
  }

  // Stack stores the index of word used to generate ngram.
  // The size of stack is the size of ngram.
  std::vector<int> stack(params->ngram_size, 0);
  // Stack index that indicates which depth the recursion is operating at.
  int stack_idx = 1;
  int num_words = words.size();

  while (stack_idx >= 0) {
    if (ShouldStepInRecursion(params, stack, stack_idx, num_words)) {
      // When current depth can fill with a new word
      // and the new word is within the max range to skip,
      // fill this word to stack, recurse into next depth.
      stack[stack_idx]++;
      stack_idx++;
      if (stack_idx < params->ngram_size) {
        stack[stack_idx] = stack[stack_idx - 1];
      }
    } else {
      if (ShouldIncludeCurrentNgram(params, stack_idx)) {
        // Add n-gram to tensor buffer when the stack has filled with enough
        // words to generate the ngram.
        std::vector<StringRef> gram(stack_idx);
        for (int i = 0; i < stack_idx; i++) {
          gram[i] = words[stack[i]];
        }
        buf.AddJoinedString(gram, ' ');
      }
      // When current depth cannot fill with a valid new word,
      // and not in last depth to generate ngram,
      // step back to previous depth to iterate to next possible word.
      stack_idx--;
    }
  }

  buf.WriteToTensorAsVector(GetOutput(context, node, 0));
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration* Register_SKIP_GRAM() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gramDTcc mht_4(mht_4_v, 340, "", "./tensorflow/lite/kernels/skip_gram.cc", "Register_SKIP_GRAM");

  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
