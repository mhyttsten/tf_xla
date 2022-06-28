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
#ifndef TENSORFLOW_LITE_CORE_SIGNATURE_RUNNER_H_
#define TENSORFLOW_LITE_CORE_SIGNATURE_RUNNER_H_
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
class MHTracer_DTPStensorflowPSlitePSsignature_runnerDTh {
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
   MHTracer_DTPStensorflowPSlitePSsignature_runnerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSsignature_runnerDTh() {
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


#include <cstddef>
#include <cstdint>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/internal/signature_def.h"

namespace tflite {
class Interpreter;  // Class for friend declarations.
class SignatureRunnerJNIHelper;  // Class for friend declarations.
class TensorHandle;              // Class for friend declarations.

/// WARNING: Experimental interface, subject to change
///
/// SignatureRunner class for running TFLite models using SignatureDef.
///
/// Usage:
///
/// <pre><code>
/// // Create model from file. Note that the model instance must outlive the
/// // interpreter instance.
/// auto model = tflite::FlatBufferModel::BuildFromFile(...);
/// if (model == nullptr) {
///   // Return error.
/// }
///
/// // Create an Interpreter with an InterpreterBuilder.
/// std::unique_ptr<tflite::Interpreter> interpreter;
/// tflite::ops::builtin::BuiltinOpResolver resolver;
/// if (InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
///   // Return failure.
/// }
///
/// // Get the list of signatures and check it.
/// auto signature_defs = interpreter->signature_def_names();
/// if (signature_defs.empty()) {
///   // Return error.
/// }
///
/// // Get pointer to the SignatureRunner instance corresponding to a signature.
/// // Note that the pointed SignatureRunner instance has lifetime same as the
/// // Interpreter instance.
/// tflite::SignatureRunner* runner =
///                interpreter->GetSignatureRunner(signature_defs[0]->c_str());
/// if (runner == nullptr) {
///   // Return error.
/// }
/// if (runner->AllocateTensors() != kTfLiteOk) {
///   // Return failure.
/// }
///
/// // Set input data. In this example, the input tensor has float type.
/// float* input = runner->input_tensor(0)->data.f;
/// for (int i = 0; i < input_size; i++) {
///   input[i] = ...;
//  }
/// runner->Invoke();
/// </code></pre>
///
/// WARNING: This class is *not* thread-safe. The client is responsible for
/// ensuring serialized interaction to avoid data races and undefined behavior.
///
/// SignatureRunner and Interpreter share the same underlying data. Calling
/// methods on an Interpreter object will affect the state in corresponding
/// SignatureRunner objects. Therefore, it is recommended not to call other
/// Interpreter methods after calling GetSignatureRunner to create
/// SignatureRunner instances.
class SignatureRunner {
 public:
  /// Returns the key for the corresponding signature.
  const std::string& signature_key() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSsignature_runnerDTh mht_0(mht_0_v, 258, "", "./tensorflow/lite/signature_runner.h", "signature_key");
 return signature_def_->signature_key; }

  /// Returns the number of inputs.
  size_t input_size() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSsignature_runnerDTh mht_1(mht_1_v, 264, "", "./tensorflow/lite/signature_runner.h", "input_size");
 return subgraph_->inputs().size(); }

  /// Returns the number of outputs.
  size_t output_size() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSsignature_runnerDTh mht_2(mht_2_v, 270, "", "./tensorflow/lite/signature_runner.h", "output_size");
 return subgraph_->outputs().size(); }

  /// Read-only access to list of signature input names.
  const std::vector<const char*>& input_names() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSsignature_runnerDTh mht_3(mht_3_v, 276, "", "./tensorflow/lite/signature_runner.h", "input_names");
 return input_names_; }

  /// Read-only access to list of signature output names.
  const std::vector<const char*>& output_names() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSsignature_runnerDTh mht_4(mht_4_v, 282, "", "./tensorflow/lite/signature_runner.h", "output_names");
 return output_names_; }

  /// Returns the input tensor identified by 'input_name' in the
  /// given signature. Returns nullptr if the given name is not valid.
  TfLiteTensor* input_tensor(const char* input_name);

  /// Returns the output tensor identified by 'output_name' in the
  /// given signature. Returns nullptr if the given name is not valid.
  const TfLiteTensor* output_tensor(const char* output_name) const;

  /// Change a dimensionality of a given tensor. Note, this is only acceptable
  /// for tensors that are inputs.
  /// Returns status of failure or success. Note that this doesn't actually
  /// resize any existing buffers. A call to AllocateTensors() is required to
  /// change the tensor input buffer.
  TfLiteStatus ResizeInputTensor(const char* input_name,
                                 const std::vector<int>& new_size);

  /// Change the dimensionality of a given tensor. This is only acceptable for
  /// tensor indices that are inputs or variables. Only unknown dimensions can
  /// be resized with this function. Unknown dimensions are indicated as `-1` in
  /// the `dims_signature` attribute of a TfLiteTensor.
  /// Returns status of failure or success. Note that this doesn't actually
  /// resize any existing buffers. A call to AllocateTensors() is required to
  /// change the tensor input buffer.
  TfLiteStatus ResizeInputTensorStrict(const char* input_name,
                                       const std::vector<int>& new_size);

  /// Updates allocations for all tensors, related to the given signature.
  TfLiteStatus AllocateTensors() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSsignature_runnerDTh mht_5(mht_5_v, 314, "", "./tensorflow/lite/signature_runner.h", "AllocateTensors");
 return subgraph_->AllocateTensors(); }

  /// Invokes the signature runner (run the graph identified by the given
  /// signature in dependency order).
  TfLiteStatus Invoke();

 private:
  // The life cycle of SignatureRunner depends on the life cycle of Subgraph,
  // which is owned by an Interpreter. Therefore, the Interpreter will takes the
  // responsibility to create and manage SignatureRunner objects to make sure
  // SignatureRunner objects don't outlive their corresponding Subgraph objects.
  SignatureRunner(const internal::SignatureDef* signature_def,
                  Subgraph* subgraph);
  friend class Interpreter;
  friend class SignatureRunnerJNIHelper;
  friend class TensorHandle;

  // The SignatureDef object is owned by the interpreter.
  const internal::SignatureDef* signature_def_;
  // The Subgraph object is owned by the interpreter.
  Subgraph* subgraph_;
  // The list of input tensor names.
  std::vector<const char*> input_names_;
  // The list of output tensor names.
  std::vector<const char*> output_names_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_SIGNATURE_RUNNER_H_
