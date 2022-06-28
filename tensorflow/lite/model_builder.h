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
/// \file
/// Deserialization infrastructure for tflite. Provides functionality
/// to go from a serialized tflite model in flatbuffer format to an
/// in-memory representation of the model.
///
#ifndef TENSORFLOW_LITE_MODEL_BUILDER_H_
#define TENSORFLOW_LITE_MODEL_BUILDER_H_
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
class MHTracer_DTPStensorflowPSlitePSmodel_builderDTh {
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
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSmodel_builderDTh() {
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


#include <stddef.h>

#include <map>
#include <memory>
#include <string>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/verifier.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

/// An RAII object that represents a read-only tflite model, copied from disk,
/// or mmapped. This uses flatbuffers as the serialization format.
///
/// NOTE: The current API requires that a FlatBufferModel instance be kept alive
/// by the client as long as it is in use by any dependent Interpreter
/// instances. As the FlatBufferModel instance is effectively immutable after
/// creation, the client may safely use a single model with multiple dependent
/// Interpreter instances, even across multiple threads (though note that each
/// Interpreter instance is *not* thread-safe).
///
/// <pre><code>
/// using namespace tflite;
/// StderrReporter error_reporter;
/// auto model = FlatBufferModel::BuildFromFile("interesting_model.tflite",
///                                             &error_reporter);
/// MyOpResolver resolver;  // You need to subclass OpResolver to provide
///                         // implementations.
/// InterpreterBuilder builder(*model, resolver);
/// std::unique_ptr<Interpreter> interpreter;
/// if(builder(&interpreter) == kTfLiteOk) {
///   .. run model inference with interpreter
/// }
/// </code></pre>
///
/// OpResolver must be defined to provide your kernel implementations to the
/// interpreter. This is environment specific and may consist of just the
/// builtin ops, or some custom operators you defined to extend tflite.
class FlatBufferModel {
 public:
  /// Builds a model based on a file.
  /// Caller retains ownership of `error_reporter` and must ensure its lifetime
  /// is longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromFile(
      const char* filename,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Verifies whether the content of the file is legit, then builds a model
  /// based on the file.
  /// The extra_verifier argument is an additional optional verifier for the
  /// file contents. By default, we always check with tflite::VerifyModelBuffer.
  /// If extra_verifier is supplied, the file contents is also checked against
  /// the extra_verifier after the check against tflite::VerifyModelBuilder.
  /// Caller retains ownership of `error_reporter` and must ensure its lifetime
  /// is longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromFile(
      const char* filename, TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Builds a model based on a pre-loaded flatbuffer.
  /// Caller retains ownership of the buffer and should keep it alive until
  /// the returned object is destroyed. Caller also retains ownership of
  /// `error_reporter` and must ensure its lifetime is longer than the
  /// FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  /// NOTE: this does NOT validate the buffer so it should NOT be called on
  /// invalid/untrusted input. Use VerifyAndBuildFromBuffer in that case
  static std::unique_ptr<FlatBufferModel> BuildFromBuffer(
      const char* caller_owned_buffer, size_t buffer_size,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Verifies whether the content of the buffer is legit, then builds a model
  /// based on the pre-loaded flatbuffer.
  /// The extra_verifier argument is an additional optional verifier for the
  /// buffer. By default, we always check with tflite::VerifyModelBuffer. If
  /// extra_verifier is supplied, the buffer is checked against the
  /// extra_verifier after the check against tflite::VerifyModelBuilder. The
  /// caller retains ownership of the buffer and should keep it alive until the
  /// returned object is destroyed. Caller retains ownership of `error_reporter`
  /// and must ensure its lifetime is longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromBuffer(
      const char* caller_owned_buffer, size_t buffer_size,
      TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Builds a model directly from an allocation.
  /// Ownership of the allocation is passed to the model, but the caller
  /// retains ownership of `error_reporter` and must ensure its lifetime is
  /// longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure (e.g., the allocation is invalid).
  static std::unique_ptr<FlatBufferModel> BuildFromAllocation(
      std::unique_ptr<Allocation> allocation,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Verifies whether the content of the allocation is legit, then builds a
  /// model based on the provided allocation.
  /// The extra_verifier argument is an additional optional verifier for the
  /// buffer. By default, we always check with tflite::VerifyModelBuffer. If
  /// extra_verifier is supplied, the buffer is checked against the
  /// extra_verifier after the check against tflite::VerifyModelBuilder.
  /// Ownership of the allocation is passed to the model, but the caller
  /// retains ownership of `error_reporter` and must ensure its lifetime is
  /// longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromAllocation(
      std::unique_ptr<Allocation> allocation,
      TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Builds a model directly from a flatbuffer pointer
  /// Caller retains ownership of the buffer and should keep it alive until the
  /// returned object is destroyed. Caller retains ownership of `error_reporter`
  /// and must ensure its lifetime is longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromModel(
      const tflite::Model* caller_owned_model_spec,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  // Releases memory or unmaps mmaped memory.
  ~FlatBufferModel();

  // Copying or assignment is disallowed to simplify ownership semantics.
  FlatBufferModel(const FlatBufferModel&) = delete;
  FlatBufferModel& operator=(const FlatBufferModel&) = delete;

  bool initialized() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTh mht_0(mht_0_v, 327, "", "./tensorflow/lite/model_builder.h", "initialized");
 return model_ != nullptr; }
  const tflite::Model* operator->() const { return model_; }
  const tflite::Model* GetModel() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTh mht_1(mht_1_v, 332, "", "./tensorflow/lite/model_builder.h", "GetModel");
 return model_; }
  ErrorReporter* error_reporter() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTh mht_2(mht_2_v, 336, "", "./tensorflow/lite/model_builder.h", "error_reporter");
 return error_reporter_; }
  const Allocation* allocation() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTh mht_3(mht_3_v, 340, "", "./tensorflow/lite/model_builder.h", "allocation");
 return allocation_.get(); }

  // Returns the minimum runtime version from the flatbuffer. This runtime
  // version encodes the minimum required interpreter version to run the
  // flatbuffer model. If the minimum version can't be determined, an empty
  // string will be returned.
  // Note that the returned minimum version is a lower-bound but not a strict
  // lower-bound; ops in the graph may not have an associated runtime version,
  // in which case the actual required runtime might be greater than the
  // reported minimum.
  std::string GetMinimumRuntime() const;

  // Return model metadata as a mapping of name & buffer strings.
  // See Metadata table in TFLite schema.
  std::map<std::string, std::string> ReadAllMetadata() const;

  /// Returns true if the model identifier is correct (otherwise false and
  /// reports an error).
  bool CheckModelIdentifier() const;

 private:
  /// Loads a model from a given allocation. FlatBufferModel will take over the
  /// ownership of `allocation`, and delete it in destructor. The ownership of
  /// `error_reporter`remains with the caller and must have lifetime at least
  /// as much as FlatBufferModel. This is to allow multiple models to use the
  /// same ErrorReporter instance.
  FlatBufferModel(std::unique_ptr<Allocation> allocation,
                  ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Loads a model from Model flatbuffer. The `model` has to remain alive and
  /// unchanged until the end of this flatbuffermodel's lifetime.
  FlatBufferModel(const Model* model, ErrorReporter* error_reporter);

  /// Flatbuffer traverser pointer. (Model* is a pointer that is within the
  /// allocated memory of the data allocated by allocation's internals.
  const tflite::Model* model_ = nullptr;
  /// The error reporter to use for model errors and subsequent errors when
  /// the interpreter is created
  ErrorReporter* error_reporter_;
  /// The allocator used for holding memory of the model. Note that this will
  /// be null if the client provides a tflite::Model directly.
  std::unique_ptr<Allocation> allocation_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MODEL_BUILDER_H_
