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
class MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc() {
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
#include "tensorflow/lite/model_builder.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <utility>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/verifier.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

namespace {

// Ensure that ErrorReporter is non-null.
ErrorReporter* ValidateErrorReporter(ErrorReporter* e) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/model_builder.cc", "ValidateErrorReporter");

  return e ? e : DefaultErrorReporter();
}

}  // namespace

#ifndef TFLITE_MCU
// Loads a model from `filename`. If `mmap_file` is true then use mmap,
// otherwise make a copy of the model in a buffer.
std::unique_ptr<Allocation> GetAllocationFromFile(
    const char* filename, ErrorReporter* error_reporter) {
  std::unique_ptr<Allocation> allocation;
  if (MMAPAllocation::IsSupported()) {
    allocation.reset(new MMAPAllocation(filename, error_reporter));
  } else {
    allocation.reset(new FileCopyAllocation(filename, error_reporter));
  }
  return allocation;
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromFile(
    const char* filename, ErrorReporter* error_reporter) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + (filename == nullptr ? std::string("nullptr") : std::string((char*)filename)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_1(mht_1_v, 231, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::BuildFromFile");

  error_reporter = ValidateErrorReporter(error_reporter);
  return BuildFromAllocation(GetAllocationFromFile(filename, error_reporter),
                             error_reporter);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromFile(
    const char* filename, TfLiteVerifier* extra_verifier,
    ErrorReporter* error_reporter) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("filename: \"" + (filename == nullptr ? std::string("nullptr") : std::string((char*)filename)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_2(mht_2_v, 243, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::VerifyAndBuildFromFile");

  error_reporter = ValidateErrorReporter(error_reporter);
  return VerifyAndBuildFromAllocation(
      GetAllocationFromFile(filename, error_reporter), extra_verifier,
      error_reporter);
}
#endif

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromBuffer(
    const char* caller_owned_buffer, size_t buffer_size,
    ErrorReporter* error_reporter) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("caller_owned_buffer: \"" + (caller_owned_buffer == nullptr ? std::string("nullptr") : std::string((char*)caller_owned_buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_3(mht_3_v, 257, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::BuildFromBuffer");

  error_reporter = ValidateErrorReporter(error_reporter);
  std::unique_ptr<Allocation> allocation(
      new MemoryAllocation(caller_owned_buffer, buffer_size, error_reporter));
  return BuildFromAllocation(std::move(allocation), error_reporter);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromBuffer(
    const char* caller_owned_buffer, size_t buffer_size,
    TfLiteVerifier* extra_verifier, ErrorReporter* error_reporter) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("caller_owned_buffer: \"" + (caller_owned_buffer == nullptr ? std::string("nullptr") : std::string((char*)caller_owned_buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_4(mht_4_v, 270, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::VerifyAndBuildFromBuffer");

  error_reporter = ValidateErrorReporter(error_reporter);
  std::unique_ptr<Allocation> allocation(
      new MemoryAllocation(caller_owned_buffer, buffer_size, error_reporter));
  return VerifyAndBuildFromAllocation(std::move(allocation), extra_verifier,
                                      error_reporter);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromAllocation(
    std::unique_ptr<Allocation> allocation, ErrorReporter* error_reporter) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_5(mht_5_v, 282, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::BuildFromAllocation");

  std::unique_ptr<FlatBufferModel> model(new FlatBufferModel(
      std::move(allocation), ValidateErrorReporter(error_reporter)));
  if (!model->initialized()) {
    model.reset();
  }
  return model;
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromAllocation(
    std::unique_ptr<Allocation> allocation, TfLiteVerifier* extra_verifier,
    ErrorReporter* error_reporter) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_6(mht_6_v, 296, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::VerifyAndBuildFromAllocation");

  error_reporter = ValidateErrorReporter(error_reporter);
  if (!allocation || !allocation->valid()) {
    TF_LITE_REPORT_ERROR(error_reporter, "The model allocation is null/empty");
    return nullptr;
  }

  flatbuffers::Verifier base_verifier(
      reinterpret_cast<const uint8_t*>(allocation->base()),
      allocation->bytes());
  if (!VerifyModelBuffer(base_verifier)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "The model is not a valid Flatbuffer buffer");
    return nullptr;
  }

  if (extra_verifier &&
      !extra_verifier->Verify(static_cast<const char*>(allocation->base()),
                              allocation->bytes(), error_reporter)) {
    // The verifier will have already logged an appropriate error message.
    return nullptr;
  }

  return BuildFromAllocation(std::move(allocation), error_reporter);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromModel(
    const tflite::Model* caller_owned_model_spec,
    ErrorReporter* error_reporter) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_7(mht_7_v, 327, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::BuildFromModel");

  error_reporter = ValidateErrorReporter(error_reporter);

  std::unique_ptr<FlatBufferModel> model(
      new FlatBufferModel(caller_owned_model_spec, error_reporter));
  if (!model->initialized()) {
    model.reset();
  }
  return model;
}

string FlatBufferModel::GetMinimumRuntime() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_8(mht_8_v, 341, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::GetMinimumRuntime");

  if (!model_ || !model_->metadata()) return "";

  for (int i = 0; i < model_->metadata()->size(); ++i) {
    auto metadata = model_->metadata()->Get(i);
    if (metadata->name()->str() == "min_runtime_version") {
      auto buf = metadata->buffer();
      auto* buffer = (*model_->buffers())[buf];
      auto* array = buffer->data();
      // Get the real length of the runtime string, since there might be
      // trailing
      // '\0's in the buffer.
      for (int len = 0; len < array->size(); ++len) {
        if (array->data()[len] == '\0') {
          return string(reinterpret_cast<const char*>(array->data()), len);
        }
      }
      // If there is no '\0' in the buffer, this indicates that the flatbuffer
      // is malformed.
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Min_runtime_version in model metadata is malformed");
      break;
    }
  }
  return "";
}

std::map<std::string, std::string> FlatBufferModel::ReadAllMetadata() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_9(mht_9_v, 372, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::ReadAllMetadata");

  std::map<std::string, std::string> keys_values;
  if (!model_ || !model_->metadata() || !model_->buffers()) return keys_values;

  for (int i = 0; i < model_->metadata()->size(); ++i) {
    auto metadata = model_->metadata()->Get(i);
    auto buf = metadata->buffer();
    const tflite::Buffer* buffer = (*model_->buffers())[buf];
    if (!buffer || !buffer->data()) continue;
    const flatbuffers::Vector<uint8_t>* array = buffer->data();
    if (!array) continue;
    std::string val =
        string(reinterpret_cast<const char*>(array->data()), array->size());
    // Skip if key or value of metadata is empty.
    if (!metadata->name() || val.empty()) continue;
    keys_values[metadata->name()->str()] = val;
  }
  return keys_values;
}

bool FlatBufferModel::CheckModelIdentifier() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_10(mht_10_v, 395, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::CheckModelIdentifier");

  if (!tflite::ModelBufferHasIdentifier(allocation_->base())) {
    const char* ident = flatbuffers::GetBufferIdentifier(allocation_->base());
    error_reporter_->Report(
        "Model provided has model identifier '%c%c%c%c', should be '%s'\n",
        ident[0], ident[1], ident[2], ident[3], tflite::ModelIdentifier());
    return false;
  }
  return true;
}

FlatBufferModel::FlatBufferModel(const Model* model,
                                 ErrorReporter* error_reporter)
    : model_(model), error_reporter_(ValidateErrorReporter(error_reporter)) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_11(mht_11_v, 411, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::FlatBufferModel");
}

FlatBufferModel::FlatBufferModel(std::unique_ptr<Allocation> allocation,
                                 ErrorReporter* error_reporter)
    : error_reporter_(ValidateErrorReporter(error_reporter)),
      allocation_(std::move(allocation)) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_12(mht_12_v, 419, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::FlatBufferModel");

  if (!allocation_ || !allocation_->valid() || !CheckModelIdentifier()) {
    return;
  }

  model_ = ::tflite::GetModel(allocation_->base());
}

FlatBufferModel::~FlatBufferModel() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSmodel_builderDTcc mht_13(mht_13_v, 430, "", "./tensorflow/lite/model_builder.cc", "FlatBufferModel::~FlatBufferModel");
}

}  // namespace tflite
