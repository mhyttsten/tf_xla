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
class MHTracer_DTPStensorflowPSsecurityPSfuzzingPScheckpoint_reader_fuzzDTcc {
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
   MHTracer_DTPStensorflowPSsecurityPSfuzzingPScheckpoint_reader_fuzzDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSsecurityPSfuzzingPScheckpoint_reader_fuzzDTcc() {
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

#include <memory>
#include <set>

#include "src/libfuzzer/libfuzzer_macro.h"  // from @com_google_libprotobuf_mutator
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/io/table_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/security/fuzzing/checkpoint_reader_fuzz_input.pb.h"

// This is a fuzzer for tensorflow::checkpoint::CheckpointReader. LevelDB
// reading and proto parsing are already fuzz-tested, so there's no need to test
// them here.

namespace {

using ::tensorflow::checkpoint::EncodeTensorNameSlice;
using ::tensorflow::checkpoint::kSavedTensorSlicesKey;

void CreateCheckpoint(const std::string& filename,
                      const tensorflow::CheckpointReaderFuzzInput& contents) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSsecurityPSfuzzingPScheckpoint_reader_fuzzDTcc mht_0(mht_0_v, 218, "", "./tensorflow/security/fuzzing/checkpoint_reader_fuzz.cc", "CreateCheckpoint");

  std::unique_ptr<tensorflow::WritableFile> writable_file;
  TF_CHECK_OK(
      tensorflow::Env::Default()->NewWritableFile(filename, &writable_file));
  tensorflow::table::Options options;
  options.compression = tensorflow::table::kNoCompression;
  tensorflow::table::TableBuilder builder(options, writable_file.get());

  // Entries must be added in sorted order.
  {
    tensorflow::SavedTensorSlices sts;
    *sts.mutable_meta() = contents.meta();
    builder.Add(kSavedTensorSlicesKey, sts.SerializeAsString());
  }
  std::map<std::string, const tensorflow::SavedSlice*> entries;
  for (const tensorflow::SavedSlice& saved_slice : contents.data()) {
    // The encoded tensor slice name is not included in the fuzz input since
    // it's difficult for the fuzzer to find the proper encoding, resulting in
    // lots of fruitless inputs with mismatched keys. Note that TensorSlice will
    // not currently crash with unverified data so long as it's only used by
    // EncodeTensorNameSlice.
    tensorflow::TensorSlice slice(saved_slice.slice());
    entries.insert(
        {EncodeTensorNameSlice(saved_slice.name(), slice), &saved_slice});
  }
  tensorflow::SavedTensorSlices sts;
  for (const auto& entry : entries) {
    *sts.mutable_data() = *entry.second;
    builder.Add(entry.first, sts.SerializeAsString());
  }
  TF_CHECK_OK(builder.Finish());
  TF_CHECK_OK(writable_file->Close());
}

int GetDataTypeSize(tensorflow::DataType data_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSsecurityPSfuzzingPScheckpoint_reader_fuzzDTcc mht_1(mht_1_v, 255, "", "./tensorflow/security/fuzzing/checkpoint_reader_fuzz.cc", "GetDataTypeSize");

  // tensorflow::DataTypeSize doesn't support several types.
  switch (data_type) {
    case tensorflow::DT_STRING:
      return sizeof(tensorflow::tstring);
    case tensorflow::DT_VARIANT:
      return sizeof(tensorflow::Variant);
    case tensorflow::DT_RESOURCE:
      return sizeof(tensorflow::ResourceHandle);
    default:
      return tensorflow::DataTypeSize(data_type);
  }
}

DEFINE_PROTO_FUZZER(const tensorflow::CheckpointReaderFuzzInput& input) {
  // Using a ram file avoids disk I/O, speeding up the fuzzer.
  const std::string filename = "ram:///checkpoint";
  CreateCheckpoint(filename, input);

  tensorflow::TF_StatusPtr status(TF_NewStatus());
  tensorflow::checkpoint::CheckpointReader reader(filename, status.get());
  if (TF_GetCode(status.get()) != TF_OK) return;

  // Load each tensor in the input.
  std::unique_ptr<tensorflow::Tensor> tensor;
  for (const auto& entry : input.meta().tensor()) {
    // Fuzz tests have a memory limit of 2 GB; skipping tensors over 1 GB is
    // sufficient to avoid OOMs.
    static constexpr double kMaxTensorSize = 1e9;
    auto data_type = reader.GetVariableToDataTypeMap().find(entry.name());
    auto shape = reader.GetVariableToShapeMap().find(entry.name());
    if (data_type != reader.GetVariableToDataTypeMap().end() &&
        shape != reader.GetVariableToShapeMap().end() &&
        static_cast<double>(GetDataTypeSize(data_type->second)) *
                shape->second.num_elements() <
            kMaxTensorSize) {
      reader.GetTensor(entry.name(), &tensor, status.get());
    }
  }
}

}  // namespace
