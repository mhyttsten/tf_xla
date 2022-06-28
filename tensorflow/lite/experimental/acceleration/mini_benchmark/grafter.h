/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINIBENCHMARK_GRAFTER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINIBENCHMARK_GRAFTER_H_
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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTh {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTh() {
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


#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/idl.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"

namespace tflite {
namespace acceleration {

// Combines the given models into one, using the FlatBufferBuilder.
//
// This is useful for constructing models that contain validation data and
// metrics.
//
// The model fields are handled as follows:
// - version is set to 3
// - operator codes are concatenated (no deduplication)
// - subgraphs are concatenated in order, rewriting operator and buffer indices
// to match the combined model. Subgraph names are set from 'subgraph_names'
// - description is taken from first model
// - buffers are concatenated
// - metadata buffer is left unset
// - metadata are concatenated
// - signature_defs are taken from the first model (as they refer to the main
// subgraph).
absl::Status CombineModels(flatbuffers::FlatBufferBuilder* fbb,
                           std::vector<const Model*> models,
                           std::vector<std::string> subgraph_names,
                           const reflection::Schema* schema);

// Convenience methods for copying flatbuffer Tables and Vectors.
//
// These are used by CombineModels above, but also needed for constructing
// validation subgraphs to be combined with models.
class FlatbufferHelper {
 public:
  FlatbufferHelper(flatbuffers::FlatBufferBuilder* fbb,
                   const reflection::Schema* schema);
  template <typename T>
  absl::Status CopyTableToVector(const std::string& name, const T* o,
                                 std::vector<flatbuffers::Offset<T>>* v) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTh mht_0(mht_0_v, 234, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/grafter.h", "CopyTableToVector");

    auto copied = CopyTable(name, o);
    if (!copied.ok()) {
      return copied.status();
    }
    v->push_back(*copied);
    return absl::OkStatus();
  }
  template <typename T>
  absl::StatusOr<flatbuffers::Offset<T>> CopyTable(const std::string& name,
                                                   const T* o) {
    if (o == nullptr) return 0;
    const reflection::Object* def = FindObject(name);
    if (!def) {
      return absl::NotFoundError(
          absl::StrFormat("Type %s not found in schema", name));
    }
    // We want to use the general copying mechanisms that operate on
    // flatbuffers::Table pointers. Flatbuffer types are not directly
    // convertible to Table, as they inherit privately from table.
    // For type* -> Table*, use reinterpret cast.
    const flatbuffers::Table* ot =
        reinterpret_cast<const flatbuffers::Table*>(o);
    // For Offset<Table *> -> Offset<type>, rely on uoffset_t conversion to
    // any flatbuffers::Offset<T>.
    return flatbuffers::CopyTable(*fbb_, *schema_, *def, *ot).o;
  }
  template <typename int_type>
  flatbuffers::Offset<flatbuffers::Vector<int_type>> CopyIntVector(
      const flatbuffers::Vector<int_type>* from) {
    if (from == nullptr) {
      return 0;
    }
    std::vector<int_type> v{from->cbegin(), from->cend()};
    return fbb_->CreateVector(v);
  }
  const reflection::Object* FindObject(const std::string& name);

 private:
  flatbuffers::FlatBufferBuilder* fbb_;
  const reflection::Schema* schema_;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINIBENCHMARK_GRAFTER_H_
