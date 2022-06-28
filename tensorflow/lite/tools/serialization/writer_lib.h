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
// Library to write a flatbuffer of a currently loaded TFLite model/subgraph.

#ifndef TENSORFLOW_LITE_TOOLS_SERIALIZATION_WRITER_LIB_H_
#define TENSORFLOW_LITE_TOOLS_SERIALIZATION_WRITER_LIB_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTh() {
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

#include <iostream>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"
#include "tensorflow/lite/tools/serialization/enum_mapping.h"
#include "tensorflow/lite/version.h"

namespace tflite {

struct OpCode {
  int builtin;
  std::string custom;
};

// Forward declaration.
class SubgraphWriter;

// Handles writing a full TFLite model (with 1 or more subgraphs) to a
// serialized TF lite file format.
// TODO(b/174708523): Support custom I/O or unused tensors later.
class ModelWriter {
 public:
  // Construct a writer for the specified `interpreter`. Then, use
  // .Write() or .GetBuffer(...) to extract the data.
  explicit ModelWriter(Interpreter* interpreter);

  // Same as above, except takes subgraphs as input.
  explicit ModelWriter(const std::vector<Subgraph*>& subgraphs);

  // For initializing the ModelWriter internal data.
  void Init(const std::vector<Subgraph*>& subgraphs);

  // Get a buffer and size of a serialized flatbuffer.
  TfLiteStatus GetBuffer(std::unique_ptr<uint8_t[]>* out, size_t* size);
  // Write the serialized flatbuffer to the prescribed `filename`.
  TfLiteStatus Write(const std::string& filename);

  // Specifies unused tensors on the target subgraph.
  void SetUnusedTensors(int subgraph_index,
                        const std::set<int>& unused_tensors);

  // Specifies custom inputs, outputs, and execution_plan to target subgraph.
  TfLiteStatus SetCustomInputOutput(int subgraph_index,
                                    const std::vector<int>& inputs,
                                    const std::vector<int>& outputs,
                                    const std::vector<int>& execution_plan);

 private:
  template <class T>
  using Offset = flatbuffers::Offset<T>;
  Offset<flatbuffers::Vector<Offset<OperatorCode>>> CreateOpCodeTable(
      flatbuffers::FlatBufferBuilder* fbb);
  Offset<flatbuffers::Vector<Offset<Buffer>>> ExportBuffers(
      flatbuffers::FlatBufferBuilder* fbb);

  // List of subgraph writers owned by this model writer.
  // There is one subgraph writer for each subgraph in the model.
  std::vector<SubgraphWriter> subgraph_writers_;

  // This data corresponds to the overall model (rather than individual
  // subgraphs), so we define common fields. Keep track of byte buffers
  std::vector<std::pair<const uint8_t*, size_t>> buffers_;
  // List of used opcodes
  std::vector<OpCode> opcodes_;
  absl::flat_hash_map<int, int> builtin_op_to_opcode_;
};

// Handles writing TensorFlow Lite running subgraph to a serialized TF lite
// file format.
// TODO(b/174708523): Reconcile into ModelWriter?
class SubgraphWriter {
 public:
  friend class ModelWriter;

  typedef flatbuffers::Offset<Operator> (*CustomWriter)(
      flatbuffers::FlatBufferBuilder* fbb, Subgraph* subgraph, int node_index,
      flatbuffers::Offset<flatbuffers::Vector<uint8_t>>* output_options,
      CustomOptionsFormat* custom_options_format);

  // Construct a subgraph writer for the specified `subgraph`. Then, use
  // .Write() or .GetBuffer(...) to extract the data.
  explicit SubgraphWriter(Subgraph* subgraph)
      : subgraph_(subgraph),
        inputs_(subgraph->inputs()),
        outputs_(subgraph->outputs()),
        execution_plan_(subgraph->execution_plan()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTh mht_0(mht_0_v, 279, "", "./tensorflow/lite/tools/serialization/writer_lib.h", "SubgraphWriter");

    buffers_ = &buffers_data_;
    opcodes_ = &opcodes_data_;
    builtin_op_to_opcode_ = &builtin_op_to_opcode_data_;
    buffers_->push_back(std::make_pair(nullptr, 0));
  }

  // Get a buffer and size of a serialized flatbuffer.
  TfLiteStatus GetBuffer(std::unique_ptr<uint8_t[]>* out, size_t* size);
  // Write the serialized flatbuffer to the prescribed `filename`.
  TfLiteStatus Write(const std::string& filename);
  // Registers a custom writer for a custom op. The customization allows the
  // caller to change the custom data.
  TfLiteStatus RegisterCustomWriter(const std::string& custom_name,
                                    CustomWriter custom_writer);
  // Tensors that are unused and shouldn't be written.
  void SetUnusedTensors(const std::set<int>& unused_tensors) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTh mht_1(mht_1_v, 298, "", "./tensorflow/lite/tools/serialization/writer_lib.h", "SetUnusedTensors");

    unused_tensors_ = unused_tensors;
  }
  // Sets custom inputs, outputs, and execution_plan so that a portion of the
  // subgraph is written to the buffer instead of the whole subgraph.
  TfLiteStatus SetCustomInputOutput(const std::vector<int>& inputs,
                                    const std::vector<int>& outputs,
                                    const std::vector<int>& execution_plan);

 private:
  // Used by ModelWriter.
  explicit SubgraphWriter(
      Subgraph* subgraph,
      std::vector<std::pair<const uint8_t*, size_t>>* external_buffers,
      std::vector<OpCode>* external_opcodes,
      absl::flat_hash_map<int, int>* external_builtin_op_to_opcode)
      : subgraph_(subgraph),
        inputs_(subgraph->inputs()),
        outputs_(subgraph->outputs()),
        execution_plan_(subgraph->execution_plan()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTh mht_2(mht_2_v, 320, "", "./tensorflow/lite/tools/serialization/writer_lib.h", "SubgraphWriter");

    buffers_ = external_buffers;
    opcodes_ = external_opcodes;
    builtin_op_to_opcode_ = external_builtin_op_to_opcode;
    buffers_->push_back(std::make_pair(nullptr, 0));
  }

  // Used by ModelWriter to populate data specific to this subgraph.
  // Global stuff (like opcodes & buffers) is populated into buffers_, opcodes_,
  // etc. & populated in the Flatbuffer by ModelWriter.
  flatbuffers::Offset<SubGraph> PopulateAndGetOffset(
      flatbuffers::FlatBufferBuilder* builder,
      const std::string& subgraph_name);

  template <class T>
  using Offset = flatbuffers::Offset<T>;
  template <class T_OUTPUT, class T_INPUT>
  Offset<flatbuffers::Vector<T_OUTPUT>> ExportVector(
      flatbuffers::FlatBufferBuilder* fbb, const T_INPUT& v);
  Offset<flatbuffers::Vector<Offset<Tensor>>> ExportTensors(
      flatbuffers::FlatBufferBuilder* fbb);
  Offset<flatbuffers::Vector<Offset<Operator>>> ExportOperators(
      flatbuffers::FlatBufferBuilder* fbb);
  Offset<flatbuffers::Vector<Offset<OperatorCode>>> CreateOpCodeTable(
      flatbuffers::FlatBufferBuilder* fbb);
  Offset<flatbuffers::Vector<Offset<Buffer>>> ExportBuffers(
      flatbuffers::FlatBufferBuilder* fbb);

  template <class T>
  std::vector<int> RemapTensorIndicesToWritten(const T& input);

  // Checks if given `input`, `output`, and `execution_plan` represents a valid
  // model within the Subgraph.
  TfLiteStatus CheckInputOutput(const std::vector<int>& inputs,
                                const std::vector<int>& outputs,
                                const std::vector<int>& execution_plan);

  int GetOpCodeForBuiltin(int builtin_op_index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTh mht_3(mht_3_v, 360, "", "./tensorflow/lite/tools/serialization/writer_lib.h", "GetOpCodeForBuiltin");

    // auto it = builtin_op_to_opcode_.find(builtin_op_index);
    std::pair<decltype(builtin_op_to_opcode_data_)::iterator, bool> result =
        builtin_op_to_opcode_->insert(
            std::make_pair(builtin_op_index, opcodes_->size()));
    if (result.second) {
      opcodes_->push_back({builtin_op_index, ""});
    }
    return result.first->second;
  }

  int GetOpCodeForCustom(const std::string& custom_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("custom_name: \"" + custom_name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTh mht_4(mht_4_v, 375, "", "./tensorflow/lite/tools/serialization/writer_lib.h", "GetOpCodeForCustom");

    std::pair<decltype(custom_op_to_opcode_)::iterator, bool> result =
        custom_op_to_opcode_.insert(
            std::make_pair(custom_name, opcodes_->size()));
    if (result.second) {
      opcodes_->push_back({BuiltinOperator_CUSTOM, custom_name});
    }
    return result.first->second;
  }

  // The subgraph we are writing
  Subgraph* subgraph_;
  // Input tensor indices to be written.
  std::vector<int> inputs_;
  // Output tensor indices to be written.
  std::vector<int> outputs_;
  // Order of nodes to be written.
  std::vector<int> execution_plan_;
  // List of op codes and mappings from builtin or custom op to opcode
  std::set<int> unused_tensors_;
  // For every tensor index in the subgraph, the index in the written.
  // This is different due to temporary and unused tensors not being written.
  std::vector<int> tensor_to_written_tensor_;
  std::unordered_map<std::string, int> custom_op_to_opcode_;
  std::unordered_map<std::string, CustomWriter> custom_op_to_writer_;

  // We use pointers for these, since they may be provided by ModelWriter.
  // Keep track of byte buffers
  std::vector<std::pair<const uint8_t*, size_t>>* buffers_;
  // List of used opcodes
  std::vector<OpCode>* opcodes_;
  absl::flat_hash_map<int, int>* builtin_op_to_opcode_;

  // These are used if SubgraphWriter is being used directly.
  std::vector<std::pair<const uint8_t*, size_t>> buffers_data_;
  // List of used opcodes
  std::vector<OpCode> opcodes_data_;
  absl::flat_hash_map<int, int> builtin_op_to_opcode_data_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_SERIALIZATION_WRITER_LIB_H_
