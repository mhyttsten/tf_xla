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
class MHTracer_DTPStensorflowPScPScheckpoint_readerDTcc {
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
   MHTracer_DTPStensorflowPScPScheckpoint_readerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPScheckpoint_readerDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/checkpoint_reader.h"

#include <unordered_set>
#include <utility>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {
namespace checkpoint {

class TensorSliceReader;

CheckpointReader::CheckpointReader(const string& filename, TF_Status* status)
    : reader_(nullptr),
      v2_reader_(nullptr),
      var_to_shape_map_(nullptr),
      var_to_data_type_map_(nullptr) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScPScheckpoint_readerDTcc mht_0(mht_0_v, 206, "", "./tensorflow/c/checkpoint_reader.cc", "CheckpointReader::CheckpointReader");

  // Depending on whether this is a V2 ckpt, initializes "reader_" or
  // "v2_reader_".
  std::vector<string> v2_path;
  if (Env::Default()->GetMatchingPaths(MetaFilename(filename), &v2_path).ok() &&
      !v2_path.empty()) {
    v2_reader_.reset(
        new BundleReader(Env::Default(), filename /* prefix to a V2 ckpt */));
    if (!v2_reader_->status().ok()) {
      Set_TF_Status_from_Status(status, v2_reader_->status());
      return;
    }
    auto result = BuildV2VarMaps();
    var_to_shape_map_.swap(result.first);
    var_to_data_type_map_.swap(result.second);
  } else {
    reader_.reset(new TensorSliceReader(filename));
    if (!reader_->status().ok()) {
      Set_TF_Status_from_Status(status, reader_->status());
      return;
    }
    var_to_shape_map_.reset(
        new TensorSliceReader::VarToShapeMap(reader_->GetVariableToShapeMap()));
    var_to_data_type_map_.reset(new TensorSliceReader::VarToDataTypeMap(
        reader_->GetVariableToDataTypeMap()));
  }
}

bool CheckpointReader::HasTensor(const string& name) const {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScPScheckpoint_readerDTcc mht_1(mht_1_v, 238, "", "./tensorflow/c/checkpoint_reader.cc", "CheckpointReader::HasTensor");

  if (reader_ != nullptr) {
    return reader_->HasTensor(name, nullptr, nullptr);
  }
  return v2_reader_->Contains(name);
}

const TensorSliceReader::VarToShapeMap&
CheckpointReader::GetVariableToShapeMap() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPScheckpoint_readerDTcc mht_2(mht_2_v, 249, "", "./tensorflow/c/checkpoint_reader.cc", "CheckpointReader::GetVariableToShapeMap");

  CHECK(var_to_shape_map_);
  return *var_to_shape_map_;
}

const TensorSliceReader::VarToDataTypeMap&
CheckpointReader::GetVariableToDataTypeMap() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPScheckpoint_readerDTcc mht_3(mht_3_v, 258, "", "./tensorflow/c/checkpoint_reader.cc", "CheckpointReader::GetVariableToDataTypeMap");

  CHECK(var_to_data_type_map_);
  return *var_to_data_type_map_;
}

const string CheckpointReader::DebugString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPScheckpoint_readerDTcc mht_4(mht_4_v, 266, "", "./tensorflow/c/checkpoint_reader.cc", "CheckpointReader::DebugString");

  if (reader_ != nullptr) return reader_->DebugString();
  return v2_reader_->DebugString();
}

void CheckpointReader::GetTensor(
    const string& name, std::unique_ptr<tensorflow::Tensor>* out_tensor,
    TF_Status* out_status) const {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScPScheckpoint_readerDTcc mht_5(mht_5_v, 277, "", "./tensorflow/c/checkpoint_reader.cc", "CheckpointReader::GetTensor");

  Status status;
  if (reader_ != nullptr) {
    status = reader_->GetTensor(name, out_tensor);
  } else {
    tensorflow::DataType dtype;
    tensorflow::TensorShape shape;
    status = v2_reader_->LookupDtypeAndShape(name, &dtype, &shape);
    if (status.ok()) {
      out_tensor->reset(new Tensor(dtype, shape));
      status = v2_reader_->Lookup(name, out_tensor->get());
      if (!status.ok()) out_tensor->reset();
    }
  }
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
}

std::pair<std::unique_ptr<TensorSliceReader::VarToShapeMap>,
          std::unique_ptr<TensorSliceReader::VarToDataTypeMap>>
CheckpointReader::BuildV2VarMaps() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPScheckpoint_readerDTcc mht_6(mht_6_v, 301, "", "./tensorflow/c/checkpoint_reader.cc", "CheckpointReader::BuildV2VarMaps");

  CHECK(v2_reader_ != nullptr);
  CHECK(v2_reader_->status().ok());

  // First pass: filters out the entries of the slices.
  std::unordered_set<string> filtered_keys;
  BundleEntryProto entry;
  v2_reader_->Seek(kHeaderEntryKey);
  for (v2_reader_->Next(); v2_reader_->Valid(); v2_reader_->Next()) {
    CHECK(entry.ParseFromArray(v2_reader_->value().data(),
                               v2_reader_->value().size()))
        << entry.InitializationErrorString();
    for (int i = 0; i < entry.slices_size(); ++i) {
      const auto& slice_proto = entry.slices(i);
      CHECK(filtered_keys
                .insert(EncodeTensorNameSlice(
                    string(v2_reader_->key()) /* full var's name */,
                    TensorSlice(slice_proto)))
                .second);
    }
  }

  // Second pass: adds the entries, ignoring the filtered keys.
  std::unique_ptr<TensorSliceReader::VarToShapeMap> var_to_shape_map(
      new TensorSliceReader::VarToShapeMap);
  std::unique_ptr<TensorSliceReader::VarToDataTypeMap> var_to_data_type_map(
      new TensorSliceReader::VarToDataTypeMap);
  v2_reader_->Seek(kHeaderEntryKey);
  for (v2_reader_->Next(); v2_reader_->Valid(); v2_reader_->Next()) {
    if (filtered_keys.count(string(v2_reader_->key())) > 0) continue;
    CHECK(entry.ParseFromArray(v2_reader_->value().data(),
                               v2_reader_->value().size()))
        << entry.InitializationErrorString();
    string key(v2_reader_->key());
    (*var_to_shape_map)[key] = TensorShape(entry.shape());
    (*var_to_data_type_map)[key] = DataType(entry.dtype());
  }
  // The returned pointers are owned by the caller.
  return std::make_pair(std::move(var_to_shape_map),
                        std::move(var_to_data_type_map));
}

}  // namespace checkpoint
}  // namespace tensorflow
