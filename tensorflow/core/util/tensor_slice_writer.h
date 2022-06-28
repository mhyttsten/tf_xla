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

// The utility to write checkpoints for google brain tensor ops and v3
// checkpoints for dist_belief.

#ifndef TENSORFLOW_CORE_UTIL_TENSOR_SLICE_WRITER_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_SLICE_WRITER_H_
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
class MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTh() {
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


#include <unordered_map>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice.pb.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

namespace checkpoint {

class TensorSliceWriter {
 public:
  // Abstract interface that TensorSliceWriter uses for building
  class Builder {
   public:
    virtual ~Builder() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTh mht_0(mht_0_v, 216, "", "./tensorflow/core/util/tensor_slice_writer.h", "~Builder");
}
    virtual void Add(StringPiece key, StringPiece value) = 0;
    virtual Status Finish(int64_t* file_size) = 0;
  };
  typedef std::function<Status(const string&, Builder**)> CreateBuilderFunction;

  TensorSliceWriter(const string& filename,
                    CreateBuilderFunction create_builder);
  virtual ~TensorSliceWriter() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTh mht_1(mht_1_v, 227, "", "./tensorflow/core/util/tensor_slice_writer.h", "~TensorSliceWriter");
}
  // Adds a slice. We support float and int32 for now.
  // TODO(yangke): add more supports
  template <typename T>
  Status Add(const string& name, const TensorShape& shape,
             const TensorSlice& slice, const T* data);
  Status Finish();

  // Allocate "num_elements" elements in "ss" and save the data in "data"
  // there.
  template <typename T>
  static Status SaveData(const T* data, int64_t num_elements, SavedSlice* ss);

  static size_t MaxBytesPerElement(DataType dt);

 private:
  static constexpr size_t kMaxMessageBytes = 1LL << 31;
  // Filling in the TensorProto in a SavedSlice will add the following
  // header bytes, in addition to the data:
  // - 1 byte: TensorProto tag and wire format
  // - <= 5 bytes: TensorProto length
  // - 1 byte: Repeated *_val tag and wire format
  // - <= 5 bytes: *_val length
  // However, we add 1KB of slack, to be conservative and guard
  // against other additions to the TensorProto.
  static constexpr size_t kTensorProtoHeaderBytes = 1 << 10;

  const string filename_;
  const CreateBuilderFunction create_builder_;
  const string tmpname_;

  // A mapping from the tensor names to their index in meta_.saved_slice_meta()
  std::unordered_map<string, int> name_to_index_;
  // The metadata that holds all the saved tensor slices.
  SavedTensorSlices sts_;
  // The data to be written to the builder
  std::map<string, string> data_;
  // Total number of slices written
  int slices_;
  TF_DISALLOW_COPY_AND_ASSIGN(TensorSliceWriter);
};

template <typename T>
Status TensorSliceWriter::Add(const string& name, const TensorShape& shape,
                              const TensorSlice& slice, const T* data) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTh mht_2(mht_2_v, 275, "", "./tensorflow/core/util/tensor_slice_writer.h", "TensorSliceWriter::Add");

  // The tensor and the slice have to be compatible
  if (shape.dims() != slice.dims()) {
    return errors::Internal("Incompatible tensor shape and slice: ", "shape = ",
                            shape.DebugString(),
                            ", slice = ", slice.DebugString());
  }
  DataType dt = DataTypeToEnum<T>::value;
  // We need to add an entry for "name" if there isn't an entry already.
  int index = gtl::FindWithDefault(name_to_index_, name, -1);
  if (index >= 0) {
    // The same tensor has been registered -- we verify that the shapes and the
    // type agree.
    const SavedSliceMeta& ssm = sts_.meta().tensor(index);
    CHECK_EQ(name, ssm.name()) << ssm.ShortDebugString();
    TensorShape ssm_shape(ssm.shape());
    if (!shape.IsSameSize(ssm_shape)) {
      return errors::Internal(
          "Mismatching shapes: existing tensor = ", ssm_shape.DebugString(),
          ", trying to add name ", name, ", shape = ", shape.DebugString());
    }
    if (dt != ssm.type()) {
      return errors::Internal(
          "Mismatching types: existing type = ", DataTypeString(ssm.type()),
          ", trying to add name ", name, ", type = ", DataTypeString(dt));
    }
  } else {
    // Insert the new tensor name with the shape information
    index = sts_.meta().tensor_size();
    name_to_index_.insert(std::make_pair(name, index));
    SavedSliceMeta* ssm = sts_.mutable_meta()->add_tensor();
    ssm->set_name(name);
    shape.AsProto(ssm->mutable_shape());
    ssm->set_type(dt);
  }
  // Now we need to add the slice info the list of slices.
  SavedSliceMeta* ssm = sts_.mutable_meta()->mutable_tensor(index);
  slice.AsProto(ssm->add_slice());

  // Now we need to add the real data.
  {
    SavedTensorSlices sts;
    SavedSlice* ss = sts.mutable_data();
    ss->set_name(name);
    slice.AsProto(ss->mutable_slice());
    TensorShape saved_shape(ssm->shape());
    TensorShape sliced_shape;
    TF_RETURN_IF_ERROR(slice.SliceTensorShape(saved_shape, &sliced_shape));
    TF_RETURN_IF_ERROR(SaveData(data, sliced_shape.num_elements(), ss));
    string key = EncodeTensorNameSlice(name, slice);
    // TODO(yangke): consider doing a two-pass thing where the first pass just
    // list the tensor slices we want to save and then another pass to actually
    // set the data. Need to figure out if the interface works well.
    std::pair<string, string> key_value(key, "");
    if (!sts.AppendToString(&key_value.second)) {
      return errors::Internal("Error writing Tensor. Possible size overflow.");
    }
    data_.insert(key_value);
  }
  ++slices_;
  return Status::OK();
}

template <typename T>
Status TensorSliceWriter::SaveData(const T* data, int64_t num_elements,
                                   SavedSlice* ss) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTh mht_3(mht_3_v, 343, "", "./tensorflow/core/util/tensor_slice_writer.h", "TensorSliceWriter::SaveData");

  size_t size_bound =
      ss->ByteSize() + kTensorProtoHeaderBytes +
      (MaxBytesPerElement(DataTypeToEnum<T>::value) * num_elements);
  if (size_bound > kMaxMessageBytes) {
    return errors::InvalidArgument(
        "Tensor slice is too large to serialize (conservative estimate: ",
        size_bound, " bytes)");
  }
  Fill(data, num_elements, ss->mutable_data());
  DCHECK_GE(ss->ByteSize(), 0);
  DCHECK_LE(ss->ByteSize(), size_bound);
  return Status::OK();
}

template <>
Status TensorSliceWriter::SaveData(const tstring* data, int64_t num_elements,
                                   SavedSlice* ss);

// Create a table builder that will write to "filename" in
// tensorflow::io::Table format.  If successful, return OK
// and set "*builder" to the allocated builder.  Otherwise, return a
// non-OK status.
Status CreateTableTensorSliceBuilder(const string& filename,
                                     TensorSliceWriter::Builder** builder);

}  // namespace checkpoint

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_SLICE_WRITER_H_
