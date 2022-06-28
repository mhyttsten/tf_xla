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
class MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc() {
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

#include "tensorflow/core/util/tensor_slice_writer.h"

#include <utility>

#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

namespace checkpoint {

namespace {

class TableBuilder : public TensorSliceWriter::Builder {
 public:
  TableBuilder(const string& name, WritableFile* f) : name_(name), file_(f) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/util/tensor_slice_writer.cc", "TableBuilder");

    table::Options option;
    option.compression = table::kNoCompression;
    builder_.reset(new table::TableBuilder(option, f));
  }
  void Add(StringPiece key, StringPiece val) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/util/tensor_slice_writer.cc", "Add");

    builder_->Add(key, val);
  }
  Status Finish(int64_t* file_size) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/util/tensor_slice_writer.cc", "Finish");

    *file_size = -1;
    Status s = builder_->Finish();
    if (s.ok()) {
      s = file_->Close();
      if (s.ok()) {
        *file_size = builder_->FileSize();
      }
    }
    if (!s.ok()) {
      s = errors::Internal("Error writing (tmp) checkpoint file: ", name_, ": ",
                           s.error_message());
    }
    builder_.reset();
    file_.reset();
    return s;
  }

 private:
  string name_;
  std::unique_ptr<WritableFile> file_;
  std::unique_ptr<table::TableBuilder> builder_;
};
}  // anonymous namespace

Status CreateTableTensorSliceBuilder(const string& name,
                                     TensorSliceWriter::Builder** builder) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/util/tensor_slice_writer.cc", "CreateTableTensorSliceBuilder");

  *builder = nullptr;
  std::unique_ptr<WritableFile> f;
  Status s = Env::Default()->NewWritableFile(name, &f);
  if (s.ok()) {
    *builder = new TableBuilder(name, f.release());
    return Status::OK();
  } else {
    return s;
  }
}

TensorSliceWriter::TensorSliceWriter(const string& filename,
                                     CreateBuilderFunction create_builder)
    : filename_(filename),
      create_builder_(std::move(create_builder)),
      tmpname_(strings::StrCat(filename, ".tempstate", random::New64())),
      slices_(0) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc mht_4(mht_4_v, 273, "", "./tensorflow/core/util/tensor_slice_writer.cc", "TensorSliceWriter::TensorSliceWriter");

  VersionDef* versions = sts_.mutable_meta()->mutable_versions();
  versions->set_producer(TF_CHECKPOINT_VERSION);
  versions->set_min_consumer(TF_CHECKPOINT_VERSION_MIN_CONSUMER);
}

Status TensorSliceWriter::Finish() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc mht_5(mht_5_v, 282, "", "./tensorflow/core/util/tensor_slice_writer.cc", "TensorSliceWriter::Finish");

  Builder* b;
  Status s = create_builder_(tmpname_, &b);
  if (!s.ok()) {
    delete b;
    return s;
  }
  std::unique_ptr<Builder> builder(b);

  // We save the saved tensor slice metadata as the first element.
  string meta;
  sts_.AppendToString(&meta);
  builder->Add(kSavedTensorSlicesKey, meta);

  // Go through all the data and add them
  for (const auto& x : data_) {
    builder->Add(x.first, x.second);
  }

  int64_t file_size;
  s = builder->Finish(&file_size);
  // We need to rename the file to the proper name
  if (s.ok()) {
    s = Env::Default()->RenameFile(tmpname_, filename_);
    if (s.ok()) {
      VLOG(1) << "Written " << slices_ << " slices for "
              << sts_.meta().tensor_size() << " tensors (" << file_size
              << " bytes) to " << filename_;
    } else {
      LOG(ERROR) << "Failed to rename file " << tmpname_ << " to " << filename_;
    }
  } else {
    Env::Default()->DeleteFile(tmpname_).IgnoreError();
  }
  return s;
}

/* static */
size_t TensorSliceWriter::MaxBytesPerElement(DataType dt) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc mht_6(mht_6_v, 323, "", "./tensorflow/core/util/tensor_slice_writer.cc", "TensorSliceWriter::MaxBytesPerElement");

  switch (dt) {
    case DT_FLOAT:
      return 4;
    case DT_DOUBLE:
      return 8;
    case DT_INT32:
      return 10;
    case DT_UINT8:
      return 2;
    case DT_INT16:
      return 10;
    case DT_INT8:
      return 10;
    case DT_COMPLEX64:
      return 8;
    case DT_INT64:
      return 10;
    case DT_BOOL:
      return 1;
    case DT_QINT8:
      return 10;
    case DT_QUINT8:
      return 2;
    case DT_QINT32:
      return 10;
    case DT_QINT16:
      return 10;
    case DT_QUINT16:
      return 3;
    case DT_UINT16:
      return 3;
    case DT_COMPLEX128:
      return 16;
    case DT_HALF:
      return 3;
    case DT_INVALID:
    case DT_STRING:
    case DT_BFLOAT16:
    default:
      LOG(FATAL) << "MaxBytesPerElement not implemented for dtype: " << dt;
  }
  return 0;
}

template <>
Status TensorSliceWriter::SaveData(const tstring* data, int64_t num_elements,
                                   SavedSlice* ss) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writerDTcc mht_7(mht_7_v, 373, "", "./tensorflow/core/util/tensor_slice_writer.cc", "TensorSliceWriter::SaveData");

  size_t size_bound = ss->ByteSize() + kTensorProtoHeaderBytes +
                      (num_elements * MaxBytesPerElement(DT_INT32));
  for (int64_t i = 0; i < num_elements; ++i) {
    size_bound += data[i].size();
  }
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

}  // namespace checkpoint

}  // namespace tensorflow
