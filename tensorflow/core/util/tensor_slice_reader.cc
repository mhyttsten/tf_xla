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
class MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc() {
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

#include "tensorflow/core/util/tensor_slice_reader.h"

#include <utility>
#include <vector>

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/iterator.h"
#include "tensorflow/core/lib/io/table.h"
#include "tensorflow/core/lib/io/table_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_slice_util.h"

namespace tensorflow {

namespace checkpoint {

TensorSliceReader::Table::~Table() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::Table::~Table");
}

namespace {
class TensorSliceReaderTable : public TensorSliceReader::Table {
 public:
  // Takes ownership of 'f'.
  explicit TensorSliceReaderTable(RandomAccessFile* f, table::Table* t)
      : file_(f), table_(t) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReaderTable");
}

  ~TensorSliceReaderTable() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_2(mht_2_v, 223, "", "./tensorflow/core/util/tensor_slice_reader.cc", "~TensorSliceReaderTable");

    delete table_;
    delete file_;
  }

  bool Get(const string& key, string* value) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_3(mht_3_v, 232, "", "./tensorflow/core/util/tensor_slice_reader.cc", "Get");

    std::unique_ptr<table::Iterator> iter(table_->NewIterator());
    iter->Seek(key);
    if (iter->Valid() && iter->key() == key) {
      StringPiece v = iter->value();
      value->assign(v.data(), v.size());
      return true;
    } else {
      return false;
    }
  }

 private:
  RandomAccessFile* file_;  // Owns.
  table::Table* table_;
};
}  // namespace

Status OpenTableTensorSliceReader(const string& fname,
                                  TensorSliceReader::Table** result) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_4(mht_4_v, 255, "", "./tensorflow/core/util/tensor_slice_reader.cc", "OpenTableTensorSliceReader");

  *result = nullptr;
  Env* env = Env::Default();
  std::unique_ptr<RandomAccessFile> f;
  Status s = env->NewRandomAccessFile(fname, &f);
  if (s.ok()) {
    uint64 file_size;
    s = env->GetFileSize(fname, &file_size);
    if (s.ok()) {
      table::Options options;
      table::Table* table;
      s = table::Table::Open(options, f.get(), file_size, &table);
      if (s.ok()) {
        *result = new TensorSliceReaderTable(f.release(), table);
        return Status::OK();
      } else {
        s = errors::CreateWithUpdatedMessage(
            s, strings::StrCat(s.error_message(),
                               ": perhaps your file is in a different "
                               "file format and you need to use a "
                               "different restore operator?"));
      }
    }
  }
  LOG(WARNING) << "Could not open " << fname << ": " << s;
  return s;
}

TensorSliceReader::TensorSliceReader(const string& filepattern)
    : TensorSliceReader(filepattern, OpenTableTensorSliceReader,
                        kLoadAllShards) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("filepattern: \"" + filepattern + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_5(mht_5_v, 289, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::TensorSliceReader");
}

TensorSliceReader::TensorSliceReader(const string& filepattern,
                                     OpenTableFunction open_function)
    : TensorSliceReader(filepattern, std::move(open_function), kLoadAllShards) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("filepattern: \"" + filepattern + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_6(mht_6_v, 297, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::TensorSliceReader");

}

TensorSliceReader::TensorSliceReader(const string& filepattern,
                                     OpenTableFunction open_function,
                                     int preferred_shard)
    : filepattern_(filepattern), open_function_(std::move(open_function)) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("filepattern: \"" + filepattern + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_7(mht_7_v, 307, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::TensorSliceReader");

  VLOG(1) << "TensorSliceReader for " << filepattern;
  Status s = Env::Default()->GetMatchingPaths(filepattern, &fnames_);
  if (!s.ok()) {
    status_ = errors::InvalidArgument(
        "Unsuccessful TensorSliceReader constructor: "
        "Failed to get matching files on ",
        filepattern, ": ", s.ToString());
    return;
  }
  if (fnames_.empty()) {
    status_ = errors::NotFound(
        "Unsuccessful TensorSliceReader constructor: "
        "Failed to find any matching files for ",
        filepattern);
    return;
  }
  sss_.resize(fnames_.size());
  for (size_t shard = 0; shard < fnames_.size(); ++shard) {
    fname_to_index_.insert(std::make_pair(fnames_[shard], shard));
  }
  if (preferred_shard == kLoadAllShards || fnames_.size() == 1 ||
      static_cast<size_t>(preferred_shard) >= fnames_.size()) {
    LoadAllShards();
  } else {
    VLOG(1) << "Loading shard " << preferred_shard << " for " << filepattern_;
    LoadShard(preferred_shard);
  }
}

void TensorSliceReader::LoadShard(int shard) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_8(mht_8_v, 340, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::LoadShard");

  CHECK_LT(shard, sss_.size());
  if (sss_[shard] || !status_.ok()) {
    return;  // Already loaded, or invalid.
  }
  string value;
  SavedTensorSlices sts;
  const string fname = fnames_[shard];
  VLOG(1) << "Reading meta data from file " << fname << "...";
  Table* table;
  Status s = open_function_(fname, &table);
  if (!s.ok()) {
    status_ = errors::DataLoss("Unable to open table file ", fname, ": ",
                               s.ToString());
    return;
  }
  sss_[shard].reset(table);
  if (!(table->Get(kSavedTensorSlicesKey, &value) &&
        ParseProtoUnlimited(&sts, value))) {
    status_ = errors::Internal(
        "Failed to find the saved tensor slices at the beginning of the "
        "checkpoint file: ",
        fname);
    return;
  }
  status_ = CheckVersions(sts.meta().versions(), TF_CHECKPOINT_VERSION,
                          TF_CHECKPOINT_VERSION_MIN_PRODUCER, "Checkpoint",
                          "checkpoint");
  if (!status_.ok()) return;
  for (const SavedSliceMeta& ssm : sts.meta().tensor()) {
    TensorShape ssm_shape;
    status_ = TensorShape::BuildTensorShapeBase(ssm.shape(), &ssm_shape);
    if (!status_.ok()) return;
    for (const TensorSliceProto& tsp : ssm.slice()) {
      TensorSlice ss_slice;
      status_ = TensorSlice::BuildTensorSlice(tsp, &ss_slice);
      if (!status_.ok()) return;
      status_ = RegisterTensorSlice(ssm.name(), ssm_shape, ssm.type(), fname,
                                    ss_slice, &tensors_);
      if (!status_.ok()) return;
    }
  }
}

void TensorSliceReader::LoadAllShards() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_9(mht_9_v, 387, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::LoadAllShards");

  VLOG(1) << "Loading all shards for " << filepattern_;
  for (size_t i = 0; i < fnames_.size() && status_.ok(); ++i) {
    LoadShard(i);
  }
  all_shards_loaded_ = true;
}

const TensorSliceSet* TensorSliceReader::FindTensorSlice(
    const string& name, const TensorSlice& slice,
    std::vector<std::pair<TensorSlice, string>>* details) const {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_10(mht_10_v, 401, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::FindTensorSlice");

  const TensorSliceSet* tss = gtl::FindPtrOrNull(tensors_, name);
  if (tss && !tss->QueryMeta(slice, details)) {
    return nullptr;
  }
  return tss;
}

TensorSliceReader::~TensorSliceReader() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_11(mht_11_v, 412, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::~TensorSliceReader");

  for (auto& temp : tensors_) {
    delete temp.second;
  }
  tensors_.clear();
}

bool TensorSliceReader::HasTensor(const string& name, TensorShape* shape,
                                  DataType* type) const {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_12(mht_12_v, 424, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::HasTensor");

  mutex_lock l(mu_);
  const TensorSliceSet* tss = gtl::FindPtrOrNull(tensors_, name);
  if (!tss && !all_shards_loaded_) {
    VLOG(1) << "Did not find tensor in preferred shard, loading all shards: "
            << name;
    LoadAllShards();
    tss = gtl::FindPtrOrNull(tensors_, name);
  }
  if (tss) {
    if (shape) {
      *shape = tss->shape();
    }
    if (type) {
      *type = tss->type();
    }
    return true;
  } else {
    return false;
  }
}

Status TensorSliceReader::GetTensor(
    const string& name, std::unique_ptr<tensorflow::Tensor>* out_tensor) const {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_13(mht_13_v, 451, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::GetTensor");

  DataType type;
  TensorShape shape;
  TensorSlice slice;
  {
    mutex_lock l(mu_);
    const TensorSliceSet* tss = gtl::FindPtrOrNull(tensors_, name);
    if (tss == nullptr) {
      return errors::NotFound(name, " not found in checkpoint file");
    }

    if (tss->Slices().size() > 1) {
      // TODO(sherrym): Support multi-slice checkpoints.
      return errors::Unimplemented("Sliced checkpoints are not supported");
    }

    type = tss->type();
    shape = tss->shape();
    slice = tss->Slices().begin()->second.slice;
  }

  std::unique_ptr<tensorflow::Tensor> t(new tensorflow::Tensor);
  Status s = tensorflow::Tensor::BuildTensor(type, shape, t.get());
  if (!s.ok()) return s;
  bool success = false;

#define READER_COPY(dt)                                                  \
  case dt:                                                               \
    success = CopySliceData(name, slice,                                 \
                            t->flat<EnumToDataType<dt>::Type>().data()); \
    break;

  switch (type) {
    READER_COPY(DT_FLOAT);
    READER_COPY(DT_DOUBLE);
    READER_COPY(DT_INT32);
    READER_COPY(DT_UINT8);
    READER_COPY(DT_INT16);
    READER_COPY(DT_INT8);
    READER_COPY(DT_INT64);
    READER_COPY(DT_STRING);
    default:
      return errors::Unimplemented("Data type not supported");
  }
#undef READER_COPY

  if (!success) {
    return errors::NotFound(name, " not found in checkpoint file");
  }
  std::swap(*out_tensor, t);

  return Status::OK();
}

TensorSliceReader::VarToShapeMap TensorSliceReader::GetVariableToShapeMap()
    const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_14(mht_14_v, 509, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::GetVariableToShapeMap");

  VarToShapeMap name_to_shape;
  if (status().ok()) {
    for (auto& e : Tensors()) {
      name_to_shape[e.first] = e.second->shape();
    }
  }
  return name_to_shape;
}

TensorSliceReader::VarToDataTypeMap
TensorSliceReader::GetVariableToDataTypeMap() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_15(mht_15_v, 523, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::GetVariableToDataTypeMap");

  VarToDataTypeMap name_to_dtype;
  if (status().ok()) {
    for (auto& e : Tensors()) {
      name_to_dtype[e.first] = e.second->type();
    }
  }
  return name_to_dtype;
}

const string TensorSliceReader::DebugString() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_readerDTcc mht_16(mht_16_v, 536, "", "./tensorflow/core/util/tensor_slice_reader.cc", "TensorSliceReader::DebugString");

  string shape_str;
  if (status().ok()) {
    for (const auto& e : Tensors()) {
      strings::StrAppend(&shape_str, e.first, " (",
                         DataType_Name(e.second->type()), ") ",
                         e.second->shape().DebugString());
      // Indicates if a tensor has more than 1 slice (i.e., it's partitioned).
      const int num_slices = e.second->Slices().size();
      if (num_slices > 1) {
        strings::StrAppend(&shape_str, ", ", num_slices, " slices");
      }
      strings::StrAppend(&shape_str, "\n");
    }
  }
  return shape_str;
}

}  // namespace checkpoint

}  // namespace tensorflow
