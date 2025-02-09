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
class MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc() {
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

#include "tensorflow/core/kernels/lookup_util.h"

#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/platform/refcount.h"

namespace tensorflow {
namespace lookup {
namespace {

using InitializerSerializer =
    ::tensorflow::lookup::InitializableLookupTable::InitializerSerializer;

static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */
static const int kLineNumber = -1;
static const int kWholeLine = -2;

Status GetNumLinesInTextFile(Env* env, const string& vocab_file,
                             int64_t* num_lines) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("vocab_file: \"" + vocab_file + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/lookup_util.cc", "GetNumLinesInTextFile");

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(vocab_file, &file));

  io::InputBuffer input_buffer(file.get(), kInputBufferSize);
  string line;
  Status s = input_buffer.ReadLine(&line);
  int64_t next_id = 0;
  while (s.ok()) {
    next_id++;
    s = input_buffer.ReadLine(&line);
  }
  if (!errors::IsOutOfRange(s)) {
    return s;
  }
  *num_lines = next_id;
  return Status::OK();
}

// Iterator that reads a text file. Each iteration process one line, it parses
// the line and populates the keys and values tensors used for initialization
// with a single key and corresponding value.
//
// What information of the line to populate the key or values is specified by
// providing key_index and value_index.
class TextFileLineIterator
    : public InitializableLookupTable::InitTableIterator {
 public:
  TextFileLineIterator()
      : valid_(false),
        vocab_size_(-1),
        status_(errors::FailedPrecondition("Not initialized")) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_1(mht_1_v, 244, "", "./tensorflow/core/kernels/lookup_util.cc", "TextFileLineIterator");
}

  // Initialize iterator.
  //
  // Prepares the file 'filename' and sets the data types to return the keys and
  // values tensors. It requires the indices of the tokens in the line given a
  // delimiter to specify where to pick the data from.
  //
  // - Index -2 means the entire line as string.
  // - Index -1 means the line number stored in int64.
  // - Index >= 0 represent index (starting at zero) of the split line based on
  //   delimiter.
  Status Init(const string& filename, int64_t vocab_size, char delimiter,
              DataType key_dtype, int64_t key_index, DataType value_dtype,
              int64_t value_index, int64_t offset, Env* env) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("filename: \"" + filename + "\"");
   mht_2_v.push_back("delimiter: '" + std::string(1, delimiter) + "'");
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_2(mht_2_v, 263, "", "./tensorflow/core/kernels/lookup_util.cc", "Init");

    filename_ = filename;
    vocab_size_ = vocab_size;
    delimiter_ = delimiter;
    key_ = Tensor(key_dtype, TensorShape({}));
    value_ = Tensor(value_dtype, TensorShape({}));
    key_index_ = key_index;
    value_index_ = value_index;
    env_ = env;

    status_ = env->NewRandomAccessFile(filename_, &file_);
    if (!status_.ok()) return status_;

    input_buffer_.reset(new io::InputBuffer(file_.get(), kInputBufferSize));
    valid_ = true;
    next_id_ = 0;
    offset_ = offset;
    ignore_split_ = std::max(key_index_, value_index_) < 0;
    Next();
    return status_;
  }

  void Next() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_3(mht_3_v, 288, "", "./tensorflow/core/kernels/lookup_util.cc", "Next");

    if (!valid_) return;

    string line;
    status_ = input_buffer_->ReadLine(&line);
    if (!status_.ok()) {
      if (errors::IsOutOfRange(status_) && vocab_size_ != -1 &&
          next_id_ != vocab_size_) {
        status_ = errors::InvalidArgument("Invalid vocab_size in ", filename_,
                                          ": expected ", vocab_size_,
                                          " but got ", next_id_);
      }
      valid_ = false;
      return;
    }
    if (vocab_size_ != -1 && next_id_ >= vocab_size_) {
      LOG(WARNING) << "Truncated " << filename_ << " before its end at "
                   << vocab_size_ << " records.";
      LOG(WARNING) << "next_id_  : " << next_id_;
      status_ = errors::OutOfRange("Finished reading ", vocab_size_,
                                   " of lines from ", filename_);
      valid_ = false;
      return;
    }
    if (line.empty()) {
      status_ = errors::InvalidArgument("Invalid content in ", filename_,
                                        ": empty line found at position ",
                                        input_buffer_->Tell(), ".");
      valid_ = false;
      return;
    }

    std::vector<string> tokens;
    if (!ignore_split_) {
      tokens = str_util::Split(line, delimiter_);
      const auto expected_size =
          static_cast<size_t>(std::max(key_index_, value_index_) + 1);
      if (tokens.size() < expected_size) {
        status_ = errors::InvalidArgument(
            "Invalid number of columns in ", filename_, " line ", next_id_,
            " (", line, ") : expected at least ", expected_size, " got ",
            tokens.size());
        valid_ = false;
        return;
      }
    }

    status_ = SetValue(line, tokens, key_index_, &key_);
    if (!status_.ok()) {
      valid_ = false;
      return;
    }
    status_ = SetValue(line, tokens, value_index_, &value_);
    if (!status_.ok()) {
      valid_ = false;
      return;
    }

    next_id_++;
  }

  bool Valid() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_4(mht_4_v, 352, "", "./tensorflow/core/kernels/lookup_util.cc", "Valid");
 return valid_; }

  const Tensor& keys() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_5(mht_5_v, 357, "", "./tensorflow/core/kernels/lookup_util.cc", "keys");
 return key_; }

  const Tensor& values() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_6(mht_6_v, 362, "", "./tensorflow/core/kernels/lookup_util.cc", "values");
 return value_; }

  Status status() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_7(mht_7_v, 367, "", "./tensorflow/core/kernels/lookup_util.cc", "status");
 return status_; }

  int64_t total_size() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_8(mht_8_v, 372, "", "./tensorflow/core/kernels/lookup_util.cc", "total_size");

    if (vocab_size_ == -1) {
      int64_t new_size = -1;
      Status status = GetNumLinesInTextFile(env_, filename_, &new_size);
      if (!status.ok()) {
        LOG(WARNING) << "Unable to get line count: " << status;
        new_size = -1;
      }
      *const_cast<int64_t*>(&vocab_size_) = new_size;
    }
    return vocab_size_;
  }

 private:
  Tensor key_;
  Tensor value_;
  bool valid_;  // true if the iterator points to an existing range.
  int64_t key_index_;
  int64_t value_index_;
  Env* env_;
  int64_t next_id_;
  int64_t offset_;
  int64_t vocab_size_;
  string filename_;
  char delimiter_;
  Status status_;
  bool ignore_split_;
  std::unique_ptr<RandomAccessFile> file_;  // must outlive input_buffer_
  std::unique_ptr<io::InputBuffer> input_buffer_;

  // Set the corresponding value from line or tokens based on 'index' into the
  // tensor 't'. The value is transformed to the given data type 'dtype'.
  Status SetValue(const string& line, const std::vector<string>& tokens,
                  int64_t index, Tensor* tensor) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("line: \"" + line + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_9(mht_9_v, 409, "", "./tensorflow/core/kernels/lookup_util.cc", "SetValue");

    if (index == kLineNumber) {
      tensor->flat<int64_t>()(0) = next_id_ + offset_;
      return Status::OK();
    }
    const string& token = (index == kWholeLine) ? line : tokens[index];
    const DataType& dtype = tensor->dtype();
    switch (dtype) {
      case DT_INT32: {
        int32_t value;
        if (!strings::safe_strto32(token.c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", token, " in line ", next_id_,
                                         " is not a valid int32.");
        }
        tensor->flat<int32>()(0) = value + offset_;
      } break;
      case DT_INT64: {
        int64_t value;
        if (!strings::safe_strto64(token.c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", token, " in line ", next_id_,
                                         " is not a valid int64.");
        }
        tensor->flat<int64_t>()(0) = value;
      } break;
      case DT_FLOAT: {
        float value;
        if (!strings::safe_strtof(token.c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", token, " in line ", next_id_,
                                         " is not a valid float.");
        }
        tensor->flat<float>()(0) = value;
      } break;
      case DT_DOUBLE: {
        double value;
        if (!strings::safe_strtod(token.c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", token, " in line ", next_id_,
                                         " is not a valid double.");
        }
        tensor->flat<double>()(0) = value;
      } break;
      case DT_STRING:
        tensor->flat<tstring>()(0) = token;
        break;
      default:
        valid_ = false;
        return errors::InvalidArgument("Data type ", DataTypeString(dtype),
                                       " not supported.");
    }
    return Status::OK();
  }

  TF_DISALLOW_COPY_AND_ASSIGN(TextFileLineIterator);
};

Status GetTableHandle(StringPiece input_name, OpKernelContext* ctx,
                      string* container, string* table_handle) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_10(mht_10_v, 471, "", "./tensorflow/core/kernels/lookup_util.cc", "GetTableHandle");

  {
    mutex* mu;
    TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
    mutex_lock l(*mu);
    Tensor tensor;
    TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Lookup table handle must be scalar, but had shape: ",
          tensor.shape().DebugString());
    }
    auto h = tensor.flat<tstring>();
    *container = h(0);
    *table_handle = h(1);
  }
  return Status::OK();
}

}  // namespace

Status GetResourceLookupTable(StringPiece input_name, OpKernelContext* ctx,
                              LookupInterface** table) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_11(mht_11_v, 496, "", "./tensorflow/core/kernels/lookup_util.cc", "GetResourceLookupTable");

  const Tensor* handle_tensor;
  TF_RETURN_IF_ERROR(ctx->input(input_name, &handle_tensor));
  const ResourceHandle& handle = handle_tensor->scalar<ResourceHandle>()();
  return LookupResource(ctx, handle, table);
}

Status GetReferenceLookupTable(StringPiece input_name, OpKernelContext* ctx,
                               LookupInterface** table) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_12(mht_12_v, 507, "", "./tensorflow/core/kernels/lookup_util.cc", "GetReferenceLookupTable");

  string container;
  string table_handle;
  TF_RETURN_IF_ERROR(
      GetTableHandle(input_name, ctx, &container, &table_handle));
  return ctx->resource_manager()->Lookup(container, table_handle, table);
}

Status GetLookupTable(StringPiece input_name, OpKernelContext* ctx,
                      LookupInterface** table) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_13(mht_13_v, 519, "", "./tensorflow/core/kernels/lookup_util.cc", "GetLookupTable");

  DataType handle_dtype;
  TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &handle_dtype));
  if (handle_dtype == DT_RESOURCE) {
    return GetResourceLookupTable(input_name, ctx, table);
  } else {
    return GetReferenceLookupTable(input_name, ctx, table);
  }
}

Status GetInitializableLookupTable(StringPiece input_name, OpKernelContext* ctx,
                                   InitializableLookupTable** table) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_14(mht_14_v, 533, "", "./tensorflow/core/kernels/lookup_util.cc", "GetInitializableLookupTable");

  LookupInterface* lookup_table;
  DataType handle_dtype;
  TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &handle_dtype));
  if (handle_dtype == DT_RESOURCE) {
    ResourceHandle handle;
    TF_RETURN_IF_ERROR(HandleFromInput(ctx, input_name, &handle));
    TF_RETURN_IF_ERROR(LookupResource(ctx, handle, &lookup_table));
    *table = lookup_table->GetInitializableLookupTable();
    if (*table == nullptr) {
      lookup_table->Unref();
      return errors::InvalidArgument("Table ", handle.container(), " ",
                                     handle.name(), " is not initializable");
    }
  } else {
    string container;
    string table_handle;
    TF_RETURN_IF_ERROR(
        GetTableHandle(input_name, ctx, &container, &table_handle));
    TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(container, table_handle,
                                                       &lookup_table));
    *table = lookup_table->GetInitializableLookupTable();
    if (*table == nullptr) {
      lookup_table->Unref();
      return errors::InvalidArgument("Table ", container, " ", table_handle,
                                     " is not initializable");
    }
  }
  return Status::OK();
}

Status CheckTableDataTypes(const LookupInterface& table, DataType key_dtype,
                           DataType value_dtype, const string& table_name) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("table_name: \"" + table_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_15(mht_15_v, 569, "", "./tensorflow/core/kernels/lookup_util.cc", "CheckTableDataTypes");

  if (table.key_dtype() != key_dtype || table.value_dtype() != value_dtype) {
    return errors::InvalidArgument(
        "Conflicting key/value dtypes ", DataTypeString(key_dtype), "->",
        DataTypeString(value_dtype), " with ",
        DataTypeString(table.key_dtype()), "-",
        DataTypeString(table.value_dtype()), " for table ", table_name);
  }
  return Status::OK();
}

// Helper function to initialize an InitializableLookupTable from a text file.
Status InitializeTableFromTextFile(const string& filename, int64_t vocab_size,
                                   char delimiter, int32_t key_index,
                                   int32_t value_index, int64_t offset,
                                   Env* env, InitializableLookupTable* table) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("filename: \"" + filename + "\"");
   mht_16_v.push_back("delimiter: '" + std::string(1, delimiter) + "'");
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_16(mht_16_v, 589, "", "./tensorflow/core/kernels/lookup_util.cc", "InitializeTableFromTextFile");

  return InitializeTableFromTextFile(filename, vocab_size, delimiter, key_index,
                                     value_index, offset, env,
                                     /*serializer=*/nullptr, table);
}

Status InitializeTableFromTextFile(
    const string& filename, int64_t vocab_size, char delimiter,
    int32_t key_index, int32_t value_index, int64_t offset, Env* env,
    std::unique_ptr<InitializableLookupTable::InitializerSerializer> serializer,
    InitializableLookupTable* table) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("filename: \"" + filename + "\"");
   mht_17_v.push_back("delimiter: '" + std::string(1, delimiter) + "'");
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_utilDTcc mht_17(mht_17_v, 604, "", "./tensorflow/core/kernels/lookup_util.cc", "InitializeTableFromTextFile");

  if (key_index == kLineNumber && table->key_dtype() != DT_INT64) {
    return errors::InvalidArgument(
        "Key index for line number requires table key dtype of int64, got ",
        DataTypeString(table->key_dtype()));
  }
  const DataType& key_dtype = table->key_dtype();
  const DataType& value_dtype = table->value_dtype();
  if (key_index == kWholeLine && !DataTypeIsInteger(key_dtype) &&
      key_dtype != DT_STRING) {
    return errors::InvalidArgument(
        "Key index for whole line requires string or integer table key, got ",
        DataTypeString(table->key_dtype()));
  }
  if (value_index == kLineNumber && value_dtype != DT_INT64) {
    return errors::InvalidArgument(
        "Value index for line number requires table value dtype of int64, got ",
        DataTypeString(table->value_dtype()));
  }
  if (value_index == kWholeLine && !DataTypeIsInteger(value_dtype) &&
      value_dtype != DT_STRING) {
    return errors::InvalidArgument(
        "Value index for whole line requires table value dtype of integer or "
        "string, got ",
        DataTypeString(table->value_dtype()));
  }

  TextFileLineIterator iter;
  TF_RETURN_IF_ERROR(iter.Init(filename, vocab_size, delimiter, key_dtype,
                               key_index, value_dtype, value_index, offset,
                               env));
  // For initialization from files, ignore if the table is already
  // initialized. The table shared name should contain the filename to
  // avoid trying to initialize the same table from the same file at the same
  // time.
  Status s = table->Initialize(iter, std::move(serializer));
  if (errors::IsFailedPrecondition(s) && table->is_initialized()) {
    LOG(INFO) << "Table trying to initialize from file " << filename
              << " is already initialized.";
    return Status::OK();
  }
  return s;
}

}  // namespace lookup
}  // namespace tensorflow
