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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsqlPSsqlite_query_connectionDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsqlPSsqlite_query_connectionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsqlPSsqlite_query_connectionDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/sql/sqlite_query_connection.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace sql {

SqliteQueryConnection::SqliteQueryConnection() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsqlPSsqlite_query_connectionDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/kernels/data/experimental/sql/sqlite_query_connection.cc", "SqliteQueryConnection::SqliteQueryConnection");
}

SqliteQueryConnection::~SqliteQueryConnection() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsqlPSsqlite_query_connectionDTcc mht_1(mht_1_v, 200, "", "./tensorflow/core/kernels/data/experimental/sql/sqlite_query_connection.cc", "SqliteQueryConnection::~SqliteQueryConnection");

  if (db_ != nullptr) db_->Unref();
}

Status SqliteQueryConnection::Open(const string& data_source_name,
                                   const string& query,
                                   const DataTypeVector& output_types) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("data_source_name: \"" + data_source_name + "\"");
   mht_2_v.push_back("query: \"" + query + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsqlPSsqlite_query_connectionDTcc mht_2(mht_2_v, 211, "", "./tensorflow/core/kernels/data/experimental/sql/sqlite_query_connection.cc", "SqliteQueryConnection::Open");

  if (db_ != nullptr) {
    return errors::FailedPrecondition(
        "Failed to open query connection: Connection already opened.");
  }
  TF_RETURN_IF_ERROR(Sqlite::Open(
      data_source_name, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, &db_));
  query_ = query;
  output_types_ = output_types;
  return Status::OK();
}

Status SqliteQueryConnection::Close() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsqlPSsqlite_query_connectionDTcc mht_3(mht_3_v, 226, "", "./tensorflow/core/kernels/data/experimental/sql/sqlite_query_connection.cc", "SqliteQueryConnection::Close");

  stmt_ = SqliteStatement();
  db_->Unref();
  db_ = nullptr;
  return Status::OK();
}

Status SqliteQueryConnection::GetNext(IteratorContext* ctx,
                                      std::vector<Tensor>* out_tensors,
                                      bool* end_of_sequence) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsqlPSsqlite_query_connectionDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/kernels/data/experimental/sql/sqlite_query_connection.cc", "SqliteQueryConnection::GetNext");

  if (!stmt_) TF_RETURN_IF_ERROR(PrepareQuery());
  TF_RETURN_IF_ERROR(stmt_.Step(end_of_sequence));
  if (!*end_of_sequence) {
    for (int i = 0; i < column_count_; i++) {
      DataType dt = output_types_[i];
      // TODO(mrry): Pass in the `IteratorContext::allocator()`.
      out_tensors->emplace_back(ctx->allocator({}), dt, TensorShape({}));
      FillTensorWithResultSetEntry(dt, i, &out_tensors->back());
    }
  }
  return Status::OK();
}

Status SqliteQueryConnection::PrepareQuery() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsqlPSsqlite_query_connectionDTcc mht_5(mht_5_v, 255, "", "./tensorflow/core/kernels/data/experimental/sql/sqlite_query_connection.cc", "SqliteQueryConnection::PrepareQuery");

  TF_RETURN_IF_ERROR(db_->Prepare(query_, &stmt_));
  int column_count = stmt_.ColumnCount();
  if (column_count != static_cast<int>(output_types_.size())) {
    stmt_ = SqliteStatement();
    return errors::InvalidArgument(tensorflow::strings::Printf(
        "The number of columns in query (%d) must match the number of "
        "elements in output_types (%zu).",
        column_count, output_types_.size()));
  }
  column_count_ = column_count;
  return Status::OK();
}

void SqliteQueryConnection::FillTensorWithResultSetEntry(
    const DataType& data_type, int column_index, Tensor* tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsqlPSsqlite_query_connectionDTcc mht_6(mht_6_v, 273, "", "./tensorflow/core/kernels/data/experimental/sql/sqlite_query_connection.cc", "SqliteQueryConnection::FillTensorWithResultSetEntry");

#define CASE(T, M)                                                 \
  case DataTypeToEnum<T>::value:                                   \
    tensor->scalar<T>()() = static_cast<T>(stmt_.M(column_index)); \
    break;
#define INT_CASE(T) CASE(T, ColumnInt)
#define DOUBLE_CASE(T) CASE(T, ColumnDouble)
#define STRING_CASE(T) CASE(T, ColumnString)
  // clang-format off
  switch (data_type) {
    TF_CALL_int8(INT_CASE)
    TF_CALL_uint8(INT_CASE)
    TF_CALL_int16(INT_CASE)
    TF_CALL_uint16(INT_CASE)
    TF_CALL_int32(INT_CASE)
    TF_CALL_uint32(INT_CASE)
    TF_CALL_int64(INT_CASE)
    TF_CALL_uint64(INT_CASE)
    TF_CALL_float(DOUBLE_CASE)
    TF_CALL_double(DOUBLE_CASE)
    TF_CALL_tstring(STRING_CASE)
    case DT_BOOL:
      tensor->scalar<bool>()() = stmt_.ColumnInt(column_index) != 0;
      break;
    // Error preemptively thrown by SqlDatasetOp::MakeDataset in this case.
    default:
      LOG(ERROR)
          << "Use of unsupported TensorFlow data type by 'SqlQueryConnection': "
          << DataTypeString(data_type) << ".";
  }
  // clang-format on
}

}  // namespace sql
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
