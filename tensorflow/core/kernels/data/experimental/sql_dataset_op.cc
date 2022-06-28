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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc() {
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
#include <utility>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/experimental/sql/driver_manager.h"
#include "tensorflow/core/kernels/data/experimental/sql/query_connection.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class SqlDatasetOp : public DatasetOpKernel {
 public:
  explicit SqlDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "SqlDatasetOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    for (const DataType& dt : output_types_) {
      OP_REQUIRES(ctx,
                  dt == DT_STRING || dt == DT_INT8 || dt == DT_INT16 ||
                      dt == DT_INT32 || dt == DT_INT64 || dt == DT_UINT8 ||
                      dt == DT_UINT16 || dt == DT_BOOL || dt == DT_DOUBLE,
                  errors::InvalidArgument(
                      "Each element of `output_types_` must be one of: "
                      "DT_STRING, DT_INT8, DT_INT16, DT_INT32, DT_INT64, "
                      "DT_UINT8, DT_UINT16, DT_BOOL, DT_DOUBLE "));
    }
    for (const PartialTensorShape& pts : output_shapes_) {
      OP_REQUIRES(ctx, pts.dims() == 0,
                  errors::InvalidArgument(
                      "Each element of `output_shapes_` must be a scalar."));
    }
  }
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "MakeDataset");

    tstring driver_name;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<tstring>(ctx, "driver_name", &driver_name));

    tstring data_source_name;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, "data_source_name",
                                                     &data_source_name));

    tstring query;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, "query", &query));

    // TODO(b/64276826) Change this check when we add support for other
    // databases.
    OP_REQUIRES(ctx, driver_name == "sqlite",
                errors::InvalidArgument(tensorflow::strings::Printf(
                    "The database type, %s, is not supported by SqlDataset. "
                    "The set of supported databases is: {'sqlite'}.",
                    driver_name.c_str())));

    *output = new Dataset(ctx, driver_name, data_source_name, query,
                          output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const string& driver_name,
            const string& data_source_name, const string& query,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          driver_name_(driver_name),
          data_source_name_(data_source_name),
          query_(query),
          output_types_(output_types),
          output_shapes_(output_shapes) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("driver_name: \"" + driver_name + "\"");
   mht_2_v.push_back("data_source_name: \"" + data_source_name + "\"");
   mht_2_v.push_back("query: \"" + query + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_2(mht_2_v, 266, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "Dataset");
}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Sql")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "output_dtypes");

      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_4(mht_4_v, 284, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "output_shapes");

      return output_shapes_;
    }

    string DebugString() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_5(mht_5_v, 291, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "DebugString");
 return "SqlDatasetOp::Dataset"; }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_6(mht_6_v, 297, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "InputDatasets");

      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_7(mht_7_v, 304, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "CheckExternalState");
 return Status::OK(); }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_8(mht_8_v, 312, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "AsGraphDefInternal");

      Node* driver_name_node;
      TF_RETURN_IF_ERROR(b->AddScalar(driver_name_, &driver_name_node));
      Node* data_source_name_node;
      TF_RETURN_IF_ERROR(
          b->AddScalar(data_source_name_, &data_source_name_node));
      Node* query_node;
      TF_RETURN_IF_ERROR(b->AddScalar(query_, &query_node));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {driver_name_node, data_source_name_node, query_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_9(mht_9_v, 332, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "Iterator");
}
      ~Iterator() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_10(mht_10_v, 336, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "~Iterator");

        if (query_connection_initialized_) {
          Status s = query_connection_->Close();
          if (!s.ok()) {
            LOG(WARNING) << "Failed to close query connection: " << s;
          }
        }
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_11(mht_11_v, 350, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);
        if (!query_connection_initialized_) {
          TF_RETURN_IF_ERROR(InitializeQueryConnection());
        }
        Status status = Status::OK();
        if (!end_of_sequence_) {
          next_calls_++;
          status =
              query_connection_->GetNext(ctx, out_tensors, &end_of_sequence_);
        }
        *end_of_sequence = end_of_sequence_;
        return status;
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeSourceNode(std::move(args));
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_12(mht_12_v, 375, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "SaveInternal");

        mutex_lock l(mu_);
        if (query_connection_initialized_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("next_calls"), next_calls_));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_13(mht_13_v, 388, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        if (reader->Contains(full_name("next_calls"))) {
          TF_RETURN_IF_ERROR(InitializeQueryConnection());
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("next_calls"), &next_calls_));
          int64_t rem_next_calls = next_calls_;
          std::vector<Tensor> out_tensors;
          end_of_sequence_ = false;
          while (rem_next_calls--) {
            TF_RETURN_IF_ERROR(query_connection_->GetNext(ctx, &out_tensors,
                                                          &end_of_sequence_));
            out_tensors.clear();
          }
        } else {
          query_connection_initialized_ = false;
          end_of_sequence_ = false;
        }
        return Status::OK();
      }

     private:
      Status InitializeQueryConnection() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsql_dataset_opDTcc mht_14(mht_14_v, 413, "", "./tensorflow/core/kernels/data/experimental/sql_dataset_op.cc", "InitializeQueryConnection");

        query_connection_initialized_ = true;
        end_of_sequence_ = false;
        query_connection_ =
            sql::DriverManager::CreateQueryConnection(dataset()->driver_name_);
        Status s = query_connection_->Open(dataset()->data_source_name_,
                                           dataset()->query_,
                                           dataset()->output_types_);
        next_calls_ = 0;
        if (!s.ok()) {
          LOG(WARNING) << "Failed to connect to database: " << s;
          return s;
        }
        return Status::OK();
      }

      mutex mu_;
      // TODO(b/129062371): explore ways to seek into a SQLite databases.
      int64_t next_calls_ TF_GUARDED_BY(mu_) = 0;
      std::unique_ptr<sql::QueryConnection> query_connection_
          TF_GUARDED_BY(mu_);
      bool query_connection_initialized_ TF_GUARDED_BY(mu_) = false;
      bool end_of_sequence_ TF_GUARDED_BY(mu_) = false;
    };
    const tstring driver_name_;
    const tstring data_source_name_;
    const tstring query_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("SqlDataset").Device(DEVICE_CPU), SqlDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalSqlDataset").Device(DEVICE_CPU),
                        SqlDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
