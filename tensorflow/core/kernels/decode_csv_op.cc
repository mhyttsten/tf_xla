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
class MHTracer_DTPStensorflowPScorePSkernelsPSdecode_csv_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_csv_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdecode_csv_opDTcc() {
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

// See docs in ../ops/parsing_ops.cc.
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

class DecodeCSVOp : public OpKernel {
 public:
  explicit DecodeCSVOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_csv_opDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/kernels/decode_csv_op.cc", "DecodeCSVOp");

    string delim;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("OUT_TYPE", &out_type_));
    OP_REQUIRES(ctx, out_type_.size() < std::numeric_limits<int>::max(),
                errors::InvalidArgument("Out type too large"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_delim", &delim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_quote_delim", &use_quote_delim_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("select_cols", &select_cols_));
    OP_REQUIRES(
        ctx, out_type_.size() == select_cols_.size() || select_cols_.empty(),
        errors::InvalidArgument("select_cols should match output size"));
    select_all_cols_ = select_cols_.empty();
    for (int i = 1; i < select_cols_.size(); i++) {
      OP_REQUIRES(ctx, select_cols_[i - 1] < select_cols_[i],
                  errors::InvalidArgument(
                      "select_cols should be strictly increasing indices"));
    }
    OP_REQUIRES(
        ctx, select_cols_.empty() || select_cols_.front() >= 0,
        errors::InvalidArgument("select_cols should be non-negative indices"));
    OP_REQUIRES(ctx, delim.size() == 1,
                errors::InvalidArgument("field_delim should be only 1 char"));
    delim_ = delim[0];
    OP_REQUIRES_OK(ctx, ctx->GetAttr("na_value", &na_value_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_csv_opDTcc mht_1(mht_1_v, 228, "", "./tensorflow/core/kernels/decode_csv_op.cc", "Compute");

    const Tensor* records;
    OpInputList record_defaults;

    OP_REQUIRES_OK(ctx, ctx->input("records", &records));
    OP_REQUIRES_OK(ctx, ctx->input_list("record_defaults", &record_defaults));

    for (int i = 0; i < record_defaults.size(); ++i) {
      OP_REQUIRES(ctx, record_defaults[i].dims() <= 1,
                  errors::InvalidArgument(
                      "Each record default should be at most rank 1"));
      OP_REQUIRES(ctx, record_defaults[i].NumElements() < 2,
                  errors::InvalidArgument(
                      "There should only be 1 default per field but field ", i,
                      " has ", record_defaults[i].NumElements()));
    }

    auto records_t = records->flat<tstring>();
    int64_t records_size = records_t.size();

    OpOutputList output;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));

    for (int i = 0; i < static_cast<int>(out_type_.size()); ++i) {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, output.allocate(i, records->shape(), &out));
    }

    for (int64_t i = 0; i < records_size; ++i) {
      const StringPiece record(records_t(i));
      std::vector<string> fields;
      ExtractFields(ctx, record, &fields);
      OP_REQUIRES(ctx, fields.size() == out_type_.size(),
                  errors::InvalidArgument("Expect ", out_type_.size(),
                                          " fields but have ", fields.size(),
                                          " in record ", i));

      // Check each field in the record
      for (int f = 0; f < static_cast<int>(out_type_.size()); ++f) {
        const DataType& dtype = out_type_[f];
        switch (dtype) {
          case DT_INT32: {
            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));

              output[f]->flat<int32>()(i) = record_defaults[f].flat<int32>()(0);
            } else {
              int32_t value;
              OP_REQUIRES(ctx, strings::safe_strto32(fields[f], &value),
                          errors::InvalidArgument(
                              "Field ", f, " in record ", i,
                              " is not a valid int32: ", fields[f]));
              output[f]->flat<int32>()(i) = value;
            }
            break;
          }
          case DT_INT64: {
            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));

              output[f]->flat<int64_t>()(i) =
                  record_defaults[f].flat<int64_t>()(0);
            } else {
              int64_t value;
              OP_REQUIRES(ctx, strings::safe_strto64(fields[f], &value),
                          errors::InvalidArgument(
                              "Field ", f, " in record ", i,
                              " is not a valid int64: ", fields[f]));
              output[f]->flat<int64_t>()(i) = value;
            }
            break;
          }
          case DT_FLOAT: {
            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));
              output[f]->flat<float>()(i) = record_defaults[f].flat<float>()(0);
            } else {
              float value;
              OP_REQUIRES(ctx, strings::safe_strtof(fields[f], &value),
                          errors::InvalidArgument(
                              "Field ", f, " in record ", i,
                              " is not a valid float: ", fields[f]));
              output[f]->flat<float>()(i) = value;
            }
            break;
          }
          case DT_DOUBLE: {
            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));
              output[f]->flat<double>()(i) =
                  record_defaults[f].flat<double>()(0);
            } else {
              double value;
              OP_REQUIRES(ctx, strings::safe_strtod(fields[f], &value),
                          errors::InvalidArgument(
                              "Field ", f, " in record ", i,
                              " is not a valid double: ", fields[f]));
              output[f]->flat<double>()(i) = value;
            }
            break;
          }
          case DT_STRING: {
            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));
              output[f]->flat<tstring>()(i) =
                  record_defaults[f].flat<tstring>()(0);
            } else {
              output[f]->flat<tstring>()(i) = std::move(fields[f]);
            }
            break;
          }
          default:
            OP_REQUIRES(ctx, false,
                        errors::InvalidArgument("csv: data type ", dtype,
                                                " not supported in field ", f));
        }
      }
    }
  }

 private:
  std::vector<DataType> out_type_;
  std::vector<int64_t> select_cols_;
  char delim_;
  bool use_quote_delim_;
  bool select_all_cols_;
  string na_value_;

  void ExtractFields(OpKernelContext* ctx, StringPiece input,
                     std::vector<string>* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_csv_opDTcc mht_2(mht_2_v, 385, "", "./tensorflow/core/kernels/decode_csv_op.cc", "ExtractFields");

    int64_t current_idx = 0;
    int64_t num_fields_parsed = 0;
    int64_t selector_idx = 0;  // Keep track of index into select_cols

    if (!input.empty()) {
      while (static_cast<size_t>(current_idx) < input.size()) {
        if (input[current_idx] == '\n' || input[current_idx] == '\r') {
          current_idx++;
          continue;
        }

        bool quoted = false;
        bool include =
            (select_all_cols_ || select_cols_[selector_idx] ==
                                     static_cast<size_t>(num_fields_parsed));

        if (use_quote_delim_ && input[current_idx] == '"') {
          quoted = true;
          current_idx++;
        }

        // This is the body of the field;
        string field;
        if (!quoted) {
          while (static_cast<size_t>(current_idx) < input.size() &&
                 input[current_idx] != delim_) {
            OP_REQUIRES(ctx,
                        (!use_quote_delim_ || input[current_idx] != '"') &&
                            input[current_idx] != '\n' &&
                            input[current_idx] != '\r',
                        errors::InvalidArgument(
                            "Unquoted fields cannot have quotes/CRLFs inside"));
            if (include) field += input[current_idx];
            current_idx++;
          }

          // Go to next field or the end
          current_idx++;
        } else if (use_quote_delim_) {
          // Quoted field needs to be ended with '"' and delim or end
          while (
              (static_cast<size_t>(current_idx) < input.size() - 1) &&
              (input[current_idx] != '"' || input[current_idx + 1] != delim_)) {
            if (input[current_idx] != '"') {
              if (include) field += input[current_idx];
              current_idx++;
            } else {
              OP_REQUIRES(
                  ctx, input[current_idx + 1] == '"',
                  errors::InvalidArgument("Quote inside a string has to be "
                                          "escaped by another quote"));
              if (include) field += '"';
              current_idx += 2;
            }
          }

          OP_REQUIRES(
              ctx,
              (static_cast<size_t>(current_idx) < input.size() &&
               input[current_idx] == '"' &&
               (static_cast<size_t>(current_idx) == input.size() - 1 ||
                input[current_idx + 1] == delim_)),
              errors::InvalidArgument("Quoted field has to end with quote "
                                      "followed by delim or end"));

          current_idx += 2;
        }

        num_fields_parsed++;
        if (include) {
          result->push_back(field);
          selector_idx++;
          if (selector_idx == select_cols_.size()) return;
        }
      }

      bool include =
          (select_all_cols_ || select_cols_[selector_idx] ==
                                   static_cast<size_t>(num_fields_parsed));
      // Check if the last field is missing
      if (include && input[input.size() - 1] == delim_)
        result->push_back(string());
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("DecodeCSV").Device(DEVICE_CPU), DecodeCSVOp);

}  // namespace tensorflow
