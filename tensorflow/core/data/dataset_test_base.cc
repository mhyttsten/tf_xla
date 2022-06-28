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
class MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/dataset_test_base.h"

#include <algorithm>
#include <complex>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace data {

string ToString(CompressionType compression_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_0(mht_0_v, 255, "", "./tensorflow/core/data/dataset_test_base.cc", "ToString");

  switch (compression_type) {
    case CompressionType::ZLIB:
      return "ZLIB";
    case CompressionType::GZIP:
      return "GZIP";
    case CompressionType::RAW:
      return "RAW";
    case CompressionType::UNCOMPRESSED:
      return "";
  }
}

io::ZlibCompressionOptions GetZlibCompressionOptions(
    CompressionType compression_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_1(mht_1_v, 272, "", "./tensorflow/core/data/dataset_test_base.cc", "GetZlibCompressionOptions");

  switch (compression_type) {
    case CompressionType::ZLIB:
      return io::ZlibCompressionOptions::DEFAULT();
    case CompressionType::GZIP:
      return io::ZlibCompressionOptions::GZIP();
    case CompressionType::RAW:
      return io::ZlibCompressionOptions::RAW();
    case CompressionType::UNCOMPRESSED:
      LOG(WARNING) << "ZlibCompressionOptions does not have an option for "
                   << ToString(compression_type);
      return io::ZlibCompressionOptions::DEFAULT();
  }
}

Status WriteDataToFile(const string& filename, const char* data) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("filename: \"" + filename + "\"");
   mht_2_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_2(mht_2_v, 292, "", "./tensorflow/core/data/dataset_test_base.cc", "WriteDataToFile");

  return WriteDataToFile(filename, data, CompressionParams());
}

Status WriteDataToFile(const string& filename, const char* data,
                       const CompressionParams& params) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("filename: \"" + filename + "\"");
   mht_3_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_3(mht_3_v, 302, "", "./tensorflow/core/data/dataset_test_base.cc", "WriteDataToFile");

  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file_writer;
  TF_RETURN_IF_ERROR(env->NewWritableFile(filename, &file_writer));
  if (params.compression_type == CompressionType::UNCOMPRESSED) {
    TF_RETURN_IF_ERROR(file_writer->Append(data));
  } else if (params.compression_type == CompressionType::ZLIB ||
             params.compression_type == CompressionType::GZIP ||
             params.compression_type == CompressionType::RAW) {
    auto zlib_compression_options =
        GetZlibCompressionOptions(params.compression_type);
    io::ZlibOutputBuffer out(file_writer.get(), params.input_buffer_size,
                             params.output_buffer_size,
                             zlib_compression_options);
    TF_RETURN_IF_ERROR(out.Init());
    TF_RETURN_IF_ERROR(out.Append(data));
    TF_RETURN_IF_ERROR(out.Flush());
    TF_RETURN_IF_ERROR(out.Close());
  } else {
    return tensorflow::errors::InvalidArgument(
        "Unsupported compression_type: ", ToString(params.compression_type));
  }

  TF_RETURN_IF_ERROR(file_writer->Flush());
  TF_RETURN_IF_ERROR(file_writer->Close());

  return Status::OK();
}

Status WriteDataToTFRecordFile(const string& filename,
                               const std::vector<absl::string_view>& records,
                               const CompressionParams& params) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_4(mht_4_v, 337, "", "./tensorflow/core/data/dataset_test_base.cc", "WriteDataToTFRecordFile");

  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file_writer;
  TF_RETURN_IF_ERROR(env->NewWritableFile(filename, &file_writer));
  auto options = io::RecordWriterOptions::CreateRecordWriterOptions(
      ToString(params.compression_type));
  options.zlib_options.input_buffer_size = params.input_buffer_size;
  io::RecordWriter record_writer(file_writer.get(), options);
  for (const auto& record : records) {
    TF_RETURN_IF_ERROR(record_writer.WriteRecord(record));
  }
  TF_RETURN_IF_ERROR(record_writer.Flush());
  TF_RETURN_IF_ERROR(record_writer.Close());
  TF_RETURN_IF_ERROR(file_writer->Flush());
  TF_RETURN_IF_ERROR(file_writer->Close());
  return Status::OK();
}

template <typename T>
Status IsEqual(const Tensor& t1, const Tensor& t2) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_5(mht_5_v, 359, "", "./tensorflow/core/data/dataset_test_base.cc", "IsEqual");

  if (t1.dtype() != t2.dtype()) {
    return tensorflow::errors::Internal(
        "Two tensors have different dtypes: ", DataTypeString(t1.dtype()),
        " vs. ", DataTypeString(t2.dtype()));
  }
  if (!t1.IsSameSize(t2)) {
    return tensorflow::errors::Internal(
        "Two tensors have different shapes: ", t1.shape().DebugString(),
        " vs. ", t2.shape().DebugString());
  }

  auto flat_t1 = t1.flat<T>();
  auto flat_t2 = t2.flat<T>();
  auto length = flat_t1.size();

  for (int i = 0; i < length; ++i) {
    if (flat_t1(i) != flat_t2(i)) {
      return tensorflow::errors::Internal(
          "Two tensors have different values "
          "at [",
          i, "]: ", flat_t1(i), " vs. ", flat_t2(i));
    }
  }
  return Status::OK();
}

DatasetOpsTestBase::DatasetOpsTestBase()
    : device_(DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0")),
      device_type_(DEVICE_CPU),
      cpu_num_(kDefaultCPUNum),
      thread_num_(kDefaultThreadNum) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_6(mht_6_v, 393, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::DatasetOpsTestBase");

  allocator_ = device_->GetAllocator(AllocatorAttributes());
}

DatasetOpsTestBase::~DatasetOpsTestBase() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_7(mht_7_v, 400, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::~DatasetOpsTestBase");

  if (dataset_) {
    dataset_->Unref();
  }
}

Status DatasetOpsTestBase::ExpectEqual(const Tensor& a, const Tensor& b) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_8(mht_8_v, 409, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::ExpectEqual");

  switch (a.dtype()) {
#define CASE(DT)                           \
  case DataTypeToEnum<DT>::value:          \
    TF_RETURN_IF_ERROR(IsEqual<DT>(a, b)); \
    break;
    TF_CALL_NUMBER_TYPES(CASE);
    TF_CALL_tstring(CASE);
    // TODO(feihugis): figure out how to support variant tensors.
#undef CASE
    default:
      return errors::Internal("Unsupported dtype: ", a.dtype());
  }
  return Status::OK();
}

template <typename T>
bool compare(const Tensor& t1, const Tensor& t2) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_9(mht_9_v, 429, "", "./tensorflow/core/data/dataset_test_base.cc", "compare");

  auto flat_t1 = t1.flat<T>();
  auto flat_t2 = t2.flat<T>();
  auto length = std::min(flat_t1.size(), flat_t2.size());
  for (int i = 0; i < length; ++i) {
    if (flat_t1(i) < flat_t2(i)) return true;
    if (flat_t1(i) > flat_t2(i)) return false;
  }
  return flat_t1.size() < length;
}

Status DatasetOpsTestBase::ExpectEqual(std::vector<Tensor> produced_tensors,
                                       std::vector<Tensor> expected_tensors,
                                       bool compare_order) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_10(mht_10_v, 445, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::ExpectEqual");

  if (produced_tensors.size() != expected_tensors.size()) {
    return Status(tensorflow::errors::Internal(
        "The two tensor vectors have different size (", produced_tensors.size(),
        " v.s. ", expected_tensors.size(), ")"));
  }

  if (produced_tensors.empty()) return Status::OK();
  if (produced_tensors[0].dtype() != expected_tensors[0].dtype()) {
    return Status(tensorflow::errors::Internal(
        "The two tensor vectors have different dtypes (",
        produced_tensors[0].dtype(), " v.s. ", expected_tensors[0].dtype(),
        ")"));
  }

  if (!compare_order) {
    const DataType& dtype = produced_tensors[0].dtype();
    switch (dtype) {
#define CASE(DT)                                                \
  case DT:                                                      \
    std::sort(produced_tensors.begin(), produced_tensors.end(), \
              compare<EnumToDataType<DT>::Type>);               \
    std::sort(expected_tensors.begin(), expected_tensors.end(), \
              compare<EnumToDataType<DT>::Type>);               \
    break;
      CASE(DT_FLOAT);
      CASE(DT_DOUBLE);
      CASE(DT_INT32);
      CASE(DT_UINT8);
      CASE(DT_INT16);
      CASE(DT_INT8);
      CASE(DT_STRING);
      CASE(DT_INT64);
      CASE(DT_BOOL);
      CASE(DT_QINT8);
      CASE(DT_QUINT8);
      CASE(DT_QINT32);
      CASE(DT_QINT16);
      CASE(DT_QUINT16);
      CASE(DT_UINT16);
      CASE(DT_HALF);
      CASE(DT_UINT32);
      CASE(DT_UINT64);
      // TODO(feihugis): support other dtypes.
#undef CASE
      default:
        return errors::Internal("Unsupported dtype: ", dtype);
    }
  }

  for (int i = 0; i < produced_tensors.size(); ++i) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::ExpectEqual(produced_tensors[i],
                                                       expected_tensors[i]));
  }
  return Status::OK();
}

Status DatasetOpsTestBase::CreateOpKernel(
    const NodeDef& node_def, std::unique_ptr<OpKernel>* op_kernel) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_11(mht_11_v, 506, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CreateOpKernel");

  OpKernel* kernel;
  Status s;

  std::shared_ptr<const NodeProperties> props;
  TF_RETURN_IF_ERROR(NodeProperties::CreateFromNodeDef(
      node_def, flr_->GetFunctionLibraryDefinition(), &props));
  TF_RETURN_IF_ERROR(tensorflow::CreateOpKernel(
      device_type_, device_.get(), allocator_, flr_,
      device_->resource_manager(), props, TF_GRAPH_DEF_VERSION, &kernel));
  op_kernel->reset(kernel);
  return Status::OK();
}

Status DatasetOpsTestBase::CreateDatasetContext(
    OpKernel* const dateset_kernel,
    gtl::InlinedVector<TensorValue, 4>* const inputs,
    std::unique_ptr<OpKernelContext::Params>* dataset_context_params,
    std::unique_ptr<OpKernelContext>* dataset_context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_12(mht_12_v, 527, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CreateDatasetContext");

  Status status = CheckOpKernelInput(*dateset_kernel, *inputs);
  if (!status.ok()) {
    VLOG(0) << "WARNING: " << status.ToString();
  }
  TF_RETURN_IF_ERROR(CreateOpKernelContext(
      dateset_kernel, inputs, dataset_context_params, dataset_context));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateDataset(OpKernel* kernel,
                                         OpKernelContext* context,
                                         DatasetBase** const dataset) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_13(mht_13_v, 542, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CreateDataset");

  TF_RETURN_IF_ERROR(RunOpKernel(kernel, context));
  // Assume that DatasetOp has only one output.
  DCHECK_EQ(context->num_outputs(), 1);
  TF_RETURN_IF_ERROR(GetDatasetFromContext(context, 0, dataset));
  return Status::OK();
}

Status DatasetOpsTestBase::RestoreIterator(
    IteratorContext* ctx, IteratorStateReader* reader,
    const string& output_prefix, const DatasetBase& dataset,
    std::unique_ptr<IteratorBase>* iterator) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("output_prefix: \"" + output_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_14(mht_14_v, 557, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::RestoreIterator");

  return dataset.MakeIteratorFromCheckpoint(ctx, output_prefix, reader,
                                            iterator);
}

Status DatasetOpsTestBase::CreateIteratorContext(
    OpKernelContext* const op_context,
    std::unique_ptr<IteratorContext>* iterator_context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_15(mht_15_v, 567, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CreateIteratorContext");

  IteratorContext::Params params(op_context);
  params.resource_mgr = op_context->resource_manager();
  function_handle_cache_ = absl::make_unique<FunctionHandleCache>(flr_);
  params.function_handle_cache = function_handle_cache_.get();
  params.cancellation_manager = cancellation_manager_.get();
  *iterator_context = absl::make_unique<IteratorContext>(params);
  return Status::OK();
}

Status DatasetOpsTestBase::GetDatasetFromContext(OpKernelContext* context,
                                                 int output_index,
                                                 DatasetBase** const dataset) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_16(mht_16_v, 582, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::GetDatasetFromContext");

  Tensor* output = context->mutable_output(output_index);
  Status status = GetDatasetFromVariantTensor(*output, dataset);
  (*dataset)->Ref();
  return status;
}

Status DatasetOpsTestBase::InitThreadPool(int thread_num) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_17(mht_17_v, 592, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::InitThreadPool");

  if (thread_num < 1) {
    return errors::InvalidArgument(
        "The `thread_num` argument should be positive but got: ", thread_num);
  }
  thread_pool_ = absl::make_unique<thread::ThreadPool>(
      Env::Default(), ThreadOptions(), "test_thread_pool", thread_num);
  return Status::OK();
}

Status DatasetOpsTestBase::InitFunctionLibraryRuntime(
    const std::vector<FunctionDef>& flib, int cpu_num) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_18(mht_18_v, 606, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::InitFunctionLibraryRuntime");

  if (cpu_num < 1) {
    return errors::InvalidArgument(
        "The `cpu_num` argument should be positive but got: ", cpu_num);
  }
  SessionOptions options;
  auto* device_count = options.config.mutable_device_count();
  device_count->insert({"CPU", cpu_num});
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
  resource_mgr_ = absl::make_unique<ResourceMgr>("default_container");

  FunctionDefLibrary proto;
  for (const auto& fdef : flib) *(proto.add_function()) = fdef;
  lib_def_ =
      absl::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(), proto);

  OptimizerOptions opts;
  pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, lib_def_.get(), opts, thread_pool_.get(),
      /*parent=*/nullptr,
      /*session_metadata=*/nullptr,
      Rendezvous::Factory{
          [](const int64_t, const DeviceMgr* device_mgr, Rendezvous** r) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_19(mht_19_v, 635, "", "./tensorflow/core/data/dataset_test_base.cc", "lambda");

            *r = new IntraProcessRendezvous(device_mgr);
            return Status::OK();
          }});
  flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  if (thread_pool_ == nullptr) {
    runner_ = [](const std::function<void()>& fn) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_20(mht_20_v, 644, "", "./tensorflow/core/data/dataset_test_base.cc", "lambda");
 fn(); };
  } else {
    runner_ = [this](std::function<void()> fn) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_21(mht_21_v, 649, "", "./tensorflow/core/data/dataset_test_base.cc", "lambda");

      thread_pool_->Schedule(std::move(fn));
    };
  }
  return Status::OK();
}

Status DatasetOpsTestBase::RunOpKernel(OpKernel* op_kernel,
                                       OpKernelContext* context) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_22(mht_22_v, 660, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::RunOpKernel");

  device_->Compute(op_kernel, context);
  return context->status();
}

Status DatasetOpsTestBase::RunFunction(
    const FunctionDef& fdef, test::function::Attrs attrs,
    const std::vector<Tensor>& args,
    const GraphConstructorOptions& graph_options, std::vector<Tensor*> rets) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_23(mht_23_v, 671, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::RunFunction");

  std::unique_ptr<Executor> exec;
  InstantiationResult result;
  auto GetOpSig = [](const string& op, const OpDef** sig) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_24(mht_24_v, 678, "", "./tensorflow/core/data/dataset_test_base.cc", "lambda");

    return OpRegistry::Global()->LookUpOpDef(op, sig);
  };
  TF_RETURN_IF_ERROR(InstantiateFunction(fdef, attrs, GetOpSig, &result));

  DataTypeVector arg_types = result.arg_types;
  DataTypeVector ret_types = result.ret_types;

  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_RETURN_IF_ERROR(
      ConvertNodeDefsToGraph(graph_options, result.nodes, g.get()));

  const int version = g->versions().producer();
  LocalExecutorParams params;
  params.function_library = flr_;
  params.device = device_.get();
  params.create_kernel = [this, version](
                             const std::shared_ptr<const NodeProperties>& props,
                             OpKernel** kernel) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_25(mht_25_v, 699, "", "./tensorflow/core/data/dataset_test_base.cc", "lambda");

    return CreateNonCachedKernel(device_.get(), this->flr_, props, version,
                                 kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_26(mht_26_v, 706, "", "./tensorflow/core/data/dataset_test_base.cc", "lambda");

    DeleteNonCachedKernel(kernel);
  };

  Executor* cur_exec;
  TF_RETURN_IF_ERROR(NewLocalExecutor(params, *g, &cur_exec));
  exec.reset(cur_exec);
  FunctionCallFrame frame(arg_types, ret_types);
  TF_RETURN_IF_ERROR(frame.SetArgs(args));
  Executor::Args exec_args;
  exec_args.call_frame = &frame;
  exec_args.runner = runner_;
  TF_RETURN_IF_ERROR(exec->Run(exec_args));
  std::vector<Tensor> computed;
  TF_RETURN_IF_ERROR(frame.GetRetvals(&computed));
  if (computed.size() != rets.size()) {
    return errors::InvalidArgument(
        "The result does not match the expected number of return outpus",
        ". Expected: ", rets.size(), ". Actual: ", computed.size());
  }
  for (int i = 0; i < rets.size(); ++i) {
    *(rets[i]) = computed[i];
  }
  return Status::OK();
}

Status DatasetOpsTestBase::CreateOpKernelContext(
    OpKernel* kernel, gtl::InlinedVector<TensorValue, 4>* inputs,
    std::unique_ptr<OpKernelContext>* context) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_27(mht_27_v, 737, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CreateOpKernelContext");

  return CreateOpKernelContext(kernel, inputs, &params_, context);
}

Status DatasetOpsTestBase::CreateOpKernelContext(
    OpKernel* kernel, gtl::InlinedVector<TensorValue, 4>* inputs,
    std::unique_ptr<OpKernelContext::Params>* context_params,
    std::unique_ptr<OpKernelContext>* context) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_28(mht_28_v, 747, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CreateOpKernelContext");

  auto params = absl::make_unique<OpKernelContext::Params>();
  cancellation_manager_ = absl::make_unique<CancellationManager>();
  params->cancellation_manager = cancellation_manager_.get();
  params->device = device_.get();
  params->frame_iter = FrameAndIter(0, 0);
  params->function_library = flr_;
  params->inputs = inputs;
  params->op_kernel = kernel;
  params->resource_manager = resource_mgr_.get();
  params->runner = &runner_;
  slice_reader_cache_ =
      absl::make_unique<checkpoint::TensorSliceReaderCacheWrapper>();
  params->slice_reader_cache = slice_reader_cache_.get();
  step_container_ =
      absl::make_unique<ScopedStepContainer>(0, [](const string&) {});
  params->step_container = step_container_.get();

  // Set the allocator attributes for the outputs.
  allocator_attrs_.clear();
  for (int index = 0; index < params->op_kernel->num_outputs(); index++) {
    AllocatorAttributes attr;
    const bool on_host =
        (params->op_kernel->output_memory_types()[index] == HOST_MEMORY);
    attr.set_on_host(on_host);
    allocator_attrs_.emplace_back(attr);
  }
  params->output_attr_array = allocator_attrs_.data();

  *context = absl::make_unique<OpKernelContext>(params.get());
  *context_params = std::move(params);
  return Status::OK();
}

Status DatasetOpsTestBase::CreateSerializationContext(
    std::unique_ptr<SerializationContext>* context) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_29(mht_29_v, 785, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CreateSerializationContext");

  *context =
      absl::make_unique<SerializationContext>(SerializationContext::Params{});
  return Status::OK();
}

Status DatasetOpsTestBase::CheckOpKernelInput(
    const OpKernel& kernel, const gtl::InlinedVector<TensorValue, 4>& inputs) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_30(mht_30_v, 795, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckOpKernelInput");

  if (kernel.num_inputs() != inputs.size()) {
    return errors::InvalidArgument("The number of input elements should be ",
                                   kernel.num_inputs(),
                                   ", but got: ", inputs.size());
  }
  return Status::OK();
}

Status DatasetOpsTestBase::AddDatasetInput(
    gtl::InlinedVector<TensorValue, 4>* inputs, DataTypeVector input_types,
    DataType dtype, const TensorShape& shape) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_31(mht_31_v, 809, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::AddDatasetInput");

  if (input_types.size() < inputs->size()) {
    return errors::InvalidArgument("Adding more inputs than types: ",
                                   inputs->size(), " vs. ", input_types.size());
  }
  bool is_ref = IsRefType(input_types[inputs->size()]);
  auto input = absl::make_unique<Tensor>(allocator_, dtype, shape);

  if (is_ref) {
    DataType expected_dtype = RemoveRefType(input_types[inputs->size()]);
    if (expected_dtype != dtype) {
      return errors::InvalidArgument("The input data type is ", dtype,
                                     " , but expected: ", expected_dtype);
    }
    inputs->push_back({&lock_for_refs_, input.get()});
  } else {
    if (input_types[inputs->size()] != dtype) {
      return errors::InvalidArgument(
          "The input data type is ", dtype,
          " , but expected: ", input_types[inputs->size()]);
    }
    inputs->push_back({nullptr, input.get()});
  }

  // TODO(jsimsa): Figure out how to avoid using a member variable to garbage
  // collect the inputs.
  tensors_.push_back(std::move(input));

  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorGetNext(
    const std::vector<Tensor>& expected_outputs, bool compare_order) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_32(mht_32_v, 844, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckIteratorGetNext");

  return CheckIteratorGetNext(iterator_.get(), iterator_ctx_.get(),
                              expected_outputs, compare_order);
}

Status DatasetOpsTestBase::CheckIteratorGetNext(
    TestIterator* iterator, const std::vector<Tensor>& expected_outputs,
    bool compare_order) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_33(mht_33_v, 854, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckIteratorGetNext");

  return CheckIteratorGetNext(iterator->iterator(), iterator->ctx(),
                              expected_outputs, compare_order);
}

Status DatasetOpsTestBase::CheckIteratorGetNext(
    IteratorBase* iterator, IteratorContext* ctx,
    const std::vector<Tensor>& expected_outputs, bool compare_order) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_34(mht_34_v, 864, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckIteratorGetNext");

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_RETURN_IF_ERROR(iterator->GetNext(ctx, &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }
  // Call GetNext one more time to make sure it still reports
  // end_of_sequence = True.
  std::vector<Tensor> unused;
  TF_RETURN_IF_ERROR(iterator->GetNext(ctx, &unused, &end_of_sequence));
  EXPECT_TRUE(end_of_sequence);

  TF_EXPECT_OK(ExpectEqual(out_tensors, expected_outputs,
                           /*compare_order=*/compare_order));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorSkip(
    int num_to_skip, int expected_num_skipped, bool get_next,
    const std::vector<Tensor>& expected_outputs, bool compare_order) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_35(mht_35_v, 888, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckIteratorSkip");

  IteratorBase* iterator = iterator_.get();
  IteratorContext* ctx = iterator_ctx_.get();

  bool end_of_sequence = false;
  int num_skipped = 0;
  TF_RETURN_IF_ERROR(
      iterator->Skip(ctx, num_to_skip, &end_of_sequence, &num_skipped));
  EXPECT_TRUE(num_skipped == expected_num_skipped);
  if (get_next) {
    EXPECT_TRUE(!end_of_sequence);
    std::vector<Tensor> out_tensors;
    TF_RETURN_IF_ERROR(iterator->GetNext(ctx, &out_tensors, &end_of_sequence));
    TF_EXPECT_OK(ExpectEqual(out_tensors, expected_outputs,
                             /*compare_order=*/compare_order));
  }
  return Status::OK();
}

Status DatasetOpsTestBase::CheckSplitProviderFullIteration(
    const DatasetParams& params, const std::vector<Tensor>& expected_outputs) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_36(mht_36_v, 911, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckSplitProviderFullIteration");

  std::unique_ptr<TestDataset> dataset;
  TF_RETURN_IF_ERROR(MakeDataset(params, &dataset));
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_RETURN_IF_ERROR(dataset->dataset()->MakeSplitProviders(&split_providers));
  std::unique_ptr<TestIterator> iterator;
  TF_RETURN_IF_ERROR(
      MakeIterator(params, *dataset, std::move(split_providers), &iterator));
  TF_RETURN_IF_ERROR(CheckIteratorGetNext(iterator.get(), expected_outputs,
                                          /*compare_order=*/true));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckSplitProviderShardedIteration(
    const DatasetParams& params, int64_t num_shards, int64_t shard_index,
    const std::vector<Tensor>& expected_outputs) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_37(mht_37_v, 929, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckSplitProviderShardedIteration");

  std::unique_ptr<TestDataset> dataset;
  TF_RETURN_IF_ERROR(MakeDataset(params, &dataset));
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_RETURN_IF_ERROR(dataset->dataset()->MakeSplitProviders(&split_providers));
  for (int i = 0; i < split_providers.size(); ++i) {
    split_providers[i] = std::make_unique<ShardingSplitProvider>(
        num_shards, shard_index, std::move(split_providers[i]));
  }
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_RETURN_IF_ERROR(
      CreateIteratorContext(dataset->op_kernel_context(), &iterator_ctx));
  IteratorContext::Params iterator_params(iterator_ctx.get());
  std::move(split_providers.begin(), split_providers.end(),
            std::back_inserter(iterator_params.split_providers));
  iterator_ctx = absl::make_unique<IteratorContext>(iterator_params);
  int mid_breakpoint = expected_outputs.size() / 2;
  int near_end_breakpoint = expected_outputs.size() - 1;
  int end_breakpoint = expected_outputs.size();
  TF_RETURN_IF_ERROR(CheckIteratorSaveAndRestore(
      dataset->dataset(), iterator_ctx.get(), params.iterator_prefix(),
      expected_outputs,
      /*breakpoints=*/
      {0, mid_breakpoint, near_end_breakpoint, end_breakpoint},
      /*compare_order=*/true));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetNodeName(
    const string& expected_dataset_node_name) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("expected_dataset_node_name: \"" + expected_dataset_node_name + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_38(mht_38_v, 962, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckDatasetNodeName");

  EXPECT_EQ(dataset_->node_name(), expected_dataset_node_name);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetTypeString(
    const string& expected_type_str) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("expected_type_str: \"" + expected_type_str + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_39(mht_39_v, 972, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckDatasetTypeString");

  EXPECT_EQ(dataset_->type_string(), expected_type_str);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetOutputDtypes(
    const DataTypeVector& expected_output_dtypes) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_40(mht_40_v, 981, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckDatasetOutputDtypes");

  TF_EXPECT_OK(
      VerifyTypesMatch(dataset_->output_dtypes(), expected_output_dtypes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetOutputShapes(
    const std::vector<PartialTensorShape>& expected_output_shapes) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_41(mht_41_v, 991, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckDatasetOutputShapes");

  TF_EXPECT_OK(VerifyShapesCompatible(dataset_->output_shapes(),
                                      expected_output_shapes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetCardinality(int expected_cardinality) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_42(mht_42_v, 1000, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckDatasetCardinality");

  EXPECT_EQ(dataset_->Cardinality(), expected_cardinality);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetOptions(
    const Options& expected_options) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_43(mht_43_v, 1009, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckDatasetOptions");

  EXPECT_EQ(dataset_->options().SerializeAsString(),
            expected_options.SerializeAsString());
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorOutputDtypes(
    const DataTypeVector& expected_output_dtypes) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_44(mht_44_v, 1019, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckIteratorOutputDtypes");

  TF_EXPECT_OK(
      VerifyTypesMatch(iterator_->output_dtypes(), expected_output_dtypes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorOutputShapes(
    const std::vector<PartialTensorShape>& expected_output_shapes) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_45(mht_45_v, 1029, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckIteratorOutputShapes");

  TF_EXPECT_OK(VerifyShapesCompatible(iterator_->output_shapes(),
                                      expected_output_shapes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorPrefix(
    const string& expected_iterator_prefix) {
   std::vector<std::string> mht_46_v;
   mht_46_v.push_back("expected_iterator_prefix: \"" + expected_iterator_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_46(mht_46_v, 1040, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckIteratorPrefix");

  EXPECT_EQ(iterator_->prefix(), expected_iterator_prefix);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorSaveAndRestore(
    DatasetBase* dataset, IteratorContext* iterator_ctx,
    const std::string& iterator_prefix,
    const std::vector<Tensor>& expected_outputs,
    const std::vector<int>& breakpoints, bool compare_order) {
   std::vector<std::string> mht_47_v;
   mht_47_v.push_back("iterator_prefix: \"" + iterator_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_47(mht_47_v, 1053, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckIteratorSaveAndRestore");

  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(dataset->MakeIterator(iterator_ctx, /*parent=*/nullptr,
                                           iterator_prefix, &iterator));
  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_RETURN_IF_ERROR(CreateSerializationContext(&serialization_ctx));
  bool end_of_sequence = false;
  int cur_iteration = 0;
  std::vector<Tensor> out_tensors;
  for (int breakpoint : breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx, &reader, iterator_prefix,
                                 *dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_RETURN_IF_ERROR(
          iterator->GetNext(iterator_ctx, &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }
  TF_EXPECT_OK(ExpectEqual(out_tensors, expected_outputs,
                           /*compare_order=*/compare_order));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorSaveAndRestore(
    const std::string& iterator_prefix,
    const std::vector<Tensor>& expected_outputs,
    const std::vector<int>& breakpoints, bool compare_order) {
   std::vector<std::string> mht_48_v;
   mht_48_v.push_back("iterator_prefix: \"" + iterator_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_48(mht_48_v, 1091, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::CheckIteratorSaveAndRestore");

  return CheckIteratorSaveAndRestore(dataset_, iterator_ctx_.get(),
                                     iterator_prefix, expected_outputs,
                                     breakpoints, compare_order);
}

Status DatasetOpsTestBase::Initialize(const DatasetParams& dataset_params) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_49(mht_49_v, 1100, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::Initialize");

  if (initialized_) {
    return errors::Internal(
        "The fields (e.g. dataset_kernel_, dataset_ctx_, dataset_, "
        "iterator_ctx_, iterator_) have already been initialized.");
  }
  TF_RETURN_IF_ERROR(InitializeRuntime(dataset_params));
  TF_RETURN_IF_ERROR(MakeDataset(dataset_params, &dataset_kernel_, &params_,
                                 &dataset_ctx_, &tensors_, &dataset_));
  TF_RETURN_IF_ERROR(CreateIteratorContext(dataset_ctx_.get(), &iterator_ctx_));
  TF_RETURN_IF_ERROR(
      dataset_->MakeIterator(iterator_ctx_.get(), /*parent=*/nullptr,
                             dataset_params.iterator_prefix(), &iterator_));
  initialized_ = true;
  return Status::OK();
}

Status DatasetOpsTestBase::InitializeRuntime(
    const DatasetParams& dataset_params) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_50(mht_50_v, 1121, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::InitializeRuntime");

  TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
  TF_RETURN_IF_ERROR(
      InitFunctionLibraryRuntime(dataset_params.func_lib(), cpu_num_));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeDataset(const DatasetParams& dataset_params,
                                       std::unique_ptr<TestDataset>* dataset) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_51(mht_51_v, 1132, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::MakeDataset");

  DatasetBase* dataset_base;
  std::unique_ptr<OpKernel> dataset_kernel;
  std::unique_ptr<OpKernelContext::Params> dataset_ctx_params;
  std::unique_ptr<OpKernelContext> dataset_ctx;
  std::vector<std::unique_ptr<Tensor>> created_tensors;
  TF_RETURN_IF_ERROR(MakeDataset(dataset_params, &dataset_kernel,
                                 &dataset_ctx_params, &dataset_ctx,
                                 &created_tensors, &dataset_base));
  *dataset = std::make_unique<TestDataset>(
      std::move(dataset_kernel), std::move(dataset_ctx_params),
      std::move(dataset_ctx), std::move(created_tensors), dataset_base);
  return Status::OK();
}

Status DatasetOpsTestBase::RunDatasetOp(
    const DatasetParams& dataset_params,
    std::unique_ptr<OpKernel>* dataset_kernel,
    std::unique_ptr<OpKernelContext::Params>* dataset_ctx_params,
    std::vector<std::unique_ptr<Tensor>>* created_tensors,
    std::unique_ptr<OpKernelContext>* dataset_ctx) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_52(mht_52_v, 1155, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::RunDatasetOp");

  std::vector<Tensor*> input_datasets;
  for (auto& input : dataset_params.input_dataset_params()) {
    std::unique_ptr<Tensor> t;
    TF_RETURN_IF_ERROR(MakeDatasetTensor(*input, created_tensors, &t));
    input_datasets.push_back(t.get());
    created_tensors->push_back(std::move(t));
  }
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto input_dataset : input_datasets) {
    inputs.emplace_back(TensorValue(input_dataset));
  }

  // Copy the input tensors, storing them in the `inputs` vectors, and storing
  // owned references to the copies in `created_tensors`.
  for (auto& input : dataset_params.GetInputTensors()) {
    auto copy = absl::make_unique<Tensor>(input);
    inputs.push_back(TensorValue(copy.get()));
    created_tensors->push_back(std::move(copy));
  }

  TF_RETURN_IF_ERROR(MakeDatasetOpKernel(dataset_params, dataset_kernel));
  TF_RETURN_IF_ERROR(CreateDatasetContext(dataset_kernel->get(), &inputs,
                                          dataset_ctx_params, dataset_ctx));
  TF_RETURN_IF_ERROR(RunOpKernel(dataset_kernel->get(), dataset_ctx->get()));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeDataset(
    const DatasetParams& dataset_params,
    std::unique_ptr<OpKernel>* dataset_kernel,
    std::unique_ptr<OpKernelContext::Params>* dataset_ctx_params,
    std::unique_ptr<OpKernelContext>* dataset_ctx,
    std::vector<std::unique_ptr<Tensor>>* created_tensors,
    DatasetBase** dataset) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_53(mht_53_v, 1192, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::MakeDataset");

  TF_RETURN_IF_ERROR(RunDatasetOp(dataset_params, dataset_kernel,
                                  dataset_ctx_params, created_tensors,
                                  dataset_ctx));
  // Assume that DatasetOp has only one output.
  DCHECK_EQ((*dataset_ctx)->num_outputs(), 1);
  TF_RETURN_IF_ERROR(GetDatasetFromContext(dataset_ctx->get(), 0, dataset));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeIterator(
    const DatasetParams& dataset_params, const TestDataset& dataset,
    std::vector<std::unique_ptr<SplitProvider>> split_providers,
    std::unique_ptr<TestIterator>* iterator) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_54(mht_54_v, 1208, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::MakeIterator");

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_RETURN_IF_ERROR(
      CreateIteratorContext(dataset.op_kernel_context(), &iterator_ctx));
  IteratorContext::Params iterator_params(iterator_ctx.get());
  std::move(split_providers.begin(), split_providers.end(),
            std::back_inserter(iterator_params.split_providers));

  iterator_ctx = absl::make_unique<IteratorContext>(iterator_params);
  std::unique_ptr<IteratorBase> iterator_base;
  TF_RETURN_IF_ERROR(dataset.dataset()->MakeIterator(
      iterator_ctx.get(), /*parent=*/nullptr, dataset_params.iterator_prefix(),
      &iterator_base));
  *iterator = std::make_unique<TestIterator>(std::move(iterator_ctx),
                                             std::move(iterator_base));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeIterator(
    const DatasetParams& dataset_params, const TestDataset& dataset,
    std::unique_ptr<TestIterator>* iterator) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_55(mht_55_v, 1231, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::MakeIterator");

  return MakeIterator(dataset_params, dataset, /*split_providers=*/{},
                      iterator);
}

Status DatasetOpsTestBase::RunDatasetOp(const DatasetParams& dataset_params,
                                        std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_56(mht_56_v, 1240, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::RunDatasetOp");

  TF_RETURN_IF_ERROR(RunDatasetOp(dataset_params, &dataset_kernel_, &params_,
                                  &tensors_, &dataset_ctx_));
  for (int i = 0; i < dataset_ctx_->num_outputs(); ++i) {
    outputs->emplace_back(*dataset_ctx_->mutable_output(i));
  }
  return Status::OK();
}

Status DatasetOpsTestBase::MakeDatasetOpKernel(
    const DatasetParams& dataset_params,
    std::unique_ptr<OpKernel>* dataset_kernel) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_57(mht_57_v, 1254, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::MakeDatasetOpKernel");

  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  std::vector<string> input_names;
  TF_RETURN_IF_ERROR(dataset_params.GetInputNames(&input_names));
  AttributeVector attributes;
  TF_RETURN_IF_ERROR(dataset_params.GetAttributes(&attributes));
  NodeDef node_def =
      test::function::NDef(dataset_params.node_name(), dataset_params.op_name(),
                           input_names, attributes);
  TF_RETURN_IF_ERROR(CreateOpKernel(node_def, dataset_kernel));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeGetOptionsOpKernel(
    const DatasetParams& dataset_params, std::unique_ptr<OpKernel>* op_kernel) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_58(mht_58_v, 1272, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::MakeGetOptionsOpKernel");

  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  std::vector<string> input_names;
  TF_RETURN_IF_ERROR(dataset_params.GetInputNames(&input_names));
  AttributeVector attributes;
  TF_RETURN_IF_ERROR(dataset_params.GetAttributes(&attributes));
  NodeDef node_def = test::function::NDef(dataset_params.node_name(),
                                          dataset_params.dataset_type(),
                                          input_names, attributes);
  TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeDatasetTensor(
    const DatasetParams& dataset_params,
    std::vector<std::unique_ptr<Tensor>>* created_tensors,
    std::unique_ptr<Tensor>* dataset) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_59(mht_59_v, 1292, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetOpsTestBase::MakeDatasetTensor");

  // Make sure all the input dataset tensors have been populated.
  std::vector<Tensor*> input_datasets;
  for (auto& input : dataset_params.input_dataset_params()) {
    std::unique_ptr<Tensor> t;
    TF_RETURN_IF_ERROR(MakeDatasetTensor(*input, created_tensors, &t));
    input_datasets.push_back(t.get());
    created_tensors->push_back(std::move(t));
  }

  AttributeVector attributes;
  TF_RETURN_IF_ERROR(dataset_params.GetAttributes(&attributes));

  auto input_tensors = dataset_params.GetInputTensors();
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(input_datasets.size() + input_tensors.size());
  for (auto input_dataset : input_datasets) {
    inputs.emplace_back(TensorValue(input_dataset));
  }
  for (auto& input_tensor : input_tensors) {
    inputs.emplace_back(TensorValue(&input_tensor));
  }

  DatasetBase* dataset_base;
  std::unique_ptr<OpKernel> dataset_kernel;
  std::unique_ptr<OpKernelContext::Params> dataset_ctx_params;
  std::unique_ptr<OpKernelContext> dataset_ctx;
  TF_RETURN_IF_ERROR(MakeDatasetOpKernel(dataset_params, &dataset_kernel));
  TF_RETURN_IF_ERROR(CreateDatasetContext(dataset_kernel.get(), &inputs,
                                          &dataset_ctx_params, &dataset_ctx));
  TF_RETURN_IF_ERROR(
      CreateDataset(dataset_kernel.get(), dataset_ctx.get(), &dataset_base));
  Tensor dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_RETURN_IF_ERROR(
      StoreDatasetInVariantTensor(dataset_base, &dataset_tensor));
  *dataset = absl::make_unique<Tensor>(dataset_tensor);
  return Status::OK();
}

DatasetParams::DatasetParams(DataTypeVector output_dtypes,
                             std::vector<PartialTensorShape> output_shapes,
                             string node_name)
    : output_dtypes_(std::move(output_dtypes)),
      output_shapes_(std::move(output_shapes)),
      node_name_(std::move(node_name)) {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_60(mht_60_v, 1340, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetParams::DatasetParams");
}

bool DatasetParams::IsDatasetTensor(const Tensor& tensor) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_61(mht_61_v, 1345, "", "./tensorflow/core/data/dataset_test_base.cc", "DatasetParams::IsDatasetTensor");

  return tensor.dtype() == DT_VARIANT &&
         TensorShapeUtils::IsScalar(tensor.shape());
}

RangeDatasetParams::RangeDatasetParams(
    int64_t start, int64_t stop, int64_t step, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes, string node_name)
    : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                    std::move(node_name)),
      start_(start),
      stop_(stop),
      step_(step) {
   std::vector<std::string> mht_62_v;
   mht_62_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_62(mht_62_v, 1361, "", "./tensorflow/core/data/dataset_test_base.cc", "RangeDatasetParams::RangeDatasetParams");
}

RangeDatasetParams::RangeDatasetParams(int64_t start, int64_t stop,
                                       int64_t step)
    : DatasetParams({DT_INT64}, {PartialTensorShape({})}, "range_dataset"),
      start_(start),
      stop_(stop),
      step_(step) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_63(mht_63_v, 1371, "", "./tensorflow/core/data/dataset_test_base.cc", "RangeDatasetParams::RangeDatasetParams");
}

RangeDatasetParams::RangeDatasetParams(int64_t start, int64_t stop,
                                       int64_t step,
                                       DataTypeVector output_dtypes)
    : DatasetParams(std::move(output_dtypes), {PartialTensorShape({})},
                    "range_dataset"),
      start_(start),
      stop_(stop),
      step_(step) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_64(mht_64_v, 1383, "", "./tensorflow/core/data/dataset_test_base.cc", "RangeDatasetParams::RangeDatasetParams");
}

std::vector<Tensor> RangeDatasetParams::GetInputTensors() const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_65(mht_65_v, 1388, "", "./tensorflow/core/data/dataset_test_base.cc", "RangeDatasetParams::GetInputTensors");

  Tensor start_tensor = CreateTensor<int64_t>(TensorShape({}), {start_});
  Tensor stop_tensor = CreateTensor<int64_t>(TensorShape({}), {stop_});
  Tensor step_tensor = CreateTensor<int64_t>(TensorShape({}), {step_});
  return {start_tensor, stop_tensor, step_tensor};
}

Status RangeDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_66(mht_66_v, 1399, "", "./tensorflow/core/data/dataset_test_base.cc", "RangeDatasetParams::GetInputNames");

  *input_names = {"start", "stop", "step"};
  return Status::OK();
}

Status RangeDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_67(mht_67_v, 1407, "", "./tensorflow/core/data/dataset_test_base.cc", "RangeDatasetParams::GetAttributes");

  *attr_vector = {{"output_types", output_dtypes_},
                  {"output_shapes", output_shapes_},
                  {"metadata", ""}};
  return Status::OK();
}

string RangeDatasetParams::dataset_type() const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_68(mht_68_v, 1417, "", "./tensorflow/core/data/dataset_test_base.cc", "RangeDatasetParams::dataset_type");
 return "Range"; }

std::vector<Tensor> BatchDatasetParams::GetInputTensors() const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_69(mht_69_v, 1422, "", "./tensorflow/core/data/dataset_test_base.cc", "BatchDatasetParams::GetInputTensors");

  Tensor batch_size = CreateTensor<int64_t>(TensorShape({}), {batch_size_});
  Tensor drop_remainder =
      CreateTensor<bool>(TensorShape({}), {drop_remainder_});
  return {batch_size, drop_remainder};
}

Status BatchDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_70(mht_70_v, 1433, "", "./tensorflow/core/data/dataset_test_base.cc", "BatchDatasetParams::GetInputNames");

  *input_names = {"input_dataset", "batch_size", "drop_remainder"};
  return Status::OK();
}

Status BatchDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_71(mht_71_v, 1441, "", "./tensorflow/core/data/dataset_test_base.cc", "BatchDatasetParams::GetAttributes");

  *attr_vector = {{"parallel_copy", parallel_copy_},
                  {"output_types", output_dtypes_},
                  {"output_shapes", output_shapes_},
                  {"metadata", ""}};
  return Status::OK();
}

string BatchDatasetParams::dataset_type() const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_72(mht_72_v, 1452, "", "./tensorflow/core/data/dataset_test_base.cc", "BatchDatasetParams::dataset_type");
 return "Batch"; }

std::vector<Tensor> MapDatasetParams::GetInputTensors() const {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_73(mht_73_v, 1457, "", "./tensorflow/core/data/dataset_test_base.cc", "MapDatasetParams::GetInputTensors");

  return other_arguments_;
}

Status MapDatasetParams::GetInputNames(std::vector<string>* input_names) const {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_74(mht_74_v, 1464, "", "./tensorflow/core/data/dataset_test_base.cc", "MapDatasetParams::GetInputNames");

  input_names->emplace_back("input_dataset");
  for (int i = 0; i < other_arguments_.size(); ++i) {
    input_names->emplace_back(absl::StrCat("other_arguments_", i));
  }
  return Status::OK();
}

Status MapDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_75(mht_75_v, 1475, "", "./tensorflow/core/data/dataset_test_base.cc", "MapDatasetParams::GetAttributes");

  *attr_vector = {{"f", func_},
                  {"Targuments", type_arguments_},
                  {"output_shapes", output_shapes_},
                  {"output_types", output_dtypes_},
                  {"use_inter_op_parallelism", use_inter_op_parallelism_},
                  {"preserve_cardinality", preserve_cardinality_},
                  {"metadata", ""}};
  return Status::OK();
}

string MapDatasetParams::dataset_type() const {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_76(mht_76_v, 1489, "", "./tensorflow/core/data/dataset_test_base.cc", "MapDatasetParams::dataset_type");
 return "Map"; }

std::vector<FunctionDef> MapDatasetParams::func_lib() const {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_77(mht_77_v, 1494, "", "./tensorflow/core/data/dataset_test_base.cc", "MapDatasetParams::func_lib");

  return func_lib_;
}

TensorSliceDatasetParams::TensorSliceDatasetParams(
    std::vector<Tensor> components, string node_name, bool is_files)
    : DatasetParams(TensorSliceDtypes(components),
                    TensorSliceShapes(components), std::move(node_name)),
      components_(std::move(components)),
      is_files_(is_files) {
   std::vector<std::string> mht_78_v;
   mht_78_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_78(mht_78_v, 1507, "", "./tensorflow/core/data/dataset_test_base.cc", "TensorSliceDatasetParams::TensorSliceDatasetParams");
}

std::vector<Tensor> TensorSliceDatasetParams::GetInputTensors() const {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_79(mht_79_v, 1512, "", "./tensorflow/core/data/dataset_test_base.cc", "TensorSliceDatasetParams::GetInputTensors");

  return components_;
}

Status TensorSliceDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_80(mht_80_v, 1520, "", "./tensorflow/core/data/dataset_test_base.cc", "TensorSliceDatasetParams::GetInputNames");

  input_names->reserve(components_.size());
  for (int i = 0; i < components_.size(); ++i) {
    input_names->emplace_back(absl::StrCat("components_", i));
  }
  return Status::OK();
}

Status TensorSliceDatasetParams::GetAttributes(
    AttributeVector* attr_vector) const {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_81(mht_81_v, 1532, "", "./tensorflow/core/data/dataset_test_base.cc", "TensorSliceDatasetParams::GetAttributes");

  *attr_vector = {{"Toutput_types", output_dtypes_},
                  {"output_shapes", output_shapes_},
                  {"is_files", is_files_},
                  {"metadata", ""}};
  return Status::OK();
}

DataTypeVector TensorSliceDatasetParams::TensorSliceDtypes(
    const std::vector<Tensor>& input_components) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_82(mht_82_v, 1544, "", "./tensorflow/core/data/dataset_test_base.cc", "TensorSliceDatasetParams::TensorSliceDtypes");

  DataTypeVector dtypes;
  for (const auto& component : input_components) {
    dtypes.emplace_back(component.dtype());
  }
  return dtypes;
}

std::vector<PartialTensorShape> TensorSliceDatasetParams::TensorSliceShapes(
    const std::vector<Tensor>& input_components) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_83(mht_83_v, 1556, "", "./tensorflow/core/data/dataset_test_base.cc", "TensorSliceDatasetParams::TensorSliceShapes");

  std::vector<PartialTensorShape> shapes;
  for (const auto& component : input_components) {
    gtl::InlinedVector<int64_t, 4> partial_dim_sizes;
    for (int i = 1; i < component.dims(); ++i) {
      partial_dim_sizes.push_back(component.dim_size(i));
    }
    shapes.emplace_back(std::move(partial_dim_sizes));
  }
  return shapes;
}

string TensorSliceDatasetParams::dataset_type() const {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_84(mht_84_v, 1571, "", "./tensorflow/core/data/dataset_test_base.cc", "TensorSliceDatasetParams::dataset_type");
 return "TensorSlice"; }

std::vector<Tensor> TakeDatasetParams::GetInputTensors() const {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_85(mht_85_v, 1576, "", "./tensorflow/core/data/dataset_test_base.cc", "TakeDatasetParams::GetInputTensors");

  return {CreateTensor<int64_t>(TensorShape({}), {count_})};
}

Status TakeDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_86(mht_86_v, 1584, "", "./tensorflow/core/data/dataset_test_base.cc", "TakeDatasetParams::GetInputNames");

  *input_names = {"input_dataset", "count"};
  return Status::OK();
}

Status TakeDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_87(mht_87_v, 1592, "", "./tensorflow/core/data/dataset_test_base.cc", "TakeDatasetParams::GetAttributes");

  *attr_vector = {{"output_shapes", output_shapes_},
                  {"output_types", output_dtypes_},
                  {"metadata", ""}};
  return Status::OK();
}

string TakeDatasetParams::dataset_type() const {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_88(mht_88_v, 1602, "", "./tensorflow/core/data/dataset_test_base.cc", "TakeDatasetParams::dataset_type");
 return "Take"; }

std::vector<Tensor> ConcatenateDatasetParams::GetInputTensors() const {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_89(mht_89_v, 1607, "", "./tensorflow/core/data/dataset_test_base.cc", "ConcatenateDatasetParams::GetInputTensors");

  return {};
}

Status ConcatenateDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_90(mht_90_v, 1615, "", "./tensorflow/core/data/dataset_test_base.cc", "ConcatenateDatasetParams::GetInputNames");

  *input_names = {"input_dataset", "another_dataset"};
  return Status::OK();
}

Status ConcatenateDatasetParams::GetAttributes(
    AttributeVector* attr_vector) const {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_91(mht_91_v, 1624, "", "./tensorflow/core/data/dataset_test_base.cc", "ConcatenateDatasetParams::GetAttributes");

  *attr_vector = {{"output_types", output_dtypes_},
                  {"output_shapes", output_shapes_},
                  {"metadata", ""}};
  return Status::OK();
}

string ConcatenateDatasetParams::dataset_type() const {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_92(mht_92_v, 1634, "", "./tensorflow/core/data/dataset_test_base.cc", "ConcatenateDatasetParams::dataset_type");
 return "Concatenate"; }

std::vector<Tensor> OptionsDatasetParams::GetInputTensors() const {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_93(mht_93_v, 1639, "", "./tensorflow/core/data/dataset_test_base.cc", "OptionsDatasetParams::GetInputTensors");
 return {}; }

Status OptionsDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_94(mht_94_v, 1645, "", "./tensorflow/core/data/dataset_test_base.cc", "OptionsDatasetParams::GetInputNames");

  input_names->emplace_back("input_dataset");
  return Status::OK();
}

Status OptionsDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_95(mht_95_v, 1653, "", "./tensorflow/core/data/dataset_test_base.cc", "OptionsDatasetParams::GetAttributes");

  *attr_vector = {{"serialized_options", serialized_options_},
                  {"output_shapes", output_shapes_},
                  {"output_types", output_dtypes_},
                  {"metadata", ""}};
  return Status::OK();
}

string OptionsDatasetParams::dataset_type() const {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_test_baseDTcc mht_96(mht_96_v, 1664, "", "./tensorflow/core/data/dataset_test_base.cc", "OptionsDatasetParams::dataset_type");
 return "Options"; }

}  // namespace data
}  // namespace tensorflow
