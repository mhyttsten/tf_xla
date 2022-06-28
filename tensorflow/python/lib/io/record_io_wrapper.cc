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
class MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc() {
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

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "pybind11/pybind11.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/pybind11_absl.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace {

namespace py = ::pybind11;

class PyRecordReader {
 public:
  // NOTE(sethtroisi): At this time PyRecordReader doesn't benefit from taking
  // RecordReaderOptions, if this changes the API can be updated at that time.
  static tensorflow::Status New(const std::string& filename,
                                const std::string& compression_type,
                                PyRecordReader** out) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   mht_0_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_0(mht_0_v, 214, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "New");

    auto tmp = new PyRecordReader(filename, compression_type);
    TF_RETURN_IF_ERROR(tmp->Reopen());
    *out = tmp;
    return tensorflow::Status::OK();
  }

  PyRecordReader() = delete;
  ~PyRecordReader() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_1(mht_1_v, 225, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "~PyRecordReader");
 Close(); }

  tensorflow::Status ReadNextRecord(tensorflow::tstring* out) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_2(mht_2_v, 230, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "ReadNextRecord");

    if (IsClosed()) {
      return tensorflow::errors::FailedPrecondition("Reader is closed.");
    }
    return reader_->ReadRecord(&offset_, out);
  }

  bool IsClosed() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_3(mht_3_v, 240, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "IsClosed");
 return file_ == nullptr && reader_ == nullptr; }

  void Close() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_4(mht_4_v, 245, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "Close");

    reader_ = nullptr;
    file_ = nullptr;
  }

  // Reopen a closed writer by re-opening the file and re-creating the reader,
  // but preserving the prior read offset. If not closed, returns an error.
  //
  // This is useful to allow "refreshing" the underlying file handle, in cases
  // where the file was replaced with a newer version containing additional data
  // that otherwise wouldn't be available via the existing file handle. This
  // allows the file to be polled continuously using the same iterator, even as
  // it grows, which supports use cases such as TensorBoard.
  tensorflow::Status Reopen() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_5(mht_5_v, 261, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "Reopen");

    if (!IsClosed()) {
      return tensorflow::errors::FailedPrecondition("Reader is not closed.");
    }
    TF_RETURN_IF_ERROR(
        tensorflow::Env::Default()->NewRandomAccessFile(filename_, &file_));
    reader_ =
        absl::make_unique<tensorflow::io::RecordReader>(file_.get(), options_);
    return tensorflow::Status::OK();
  }

 private:
  static constexpr tensorflow::uint64 kReaderBufferSize = 16 * 1024 * 1024;

  PyRecordReader(const std::string& filename,
                 const std::string& compression_type)
      : filename_(filename),
        options_(CreateOptions(compression_type)),
        offset_(0),
        file_(nullptr),
        reader_(nullptr) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("filename: \"" + filename + "\"");
   mht_6_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_6(mht_6_v, 286, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "PyRecordReader");
}

  static tensorflow::io::RecordReaderOptions CreateOptions(
      const std::string& compression_type) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_7(mht_7_v, 293, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "CreateOptions");

    auto options =
        tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions(
            compression_type);
    options.buffer_size = kReaderBufferSize;
    return options;
  }

  const std::string filename_;
  const tensorflow::io::RecordReaderOptions options_;
  tensorflow::uint64 offset_;
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  std::unique_ptr<tensorflow::io::RecordReader> reader_;

  TF_DISALLOW_COPY_AND_ASSIGN(PyRecordReader);
};

class PyRecordRandomReader {
 public:
  static tensorflow::Status New(const std::string& filename,
                                PyRecordRandomReader** out) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_8(mht_8_v, 317, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "New");

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(
        tensorflow::Env::Default()->NewRandomAccessFile(filename, &file));
    auto options =
        tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions("");
    options.buffer_size = kReaderBufferSize;
    auto reader =
        absl::make_unique<tensorflow::io::RecordReader>(file.get(), options);
    *out = new PyRecordRandomReader(std::move(file), std::move(reader));
    return tensorflow::Status::OK();
  }

  PyRecordRandomReader() = delete;
  ~PyRecordRandomReader() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_9(mht_9_v, 334, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "~PyRecordRandomReader");
 Close(); }

  tensorflow::Status ReadRecord(tensorflow::uint64* offset,
                                tensorflow::tstring* out) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_10(mht_10_v, 340, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "ReadRecord");

    if (IsClosed()) {
      return tensorflow::errors::FailedPrecondition(
          "Random TFRecord Reader is closed.");
    }
    return reader_->ReadRecord(offset, out);
  }

  bool IsClosed() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_11(mht_11_v, 351, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "IsClosed");
 return file_ == nullptr && reader_ == nullptr; }

  void Close() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_12(mht_12_v, 356, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "Close");

    reader_ = nullptr;
    file_ = nullptr;
  }

 private:
  static constexpr tensorflow::uint64 kReaderBufferSize = 16 * 1024 * 1024;

  PyRecordRandomReader(std::unique_ptr<tensorflow::RandomAccessFile> file,
                       std::unique_ptr<tensorflow::io::RecordReader> reader)
      : file_(std::move(file)), reader_(std::move(reader)) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_13(mht_13_v, 369, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "PyRecordRandomReader");
}

  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  std::unique_ptr<tensorflow::io::RecordReader> reader_;

  TF_DISALLOW_COPY_AND_ASSIGN(PyRecordRandomReader);
};

class PyRecordWriter {
 public:
  static tensorflow::Status New(
      const std::string& filename,
      const tensorflow::io::RecordWriterOptions& options,
      PyRecordWriter** out) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_14(mht_14_v, 386, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "New");

    std::unique_ptr<tensorflow::WritableFile> file;
    TF_RETURN_IF_ERROR(
        tensorflow::Env::Default()->NewWritableFile(filename, &file));
    auto writer =
        absl::make_unique<tensorflow::io::RecordWriter>(file.get(), options);
    *out = new PyRecordWriter(std::move(file), std::move(writer));
    return tensorflow::Status::OK();
  }

  PyRecordWriter() = delete;
  ~PyRecordWriter() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_15(mht_15_v, 400, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "~PyRecordWriter");
 Close(); }

  tensorflow::Status WriteRecord(tensorflow::StringPiece record) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_16(mht_16_v, 405, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "WriteRecord");

    if (IsClosed()) {
      return tensorflow::errors::FailedPrecondition("Writer is closed.");
    }
    return writer_->WriteRecord(record);
  }

  tensorflow::Status Flush() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_17(mht_17_v, 415, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "Flush");

    if (IsClosed()) {
      return tensorflow::errors::FailedPrecondition("Writer is closed.");
    }

    auto status = writer_->Flush();
    if (status.ok()) {
      // Per the RecordWriter contract, flushing the RecordWriter does not
      // flush the underlying file.  Here we need to do both.
      return file_->Flush();
    }
    return status;
  }

  bool IsClosed() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_18(mht_18_v, 432, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "IsClosed");
 return file_ == nullptr && writer_ == nullptr; }

  tensorflow::Status Close() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_19(mht_19_v, 437, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "Close");

    if (writer_ != nullptr) {
      auto status = writer_->Close();
      writer_ = nullptr;
      if (!status.ok()) return status;
    }
    if (file_ != nullptr) {
      auto status = file_->Close();
      file_ = nullptr;
      if (!status.ok()) return status;
    }
    return tensorflow::Status::OK();
  }

 private:
  PyRecordWriter(std::unique_ptr<tensorflow::WritableFile> file,
                 std::unique_ptr<tensorflow::io::RecordWriter> writer)
      : file_(std::move(file)), writer_(std::move(writer)) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSrecord_io_wrapperDTcc mht_20(mht_20_v, 457, "", "./tensorflow/python/lib/io/record_io_wrapper.cc", "PyRecordWriter");
}

  std::unique_ptr<tensorflow::WritableFile> file_;
  std::unique_ptr<tensorflow::io::RecordWriter> writer_;

  TF_DISALLOW_COPY_AND_ASSIGN(PyRecordWriter);
};

PYBIND11_MODULE(_pywrap_record_io, m) {
  py::class_<PyRecordReader>(m, "RecordIterator")
      .def(py::init(
          [](const std::string& filename, const std::string& compression_type) {
            tensorflow::Status status;
            PyRecordReader* self = nullptr;
            {
              py::gil_scoped_release release;
              status = PyRecordReader::New(filename, compression_type, &self);
            }
            MaybeRaiseRegisteredFromStatus(status);
            return self;
          }))
      .def("__iter__", [](const py::object& self) { return self; })
      .def("__next__",
           [](PyRecordReader* self) {
             if (self->IsClosed()) {
               throw py::stop_iteration();
             }

             tensorflow::tstring record;
             tensorflow::Status status;
             {
               py::gil_scoped_release release;
               status = self->ReadNextRecord(&record);
             }
             if (tensorflow::errors::IsOutOfRange(status)) {
               // Don't close because the file being read could be updated
               // in-between
               // __next__ calls.
               throw py::stop_iteration();
             }
             MaybeRaiseRegisteredFromStatus(status);
             return py::bytes(record);
           })
      .def("close", [](PyRecordReader* self) { self->Close(); })
      .def("reopen", [](PyRecordReader* self) {
        tensorflow::Status status;
        {
          py::gil_scoped_release release;
          status = self->Reopen();
        }
        MaybeRaiseRegisteredFromStatus(status);
      });

  py::class_<PyRecordRandomReader>(m, "RandomRecordReader")
      .def(py::init([](const std::string& filename) {
        tensorflow::Status status;
        PyRecordRandomReader* self = nullptr;
        {
          py::gil_scoped_release release;
          status = PyRecordRandomReader::New(filename, &self);
        }
        MaybeRaiseRegisteredFromStatus(status);
        return self;
      }))
      .def("read",
           [](PyRecordRandomReader* self, tensorflow::uint64 offset) {
             tensorflow::uint64 temp_offset = offset;
             tensorflow::tstring record;
             tensorflow::Status status;
             {
               py::gil_scoped_release release;
               status = self->ReadRecord(&temp_offset, &record);
             }
             if (tensorflow::errors::IsOutOfRange(status)) {
               throw py::index_error(tensorflow::strings::StrCat(
                   "Out of range at reading offset ", offset));
             }
             MaybeRaiseRegisteredFromStatus(status);
             return py::make_tuple(py::bytes(record), temp_offset);
           })
      .def("close", [](PyRecordRandomReader* self) { self->Close(); });

  using tensorflow::io::ZlibCompressionOptions;
  py::class_<ZlibCompressionOptions>(m, "ZlibCompressionOptions")
      .def_readwrite("flush_mode", &ZlibCompressionOptions::flush_mode)
      .def_readwrite("input_buffer_size",
                     &ZlibCompressionOptions::input_buffer_size)
      .def_readwrite("output_buffer_size",
                     &ZlibCompressionOptions::output_buffer_size)
      .def_readwrite("window_bits", &ZlibCompressionOptions::window_bits)
      .def_readwrite("compression_level",
                     &ZlibCompressionOptions::compression_level)
      .def_readwrite("compression_method",
                     &ZlibCompressionOptions::compression_method)
      .def_readwrite("mem_level", &ZlibCompressionOptions::mem_level)
      .def_readwrite("compression_strategy",
                     &ZlibCompressionOptions::compression_strategy);

  using tensorflow::io::RecordWriterOptions;
  py::class_<RecordWriterOptions>(m, "RecordWriterOptions")
      .def(py::init(&RecordWriterOptions::CreateRecordWriterOptions))
      .def_readonly("compression_type", &RecordWriterOptions::compression_type)
      .def_readonly("zlib_options", &RecordWriterOptions::zlib_options);

  using tensorflow::MaybeRaiseRegisteredFromStatus;

  py::class_<PyRecordWriter>(m, "RecordWriter")
      .def(py::init(
          [](const std::string& filename, const RecordWriterOptions& options) {
            PyRecordWriter* self = nullptr;
            tensorflow::Status status;
            {
              py::gil_scoped_release release;
              status = PyRecordWriter::New(filename, options, &self);
            }
            MaybeRaiseRegisteredFromStatus(status);
            return self;
          }))
      .def("__enter__", [](const py::object& self) { return self; })
      .def("__exit__",
           [](PyRecordWriter* self, py::args) {
             MaybeRaiseRegisteredFromStatus(self->Close());
           })
      .def(
          "write",
          [](PyRecordWriter* self, tensorflow::StringPiece record) {
            tensorflow::Status status;
            {
              py::gil_scoped_release release;
              status = self->WriteRecord(record);
            }
            MaybeRaiseRegisteredFromStatus(status);
          },
          py::arg("record"))
      .def("flush",
           [](PyRecordWriter* self) {
             MaybeRaiseRegisteredFromStatus(self->Flush());
           })
      .def("close", [](PyRecordWriter* self) {
        MaybeRaiseRegisteredFromStatus(self->Close());
      });
}

}  // namespace
