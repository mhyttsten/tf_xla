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
class MHTracer_DTPStensorflowPSpythonPSlibPSioPSfile_io_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSfile_io_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSlibPSioPSfile_io_wrapperDTcc() {
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
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/python/lib/core/pybind11_absl.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace tensorflow {
struct PyTransactionToken {
  TransactionToken* token_;
};

inline TransactionToken* TokenFromPyToken(PyTransactionToken* t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSlibPSioPSfile_io_wrapperDTcc mht_0(mht_0_v, 209, "", "./tensorflow/python/lib/io/file_io_wrapper.cc", "TokenFromPyToken");

  return (t ? t->token_ : nullptr);
}
}  // namespace tensorflow

namespace {
namespace py = pybind11;

PYBIND11_MODULE(_pywrap_file_io, m) {
  using tensorflow::PyTransactionToken;
  using tensorflow::TransactionToken;
  py::class_<PyTransactionToken>(m, "TransactionToken")
      .def("__repr__", [](const PyTransactionToken* t) {
        if (t->token_) {
          return std::string(t->token_->owner->DecodeTransaction(t->token_));
        }
        return std::string("Invalid token!");
      });

  m.def(
      "FileExists",
      [](const std::string& filename, PyTransactionToken* token) {
        tensorflow::Status status;
        {
          py::gil_scoped_release release;
          status = tensorflow::Env::Default()->FileExists(filename);
        }
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("filename"), py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "DeleteFile",
      [](const std::string& filename, PyTransactionToken* token) {
        py::gil_scoped_release release;
        tensorflow::Status status =
            tensorflow::Env::Default()->DeleteFile(filename);
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("filename"), py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "ReadFileToString",
      [](const std::string& filename, PyTransactionToken* token) {
        std::string data;
        py::gil_scoped_release release;
        const auto status =
            ReadFileToString(tensorflow::Env::Default(), filename, &data);
        pybind11::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        return py::bytes(data);
      },
      py::arg("filename"), py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "WriteStringToFile",
      [](const std::string& filename, tensorflow::StringPiece data,
         PyTransactionToken* token) {
        py::gil_scoped_release release;
        const auto status =
            WriteStringToFile(tensorflow::Env::Default(), filename, data);
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("filename"), py::arg("data"),
      py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "GetChildren",
      [](const std::string& dirname, PyTransactionToken* token) {
        std::vector<std::string> results;
        py::gil_scoped_release release;
        const auto status =
            tensorflow::Env::Default()->GetChildren(dirname, &results);
        pybind11::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        return results;
      },
      py::arg("dirname"), py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "GetMatchingFiles",
      [](const std::string& pattern, PyTransactionToken* token) {
        std::vector<std::string> results;
        py::gil_scoped_release release;
        const auto status =
            tensorflow::Env::Default()->GetMatchingPaths(pattern, &results);
        pybind11::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        return results;
      },
      py::arg("pattern"), py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "CreateDir",
      [](const std::string& dirname, PyTransactionToken* token) {
        py::gil_scoped_release release;
        const auto status = tensorflow::Env::Default()->CreateDir(dirname);
        if (tensorflow::errors::IsAlreadyExists(status)) {
          return;
        }
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("dirname"), py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "RecursivelyCreateDir",
      [](const std::string& dirname, PyTransactionToken* token) {
        py::gil_scoped_release release;
        const auto status =
            tensorflow::Env::Default()->RecursivelyCreateDir(dirname);
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("dirname"), py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "CopyFile",
      [](const std::string& src, const std::string& target, bool overwrite,
         PyTransactionToken* token) {
        py::gil_scoped_release release;
        auto* env = tensorflow::Env::Default();
        tensorflow::Status status;
        if (!overwrite && env->FileExists(target).ok()) {
          status = tensorflow::errors::AlreadyExists("file already exists");
        } else {
          status = env->CopyFile(src, target);
        }
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("src"), py::arg("target"), py::arg("overwrite"),
      py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "RenameFile",
      [](const std::string& src, const std::string& target, bool overwrite,
         PyTransactionToken* token) {
        py::gil_scoped_release release;
        auto* env = tensorflow::Env::Default();
        tensorflow::Status status;
        if (!overwrite && env->FileExists(target).ok()) {
          status = tensorflow::errors::AlreadyExists("file already exists");
        } else {
          status = env->RenameFile(src, target);
        }
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("src"), py::arg("target"), py::arg("overwrite"),
      py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "DeleteRecursively",
      [](const std::string& dirname, PyTransactionToken* token) {
        py::gil_scoped_release release;
        int64_t undeleted_files;
        int64_t undeleted_dirs;
        auto status = tensorflow::Env::Default()->DeleteRecursively(
            dirname, &undeleted_files, &undeleted_dirs);
        if (status.ok() && (undeleted_files > 0 || undeleted_dirs > 0)) {
          status = tensorflow::errors::PermissionDenied(
              "could not fully delete dir");
        }
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("dirname"), py::arg("token") = (PyTransactionToken*)nullptr);
  m.def(
      "IsDirectory",
      [](const std::string& dirname, PyTransactionToken* token) {
        py::gil_scoped_release release;
        const auto status = tensorflow::Env::Default()->IsDirectory(dirname);
        // FAILED_PRECONDITION response means path exists but isn't a dir.
        if (tensorflow::errors::IsFailedPrecondition(status)) {
          return false;
        }

        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
        return true;
      },
      py::arg("dirname"), py::arg("token") = (PyTransactionToken*)nullptr);
  m.def("HasAtomicMove", [](const std::string& path) {
    py::gil_scoped_release release;
    bool has_atomic_move;
    const auto status =
        tensorflow::Env::Default()->HasAtomicMove(path, &has_atomic_move);
    tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
    return has_atomic_move;
  });

  py::class_<tensorflow::FileStatistics>(m, "FileStatistics")
      .def_readonly("length", &tensorflow::FileStatistics::length)
      .def_readonly("mtime_nsec", &tensorflow::FileStatistics::mtime_nsec)
      .def_readonly("is_directory", &tensorflow::FileStatistics::is_directory);

  m.def(
      "Stat",
      [](const std::string& filename, PyTransactionToken* token) {
        py::gil_scoped_release release;
        std::unique_ptr<tensorflow::FileStatistics> self(
            new tensorflow::FileStatistics);
        const auto status =
            tensorflow::Env::Default()->Stat(filename, self.get());
        py::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        return self.release();
      },
      py::arg("filename"), py::arg("token") = (PyTransactionToken*)nullptr);

  m.def("GetRegisteredSchemes", []() {
    std::vector<std::string> results;
    py::gil_scoped_release release;
    const auto status =
        tensorflow::Env::Default()->GetRegisteredFileSystemSchemes(&results);
    pybind11::gil_scoped_acquire acquire;
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    return results;
  });

  using tensorflow::WritableFile;
  py::class_<WritableFile>(m, "WritableFile")
      .def(py::init([](const std::string& filename, const std::string& mode,
                       PyTransactionToken* token) {
             py::gil_scoped_release release;
             auto* env = tensorflow::Env::Default();
             std::unique_ptr<WritableFile> self;
             const auto status = mode.find('a') == std::string::npos
                                     ? env->NewWritableFile(filename, &self)
                                     : env->NewAppendableFile(filename, &self);
             py::gil_scoped_acquire acquire;
             tensorflow::MaybeRaiseRegisteredFromStatus(status);
             return self.release();
           }),
           py::arg("filename"), py::arg("mode"),
           py::arg("token") = (PyTransactionToken*)nullptr)
      .def("append",
           [](WritableFile* self, tensorflow::StringPiece data) {
             const auto status = self->Append(data);
             tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
           })
      // TODO(slebedev): Make WritableFile::Tell const and change self
      // to be a reference.
      .def("tell",
           [](WritableFile* self) {
             int64_t pos = -1;
             py::gil_scoped_release release;
             const auto status = self->Tell(&pos);
             tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
             return pos;
           })
      .def("flush",
           [](WritableFile* self) {
             py::gil_scoped_release release;
             tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(self->Flush());
           })
      .def("close", [](WritableFile* self) {
        py::gil_scoped_release release;
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(self->Close());
      });

  using tensorflow::io::BufferedInputStream;
  py::class_<BufferedInputStream>(m, "BufferedInputStream")
      .def(py::init([](const std::string& filename, size_t buffer_size,
                       PyTransactionToken* token) {
             py::gil_scoped_release release;
             std::unique_ptr<tensorflow::RandomAccessFile> file;
             const auto status =
                 tensorflow::Env::Default()->NewRandomAccessFile(filename,
                                                                 &file);
             tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
             std::unique_ptr<tensorflow::io::RandomAccessInputStream>
                 input_stream(new tensorflow::io::RandomAccessInputStream(
                     file.release(),
                     /*owns_file=*/true));
             py::gil_scoped_acquire acquire;
             return new BufferedInputStream(input_stream.release(), buffer_size,
                                            /*owns_input_stream=*/true);
           }),
           py::arg("filename"), py::arg("buffer_size"),
           py::arg("token") = (PyTransactionToken*)nullptr)
      .def("read",
           [](BufferedInputStream* self, int64_t bytes_to_read) {
             py::gil_scoped_release release;
             tensorflow::tstring result;
             const auto status = self->ReadNBytes(bytes_to_read, &result);
             if (!status.ok() && !tensorflow::errors::IsOutOfRange(status)) {
               result.clear();
               tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
             }
             py::gil_scoped_acquire acquire;
             return py::bytes(result);
           })
      .def("readline",
           [](BufferedInputStream* self) {
             py::gil_scoped_release release;
             auto output = self->ReadLineAsString();
             py::gil_scoped_acquire acquire;
             return py::bytes(output);
           })
      .def("seek",
           [](BufferedInputStream* self, int64_t pos) {
             py::gil_scoped_release release;
             tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(self->Seek(pos));
           })
      .def("tell", [](BufferedInputStream* self) {
        py::gil_scoped_release release;
        return self->Tell();
      });
}
}  // namespace
