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
class MHTracer_DTPStensorflowPScorePSkernelsPSlmdb_reader_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlmdb_reader_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlmdb_reader_opDTcc() {
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

#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

#include <sys/stat.h>
#include "lmdb.h"

namespace tensorflow {

#define MDB_CHECK(val) CHECK_EQ(val, MDB_SUCCESS) << mdb_strerror(val)

class LMDBReader : public ReaderBase {
 public:
  LMDBReader(const string& node_name, Env* /*unused*/)
      : ReaderBase(strings::StrCat("LMDBReader '", node_name, "'")),
        mdb_env_(nullptr),
        mdb_dbi_(0),
        mdb_txn_(nullptr),
        mdb_cursor_(nullptr) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSlmdb_reader_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/lmdb_reader_op.cc", "LMDBReader");
}

  Status OnWorkStartedLocked() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlmdb_reader_opDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/kernels/lmdb_reader_op.cc", "OnWorkStartedLocked");

    MDB_CHECK(mdb_env_create(&mdb_env_));
    int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;

    // Check if the LMDB filename is actually a file instead of a directory.
    // If so, set appropriate flags so we can open it.
    struct stat source_stat;
    if (stat(current_work().c_str(), &source_stat) == 0 &&
        (source_stat.st_mode & S_IFREG)) {
      flags |= MDB_NOSUBDIR;
    }

    MDB_CHECK(mdb_env_open(mdb_env_, current_work().c_str(), flags, 0664));
    MDB_CHECK(mdb_txn_begin(mdb_env_, nullptr, MDB_RDONLY, &mdb_txn_));
    MDB_CHECK(mdb_dbi_open(mdb_txn_, nullptr, 0, &mdb_dbi_));

    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlmdb_reader_opDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/kernels/lmdb_reader_op.cc", "OnWorkFinishedLocked");

    if (mdb_env_ != nullptr) {
      if (mdb_cursor_) {
        mdb_cursor_close(mdb_cursor_);
        mdb_cursor_ = nullptr;
      }
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_txn_abort(mdb_txn_);
      mdb_env_close(mdb_env_);
      mdb_txn_ = nullptr;
      mdb_dbi_ = 0;
      mdb_env_ = nullptr;
    }
    return Status::OK();
  }

  Status ReadLocked(tstring* key, tstring* value, bool* produced,
                    bool* at_end) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlmdb_reader_opDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/kernels/lmdb_reader_op.cc", "ReadLocked");

    if (mdb_cursor_ == nullptr) {
      MDB_CHECK(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_));
      if (Seek(MDB_FIRST) == false) {
        *at_end = true;
        return Status::OK();
      }
    } else {
      if (Seek(MDB_NEXT) == false) {
        *at_end = true;
        return Status::OK();
      }
    }
    *key =
        tstring(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
    *value = tstring(static_cast<const char*>(mdb_value_.mv_data),
                     mdb_value_.mv_size);
    *produced = true;
    return Status::OK();
  }

  Status ResetLocked() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlmdb_reader_opDTcc mht_4(mht_4_v, 275, "", "./tensorflow/core/kernels/lmdb_reader_op.cc", "ResetLocked");

    CHECK_EQ(Seek(MDB_FIRST), true);
    return ReaderBase::ResetLocked();
  }

 private:
  bool Seek(MDB_cursor_op op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlmdb_reader_opDTcc mht_5(mht_5_v, 284, "", "./tensorflow/core/kernels/lmdb_reader_op.cc", "Seek");

    CHECK_NOTNULL(mdb_cursor_);
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      return false;
    } else {
      MDB_CHECK(mdb_status);
      return true;
    }
  }

  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;

  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

class LMDBReaderOp : public ReaderOpKernel {
 public:
  explicit LMDBReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlmdb_reader_opDTcc mht_6(mht_6_v, 309, "", "./tensorflow/core/kernels/lmdb_reader_op.cc", "LMDBReaderOp");

    Env* env = context->env();
    SetReaderFactory([this, env]() { return new LMDBReader(name(), env); });
  }
};

REGISTER_KERNEL_BUILDER(Name("LMDBReader").Device(DEVICE_CPU), LMDBReaderOp);

}  // namespace tensorflow
