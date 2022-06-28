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
class MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc() {
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

#include "tensorflow/core/lib/io/two_level_iterator.h"

#include "tensorflow/core/lib/io/block.h"
#include "tensorflow/core/lib/io/format.h"
#include "tensorflow/core/lib/io/iterator.h"
#include "tensorflow/core/lib/io/table.h"

namespace tensorflow {
namespace table {

namespace {

typedef Iterator* (*BlockFunction)(void*, const StringPiece&);

class TwoLevelIterator : public Iterator {
 public:
  TwoLevelIterator(Iterator* index_iter, BlockFunction block_function,
                   void* arg);

  ~TwoLevelIterator() override;

  void Seek(const StringPiece& target) override;
  void SeekToFirst() override;
  void Next() override;

  bool Valid() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "Valid");

    return (data_iter_ == nullptr) ? false : data_iter_->Valid();
  }
  StringPiece key() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "key");

    assert(Valid());
    return data_iter_->key();
  }
  StringPiece value() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_2(mht_2_v, 223, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "value");

    assert(Valid());
    return data_iter_->value();
  }
  Status status() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_3(mht_3_v, 230, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "status");

    // It'd be nice if status() returned a const Status& instead of a
    // Status
    if (!index_iter_->status().ok()) {
      return index_iter_->status();
    } else if (data_iter_ != nullptr && !data_iter_->status().ok()) {
      return data_iter_->status();
    } else {
      return status_;
    }
  }

 private:
  void SaveError(const Status& s) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_4(mht_4_v, 246, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "SaveError");

    if (status_.ok() && !s.ok()) status_ = s;
  }
  void SkipEmptyDataBlocksForward();
  void SetDataIterator(Iterator* data_iter);
  void InitDataBlock();

  BlockFunction block_function_;
  void* arg_;
  Status status_;
  Iterator* index_iter_;
  Iterator* data_iter_;  // May be NULL
  // If data_iter_ is non-NULL, then "data_block_handle_" holds the
  // "index_value" passed to block_function_ to create the data_iter_.
  string data_block_handle_;
};

TwoLevelIterator::TwoLevelIterator(Iterator* index_iter,
                                   BlockFunction block_function, void* arg)
    : block_function_(block_function),
      arg_(arg),
      index_iter_(index_iter),
      data_iter_(nullptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_5(mht_5_v, 271, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "TwoLevelIterator::TwoLevelIterator");
}

TwoLevelIterator::~TwoLevelIterator() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_6(mht_6_v, 276, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "TwoLevelIterator::~TwoLevelIterator");

  delete index_iter_;
  delete data_iter_;
}

void TwoLevelIterator::Seek(const StringPiece& target) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_7(mht_7_v, 284, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "TwoLevelIterator::Seek");

  index_iter_->Seek(target);
  InitDataBlock();
  if (data_iter_ != nullptr) data_iter_->Seek(target);
  SkipEmptyDataBlocksForward();
}

void TwoLevelIterator::SeekToFirst() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_8(mht_8_v, 294, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "TwoLevelIterator::SeekToFirst");

  index_iter_->SeekToFirst();
  InitDataBlock();
  if (data_iter_ != nullptr) data_iter_->SeekToFirst();
  SkipEmptyDataBlocksForward();
}

void TwoLevelIterator::Next() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_9(mht_9_v, 304, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "TwoLevelIterator::Next");

  assert(Valid());
  data_iter_->Next();
  SkipEmptyDataBlocksForward();
}

void TwoLevelIterator::SkipEmptyDataBlocksForward() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_10(mht_10_v, 313, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "TwoLevelIterator::SkipEmptyDataBlocksForward");

  while (data_iter_ == nullptr || !data_iter_->Valid()) {
    // Move to next block
    if (!index_iter_->Valid()) {
      SetDataIterator(nullptr);
      return;
    }
    index_iter_->Next();
    InitDataBlock();
    if (data_iter_ != nullptr) data_iter_->SeekToFirst();
  }
}

void TwoLevelIterator::SetDataIterator(Iterator* data_iter) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_11(mht_11_v, 329, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "TwoLevelIterator::SetDataIterator");

  if (data_iter_ != nullptr) {
    SaveError(data_iter_->status());
    delete data_iter_;
  }
  data_iter_ = data_iter;
}

void TwoLevelIterator::InitDataBlock() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_12(mht_12_v, 340, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "TwoLevelIterator::InitDataBlock");

  if (!index_iter_->Valid()) {
    SetDataIterator(nullptr);
  } else {
    StringPiece handle = index_iter_->value();
    if (data_iter_ != nullptr && handle.compare(data_block_handle_) == 0) {
      // data_iter_ is already constructed with this iterator, so
      // no need to change anything
    } else {
      Iterator* iter = (*block_function_)(arg_, handle);
      data_block_handle_.assign(handle.data(), handle.size());
      SetDataIterator(iter);
    }
  }
}

}  // namespace

Iterator* NewTwoLevelIterator(Iterator* index_iter,
                              BlockFunction block_function, void* arg) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStwo_level_iteratorDTcc mht_13(mht_13_v, 362, "", "./tensorflow/core/lib/io/two_level_iterator.cc", "NewTwoLevelIterator");

  return new TwoLevelIterator(index_iter, block_function, arg);
}

}  // namespace table
}  // namespace tensorflow
