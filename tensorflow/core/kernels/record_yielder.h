/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_RECORD_YIELDER_H_
#define TENSORFLOW_CORE_KERNELS_RECORD_YIELDER_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTh() {
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


#include <atomic>
#include <random>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// RecordYielder produces value records from a set of tfrecord files
// in a random order.
//
// It guarantees that:
//   1) all records in tfrecords are yielded within every epoch;
//   2) each record is yielded only once within every epoch;
//   3) the order in which records are yielded is highly randomized.
//   4) the peak memory usage is roughly avg record size *
//      (opts.bufsize + opts.parallelism * 16).
//
// Usage example:
//   RecordYielder::Options opts;
//   opts.file_pattern = "input-*";
//   opts.seed = 301;
//   opts.bufsize = 1000000;    // A randomized buffer with 1M records.
//   opts.parallelism = 8;      // Uses 8 tfrecord iterators to iterate
//                              // through all files.
//   RecordYielder yielder(opts);
//   string val;
//   while (true) {
//     yielder.YieldOne(&val);
//     // process val
//   }
//
// RecordYielder can be accessed by multiple threads concurrently.
class RecordYielder {
 public:
  struct Options {
    // Glob pattern for tfrecords.
    string file_pattern;

    // Random seed. It determines how data files are shuffled and how
    // records are shuffled.
    int64_t seed = 0;

    // Each epoch, all files are first shuffled according to the
    // random seed and the epoch number, and then all files are
    // left-shifted by file_shuffle_shift_ratio * num_files slots.  If
    // file_shuffle_shift_ratio is not within [0, 1), the
    // implementation clip it to [0, 1).
    float file_shuffle_shift_ratio = 0;

    // Randomization buffer keeps these many records.
    uint64 bufsize = 1;

    // Uses these many concurrent tfrecord iterators to iterate through
    // tfrecords.
    int32 parallelism = 1;

    string compression_type;
  };

  explicit RecordYielder(OpKernelConstruction* context,
                         const RecordYielder::Options& opts);
  ~RecordYielder();

  RecordYielder(const RecordYielder&) = delete;
  RecordYielder& operator=(const RecordYielder&) = delete;

  // Yields one 'value'.
  Status YieldOne(tstring* value);

  // Returns the current epoch number.
  int64_t current_epoch() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTh mht_0(mht_0_v, 265, "", "./tensorflow/core/kernels/record_yielder.h", "current_epoch");
 return epoch_; }

 private:
  typedef RecordYielder ME;

  Options opts_;

  // Backgrounds threads. Owned.
  thread::ThreadPool* thread_;

  // Epoch number.
  std::atomic<int64_t> epoch_;

  mutex mu_;

  // Turned to true when this is deleted.
  bool stop_ TF_GUARDED_BY(mu_) = false;
  Status status_ TF_GUARDED_BY(mu_);

  // PRG used for randomization.
  std::mt19937_64 rnd_ TF_GUARDED_BY(mu_);

  // Randomization buffer.
  std::vector<string> buf_ TF_GUARDED_BY(mu_);

  // True iff we are draining an epoch.
  bool epoch_end_ = false;

  int64_t num_records_added_in_epoch_ = 0;
  int64_t num_records_yielded_in_epoch_ = 0;

  // Trigger when the main loop has exited.
  Notification main_loop_done_;

  // condition_variables.
  condition_variable buf_empty_;
  bool BufEmpty() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || buf_.empty();
  }

  condition_variable buf_not_full_;
  bool BufNotFull() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || buf_.size() < opts_.bufsize;
  }

  condition_variable buf_enough_;
  bool BufEnough() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    // NOTE: Unless we are finishing an epoch, we want to make sure
    // the buf_ contains enough randomized elements before yielding
    // any.
    return stop_ || !status_.ok() || (epoch_end_ && !buf_.empty()) ||
           (!epoch_end_ &&
            buf_.size() >= std::max<uint64>(1, opts_.bufsize / 2));
  }

  void MainLoop();
  struct Shard;
  void ShardLoop(Shard* shard);
  bool ShouldFinish(const Status& s);
  bool Add(std::vector<string>* values);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RECORD_YIELDER_H_
