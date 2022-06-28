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
class MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc() {
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

#include "tensorflow/core/kernels/record_yielder.h"

#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

RecordYielder::RecordYielder(OpKernelConstruction* context,
                             const RecordYielder::Options& opts)
    : opts_(opts),
      thread_(new thread::ThreadPool(context->env(), ThreadOptions(),
                                     "record_yielder", 1 + opts.parallelism,
                                     /* low_latency_hint */ false)),
      epoch_(0),
      rnd_(opts.seed) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/record_yielder.cc", "RecordYielder::RecordYielder");

  thread_->Schedule([this]() { MainLoop(); });
}

RecordYielder::~RecordYielder() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/kernels/record_yielder.cc", "RecordYielder::~RecordYielder");

  {
    mutex_lock l(mu_);
    stop_ = true;
    buf_empty_.notify_all();
    buf_enough_.notify_all();
    buf_not_full_.notify_all();
  }
  main_loop_done_.WaitForNotification();
  delete thread_;
}

Status RecordYielder::YieldOne(tstring* value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/kernels/record_yielder.cc", "RecordYielder::YieldOne");

  mutex_lock l(mu_);
  while (!BufEnough() && status_.ok()) {
    buf_enough_.wait(l);
  }
  if (status_.ok()) {
    bool notify_no_longer_full = !BufNotFull();
    CHECK(!stop_ && !buf_.empty());
    *value = std::move(buf_.back());
    buf_.pop_back();
    ++num_records_yielded_in_epoch_;
    // Assumption is that an epoch always has something in the buffer
    // until it ends.  If the input pipeline was slower than the consumers
    // by a lot this might not be true.  Not sure how to handle.
    if (buf_.empty()) {
      buf_empty_.notify_all();
    }
    if (notify_no_longer_full) {
      buf_not_full_.notify_all();
    }
  }
  return status_;
}

struct RecordYielder::Shard {
  int index;                      // Shard index.
  std::vector<tstring> filenames;  // File names given to this shard.
  Notification done;              // Notified when this shard is done.
  Status status;                  // Shard status.
};

bool RecordYielder::ShouldFinish(const Status& s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc mht_3(mht_3_v, 256, "", "./tensorflow/core/kernels/record_yielder.cc", "RecordYielder::ShouldFinish");

  mutex_lock l(mu_);
  status_.Update(s);
  return stop_ || !status_.ok();
}

static Status MatchFiles(const string& patterns,
                         std::vector<string>* filenames) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("patterns: \"" + patterns + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc mht_4(mht_4_v, 267, "", "./tensorflow/core/kernels/record_yielder.cc", "MatchFiles");

  for (const auto& file_pattern : str_util::Split(patterns, ',')) {
    std::vector<string> tmp_filenames;
    TF_RETURN_IF_ERROR(
        Env::Default()->GetMatchingPaths(file_pattern, &tmp_filenames));
    filenames->insert(filenames->end(),
                      std::make_move_iterator(tmp_filenames.begin()),
                      std::make_move_iterator(tmp_filenames.end()));
  }
  return Status::OK();
}

void RecordYielder::MainLoop() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc mht_5(mht_5_v, 282, "", "./tensorflow/core/kernels/record_yielder.cc", "RecordYielder::MainLoop");

  while (true) {
    ++epoch_;
    num_records_yielded_in_epoch_ = 0;
    num_records_added_in_epoch_ = 0;

    // Finds all files.
    std::vector<string> filenames;
    Status s = MatchFiles(opts_.file_pattern, &filenames);

    if (filenames.empty()) {
      s = errors::NotFound("Found no files at ", opts_.file_pattern);
      if (ShouldFinish(s)) {
        buf_enough_.notify_all();
        break;
      }
    }

    if (ShouldFinish(s)) break;

    // Shuffles these files according to the epoch # and random seed.
    std::mt19937_64 shuffle_rnd(
        Hash64(reinterpret_cast<char*>(&epoch_), sizeof(epoch_), opts_.seed));
    std::shuffle(filenames.begin(), filenames.end(), shuffle_rnd);

    // Left-shift the filename list.
    const std::vector<string>::size_type num = filenames.size();
    int64_t shift;
    if (0 <= opts_.file_shuffle_shift_ratio &&
        opts_.file_shuffle_shift_ratio < 1) {
      shift = opts_.file_shuffle_shift_ratio * num;
      std::rotate(filenames.begin(), filenames.begin() + shift,
                  filenames.end());
    }

    // Shards files and use one thread to go through each shard.
    const int N = opts_.parallelism;
    std::vector<Shard> shards(N);
    for (int i = 0; i < N; ++i) {
      Shard* shard = &shards[i];
      shard->index = i;
      for (std::vector<string>::size_type j = i; j < filenames.size(); j += N) {
        shard->filenames.push_back(filenames[j]);
      }
      thread_->Schedule([this, shard]() { ShardLoop(shard); });
    }
    for (int i = 0; i < N; ++i) {
      shards[i].done.WaitForNotification();
      s.Update(shards[i].status);
    }

    if (num_records_added_in_epoch_ < opts_.bufsize) {
      mutex_lock l(mu_);
      opts_.bufsize = num_records_added_in_epoch_;
    }

    if (ShouldFinish(s)) {
      buf_enough_.notify_all();
      break;
    }

    // Starts the next epoch once all buffered records are consumed.
    {
      mutex_lock l(mu_);
      epoch_end_ = true;
      if (BufEnough()) {
        buf_enough_.notify_all();
      }
      while (!BufEmpty()) {
        buf_empty_.wait(l);
      }
      epoch_end_ = false;
    }
  }
  main_loop_done_.Notify();
}

bool RecordYielder::Add(std::vector<string>* values) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc mht_6(mht_6_v, 362, "", "./tensorflow/core/kernels/record_yielder.cc", "RecordYielder::Add");

  mutex_lock l(mu_);
  while (!BufNotFull()) {
    buf_not_full_.wait(l);
  }
  while (BufNotFull() && !values->empty()) {
    // Adds values->back(). Swaps its position with another random
    // element.
    auto index = rnd_() % (buf_.size() + 1);
    if (index == buf_.size()) {
      buf_.push_back(std::move(values->back()));
    } else {
      buf_.push_back(std::move(buf_[index]));
      buf_[index] = std::move(values->back());
    }
    values->pop_back();
    num_records_added_in_epoch_++;
  }
  if (BufEnough()) {
    buf_enough_.notify_all();
  }
  return stop_;
}

void RecordYielder::ShardLoop(Shard* shard) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrecord_yielderDTcc mht_7(mht_7_v, 389, "", "./tensorflow/core/kernels/record_yielder.cc", "RecordYielder::ShardLoop");

  std::vector<string> values;
  const int64_t kRecords = 16;
  for (const string& filename : shard->filenames) {
    std::unique_ptr<RandomAccessFile> file;
    if (ShouldFinish(Status::OK())) break;
    Status s = Env::Default()->NewRandomAccessFile(filename, &file);
    if (!s.ok()) {
      shard->status = errors::InvalidArgument("Can't open ", filename);
      break;
    }
    io::RecordReaderOptions options =
        io::RecordReaderOptions::CreateRecordReaderOptions(
            opts_.compression_type);
    io::RecordReader rdr(file.get(), options);
    uint64 offset = 0;
    tstring record;
    while (true) {
      Status s = rdr.ReadRecord(&offset, &record);
      if (s.ok()) {
        values.emplace_back(std::move(record));
        if (values.size() >= kRecords && Add(&values)) {
          shard->status = errors::Aborted("stopped");
          break;
        }
      } else if (errors::IsOutOfRange(s)) {
        break;
      } else {
        shard->status = s;
        break;
      }
    }
  }
  // Adds the remaining values of this shard to buf_.
  while (!values.empty()) {
    Add(&values);
  }
  shard->done.Notify();
}

}  // namespace tensorflow
