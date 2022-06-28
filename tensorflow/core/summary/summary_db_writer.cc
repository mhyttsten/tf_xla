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
class MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc {
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
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc() {
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
#include "tensorflow/core/summary/summary_db_writer.h"

#include <deque>

#include "tensorflow/core/summary/summary_converter.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/event.pb.h"

// TODO(jart): Break this up into multiple files with excellent unit tests.
// TODO(jart): Make decision to write in separate op.
// TODO(jart): Add really good busy handling.

// clang-format off
#define CALL_SUPPORTED_TYPES(m) \
  TF_CALL_tstring(m)             \
  TF_CALL_half(m)               \
  TF_CALL_float(m)              \
  TF_CALL_double(m)             \
  TF_CALL_complex64(m)          \
  TF_CALL_complex128(m)         \
  TF_CALL_int8(m)               \
  TF_CALL_int16(m)              \
  TF_CALL_int32(m)              \
  TF_CALL_int64(m)              \
  TF_CALL_uint8(m)              \
  TF_CALL_uint16(m)             \
  TF_CALL_uint32(m)             \
  TF_CALL_uint64(m)
// clang-format on

namespace tensorflow {
namespace {

// https://www.sqlite.org/fileformat.html#record_format
const uint64 kIdTiers[] = {
    0x7fffffULL,        // 23-bit (3 bytes on disk)
    0x7fffffffULL,      // 31-bit (4 bytes on disk)
    0x7fffffffffffULL,  // 47-bit (5 bytes on disk)
                        // remaining bits for future use
};
const int kMaxIdTier = sizeof(kIdTiers) / sizeof(uint64);
const int kIdCollisionDelayMicros = 10;
const int kMaxIdCollisions = 21;  // sum(2**i*10µs for i in range(21))~=21s
const int64_t kAbsent = 0LL;

const char* kScalarPluginName = "scalars";
const char* kImagePluginName = "images";
const char* kAudioPluginName = "audio";
const char* kHistogramPluginName = "histograms";

const int64_t kReserveMinBytes = 32;
const double kReserveMultiplier = 1.5;
const int64_t kPreallocateRows = 1000;

// Flush is a misnomer because what we're actually doing is having lots
// of commits inside any SqliteTransaction that writes potentially
// hundreds of megs but doesn't need the transaction to maintain its
// invariants. This ensures the WAL read penalty is small and might
// allow writers in other processes a chance to schedule.
const uint64 kFlushBytes = 1024 * 1024;

double DoubleTime(uint64 micros) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_0(mht_0_v, 251, "", "./tensorflow/core/summary/summary_db_writer.cc", "DoubleTime");

  // TODO(@jart): Follow precise definitions for time laid out in schema.
  // TODO(@jart): Use monotonic clock from gRPC codebase.
  return static_cast<double>(micros) / 1.0e6;
}

string StringifyShape(const TensorShape& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_1(mht_1_v, 260, "", "./tensorflow/core/summary/summary_db_writer.cc", "StringifyShape");

  string result;
  bool first = true;
  for (const auto& dim : shape) {
    if (first) {
      first = false;
    } else {
      strings::StrAppend(&result, ",");
    }
    strings::StrAppend(&result, dim.size);
  }
  return result;
}

Status CheckSupportedType(const Tensor& t) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_2(mht_2_v, 277, "", "./tensorflow/core/summary/summary_db_writer.cc", "CheckSupportedType");

#define CASE(T)                  \
  case DataTypeToEnum<T>::value: \
    break;
  switch (t.dtype()) {
    CALL_SUPPORTED_TYPES(CASE)
    default:
      return errors::Unimplemented(DataTypeString(t.dtype()),
                                   " tensors unsupported on platform");
  }
  return Status::OK();
#undef CASE
}

Tensor AsScalar(const Tensor& t) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_3(mht_3_v, 294, "", "./tensorflow/core/summary/summary_db_writer.cc", "AsScalar");

  Tensor t2{t.dtype(), {}};
#define CASE(T)                        \
  case DataTypeToEnum<T>::value:       \
    t2.scalar<T>()() = t.flat<T>()(0); \
    break;
  switch (t.dtype()) {
    CALL_SUPPORTED_TYPES(CASE)
    default:
      t2 = {DT_FLOAT, {}};
      t2.scalar<float>()() = NAN;
      break;
  }
  return t2;
#undef CASE
}

void PatchPluginName(SummaryMetadata* metadata, const char* name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_4(mht_4_v, 315, "", "./tensorflow/core/summary/summary_db_writer.cc", "PatchPluginName");

  if (metadata->plugin_data().plugin_name().empty()) {
    metadata->mutable_plugin_data()->set_plugin_name(name);
  }
}

Status SetDescription(Sqlite* db, int64_t id, const StringPiece& markdown) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_5(mht_5_v, 324, "", "./tensorflow/core/summary/summary_db_writer.cc", "SetDescription");

  const char* sql = R"sql(
    INSERT OR REPLACE INTO Descriptions (id, description) VALUES (?, ?)
  )sql";
  SqliteStatement insert_desc;
  TF_RETURN_IF_ERROR(db->Prepare(sql, &insert_desc));
  insert_desc.BindInt(1, id);
  insert_desc.BindText(2, markdown);
  return insert_desc.StepAndReset();
}

/// \brief Generates unique IDs randomly in the [1,2**63-1] range.
///
/// This class starts off generating IDs in the [1,2**23-1] range,
/// because it's human friendly and occupies 4 bytes max on disk with
/// SQLite's zigzag varint encoding. Then, each time a collision
/// happens, the random space is increased by 8 bits.
///
/// This class uses exponential back-off so writes gradually slow down
/// as IDs become exhausted but reads are still possible.
///
/// This class is thread safe.
class IdAllocator {
 public:
  IdAllocator(Env* env, Sqlite* db) : env_{env}, db_{db} {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_6(mht_6_v, 351, "", "./tensorflow/core/summary/summary_db_writer.cc", "IdAllocator");

    DCHECK(env_ != nullptr);
    DCHECK(db_ != nullptr);
  }

  Status CreateNewId(int64_t* id) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    Status s;
    SqliteStatement stmt;
    TF_RETURN_IF_ERROR(db_->Prepare("INSERT INTO Ids (id) VALUES (?)", &stmt));
    for (int i = 0; i < kMaxIdCollisions; ++i) {
      int64_t tid = MakeRandomId();
      stmt.BindInt(1, tid);
      s = stmt.StepAndReset();
      if (s.ok()) {
        *id = tid;
        break;
      }
      // SQLITE_CONSTRAINT maps to INVALID_ARGUMENT in sqlite.cc
      if (s.code() != error::INVALID_ARGUMENT) break;
      if (tier_ < kMaxIdTier) {
        LOG(INFO) << "IdAllocator collision at tier " << tier_ << " (of "
                  << kMaxIdTier << ") so auto-adjusting to a higher tier";
        ++tier_;
      } else {
        LOG(WARNING) << "IdAllocator (attempt #" << i << ") "
                     << "resulted in a collision at the highest tier; this "
                        "is problematic if it happens often; you can try "
                        "pruning the Ids table; you can also file a bug "
                        "asking for the ID space to be increased; otherwise "
                        "writes will gradually slow down over time until they "
                        "become impossible";
      }
      env_->SleepForMicroseconds((1 << i) * kIdCollisionDelayMicros);
    }
    return s;
  }

 private:
  int64_t MakeRandomId() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_7(mht_7_v, 393, "", "./tensorflow/core/summary/summary_db_writer.cc", "MakeRandomId");

    int64_t id = static_cast<int64_t>(random::New64() & kIdTiers[tier_]);
    if (id == kAbsent) ++id;
    return id;
  }

  mutex mu_;
  Env* const env_;
  Sqlite* const db_;
  int tier_ TF_GUARDED_BY(mu_) = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(IdAllocator);
};

class GraphWriter {
 public:
  static Status Save(Sqlite* db, SqliteTransaction* txn, IdAllocator* ids,
                     GraphDef* graph, uint64 now, int64_t run_id,
                     int64_t* graph_id)
      SQLITE_EXCLUSIVE_TRANSACTIONS_REQUIRED(*db) {
    TF_RETURN_IF_ERROR(ids->CreateNewId(graph_id));
    GraphWriter saver{db, txn, graph, now, *graph_id};
    saver.MapNameToNodeId();
    TF_RETURN_WITH_CONTEXT_IF_ERROR(saver.SaveNodeInputs(), "SaveNodeInputs");
    TF_RETURN_WITH_CONTEXT_IF_ERROR(saver.SaveNodes(), "SaveNodes");
    TF_RETURN_WITH_CONTEXT_IF_ERROR(saver.SaveGraph(run_id), "SaveGraph");
    return Status::OK();
  }

 private:
  GraphWriter(Sqlite* db, SqliteTransaction* txn, GraphDef* graph, uint64 now,
              int64_t graph_id)
      : db_(db), txn_(txn), graph_(graph), now_(now), graph_id_(graph_id) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_8(mht_8_v, 428, "", "./tensorflow/core/summary/summary_db_writer.cc", "GraphWriter");
}

  void MapNameToNodeId() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_9(mht_9_v, 433, "", "./tensorflow/core/summary/summary_db_writer.cc", "MapNameToNodeId");

    size_t toto = static_cast<size_t>(graph_->node_size());
    name_copies_.reserve(toto);
    name_to_node_id_.reserve(toto);
    for (int node_id = 0; node_id < graph_->node_size(); ++node_id) {
      // Copy name into memory region, since we call clear_name() later.
      // Then wrap in StringPiece so we can compare slices without copy.
      name_copies_.emplace_back(graph_->node(node_id).name());
      name_to_node_id_.emplace(name_copies_.back(), node_id);
    }
  }

  Status SaveNodeInputs() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_10(mht_10_v, 448, "", "./tensorflow/core/summary/summary_db_writer.cc", "SaveNodeInputs");

    const char* sql = R"sql(
      INSERT INTO NodeInputs (
        graph_id,
        node_id,
        idx,
        input_node_id,
        input_node_idx,
        is_control
      ) VALUES (?, ?, ?, ?, ?, ?)
    )sql";
    SqliteStatement insert;
    TF_RETURN_IF_ERROR(db_->Prepare(sql, &insert));
    for (int node_id = 0; node_id < graph_->node_size(); ++node_id) {
      const NodeDef& node = graph_->node(node_id);
      for (int idx = 0; idx < node.input_size(); ++idx) {
        StringPiece name = node.input(idx);
        int64_t input_node_id;
        int64_t input_node_idx = 0;
        int64_t is_control = 0;
        size_t i = name.rfind(':');
        if (i != StringPiece::npos) {
          if (!strings::safe_strto64(name.substr(i + 1, name.size() - i - 1),
                                     &input_node_idx)) {
            return errors::DataLoss("Bad NodeDef.input: ", name);
          }
          name.remove_suffix(name.size() - i);
        }
        if (!name.empty() && name[0] == '^') {
          name.remove_prefix(1);
          is_control = 1;
        }
        auto e = name_to_node_id_.find(name);
        if (e == name_to_node_id_.end()) {
          return errors::DataLoss("Could not find node: ", name);
        }
        input_node_id = e->second;
        insert.BindInt(1, graph_id_);
        insert.BindInt(2, node_id);
        insert.BindInt(3, idx);
        insert.BindInt(4, input_node_id);
        insert.BindInt(5, input_node_idx);
        insert.BindInt(6, is_control);
        unflushed_bytes_ += insert.size();
        TF_RETURN_WITH_CONTEXT_IF_ERROR(insert.StepAndReset(), node.name(),
                                        " -> ", name);
        TF_RETURN_IF_ERROR(MaybeFlush());
      }
    }
    return Status::OK();
  }

  Status SaveNodes() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_11(mht_11_v, 503, "", "./tensorflow/core/summary/summary_db_writer.cc", "SaveNodes");

    const char* sql = R"sql(
      INSERT INTO Nodes (
        graph_id,
        node_id,
        node_name,
        op,
        device,
        node_def)
      VALUES (?, ?, ?, ?, ?, ?)
    )sql";
    SqliteStatement insert;
    TF_RETURN_IF_ERROR(db_->Prepare(sql, &insert));
    for (int node_id = 0; node_id < graph_->node_size(); ++node_id) {
      NodeDef* node = graph_->mutable_node(node_id);
      insert.BindInt(1, graph_id_);
      insert.BindInt(2, node_id);
      insert.BindText(3, node->name());
      insert.BindText(4, node->op());
      insert.BindText(5, node->device());
      node->clear_name();
      node->clear_op();
      node->clear_device();
      node->clear_input();
      string node_def;
      if (node->SerializeToString(&node_def)) {
        insert.BindBlobUnsafe(6, node_def);
      }
      unflushed_bytes_ += insert.size();
      TF_RETURN_WITH_CONTEXT_IF_ERROR(insert.StepAndReset(), node->name());
      TF_RETURN_IF_ERROR(MaybeFlush());
    }
    return Status::OK();
  }

  Status SaveGraph(int64_t run_id) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_12(mht_12_v, 541, "", "./tensorflow/core/summary/summary_db_writer.cc", "SaveGraph");

    const char* sql = R"sql(
      INSERT OR REPLACE INTO Graphs (
        run_id,
        graph_id,
        inserted_time,
        graph_def
      ) VALUES (?, ?, ?, ?)
    )sql";
    SqliteStatement insert;
    TF_RETURN_IF_ERROR(db_->Prepare(sql, &insert));
    if (run_id != kAbsent) insert.BindInt(1, run_id);
    insert.BindInt(2, graph_id_);
    insert.BindDouble(3, DoubleTime(now_));
    graph_->clear_node();
    string graph_def;
    if (graph_->SerializeToString(&graph_def)) {
      insert.BindBlobUnsafe(4, graph_def);
    }
    return insert.StepAndReset();
  }

  Status MaybeFlush() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_13(mht_13_v, 566, "", "./tensorflow/core/summary/summary_db_writer.cc", "MaybeFlush");

    if (unflushed_bytes_ >= kFlushBytes) {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(txn_->Commit(), "flushing ",
                                      unflushed_bytes_, " bytes");
      unflushed_bytes_ = 0;
    }
    return Status::OK();
  }

  Sqlite* const db_;
  SqliteTransaction* const txn_;
  uint64 unflushed_bytes_ = 0;
  GraphDef* const graph_;
  const uint64 now_;
  const int64_t graph_id_;
  std::vector<string> name_copies_;
  std::unordered_map<StringPiece, int64_t, StringPieceHasher> name_to_node_id_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphWriter);
};

/// \brief Run metadata manager.
///
/// This class gives us Tag IDs we can pass to SeriesWriter. In order
/// to do that, rows are created in the Ids, Tags, Runs, Experiments,
/// and Users tables.
///
/// This class is thread safe.
class RunMetadata {
 public:
  RunMetadata(IdAllocator* ids, const string& experiment_name,
              const string& run_name, const string& user_name)
      : ids_{ids},
        experiment_name_{experiment_name},
        run_name_{run_name},
        user_name_{user_name} {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("experiment_name: \"" + experiment_name + "\"");
   mht_14_v.push_back("run_name: \"" + run_name + "\"");
   mht_14_v.push_back("user_name: \"" + user_name + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_14(mht_14_v, 607, "", "./tensorflow/core/summary/summary_db_writer.cc", "RunMetadata");

    DCHECK(ids_ != nullptr);
  }

  const string& experiment_name() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_15(mht_15_v, 614, "", "./tensorflow/core/summary/summary_db_writer.cc", "experiment_name");
 return experiment_name_; }
  const string& run_name() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_16(mht_16_v, 618, "", "./tensorflow/core/summary/summary_db_writer.cc", "run_name");
 return run_name_; }
  const string& user_name() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_17(mht_17_v, 622, "", "./tensorflow/core/summary/summary_db_writer.cc", "user_name");
 return user_name_; }

  int64_t run_id() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    return run_id_;
  }

  Status SetGraph(Sqlite* db, uint64 now, double computed_time,
                  std::unique_ptr<GraphDef> g) SQLITE_TRANSACTIONS_EXCLUDED(*db)
      TF_LOCKS_EXCLUDED(mu_) {
    int64_t run_id;
    {
      mutex_lock lock(mu_);
      TF_RETURN_IF_ERROR(InitializeRun(db, now, computed_time));
      run_id = run_id_;
    }
    int64_t graph_id;
    SqliteTransaction txn(*db);  // only to increase performance
    TF_RETURN_IF_ERROR(
        GraphWriter::Save(db, &txn, ids_, g.get(), now, run_id, &graph_id));
    return txn.Commit();
  }

  Status GetTagId(Sqlite* db, uint64 now, double computed_time,
                  const string& tag_name, int64_t* tag_id,
                  const SummaryMetadata& metadata) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    TF_RETURN_IF_ERROR(InitializeRun(db, now, computed_time));
    auto e = tag_ids_.find(tag_name);
    if (e != tag_ids_.end()) {
      *tag_id = e->second;
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(ids_->CreateNewId(tag_id));
    tag_ids_[tag_name] = *tag_id;
    TF_RETURN_IF_ERROR(
        SetDescription(db, *tag_id, metadata.summary_description()));
    const char* sql = R"sql(
      INSERT INTO Tags (
        run_id,
        tag_id,
        tag_name,
        inserted_time,
        display_name,
        plugin_name,
        plugin_data
      ) VALUES (
        :run_id,
        :tag_id,
        :tag_name,
        :inserted_time,
        :display_name,
        :plugin_name,
        :plugin_data
      )
    )sql";
    SqliteStatement insert;
    TF_RETURN_IF_ERROR(db->Prepare(sql, &insert));
    if (run_id_ != kAbsent) insert.BindInt(":run_id", run_id_);
    insert.BindInt(":tag_id", *tag_id);
    insert.BindTextUnsafe(":tag_name", tag_name);
    insert.BindDouble(":inserted_time", DoubleTime(now));
    insert.BindTextUnsafe(":display_name", metadata.display_name());
    insert.BindTextUnsafe(":plugin_name", metadata.plugin_data().plugin_name());
    insert.BindBlobUnsafe(":plugin_data", metadata.plugin_data().content());
    return insert.StepAndReset();
  }

 private:
  Status InitializeUser(Sqlite* db, uint64 now)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_18(mht_18_v, 695, "", "./tensorflow/core/summary/summary_db_writer.cc", "InitializeUser");

    if (user_id_ != kAbsent || user_name_.empty()) return Status::OK();
    const char* get_sql = R"sql(
      SELECT user_id FROM Users WHERE user_name = ?
    )sql";
    SqliteStatement get;
    TF_RETURN_IF_ERROR(db->Prepare(get_sql, &get));
    get.BindText(1, user_name_);
    bool is_done;
    TF_RETURN_IF_ERROR(get.Step(&is_done));
    if (!is_done) {
      user_id_ = get.ColumnInt(0);
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(ids_->CreateNewId(&user_id_));
    const char* insert_sql = R"sql(
      INSERT INTO Users (
        user_id,
        user_name,
        inserted_time
      ) VALUES (?, ?, ?)
    )sql";
    SqliteStatement insert;
    TF_RETURN_IF_ERROR(db->Prepare(insert_sql, &insert));
    insert.BindInt(1, user_id_);
    insert.BindText(2, user_name_);
    insert.BindDouble(3, DoubleTime(now));
    TF_RETURN_IF_ERROR(insert.StepAndReset());
    return Status::OK();
  }

  Status InitializeExperiment(Sqlite* db, uint64 now, double computed_time)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_19(mht_19_v, 730, "", "./tensorflow/core/summary/summary_db_writer.cc", "InitializeExperiment");

    if (experiment_name_.empty()) return Status::OK();
    if (experiment_id_ == kAbsent) {
      TF_RETURN_IF_ERROR(InitializeUser(db, now));
      const char* get_sql = R"sql(
        SELECT
          experiment_id,
          started_time
        FROM
          Experiments
        WHERE
          user_id IS ?
          AND experiment_name = ?
      )sql";
      SqliteStatement get;
      TF_RETURN_IF_ERROR(db->Prepare(get_sql, &get));
      if (user_id_ != kAbsent) get.BindInt(1, user_id_);
      get.BindText(2, experiment_name_);
      bool is_done;
      TF_RETURN_IF_ERROR(get.Step(&is_done));
      if (!is_done) {
        experiment_id_ = get.ColumnInt(0);
        experiment_started_time_ = get.ColumnInt(1);
      } else {
        TF_RETURN_IF_ERROR(ids_->CreateNewId(&experiment_id_));
        experiment_started_time_ = computed_time;
        const char* insert_sql = R"sql(
          INSERT INTO Experiments (
            user_id,
            experiment_id,
            experiment_name,
            inserted_time,
            started_time,
            is_watching
          ) VALUES (?, ?, ?, ?, ?, ?)
        )sql";
        SqliteStatement insert;
        TF_RETURN_IF_ERROR(db->Prepare(insert_sql, &insert));
        if (user_id_ != kAbsent) insert.BindInt(1, user_id_);
        insert.BindInt(2, experiment_id_);
        insert.BindText(3, experiment_name_);
        insert.BindDouble(4, DoubleTime(now));
        insert.BindDouble(5, computed_time);
        insert.BindInt(6, 0);
        TF_RETURN_IF_ERROR(insert.StepAndReset());
      }
    }
    if (computed_time < experiment_started_time_) {
      experiment_started_time_ = computed_time;
      const char* update_sql = R"sql(
        UPDATE
          Experiments
        SET
          started_time = ?
        WHERE
          experiment_id = ?
      )sql";
      SqliteStatement update;
      TF_RETURN_IF_ERROR(db->Prepare(update_sql, &update));
      update.BindDouble(1, computed_time);
      update.BindInt(2, experiment_id_);
      TF_RETURN_IF_ERROR(update.StepAndReset());
    }
    return Status::OK();
  }

  Status InitializeRun(Sqlite* db, uint64 now, double computed_time)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_20(mht_20_v, 800, "", "./tensorflow/core/summary/summary_db_writer.cc", "InitializeRun");

    if (run_name_.empty()) return Status::OK();
    TF_RETURN_IF_ERROR(InitializeExperiment(db, now, computed_time));
    if (run_id_ == kAbsent) {
      TF_RETURN_IF_ERROR(ids_->CreateNewId(&run_id_));
      run_started_time_ = computed_time;
      const char* insert_sql = R"sql(
        INSERT OR REPLACE INTO Runs (
          experiment_id,
          run_id,
          run_name,
          inserted_time,
          started_time
        ) VALUES (?, ?, ?, ?, ?)
      )sql";
      SqliteStatement insert;
      TF_RETURN_IF_ERROR(db->Prepare(insert_sql, &insert));
      if (experiment_id_ != kAbsent) insert.BindInt(1, experiment_id_);
      insert.BindInt(2, run_id_);
      insert.BindText(3, run_name_);
      insert.BindDouble(4, DoubleTime(now));
      insert.BindDouble(5, computed_time);
      TF_RETURN_IF_ERROR(insert.StepAndReset());
    }
    if (computed_time < run_started_time_) {
      run_started_time_ = computed_time;
      const char* update_sql = R"sql(
        UPDATE
          Runs
        SET
          started_time = ?
        WHERE
          run_id = ?
      )sql";
      SqliteStatement update;
      TF_RETURN_IF_ERROR(db->Prepare(update_sql, &update));
      update.BindDouble(1, computed_time);
      update.BindInt(2, run_id_);
      TF_RETURN_IF_ERROR(update.StepAndReset());
    }
    return Status::OK();
  }

  mutex mu_;
  IdAllocator* const ids_;
  const string experiment_name_;
  const string run_name_;
  const string user_name_;
  int64_t experiment_id_ TF_GUARDED_BY(mu_) = kAbsent;
  int64_t run_id_ TF_GUARDED_BY(mu_) = kAbsent;
  int64_t user_id_ TF_GUARDED_BY(mu_) = kAbsent;
  double experiment_started_time_ TF_GUARDED_BY(mu_) = 0.0;
  double run_started_time_ TF_GUARDED_BY(mu_) = 0.0;
  std::unordered_map<string, int64_t> tag_ids_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RunMetadata);
};

/// \brief Tensor writer for a single series, e.g. Tag.
///
/// This class is thread safe.
class SeriesWriter {
 public:
  SeriesWriter(int64_t series, RunMetadata* meta)
      : series_{series}, meta_{meta} {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_21(mht_21_v, 867, "", "./tensorflow/core/summary/summary_db_writer.cc", "SeriesWriter");

    DCHECK(series_ > 0);
  }

  Status Append(Sqlite* db, int64_t step, uint64 now, double computed_time,
                const Tensor& t) SQLITE_TRANSACTIONS_EXCLUDED(*db)
      TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    if (rowids_.empty()) {
      Status s = Reserve(db, t);
      if (!s.ok()) {
        rowids_.clear();
        return s;
      }
    }
    int64_t rowid = rowids_.front();
    Status s = Write(db, rowid, step, computed_time, t);
    if (s.ok()) {
      ++count_;
    }
    rowids_.pop_front();
    return s;
  }

  Status Finish(Sqlite* db) SQLITE_TRANSACTIONS_EXCLUDED(*db)
      TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    // Delete unused pre-allocated Tensors.
    if (!rowids_.empty()) {
      SqliteTransaction txn(*db);
      const char* sql = R"sql(
        DELETE FROM Tensors WHERE rowid = ?
      )sql";
      SqliteStatement deleter;
      TF_RETURN_IF_ERROR(db->Prepare(sql, &deleter));
      for (size_t i = count_; i < rowids_.size(); ++i) {
        deleter.BindInt(1, rowids_.front());
        TF_RETURN_IF_ERROR(deleter.StepAndReset());
        rowids_.pop_front();
      }
      TF_RETURN_IF_ERROR(txn.Commit());
      rowids_.clear();
    }
    return Status::OK();
  }

 private:
  Status Write(Sqlite* db, int64_t rowid, int64_t step, double computed_time,
               const Tensor& t) SQLITE_TRANSACTIONS_EXCLUDED(*db) {
    if (t.dtype() == DT_STRING) {
      if (t.dims() == 0) {
        return Update(db, step, computed_time, t, t.scalar<tstring>()(), rowid);
      } else {
        SqliteTransaction txn(*db);
        TF_RETURN_IF_ERROR(
            Update(db, step, computed_time, t, StringPiece(), rowid));
        TF_RETURN_IF_ERROR(UpdateNdString(db, t, rowid));
        return txn.Commit();
      }
    } else {
      return Update(db, step, computed_time, t, t.tensor_data(), rowid);
    }
  }

  Status Update(Sqlite* db, int64_t step, double computed_time, const Tensor& t,
                const StringPiece& data, int64_t rowid) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_22(mht_22_v, 935, "", "./tensorflow/core/summary/summary_db_writer.cc", "Update");

    const char* sql = R"sql(
      UPDATE OR REPLACE
        Tensors
      SET
        step = ?,
        computed_time = ?,
        dtype = ?,
        shape = ?,
        data = ?
      WHERE
        rowid = ?
    )sql";
    SqliteStatement stmt;
    TF_RETURN_IF_ERROR(db->Prepare(sql, &stmt));
    stmt.BindInt(1, step);
    stmt.BindDouble(2, computed_time);
    stmt.BindInt(3, t.dtype());
    stmt.BindText(4, StringifyShape(t.shape()));
    stmt.BindBlobUnsafe(5, data);
    stmt.BindInt(6, rowid);
    TF_RETURN_IF_ERROR(stmt.StepAndReset());
    return Status::OK();
  }

  Status UpdateNdString(Sqlite* db, const Tensor& t, int64_t tensor_rowid)
      SQLITE_EXCLUSIVE_TRANSACTIONS_REQUIRED(*db) {
    DCHECK_EQ(t.dtype(), DT_STRING);
    DCHECK_GT(t.dims(), 0);
    const char* deleter_sql = R"sql(
      DELETE FROM TensorStrings WHERE tensor_rowid = ?
    )sql";
    SqliteStatement deleter;
    TF_RETURN_IF_ERROR(db->Prepare(deleter_sql, &deleter));
    deleter.BindInt(1, tensor_rowid);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(deleter.StepAndReset(), tensor_rowid);
    const char* inserter_sql = R"sql(
      INSERT INTO TensorStrings (
        tensor_rowid,
        idx,
        data
      ) VALUES (?, ?, ?)
    )sql";
    SqliteStatement inserter;
    TF_RETURN_IF_ERROR(db->Prepare(inserter_sql, &inserter));
    auto flat = t.flat<tstring>();
    for (int64_t i = 0; i < flat.size(); ++i) {
      inserter.BindInt(1, tensor_rowid);
      inserter.BindInt(2, i);
      inserter.BindBlobUnsafe(3, flat(i));
      TF_RETURN_WITH_CONTEXT_IF_ERROR(inserter.StepAndReset(), "i=", i);
    }
    return Status::OK();
  }

  Status Reserve(Sqlite* db, const Tensor& t) SQLITE_TRANSACTIONS_EXCLUDED(*db)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    SqliteTransaction txn(*db);  // only for performance
    unflushed_bytes_ = 0;
    if (t.dtype() == DT_STRING) {
      if (t.dims() == 0) {
        TF_RETURN_IF_ERROR(ReserveData(db, &txn, t.scalar<tstring>()().size()));
      } else {
        TF_RETURN_IF_ERROR(ReserveTensors(db, &txn, kReserveMinBytes));
      }
    } else {
      TF_RETURN_IF_ERROR(ReserveData(db, &txn, t.tensor_data().size()));
    }
    return txn.Commit();
  }

  Status ReserveData(Sqlite* db, SqliteTransaction* txn, size_t size)
      SQLITE_EXCLUSIVE_TRANSACTIONS_REQUIRED(*db)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    int64_t space =
        static_cast<int64_t>(static_cast<double>(size) * kReserveMultiplier);
    if (space < kReserveMinBytes) space = kReserveMinBytes;
    return ReserveTensors(db, txn, space);
  }

  Status ReserveTensors(Sqlite* db, SqliteTransaction* txn,
                        int64_t reserved_bytes)
      SQLITE_EXCLUSIVE_TRANSACTIONS_REQUIRED(*db)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    const char* sql = R"sql(
      INSERT INTO Tensors (
        series,
        data
      ) VALUES (?, ZEROBLOB(?))
    )sql";
    SqliteStatement insert;
    TF_RETURN_IF_ERROR(db->Prepare(sql, &insert));
    // TODO(jart): Maybe preallocate index pages by setting step. This
    //             is tricky because UPDATE OR REPLACE can have a side
    //             effect of deleting preallocated rows.
    for (int64_t i = 0; i < kPreallocateRows; ++i) {
      insert.BindInt(1, series_);
      insert.BindInt(2, reserved_bytes);
      TF_RETURN_WITH_CONTEXT_IF_ERROR(insert.StepAndReset(), "i=", i);
      rowids_.push_back(db->last_insert_rowid());
      unflushed_bytes_ += reserved_bytes;
      TF_RETURN_IF_ERROR(MaybeFlush(db, txn));
    }
    return Status::OK();
  }

  Status MaybeFlush(Sqlite* db, SqliteTransaction* txn)
      SQLITE_EXCLUSIVE_TRANSACTIONS_REQUIRED(*db)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (unflushed_bytes_ >= kFlushBytes) {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(txn->Commit(), "flushing ",
                                      unflushed_bytes_, " bytes");
      unflushed_bytes_ = 0;
    }
    return Status::OK();
  }

  mutex mu_;
  const int64_t series_;
  RunMetadata* const meta_;
  uint64 count_ TF_GUARDED_BY(mu_) = 0;
  std::deque<int64_t> rowids_ TF_GUARDED_BY(mu_);
  uint64 unflushed_bytes_ TF_GUARDED_BY(mu_) = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(SeriesWriter);
};

/// \brief Tensor writer for a single Run.
///
/// This class farms out tensors to SeriesWriter instances. It also
/// keeps track of whether or not someone is watching the TensorBoard
/// GUI, so it can avoid writes when possible.
///
/// This class is thread safe.
class RunWriter {
 public:
  explicit RunWriter(RunMetadata* meta) : meta_{meta} {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_23(mht_23_v, 1074, "", "./tensorflow/core/summary/summary_db_writer.cc", "RunWriter");
}

  Status Append(Sqlite* db, int64_t tag_id, int64_t step, uint64 now,
                double computed_time, const Tensor& t)
      SQLITE_TRANSACTIONS_EXCLUDED(*db) TF_LOCKS_EXCLUDED(mu_) {
    SeriesWriter* writer = GetSeriesWriter(tag_id);
    return writer->Append(db, step, now, computed_time, t);
  }

  Status Finish(Sqlite* db) SQLITE_TRANSACTIONS_EXCLUDED(*db)
      TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    if (series_writers_.empty()) return Status::OK();
    for (auto i = series_writers_.begin(); i != series_writers_.end(); ++i) {
      if (!i->second) continue;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(i->second->Finish(db),
                                      "finish tag_id=", i->first);
      i->second.reset();
    }
    return Status::OK();
  }

 private:
  SeriesWriter* GetSeriesWriter(int64_t tag_id) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock sl(mu_);
    auto spot = series_writers_.find(tag_id);
    if (spot == series_writers_.end()) {
      SeriesWriter* writer = new SeriesWriter(tag_id, meta_);
      series_writers_[tag_id].reset(writer);
      return writer;
    } else {
      return spot->second.get();
    }
  }

  mutex mu_;
  RunMetadata* const meta_;
  std::unordered_map<int64_t, std::unique_ptr<SeriesWriter>> series_writers_
      TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RunWriter);
};

/// \brief SQLite implementation of SummaryWriterInterface.
///
/// This class is thread safe.
class SummaryDbWriter : public SummaryWriterInterface {
 public:
  SummaryDbWriter(Env* env, Sqlite* db, const string& experiment_name,
                  const string& run_name, const string& user_name)
      : SummaryWriterInterface(),
        env_{env},
        db_{db},
        ids_{env_, db_},
        meta_{&ids_, experiment_name, run_name, user_name},
        run_{&meta_} {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("experiment_name: \"" + experiment_name + "\"");
   mht_24_v.push_back("run_name: \"" + run_name + "\"");
   mht_24_v.push_back("user_name: \"" + user_name + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_24(mht_24_v, 1135, "", "./tensorflow/core/summary/summary_db_writer.cc", "SummaryDbWriter");

    DCHECK(env_ != nullptr);
    db_->Ref();
  }

  ~SummaryDbWriter() override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_25(mht_25_v, 1143, "", "./tensorflow/core/summary/summary_db_writer.cc", "~SummaryDbWriter");

    core::ScopedUnref unref(db_);
    Status s = run_.Finish(db_);
    if (!s.ok()) {
      // TODO(jart): Retry on transient errors here.
      LOG(ERROR) << s.ToString();
    }
    int64_t run_id = meta_.run_id();
    if (run_id == kAbsent) return;
    const char* sql = R"sql(
      UPDATE Runs SET finished_time = ? WHERE run_id = ?
    )sql";
    SqliteStatement update;
    s = db_->Prepare(sql, &update);
    if (s.ok()) {
      update.BindDouble(1, DoubleTime(env_->NowMicros()));
      update.BindInt(2, run_id);
      s = update.StepAndReset();
    }
    if (!s.ok()) {
      LOG(ERROR) << "Failed to set Runs[" << run_id
                 << "].finish_time: " << s.ToString();
    }
  }

  Status Flush() override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_26(mht_26_v, 1171, "", "./tensorflow/core/summary/summary_db_writer.cc", "Flush");
 return Status::OK(); }

  Status WriteTensor(int64_t global_step, Tensor t, const string& tag,
                     const string& serialized_metadata) override {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("tag: \"" + tag + "\"");
   mht_27_v.push_back("serialized_metadata: \"" + serialized_metadata + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_27(mht_27_v, 1179, "", "./tensorflow/core/summary/summary_db_writer.cc", "WriteTensor");

    TF_RETURN_IF_ERROR(CheckSupportedType(t));
    SummaryMetadata metadata;
    if (!metadata.ParseFromString(serialized_metadata)) {
      return errors::InvalidArgument("Bad serialized_metadata");
    }
    return Write(global_step, t, tag, metadata);
  }

  Status WriteScalar(int64_t global_step, Tensor t,
                     const string& tag) override {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_28(mht_28_v, 1193, "", "./tensorflow/core/summary/summary_db_writer.cc", "WriteScalar");

    TF_RETURN_IF_ERROR(CheckSupportedType(t));
    SummaryMetadata metadata;
    PatchPluginName(&metadata, kScalarPluginName);
    return Write(global_step, AsScalar(t), tag, metadata);
  }

  Status WriteGraph(int64_t global_step, std::unique_ptr<GraphDef> g) override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_29(mht_29_v, 1203, "", "./tensorflow/core/summary/summary_db_writer.cc", "WriteGraph");

    uint64 now = env_->NowMicros();
    return meta_.SetGraph(db_, now, DoubleTime(now), std::move(g));
  }

  Status WriteEvent(std::unique_ptr<Event> e) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_30(mht_30_v, 1211, "", "./tensorflow/core/summary/summary_db_writer.cc", "WriteEvent");

    return MigrateEvent(std::move(e));
  }

  Status WriteHistogram(int64_t global_step, Tensor t,
                        const string& tag) override {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_31(mht_31_v, 1220, "", "./tensorflow/core/summary/summary_db_writer.cc", "WriteHistogram");

    uint64 now = env_->NowMicros();
    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(DoubleTime(now));
    TF_RETURN_IF_ERROR(
        AddTensorAsHistogramToSummary(t, tag, e->mutable_summary()));
    return MigrateEvent(std::move(e));
  }

  Status WriteImage(int64_t global_step, Tensor t, const string& tag,
                    int max_images, Tensor bad_color) override {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_32(mht_32_v, 1235, "", "./tensorflow/core/summary/summary_db_writer.cc", "WriteImage");

    uint64 now = env_->NowMicros();
    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(DoubleTime(now));
    TF_RETURN_IF_ERROR(AddTensorAsImageToSummary(t, tag, max_images, bad_color,
                                                 e->mutable_summary()));
    return MigrateEvent(std::move(e));
  }

  Status WriteAudio(int64_t global_step, Tensor t, const string& tag,
                    int max_outputs, float sample_rate) override {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_33(mht_33_v, 1250, "", "./tensorflow/core/summary/summary_db_writer.cc", "WriteAudio");

    uint64 now = env_->NowMicros();
    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(DoubleTime(now));
    TF_RETURN_IF_ERROR(AddTensorAsAudioToSummary(
        t, tag, max_outputs, sample_rate, e->mutable_summary()));
    return MigrateEvent(std::move(e));
  }

  string DebugString() const override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_34(mht_34_v, 1263, "", "./tensorflow/core/summary/summary_db_writer.cc", "DebugString");
 return "SummaryDbWriter"; }

 private:
  Status Write(int64_t step, const Tensor& t, const string& tag,
               const SummaryMetadata& metadata) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_35(mht_35_v, 1271, "", "./tensorflow/core/summary/summary_db_writer.cc", "Write");

    uint64 now = env_->NowMicros();
    double computed_time = DoubleTime(now);
    int64_t tag_id;
    TF_RETURN_IF_ERROR(
        meta_.GetTagId(db_, now, computed_time, tag, &tag_id, metadata));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        run_.Append(db_, tag_id, step, now, computed_time, t),
        meta_.user_name(), "/", meta_.experiment_name(), "/", meta_.run_name(),
        "/", tag, "@", step);
    return Status::OK();
  }

  Status MigrateEvent(std::unique_ptr<Event> e) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_36(mht_36_v, 1287, "", "./tensorflow/core/summary/summary_db_writer.cc", "MigrateEvent");

    switch (e->what_case()) {
      case Event::WhatCase::kSummary: {
        uint64 now = env_->NowMicros();
        auto summaries = e->mutable_summary();
        for (int i = 0; i < summaries->value_size(); ++i) {
          Summary::Value* value = summaries->mutable_value(i);
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              MigrateSummary(e.get(), value, now), meta_.user_name(), "/",
              meta_.experiment_name(), "/", meta_.run_name(), "/", value->tag(),
              "@", e->step());
        }
        break;
      }
      case Event::WhatCase::kGraphDef:
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            MigrateGraph(e.get(), e->graph_def()), meta_.user_name(), "/",
            meta_.experiment_name(), "/", meta_.run_name(), "/__graph__@",
            e->step());
        break;
      default:
        // TODO(@jart): Handle other stuff.
        break;
    }
    return Status::OK();
  }

  Status MigrateGraph(const Event* e, const string& graph_def) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("graph_def: \"" + graph_def + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_37(mht_37_v, 1318, "", "./tensorflow/core/summary/summary_db_writer.cc", "MigrateGraph");

    uint64 now = env_->NowMicros();
    std::unique_ptr<GraphDef> graph{new GraphDef};
    if (!ParseProtoUnlimited(graph.get(), graph_def)) {
      return errors::InvalidArgument("bad proto");
    }
    return meta_.SetGraph(db_, now, e->wall_time(), std::move(graph));
  }

  Status MigrateSummary(const Event* e, Summary::Value* s, uint64 now) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_38(mht_38_v, 1330, "", "./tensorflow/core/summary/summary_db_writer.cc", "MigrateSummary");

    switch (s->value_case()) {
      case Summary::Value::ValueCase::kTensor:
        TF_RETURN_WITH_CONTEXT_IF_ERROR(MigrateTensor(e, s, now), "tensor");
        break;
      case Summary::Value::ValueCase::kSimpleValue:
        TF_RETURN_WITH_CONTEXT_IF_ERROR(MigrateScalar(e, s, now), "scalar");
        break;
      case Summary::Value::ValueCase::kHisto:
        TF_RETURN_WITH_CONTEXT_IF_ERROR(MigrateHistogram(e, s, now), "histo");
        break;
      case Summary::Value::ValueCase::kImage:
        TF_RETURN_WITH_CONTEXT_IF_ERROR(MigrateImage(e, s, now), "image");
        break;
      case Summary::Value::ValueCase::kAudio:
        TF_RETURN_WITH_CONTEXT_IF_ERROR(MigrateAudio(e, s, now), "audio");
        break;
      default:
        break;
    }
    return Status::OK();
  }

  Status MigrateTensor(const Event* e, Summary::Value* s, uint64 now) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_39(mht_39_v, 1356, "", "./tensorflow/core/summary/summary_db_writer.cc", "MigrateTensor");

    Tensor t;
    if (!t.FromProto(s->tensor())) return errors::InvalidArgument("bad proto");
    TF_RETURN_IF_ERROR(CheckSupportedType(t));
    int64_t tag_id;
    TF_RETURN_IF_ERROR(meta_.GetTagId(db_, now, e->wall_time(), s->tag(),
                                      &tag_id, s->metadata()));
    return run_.Append(db_, tag_id, e->step(), now, e->wall_time(), t);
  }

  // TODO(jart): Refactor Summary -> Tensor logic into separate file.

  Status MigrateScalar(const Event* e, Summary::Value* s, uint64 now) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_40(mht_40_v, 1371, "", "./tensorflow/core/summary/summary_db_writer.cc", "MigrateScalar");

    // See tensorboard/plugins/scalar/summary.py and data_compat.py
    Tensor t{DT_FLOAT, {}};
    t.scalar<float>()() = s->simple_value();
    int64_t tag_id;
    PatchPluginName(s->mutable_metadata(), kScalarPluginName);
    TF_RETURN_IF_ERROR(meta_.GetTagId(db_, now, e->wall_time(), s->tag(),
                                      &tag_id, s->metadata()));
    return run_.Append(db_, tag_id, e->step(), now, e->wall_time(), t);
  }

  Status MigrateHistogram(const Event* e, Summary::Value* s, uint64 now) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_41(mht_41_v, 1385, "", "./tensorflow/core/summary/summary_db_writer.cc", "MigrateHistogram");

    const HistogramProto& histo = s->histo();
    int k = histo.bucket_size();
    if (k != histo.bucket_limit_size()) {
      return errors::InvalidArgument("size mismatch");
    }
    // See tensorboard/plugins/histogram/summary.py and data_compat.py
    Tensor t{DT_DOUBLE, {k, 3}};
    auto data = t.flat<double>();
    for (int i = 0, j = 0; i < k; ++i) {
      // TODO(nickfelt): reconcile with TensorBoard's data_compat.py
      // From summary.proto
      // Parallel arrays encoding the bucket boundaries and the bucket values.
      // bucket(i) is the count for the bucket i.  The range for
      // a bucket is:
      //   i == 0:  -DBL_MAX .. bucket_limit(0)
      //   i != 0:  bucket_limit(i-1) .. bucket_limit(i)
      double left_edge = (i == 0) ? std::numeric_limits<double>::min()
                                  : histo.bucket_limit(i - 1);

      data(j++) = left_edge;
      data(j++) = histo.bucket_limit(i);
      data(j++) = histo.bucket(i);
    }
    int64_t tag_id;
    PatchPluginName(s->mutable_metadata(), kHistogramPluginName);
    TF_RETURN_IF_ERROR(meta_.GetTagId(db_, now, e->wall_time(), s->tag(),
                                      &tag_id, s->metadata()));
    return run_.Append(db_, tag_id, e->step(), now, e->wall_time(), t);
  }

  Status MigrateImage(const Event* e, Summary::Value* s, uint64 now) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_42(mht_42_v, 1419, "", "./tensorflow/core/summary/summary_db_writer.cc", "MigrateImage");

    // See tensorboard/plugins/image/summary.py and data_compat.py
    Tensor t{DT_STRING, {3}};
    auto img = s->mutable_image();
    t.flat<tstring>()(0) = strings::StrCat(img->width());
    t.flat<tstring>()(1) = strings::StrCat(img->height());
    t.flat<tstring>()(2) = std::move(*img->mutable_encoded_image_string());
    int64_t tag_id;
    PatchPluginName(s->mutable_metadata(), kImagePluginName);
    TF_RETURN_IF_ERROR(meta_.GetTagId(db_, now, e->wall_time(), s->tag(),
                                      &tag_id, s->metadata()));
    return run_.Append(db_, tag_id, e->step(), now, e->wall_time(), t);
  }

  Status MigrateAudio(const Event* e, Summary::Value* s, uint64 now) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_43(mht_43_v, 1436, "", "./tensorflow/core/summary/summary_db_writer.cc", "MigrateAudio");

    // See tensorboard/plugins/audio/summary.py and data_compat.py
    Tensor t{DT_STRING, {1, 2}};
    auto wav = s->mutable_audio();
    t.flat<tstring>()(0) = std::move(*wav->mutable_encoded_audio_string());
    t.flat<tstring>()(1) = "";
    int64_t tag_id;
    PatchPluginName(s->mutable_metadata(), kAudioPluginName);
    TF_RETURN_IF_ERROR(meta_.GetTagId(db_, now, e->wall_time(), s->tag(),
                                      &tag_id, s->metadata()));
    return run_.Append(db_, tag_id, e->step(), now, e->wall_time(), t);
  }

  Env* const env_;
  Sqlite* const db_;
  IdAllocator ids_;
  RunMetadata meta_;
  RunWriter run_;
};

}  // namespace

Status CreateSummaryDbWriter(Sqlite* db, const string& experiment_name,
                             const string& run_name, const string& user_name,
                             Env* env, SummaryWriterInterface** result) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("experiment_name: \"" + experiment_name + "\"");
   mht_44_v.push_back("run_name: \"" + run_name + "\"");
   mht_44_v.push_back("user_name: \"" + user_name + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writerDTcc mht_44(mht_44_v, 1466, "", "./tensorflow/core/summary/summary_db_writer.cc", "CreateSummaryDbWriter");

  *result = new SummaryDbWriter(env, db, experiment_name, run_name, user_name);
  return Status::OK();
}

}  // namespace tensorflow
