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
class MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc() {
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
#include "tensorflow/core/lib/db/sqlite.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

extern "C" int sqlite3_snapfn_init(sqlite3*, const char**, const void*);

namespace tensorflow {
namespace {

error::Code GetTfErrorCode(int code) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/lib/db/sqlite.cc", "GetTfErrorCode");

  // See: https://sqlite.org/rescode.html
  switch (code & 0xff) {
    case SQLITE_OK:    // Successful result
    case SQLITE_ROW:   // Step has another row ready
    case SQLITE_DONE:  // Step has finished executing
      return error::OK;
    case SQLITE_ABORT:  // Callback routine requested an abort
      return error::ABORTED;
    case SQLITE_READONLY:  // Attempt to write a readonly database
    case SQLITE_MISMATCH:  // Data type mismatch
      return error::FAILED_PRECONDITION;
    case SQLITE_MISUSE:    // Library used incorrectly
    case SQLITE_INTERNAL:  // Internal logic error in SQLite
      return error::INTERNAL;
    case SQLITE_RANGE:  // 2nd parameter to sqlite3_bind out of range
      return error::OUT_OF_RANGE;
    case SQLITE_CANTOPEN:    // Unable to open the database file
    case SQLITE_CONSTRAINT:  // Abort due to constraint violation
    case SQLITE_NOTFOUND:    // Unknown opcode or statement parameter name
    case SQLITE_NOTADB:      // File opened that is not a database file
      return error::INVALID_ARGUMENT;
    case SQLITE_CORRUPT:  // The database disk image is malformed
      return error::DATA_LOSS;
    case SQLITE_AUTH:  // Authorization denied
    case SQLITE_PERM:  // Access permission denied
      return error::PERMISSION_DENIED;
    case SQLITE_FULL:    // Insertion failed because database is full
    case SQLITE_TOOBIG:  // String or BLOB exceeds size limit
    case SQLITE_NOLFS:   // Uses OS features not supported on host
      return error::RESOURCE_EXHAUSTED;
    case SQLITE_BUSY:      // The database file is locked
    case SQLITE_LOCKED:    // A table in the database is locked
    case SQLITE_PROTOCOL:  // Database lock protocol error
    case SQLITE_NOMEM:     // Out of heap or perhaps lookaside memory
      return error::UNAVAILABLE;
    case SQLITE_INTERRUPT:  // Operation terminated by sqlite3_interrupt
      return error::CANCELLED;
    case SQLITE_ERROR:   // SQL error or missing database
    case SQLITE_IOERR:   // Some kind of disk I/O error occurred
    case SQLITE_SCHEMA:  // The database schema changed
    default:
      return error::UNKNOWN;
  }
}

template <typename... Args>
Status PrintfStatus(int rc, const char* fmt, Args&&... args) {
  return {GetTfErrorCode(rc),
          strings::Printf(fmt, std::forward<Args>(args)...)};
}

sqlite3_stmt* PrepareRawOrDie(sqlite3* db, const char* sql) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("sql: \"" + (sql == nullptr ? std::string("nullptr") : std::string((char*)sql)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_1(mht_1_v, 250, "", "./tensorflow/core/lib/db/sqlite.cc", "PrepareRawOrDie");

  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
  CHECK_EQ(SQLITE_OK, rc) << sql;
  return stmt;
}

Status SetPragma(Sqlite* db, const char* pragma, const StringPiece& value) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("pragma: \"" + (pragma == nullptr ? std::string("nullptr") : std::string((char*)pragma)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_2(mht_2_v, 261, "", "./tensorflow/core/lib/db/sqlite.cc", "SetPragma");

  if (value.empty()) return Status::OK();
  for (auto p = value.begin(); p < value.end(); ++p) {
    if (!(('0' <= *p && *p <= '9') || ('A' <= *p && *p <= 'Z') ||
          ('a' <= *p && *p <= 'z') || *p == '-')) {
      return errors::InvalidArgument("Illegal pragma character");
    }
  }
  SqliteStatement stmt;
  TF_RETURN_IF_ERROR(  // We can't use Bind*() pragma statements.
      db->Prepare(strings::StrCat("PRAGMA ", pragma, "=", value), &stmt));
  bool unused_done;
  return stmt.Step(&unused_done);
}

const StringPiece GetEnv(const char* var) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("var: \"" + (var == nullptr ? std::string("nullptr") : std::string((char*)var)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_3(mht_3_v, 280, "", "./tensorflow/core/lib/db/sqlite.cc", "GetEnv");

  const char* val = std::getenv(var);
  return (val == nullptr) ? StringPiece() : StringPiece(val);
}

Status EnvPragma(Sqlite* db, const char* pragma, const char* var) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("pragma: \"" + (pragma == nullptr ? std::string("nullptr") : std::string((char*)pragma)) + "\"");
   mht_4_v.push_back("var: \"" + (var == nullptr ? std::string("nullptr") : std::string((char*)var)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_4(mht_4_v, 290, "", "./tensorflow/core/lib/db/sqlite.cc", "EnvPragma");

  TF_RETURN_WITH_CONTEXT_IF_ERROR(SetPragma(db, pragma, GetEnv(var)), "getenv(",
                                  var, ")");
  return Status::OK();
}

}  // namespace

/* static */
Status Sqlite::Open(const string& path, int flags, Sqlite** db) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_5(mht_5_v, 303, "", "./tensorflow/core/lib/db/sqlite.cc", "Sqlite::Open");

  flags |= SQLITE_OPEN_PRIVATECACHE;
  flags |= SQLITE_OPEN_URI;
  sqlite3* sqlite = nullptr;
  int rc = sqlite3_open_v2(path.c_str(), &sqlite, flags, nullptr);
  if (rc != SQLITE_OK) {
    *db = nullptr;
    return PrintfStatus(rc, "Sqlite::Open(%s) failed: %s", path.c_str(),
                        sqlite3_errstr(rc));
  }
  CHECK_EQ(SQLITE_OK, sqlite3_extended_result_codes(sqlite, 1));
  CHECK_EQ(SQLITE_OK, sqlite3_snapfn_init(sqlite, nullptr, nullptr));
  // Prepare these tiny privileged statements for SqliteTransaction
  // so it can do less work, particularly in its constructor, per
  // Google C++ Style.
  sqlite3_stmt* begin = PrepareRawOrDie(sqlite, "BEGIN");
  sqlite3_stmt* commit = PrepareRawOrDie(sqlite, "COMMIT");
  sqlite3_stmt* rollback = PrepareRawOrDie(sqlite, "ROLLBACK");
  *db = new Sqlite(sqlite, begin, commit, rollback);
  Status s = Status::OK();
  // Up until 2016 the default SQLite page_size was 1024. This ensures
  // the new default regardless of linkage unless configured otherwise.
  s.Update(SetPragma(*db, "page_size", "4096"));
  // TensorFlow is designed to work well in all SQLite modes. However
  // users might find tuning some these pragmas rewarding, depending on
  // various considerations. Pragmas are set on a best-effort basis and
  // might be ignored.
  s.Update(EnvPragma(*db, "secure_delete", "TF_SQLITE_SECURE_DELETE"));
  s.Update(EnvPragma(*db, "page_size", "TF_SQLITE_PAGE_SIZE"));
  s.Update(EnvPragma(*db, "journal_mode", "TF_SQLITE_JOURNAL_MODE"));
  s.Update(EnvPragma(*db, "synchronous", "TF_SQLITE_SYNCHRONOUS"));
  s.Update(EnvPragma(*db, "mmap_size", "TF_SQLITE_MMAP_SIZE"));
  s.Update(EnvPragma(*db, "locking_mode", "TF_SQLITE_LOCKING_MODE"));
  s.Update(EnvPragma(*db, "cache_size", "TF_SQLITE_CACHE_SIZE"));
  s.Update(EnvPragma(*db, "auto_vacuum", "TF_SQLITE_AUTO_VACUUM"));
  DCHECK((*db)->RefCountIsOne());
  if (!s.ok()) {
    (*db)->Unref();
    *db = nullptr;
  }
  return s;
}

Sqlite::~Sqlite() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_6(mht_6_v, 349, "", "./tensorflow/core/lib/db/sqlite.cc", "Sqlite::~Sqlite");

  sqlite3_finalize(rollback_);
  sqlite3_finalize(commit_);
  sqlite3_finalize(begin_);
  CHECK_EQ(SQLITE_OK, sqlite3_close(db_));
}

Status Sqlite::Prepare(const StringPiece& sql, SqliteStatement* stmt) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_7(mht_7_v, 359, "", "./tensorflow/core/lib/db/sqlite.cc", "Sqlite::Prepare");

  SqliteLock lock(*this);
  sqlite3_stmt* ps = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql.data(), static_cast<int>(sql.size()),
                              &ps, nullptr);
  if (rc != SQLITE_OK) {
    *stmt = SqliteStatement();
    return PrintfStatus(rc, "Prepare() failed: [%d] %s: %.*s", rc, errmsg(),
                        sql.size(), sql.data());
  }
  *stmt = SqliteStatement(this, ps);
  return Status::OK();
}

Status SqliteStatement::Step(bool* is_done) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_8(mht_8_v, 376, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteStatement::Step");

  DCHECK(stmt_ != nullptr);
  if (TF_PREDICT_FALSE(bind_error_ != SQLITE_OK)) {
    *is_done = true;
    return PrintfStatus(bind_error_, "Bind(%d) failed: %s: %s",
                        bind_error_parameter_, sqlite3_errstr(bind_error_),
                        sql());
  }
  SqliteLock lock(*db_);
  int rc = sqlite3_step(stmt_);
  switch (rc) {
    case SQLITE_ROW:
      *is_done = false;
      return Status::OK();
    case SQLITE_DONE:
      *is_done = true;
      return Status::OK();
    default:
      *is_done = true;
      return PrintfStatus(rc, "Step() failed: [%d] %s: %s", rc, db_->errmsg(),
                          sql());
  }
}

bool SqliteStatement::StepOrDie() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_9(mht_9_v, 403, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteStatement::StepOrDie");

  bool is_done;
  TF_CHECK_OK(Step(&is_done));
  return !is_done;
}

Status SqliteStatement::StepOnce() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_10(mht_10_v, 412, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteStatement::StepOnce");

  bool is_done;
  TF_RETURN_IF_ERROR(Step(&is_done));
  if (TF_PREDICT_FALSE(is_done)) {
    return errors::Internal("No rows returned: ", sql());
  }
  return Status::OK();
}

const SqliteStatement& SqliteStatement::StepOnceOrDie() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_11(mht_11_v, 424, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteStatement::StepOnceOrDie");

  TF_CHECK_OK(StepOnce());
  return *this;
}

Status SqliteStatement::StepAndReset() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_12(mht_12_v, 432, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteStatement::StepAndReset");

  bool is_done;
  Status s = Step(&is_done);
  if (TF_PREDICT_FALSE(s.ok() && !is_done)) {
    s = errors::Internal("Unexpected row: ", sql());
  }
  Reset();
  return s;
}

void SqliteStatement::StepAndResetOrDie() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_13(mht_13_v, 445, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteStatement::StepAndResetOrDie");
 TF_CHECK_OK(StepAndReset()); }

void SqliteStatement::Reset() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_14(mht_14_v, 450, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteStatement::Reset");

  if (TF_PREDICT_TRUE(stmt_ != nullptr)) {
    sqlite3_reset(stmt_);
    sqlite3_clear_bindings(stmt_);
  }
  bind_error_ = SQLITE_OK;
  size_ = 0;
}

SqliteTransaction::SqliteTransaction(Sqlite& db) : db_(&db) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_15(mht_15_v, 462, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteTransaction::SqliteTransaction");

  sqlite3_mutex_enter(sqlite3_db_mutex(db_->db_));
  CHECK(!db_->is_in_transaction_);
  db_->is_in_transaction_ = true;
  Begin();
}

SqliteTransaction::~SqliteTransaction() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_16(mht_16_v, 472, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteTransaction::~SqliteTransaction");

  // Rollback should only return an error if there's no transaction.
  // Since the API performs auto-rollbacks in some cases, we ignore.
  sqlite3_step(db_->rollback_);
  sqlite3_reset(db_->rollback_);
  sqlite3_reset(db_->begin_);
  db_->is_in_transaction_ = false;
  sqlite3_mutex_leave(sqlite3_db_mutex(db_->db_));
}

void SqliteTransaction::Begin() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_17(mht_17_v, 485, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteTransaction::Begin");

  // This shouldn't allocate memory or perform I/O. All it does is
  // execute OP_AutoCommit(0, 0) a.k.a. BEGIN DEFERRED which flips
  // the sqlite3::autoCommit bit.
  if (sqlite3_step(db_->begin_) != SQLITE_DONE) {
    // It shouldn't be possible for this to fail since we already
    // performed the reentrancy check.
    LOG(FATAL) << "BEGIN failed: " << sqlite3_errmsg(db_->db_);
  }
}

Status SqliteTransaction::Commit() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSlibPSdbPSsqliteDTcc mht_18(mht_18_v, 499, "", "./tensorflow/core/lib/db/sqlite.cc", "SqliteTransaction::Commit");

  int rc = sqlite3_step(db_->commit_);
  if (rc != SQLITE_DONE) {
    return PrintfStatus(rc, "COMMIT failed: [%d] %s", rc,
                        sqlite3_errmsg(db_->db_));
  }
  sqlite3_reset(db_->commit_);
  sqlite3_reset(db_->begin_);
  Begin();
  return Status::OK();
}

}  // namespace tensorflow
