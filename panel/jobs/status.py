"""Job status tracking with SQLite persistence."""
import sqlite3
import os
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from config import DATA_DIR
DATA_DIR.mkdir(exist_ok=True)

# Output files directory
OUTPUT_DIR = DATA_DIR / 'job_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# SQLite database for job status
DB_PATH = DATA_DIR / 'job_status.db'

def _get_db():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    """Initialize database schema."""
    conn = _get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            tool_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error TEXT,
            pid INTEGER
        )
    ''')
    # Add pid column if it doesn't exist (migration)
    try:
        conn.execute('ALTER TABLE jobs ADD COLUMN pid INTEGER')
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()

# Initialize on import
_init_db()

@dataclass
class JobStatus:
    """Job status data."""
    id: str
    tool_id: str
    tool_name: str
    status: str  # pending, running, completed, failed, cancelled
    created_at: str
    updated_at: str
    error: Optional[str] = None
    pid: Optional[int] = None

def create_job(job_id: str, tool_id: str, tool_name: str) -> JobStatus:
    """Create a new job record."""
    now = datetime.now().isoformat()
    conn = _get_db()
    conn.execute(
        'INSERT INTO jobs (id, tool_id, tool_name, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)',
        (job_id, tool_id, tool_name, 'pending', now, now)
    )
    conn.commit()
    conn.close()
    
    # Create empty output file (write_text clears any existing content)
    output_path = OUTPUT_DIR / f"{job_id}.txt"
    output_path.write_text('')
    
    return JobStatus(
        id=job_id,
        tool_id=tool_id,
        tool_name=tool_name,
        status='pending',
        created_at=now,
        updated_at=now
    )

def update_job(job_id: str, status: Optional[str] = None, error: Optional[str] = None, pid: Optional[int] = None):
    """Update job status."""
    now = datetime.now().isoformat()
    conn = _get_db()
    
    updates = ['updated_at = ?']
    params = [now]
    
    if status:
        updates.append('status = ?')
        params.append(status)
    if error:
        updates.append('error = ?')
        params.append(error)
    if pid is not None:
        updates.append('pid = ?')
        params.append(pid)
    
    params.append(job_id)
    conn.execute(f'UPDATE jobs SET {", ".join(updates)} WHERE id = ?', params)
    conn.commit()
    conn.close()

def get_job(job_id: str) -> Optional[JobStatus]:
    """Get job by ID."""
    conn = _get_db()
    row = conn.execute('SELECT * FROM jobs WHERE id = ?', (job_id,)).fetchone()
    conn.close()
    
    if row:
        return JobStatus(
            id=row['id'],
            tool_id=row['tool_id'],
            tool_name=row['tool_name'],
            status=row['status'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            error=row['error'],
            pid=row['pid'] if 'pid' in row.keys() else None
        )
    return None

def list_jobs(limit: int = 20) -> List[JobStatus]:
    """List recent jobs, newest first."""
    conn = _get_db()
    rows = conn.execute(
        'SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?',
        (limit,)
    ).fetchall()
    conn.close()
    
    return [
        JobStatus(
            id=row['id'],
            tool_id=row['tool_id'],
            tool_name=row['tool_name'],
            status=row['status'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            error=row['error'],
            pid=row['pid'] if 'pid' in row.keys() else None
        )
        for row in rows
    ]

def cancel_job(job_id: str) -> bool:
    """Cancel a running job by killing its process.
    
    Strategy to avoid race conditions:
    1. Set status to 'cancelled' FIRST (before killing)
    2. Send SIGTERM, wait up to 3 seconds
    3. If still alive, send SIGKILL
    4. Re-confirm status at end
    """
    import time
    
    job = get_job(job_id)
    if not job:
        return False
    
    if job.status not in ('pending', 'running'):
        return False  # Already finished
    
    # Set status FIRST to win any race with the worker
    update_job(job_id, status='cancelled', error='Cancelled by user')
    append_output(job_id, "\n" + "=" * 60 + "\n")
    append_output(job_id, "🛑 Cancelling job...\n")
    
    if job.pid:
        def is_alive(pid):
            try:
                os.kill(pid, 0)
                return True
            except (ProcessLookupError, PermissionError, OSError):
                return False
        
        def kill_it(pid, sig):
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, sig)
                except:
                    pass
        
        # Send SIGTERM
        kill_it(job.pid, signal.SIGTERM)
        
        # Wait up to 3 seconds
        for _ in range(6):
            if not is_alive(job.pid):
                break
            time.sleep(0.5)
        
        # If still alive, SIGKILL
        if is_alive(job.pid):
            append_output(job_id, "⚠️ Process not responding, forcing kill...\n")
            kill_it(job.pid, signal.SIGKILL)
            time.sleep(0.5)
    
    append_output(job_id, "🛑 Job cancelled\n")
    
    # Re-confirm status (in case worker overwrote it)
    update_job(job_id, status='cancelled')
    return True

def append_output(job_id: str, line: str):
    """Append a line to job output file."""
    output_path = OUTPUT_DIR / f"{job_id}.txt"
    with open(output_path, 'a') as f:
        f.write(line)

def get_output(job_id: str) -> str:
    """Read job output file."""
    output_path = OUTPUT_DIR / f"{job_id}.txt"
    if output_path.exists():
        return output_path.read_text()
    return ""


def delete_job(job_id: str) -> bool:
    """Delete a job record and its output file."""
    job = get_job(job_id)
    if not job:
        return False
    
    # Don't delete running jobs
    if job.status in ('pending', 'running'):
        return False
    
    # Delete from database
    conn = _get_db()
    conn.execute('DELETE FROM jobs WHERE id = ?', (job_id,))
    conn.commit()
    conn.close()
    
    # Delete output file
    output_path = OUTPUT_DIR / f"{job_id}.txt"
    if output_path.exists():
        output_path.unlink()
    
    return True


def clear_completed_jobs() -> int:
    """Delete all completed/failed/cancelled jobs. Returns count deleted."""
    conn = _get_db()
    
    # Get IDs of jobs to delete
    rows = conn.execute(
        "SELECT id FROM jobs WHERE status IN ('completed', 'failed', 'cancelled')"
    ).fetchall()
    job_ids = [row['id'] for row in rows]
    
    # Delete from database
    conn.execute("DELETE FROM jobs WHERE status IN ('completed', 'failed', 'cancelled')")
    conn.commit()
    conn.close()
    
    # Delete output files
    for job_id in job_ids:
        output_path = OUTPUT_DIR / f"{job_id}.txt"
        if output_path.exists():
            output_path.unlink()
    
    return len(job_ids)


def force_cancel_all_active() -> int:
    """Force cancel all pending/running jobs. Used to reset worker state.
    
    Returns count of jobs cancelled.
    """
    conn = _get_db()
    rows = conn.execute(
        "SELECT id, pid FROM jobs WHERE status IN ('pending', 'running')"
    ).fetchall()
    conn.close()
    
    count = 0
    for row in rows:
        job_id = row['id']
        pid = row['pid']
        
        # Kill the process if it exists
        if pid:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except:
                try:
                    os.kill(pid, signal.SIGKILL)
                except:
                    pass
        
        # Mark as cancelled
        update_job(job_id, status='cancelled', error='Force cancelled (worker reset)')
        append_output(job_id, "\n🔄 Force cancelled - worker reset\n")
        count += 1
    
    return count
