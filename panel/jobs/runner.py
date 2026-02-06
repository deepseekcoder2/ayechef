"""Generic job runner - works with ANY tool from the registry."""
from pathlib import Path
import subprocess
import os
import signal
import select
import sys

from .huey_config import huey
from .status import update_job, append_output, get_job
from panel.tools.registry import TOOLS, build_command

# Project root for running commands
PROJECT_ROOT = Path(__file__).parent.parent.parent

@huey.task()
def run_tool_job(job_id: str, tool_id: str, form_data: dict):
    """
    Generic job runner - works with ANY tool from registry.
    
    This is the ONLY job function. All tools use this same runner.
    The tool registry defines what command to run.
    """
    tool = TOOLS.get(tool_id)
    if not tool:
        update_job(job_id, status='failed', error=f'Unknown tool: {tool_id}')
        return
    
    cmd = build_command(tool, form_data)
    
    update_job(job_id, status='running')
    append_output(job_id, f"$ {' '.join(cmd)}\n")
    append_output(job_id, f"Working directory: {PROJECT_ROOT}\n")
    append_output(job_id, "-" * 60 + "\n")
    
    try:
        # Run command in new process group so we can kill all children
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            cwd=str(PROJECT_ROOT),
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Save PID so we can cancel later
        update_job(job_id, pid=process.pid)
        
        # Stream output with non-blocking reads to handle cancellation properly
        cancelled = False
        while True:
            # Check if job was cancelled
            job = get_job(job_id)
            if job and job.status == 'cancelled':
                cancelled = True
                # Kill the entire process group
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
                break
            
            # Use select for non-blocking read with timeout (works on Unix)
            if sys.platform != 'win32' and process.stdout:
                readable, _, _ = select.select([process.stdout], [], [], 0.5)
                if readable:
                    line = process.stdout.readline()
                    if line:
                        append_output(job_id, line)
                    elif process.poll() is not None:
                        # Process finished and no more output
                        break
                elif process.poll() is not None:
                    # Process finished
                    break
            else:
                # Fallback for Windows or if select doesn't work
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    append_output(job_id, line)
        
        # Wait for process with timeout to prevent hanging
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if still alive
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            process.wait(timeout=2)
        
        # Check final status - don't overwrite if already cancelled
        job = get_job(job_id)
        if job and job.status == 'cancelled':
            return  # Don't overwrite cancelled status
        
        append_output(job_id, "-" * 60 + "\n")
        
        if process.returncode == 0:
            # Double-check status before updating (race condition protection)
            job = get_job(job_id)
            if job and job.status != 'cancelled':
                append_output(job_id, f"✅ Completed successfully\n")
                update_job(job_id, status='completed')
        elif process.returncode in (-15, -9):  # SIGTERM or SIGKILL
            # Double-check - cancel_job() may have already set this
            job = get_job(job_id)
            if job and job.status != 'cancelled':
                append_output(job_id, f"🛑 Stopped\n")
                update_job(job_id, status='cancelled', error='Stopped by user')
        else:
            # Double-check before marking failed
            job = get_job(job_id)
            if job and job.status != 'cancelled':
                append_output(job_id, f"❌ Failed with exit code: {process.returncode}\n")
                update_job(job_id, status='failed', error=f'Exit code: {process.returncode}')
    
    except Exception as e:
        append_output(job_id, f"❌ Exception: {str(e)}\n")
        update_job(job_id, status='failed', error=str(e))
