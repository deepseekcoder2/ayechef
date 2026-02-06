"""Job routes - status listing and streaming output."""
from flask import Blueprint, render_template, Response, jsonify, redirect, url_for
from panel.jobs import get_job, list_jobs, get_output, cancel_job, delete_job, clear_completed_jobs
from config import DATA_DIR
import time

bp = Blueprint('jobs', __name__)

MEAL_PLAN_FILE = DATA_DIR / 'meal_plan.txt'


def get_whatsapp_message():
    """Read the generated WhatsApp message if it exists."""
    if MEAL_PLAN_FILE.exists():
        try:
            return MEAL_PLAN_FILE.read_text(encoding='utf-8')
        except:
            pass
    return None


@bp.route('/', strict_slashes=False)
def job_list():
    """List all jobs, separated into active and recent."""
    jobs = list_jobs(limit=50)
    active = [j for j in jobs if j.status in ('pending', 'running')]
    recent = [j for j in jobs if j.status in ('completed', 'failed', 'cancelled')]
    return render_template('jobs/list.html', active=active, recent=recent)


@bp.route('/<job_id>', strict_slashes=False)
def job_detail(job_id):
    """Show job detail with streaming output."""
    job = get_job(job_id)
    if not job:
        return "Job not found", 404
    
    # Get initial output
    initial_output = get_output(job_id)
    
    # Check for WhatsApp message if this is a completed meal plan job
    whatsapp_message = None
    if job.tool_id == 'plan_week' and job.status == 'completed':
        whatsapp_message = get_whatsapp_message()
    
    return render_template('jobs/detail.html', 
                          job=job, 
                          initial_output=initial_output,
                          whatsapp_message=whatsapp_message)


@bp.route('/<job_id>/stream')
def job_stream(job_id):
    """Server-Sent Events stream of job output."""
    def generate():
        # Initial comment to establish connection (browsers/proxies may buffer otherwise)
        yield ": keepalive\n\n"

        # Some reverse proxies / browsers will drop an idle SSE connection if no bytes
        # are transmitted for a while (even though the job is still running). To avoid
        # the UI appearing "stuck", we periodically emit SSE comments during quiet
        # phases. This does NOT imply new job output; it just keeps the connection alive.
        KEEPALIVE_INTERVAL_S = 15.0
        last_activity = time.monotonic()
        last_pos = 0
        while True:
            job = get_job(job_id)
            output = get_output(job_id)
            
            # Send new output since last position
            if len(output) > last_pos:
                new_content = output[last_pos:]
                for line in new_content.split('\n'):
                    if line:
                        escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        yield f"data: <div>{escaped}</div>\n\n"
                last_pos = len(output)
                last_activity = time.monotonic()

            # Periodic keepalive to prevent idle timeouts (SSE comments start with ':')
            now = time.monotonic()
            if now - last_activity >= KEEPALIVE_INTERVAL_S:
                yield ": keepalive\n\n"
                last_activity = now
            
            # Check if job is done
            if job and job.status in ('completed', 'failed', 'cancelled'):
                status_class = 'text-green-400' if job.status == 'completed' else 'text-red-400'
                status_icon = '✅' if job.status == 'completed' else '❌'
                
                # Send done event with HTML for display and data attribute for JS handling
                # Include data-status for JS to detect special cases (like plan_week)
                if job.tool_id == 'plan_week' and job.status == 'completed':
                    yield f"event: done\ndata: <div class=\"{status_class} font-bold mt-4\" data-plan-complete=\"true\">{status_icon} Job {job.status}</div>\n\n"
                else:
                    yield f"event: done\ndata: <div class=\"{status_class} font-bold mt-4\">{status_icon} Job {job.status}</div>\n\n"
                break
            
            if not job:
                yield f"event: done\ndata: <div class=\"text-red-400 font-bold mt-4\">Job not found</div>\n\n"
                break
            
            time.sleep(0.3)
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@bp.route('/<job_id>/whatsapp')
def job_whatsapp(job_id):
    """Get WhatsApp message for a completed job."""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.tool_id != 'plan_week':
        return jsonify({'error': 'Not a meal plan job'}), 400
    if job.status != 'completed':
        return jsonify({'error': 'Job not completed'}), 400
    
    message = get_whatsapp_message()
    if message:
        return jsonify({'message': message})
    return jsonify({'error': 'No message found'}), 404


@bp.route('/<job_id>/status')
def job_status_api(job_id):
    """API endpoint for job status."""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'not found'}), 404
    return jsonify({
        'id': job.id,
        'status': job.status,
        'tool_id': job.tool_id,
        'tool_name': job.tool_name,
        'created_at': job.created_at,
        'error': job.error
    })


@bp.route('/<job_id>/cancel', methods=['POST'])
def job_cancel(job_id):
    """Cancel a running job."""
    success = cancel_job(job_id)
    if success:
        return jsonify({'success': True, 'message': 'Job cancelled'})
    return jsonify({'success': False, 'error': 'Could not cancel job'}), 400


@bp.route('/<job_id>/delete', methods=['POST'])
def job_delete(job_id):
    """Delete a completed job."""
    success = delete_job(job_id)
    if success:
        return jsonify({'success': True, 'message': 'Job deleted'})
    return jsonify({'success': False, 'error': 'Could not delete job (may be still running)'}), 400


@bp.route('/clear', methods=['POST'])
def jobs_clear():
    """Clear all completed/failed/cancelled jobs."""
    count = clear_completed_jobs()
    return jsonify({'success': True, 'deleted': count})
