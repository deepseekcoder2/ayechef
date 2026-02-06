"""Flask application factory."""
from flask import Flask, render_template, render_template_string, request
from pathlib import Path
import json
import os


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    
    # Custom Jinja filters
    @app.template_filter('from_json')
    def from_json_filter(value):
        """Parse JSON string in templates."""
        return json.loads(value) if value else []
    
    # Ensure data directory exists
    from config import DATA_DIR
    DATA_DIR.mkdir(exist_ok=True)
    
    @app.context_processor
    def inject_theme():
        """Inject theme data into all templates."""
        from panel.themes import get_theme, get_themes_json, validate_theme_id, DEFAULT_THEME_ID
        from config import CONFIG_PATH
        import yaml
        
        # Load theme preference from config
        theme_id = DEFAULT_THEME_ID
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH) as f:
                    config = yaml.safe_load(f) or {}
                theme_id = config.get('ui', {}).get('theme', DEFAULT_THEME_ID)
        except Exception:
            pass
        
        # Validate and get theme
        theme_id = validate_theme_id(theme_id)
        theme = get_theme(theme_id)
        themes_json = get_themes_json()
        
        return {
            'theme': theme,
            'themes_json': themes_json,
            'current_theme_id': theme_id
        }
    
    @app.before_request
    def check_credentials():
        """Check if required credentials are configured before each request."""
        # Import inside function to avoid circular imports
        from config import get_credential_status
        
        # Allow these paths without credentials
        allowed_prefixes = ('/settings', '/credentials', '/status', '/static/', '/api/health')
        allowed_exact = ('/',)
        if request.path in allowed_exact or any(request.path.startswith(prefix) for prefix in allowed_prefixes):
            return None
        
        # Check credential status
        status = get_credential_status()
        
        # If all credentials are configured, proceed normally
        if status['mealie_token']['configured'] and status['openrouter_api_key']['configured']:
            return None
        
        # Build list of missing credentials
        missing_items = []
        if not status['mealie_token']['configured']:
            missing_items.append('Mealie API Token')
        if not status['openrouter_api_key']['configured']:
            missing_items.append('OpenRouter API Key')
        
        # Render blocking page using themed template
        return render_template('credentials_required.html', missing_items=missing_items)
    
    # Register blueprints
    from panel.routes import register_blueprints
    register_blueprints(app)
    
    return app


# For gunicorn: gunicorn -b 0.0.0.0:8080 panel.app:app
app = create_app()
