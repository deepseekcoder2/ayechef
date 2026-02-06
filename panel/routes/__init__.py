"""Blueprint registration."""


def register_blueprints(app):
    """Register all route blueprints."""
    from .main import bp as main_bp
    from .jobs import bp as jobs_bp
    from .import_recipes import bp as import_bp
    from .diagnostics import bp as diagnostics_bp
    from .insights import bp as insights_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(jobs_bp, url_prefix='/jobs')
    app.register_blueprint(import_bp, url_prefix='/import')
    app.register_blueprint(diagnostics_bp, url_prefix='/status')
    app.register_blueprint(insights_bp, url_prefix='/insights')