"""Legacy legal API application; shares handlers and implementation with evidence_rag."""
from evidence_rag.api import *
from evidence_rag.api import legal_app as app
app.state.default_profile='legal'
app.state.default_profile_explicit=True
