from pathlib import Path

def test_workspace_exposes_event_date_and_version_effective_metadata():
 html=(Path(__file__).parents[2]/'evidence_rag/static/index.html').read_text()
 js=(Path(__file__).parents[2]/'evidence_rag/static/workspace.mjs').read_text()
 assert 'id="event-date"' in html and 'type="date"' in html
 assert "event_date: eventDate.value" in js
 assert 'revision_selection' in js and 'item.temporal' in js and 'validity' in js

def test_workspace_does_not_reintroduce_token_input():
 html=(Path(__file__).parents[2]/'evidence_rag/static/index.html').read_text()
 assert '令牌' not in html and 'type="password"' not in html
