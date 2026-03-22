#!/usr/bin/env python3
"""Web frontend for the paper generator with co-planning & chapter approval."""

import os
import sys
import json
import uuid
import time as _time
import threading
from pathlib import Path

from flask import Flask, request, jsonify, send_file, Response, stream_with_context

# Ensure project root is on path so draft_generator can import utils/config
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__, static_folder='static', template_folder='static')

# Track running jobs
_jobs: dict = {}
_jobs_lock = threading.Lock()

# Papers persistence
_PAPERS_FILE = Path(__file__).parent / 'papers.json'

def _load_papers() -> list:
    if _PAPERS_FILE.exists():
        try:
            return json.loads(_PAPERS_FILE.read_text(encoding='utf-8'))
        except Exception:
            return []
    return []

def _save_papers(papers: list):
    _PAPERS_FILE.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding='utf-8')

def _upsert_paper(job_id: str, topic: str, status: str, **extra):
    papers = _load_papers()
    for p in papers:
        if p['job_id'] == job_id:
            p['status'] = status
            p['updated_at'] = _time.time()
            p.update(extra)
            _save_papers(papers)
            return
    papers.insert(0, {
        'job_id': job_id, 'topic': topic, 'status': status,
        'created_at': _time.time(), 'updated_at': _time.time(), **extra
    })
    _save_papers(papers)

# Map log messages to structured phases for frontend
_PHASE_MAP = [
    ('Loading AI model',              'init',     'Loading AI model'),
    ('Starting academic research',    'research', 'Starting research'),
    ('Research summaries complete',   'research', 'Research summaries done'),
    ('All research papers organized', 'research', 'Papers organized'),
    ('Research gaps identified',      'research', 'Research gaps identified'),
    ('Designing thesis structure',    'outline',  'Designing structure'),
    ('Outline created',               'outline',  'Outline created'),
    ('Starting chapter composition',  'writing',  'Starting writing'),
    ('Introduction complete',         'writing',  'Introduction complete'),
    ('Literature Review complete',    'writing',  'Literature Review complete'),
    ('Methodology complete',          'writing',  'Methodology complete'),
    ('Analysis & Results complete',   'writing',  'Analysis & Results complete'),
    ('Discussion complete',           'writing',  'Discussion complete'),
    ('Conclusion complete',           'writing',  'Conclusion complete'),
    ('Starting document compilation', 'compile',  'Compiling document'),
    ('Abstract generated',            'compile',  'Abstract generated'),
    ('Starting document export',      'export',   'Exporting files'),
    ('Generation complete',           'done',     'Generation complete'),
]

# Chapter names for tracking
_CHAPTER_NAMES = [
    'Introduction', 'Literature Review', 'Methodology',
    'Analysis & Results', 'Discussion', 'Conclusion',
]


def _run_generation(job_id: str, params: dict):
    """Run paper generation with chapter-level pause/resume support."""
    gate = threading.Event()
    gate.set()  # Initially open

    with _jobs_lock:
        _jobs[job_id]['_gate'] = gate

    def log(msg: str):
        with _jobs_lock:
            _jobs[job_id]['log'].append(msg)
            for keyword, phase, label in _PHASE_MAP:
                if keyword in msg:
                    _jobs[job_id]['steps'].append({
                        'phase': phase, 'label': label,
                        'time': _time.time(), 'msg': msg,
                    })
                    break

    try:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'running'

        from draft_generator import generate_draft
        from pathlib import Path

        output_dir = Path('/tmp') / f'opendraft_{job_id}'
        output_formats = params.get('formats', ['pdf', 'docx', 'md'])

        class _Streamer:
            def stream_milestone(self, msg): log(msg)

            def stream_outline_complete(self, outline_path=None, chapters_count=0, **kw):
                # If outline was already provided (pre-approved by user), skip review
                if params.get('outline'):
                    log('📋 Using pre-approved outline, skipping review')
                    return

                # Read outline content and pause for user review
                outline_text = ''
                if outline_path and Path(str(outline_path)).exists():
                    outline_text = Path(str(outline_path)).read_text(encoding='utf-8')
                elif output_dir.exists():
                    # Try to find outline in drafts folder
                    for f in (output_dir / 'drafts').glob('*outline*'):
                        outline_text = f.read_text(encoding='utf-8')
                        break
                with _jobs_lock:
                    _jobs[job_id]['status'] = 'outline_review'
                    _jobs[job_id]['outline'] = outline_text
                    _jobs[job_id]['chapters_count'] = chapters_count
                log('📋 Outline ready for review')
                # Block until user approves
                gate.clear()
                gate.wait()
                with _jobs_lock:
                    _jobs[job_id]['status'] = 'running'

            def stream_research_complete(self, *a, **kw): pass

            def stream_chapter_complete(self, chapter_num=None, chapter_name=None, chapter_path=None, **kw):
                # Read the specific chapter file
                chapter_content = ''
                if chapter_path and Path(chapter_path).exists():
                    chapter_content = Path(chapter_path).read_text(encoding='utf-8')
                elif output_dir.exists():
                    drafts = output_dir / 'drafts'
                    if drafts.exists():
                        for f in sorted(drafts.glob('*.md'), key=lambda p: p.stat().st_mtime, reverse=True):
                            chapter_content = f.read_text(encoding='utf-8')
                            break
                # Use chapter number as section key for reliable file matching
                section_key = f'{chapter_num:02d}_chapter' if chapter_num else (chapter_name or 'chapter')
                with _jobs_lock:
                    _jobs[job_id]['status'] = 'chapter_review'
                    _jobs[job_id]['review_chapter'] = section_key
                    _jobs[job_id]['review_chapter_name'] = chapter_name or f'Chapter {chapter_num}'
                    _jobs[job_id]['review_content'] = chapter_content[:8000]
                log(f'📖 {chapter_name or "Chapter"} ready for review')
                # Block until user approves
                gate.clear()
                gate.wait()
                with _jobs_lock:
                    _jobs[job_id]['status'] = 'running'

        class _Tracker:
            cancelled = False
            def log_activity(self, msg, **kw): log(msg)
            def update_phase(self, *a, **kw): pass
            def update_research(self, *a, **kw): pass
            def update_exporting(self, *a, **kw): pass
            def log_source_found(self, title='', authors=None, year='', doi=None, url=None, **kw):
                author_str = ', '.join(authors[:2]) if authors else ''
                if len(authors or []) > 2:
                    author_str += ' et al.'
                ref = f'{author_str} ({year})' if year else author_str
                # Build link for frontend
                link = ''
                if doi:
                    link = f'https://doi.org/{doi}'
                elif url:
                    link = url
                msg = f'📄 Found: {title[:80]}' + (f' — {ref}' if ref else '')
                if link:
                    msg += f' ||{link}||'  # Special marker for frontend to parse
                log(msg)
            def send_heartbeat(self, *a, **kw): pass
            def check_cancellation(self): pass
            def mark_completed(self): pass
            def mark_failed(self, msg): log(f'FAILED: {msg}')

        pdf_path, docx_path = generate_draft(
            topic=params['topic'],
            language=params.get('language', 'en'),
            academic_level=params.get('level', 'research_paper'),
            output_dir=output_dir,
            skip_validation=True,
            verbose=False,
            tracker=_Tracker(),
            streamer=_Streamer(),
            blurb=params.get('blurb') or None,
            author_name=params.get('author_name') or None,
            institution=params.get('institution') or None,
            department=params.get('department') or None,
            faculty=params.get('faculty') or None,
            advisor=params.get('advisor') or None,
            second_examiner=params.get('second_examiner') or None,
            location=params.get('location') or None,
            student_id=params.get('student_id') or None,
            output_formats=output_formats,
            target_words=params.get('target_words') or None,
            target_citations=int(params['target_citations']) if params.get('target_citations') else None,
            provided_outline=params.get('outline') or None,
        )

        # Find MD file
        exports = output_dir / 'exports'
        md_files = list(exports.glob('*.md'))
        md_path = md_files[0] if md_files else None

        with _jobs_lock:
            _jobs[job_id]['status'] = 'done'
            _jobs[job_id]['pdf'] = str(pdf_path) if pdf_path and pdf_path.exists() else None
            _jobs[job_id]['docx'] = str(docx_path) if docx_path and docx_path.exists() else None
            _jobs[job_id]['md'] = str(md_path) if md_path and md_path.exists() else None
        _upsert_paper(job_id, params.get('topic', ''), 'done',
                       pdf=str(pdf_path) if pdf_path and pdf_path.exists() else None,
                       docx=str(docx_path) if docx_path and docx_path.exists() else None,
                       md=str(md_path) if md_path and md_path.exists() else None)

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'error'
            _jobs[job_id]['error'] = str(e)
        _upsert_paper(job_id, params.get('topic', ''), 'error', error=str(e))
        log(f'ERROR: {e}')


# ===== API ROUTES =====

@app.route('/')
def index():
    return send_file('static/index.html')


@app.route('/api/outline', methods=['POST'])
def generate_outline():
    """Quick outline generation (~10-30s) using AI model directly."""
    params = request.json or {}
    topic = params.get('topic', '').strip()
    if not topic:
        return jsonify({'error': 'Topic is required'}), 400

    language = params.get('language', 'zh')
    level = params.get('level', 'research_paper')
    target_words = params.get('target_words', '')
    target_citations = params.get('target_citations', '')

    lang_labels = {
        'zh': 'Chinese', 'en': 'English', 'de': 'German',
        'fr': 'French', 'es': 'Spanish', 'ja': 'Japanese',
    }
    lang_name = lang_labels.get(language, 'English')

    level_labels = {
        'research_paper': 'Research Paper',
        'bachelor': 'Bachelor Thesis',
        'master': 'Master Thesis',
        'phd': 'PhD Dissertation',
    }
    level_name = level_labels.get(level, 'Research Paper')

    word_info = f'\n- Target length: approximately {target_words} words' if target_words else ''
    cite_info = f'\n- Target citations: approximately {target_citations} references' if target_citations else ''

    prompt = f"""Generate a detailed academic paper outline for the following topic.

Topic: {topic}
Paper type: {level_name}
Language: {lang_name}{word_info}{cite_info}

Requirements:
1. Create a well-structured outline with chapter and section headings
2. Use markdown format with ## for chapters and ### for sections
3. Under each section, add 1-2 bullet points describing key content
4. Include these standard academic sections:
   - Chapter 1: Introduction (background, problem statement, research objectives, significance)
   - Chapter 2: Literature Review (key theories, existing research, research gaps)
   - Chapter 3: Methodology (research design, data collection, analysis methods)
   - Chapter 4: Analysis & Results (findings, data presentation)
   - Chapter 5: Discussion (interpretation, implications, limitations)
   - Chapter 6: Conclusion (summary, contributions, future research)
5. Adapt chapter names and content to fit the specific topic
6. Write all content in {lang_name}

Return ONLY the outline in markdown format, no other text."""

    try:
        from utils.agent_runner import setup_model
        import re as _re
        model = setup_model()
        response = model.generate_content(prompt)
        raw = response.text

        # Separate AI thinking from the actual outline
        # The outline starts at the first markdown heading (## or #)
        thinking_text = ''
        outline = raw

        heading_match = _re.search(r'^(#{1,3}\s)', raw, _re.MULTILINE)
        if heading_match and heading_match.start() > 0:
            thinking_text = raw[:heading_match.start()].strip()
            outline = raw[heading_match.start():].strip()

        # Translate thinking to target language if needed
        thinking_summary = ''
        if thinking_text and language != 'en':
            # Generate a brief translated summary of the thinking process
            try:
                summary_prompt = f"Summarize the following AI reasoning process in 3-5 short bullet points in {lang_name}. Each point should be one concise sentence. Return ONLY the bullet points, no other text.\n\n{thinking_text[:2000]}"
                summary_resp = model.generate_content(summary_prompt)
                thinking_summary = summary_resp.text.strip()
            except Exception:
                thinking_summary = thinking_text[:500]
        elif thinking_text:
            thinking_summary = thinking_text[:500]

        return jsonify({
            'outline': outline,
            'thinking': thinking_summary,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate():
    """Start paper generation. Accepts optional 'outline' param for pre-approved outline."""
    params = request.json or {}
    if not params.get('topic', '').strip():
        return jsonify({'error': 'Topic is required'}), 400

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            'status': 'pending', 'log': [], 'steps': [],
            'pdf': None, 'docx': None, 'md': None,
            'topic': params.get('topic', ''),
            'start_time': _time.time(),
            'outline': None, 'review_chapter': None, 'review_content': None,
        }

    _upsert_paper(job_id, params.get('topic', ''), 'running',
                   language=params.get('language', 'zh'),
                   level=params.get('level', 'research_paper'))

    t = threading.Thread(target=_run_generation, args=(job_id, params), daemon=True)
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/api/approve/<job_id>', methods=['POST'])
def approve(job_id):
    """Resume a paused job (outline_review or chapter_review)."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    gate = job.get('_gate')
    if not gate:
        return jsonify({'error': 'No gate found'}), 400

    # Accept optional edited content
    data = request.json or {}
    edited = data.get('content')
    if edited and job.get('status') == 'outline_review':
        # Save edited outline back to the output dir
        output_dir = Path('/tmp') / f'opendraft_{job_id}' / 'drafts'
        if output_dir.exists():
            for f in output_dir.glob('*outline*'):
                f.write_text(edited, encoding='utf-8')
                break

    gate.set()  # Unblock the generation thread
    return jsonify({'ok': True})


@app.route('/api/status/<job_id>')
def status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    # Return a serializable copy
    safe = {}
    for k, v in job.items():
        if k.startswith('_'):
            continue  # Skip internal objects (threading.Event etc.)
        if k == 'steps':
            safe[k] = [{kk: vv for kk, vv in s.items() if kk != 'time'} for s in v]
        else:
            safe[k] = v
    return jsonify(safe)


@app.route('/api/stream/<job_id>')
def stream(job_id):
    """SSE stream with support for review pauses."""
    def generate_events():
        sent_logs = 0
        sent_steps = 0
        last_status = None
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
            if not job:
                yield 'data: {"error": "job not found"}\n\n'
                break
            logs = job['log']
            steps = job['steps']
            cur_status = job['status']

            # Send new logs
            while sent_logs < len(logs):
                msg = logs[sent_logs].replace('\n', ' ')
                yield f'data: {json.dumps({"type": "log", "msg": msg})}\n\n'
                sent_logs += 1

            # Send new steps
            while sent_steps < len(steps):
                s = steps[sent_steps]
                yield f'data: {json.dumps({"type": "step", "phase": s["phase"], "label": s["label"], "msg": s["msg"]})}\n\n'
                sent_steps += 1

            # Notify frontend of review pauses
            if cur_status != last_status:
                if cur_status == 'outline_review':
                    yield f'data: {json.dumps({"type": "outline_review", "outline": job.get("outline", ""), "chapters_count": job.get("chapters_count", 0)})}\n\n'
                elif cur_status == 'chapter_review':
                    yield f'data: {json.dumps({"type": "chapter_review", "chapter": job.get("review_chapter", ""), "chapter_name": job.get("review_chapter_name", ""), "content": job.get("review_content", "")})}\n\n'
                last_status = cur_status

            if cur_status in ('done', 'error'):
                yield f'data: {json.dumps({"type": "done", "status": cur_status, "pdf": job.get("pdf"), "docx": job.get("docx"), "md": job.get("md"), "error": job.get("error")})}\n\n'
                break
            _time.sleep(0.5)

    return Response(stream_with_context(generate_events()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/preview/<job_id>')
def preview(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    md_path = job.get('md')
    if not md_path or not Path(md_path).exists():
        return jsonify({'error': 'Markdown file not available'}), 404

    import markdown as md_lib
    import re

    raw = Path(md_path).read_text(encoding='utf-8')
    meta = {}
    body = raw
    yaml_match = re.match(r'^---\n(.*?)\n---\n', raw, re.DOTALL)
    if yaml_match:
        for line in yaml_match.group(1).splitlines():
            if ':' in line:
                k, _, v = line.partition(':')
                meta[k.strip()] = v.strip().strip('"')
        body = raw[yaml_match.end():]

    content_html = md_lib.markdown(body, extensions=['tables', 'fenced_code', 'toc'])
    title = meta.get('title', 'Paper')
    author = meta.get('author', '')
    date = meta.get('date', '')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><title>{title}</title>
<style>
  @page {{ margin: 2.5cm 2.8cm; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Times New Roman', 'SimSun', serif; font-size: 12pt; line-height: 1.8; color: #111; background: #fff; max-width: 800px; margin: 0 auto; padding: 2rem 2.5rem; }}
  .cover {{ text-align: center; padding: 4rem 0 3rem; border-bottom: 2px solid #111; margin-bottom: 2.5rem; page-break-after: always; }}
  .cover h1 {{ font-size: 1.6rem; font-weight: bold; margin-bottom: 1.5rem; line-height: 1.4; }}
  .cover .meta {{ font-size: 0.95rem; color: #444; line-height: 2; }}
  h1, h2, h3, h4 {{ font-family: Arial, 'Microsoft YaHei', sans-serif; margin: 1.5rem 0 0.6rem; }}
  h1 {{ font-size: 1.4rem; border-bottom: 1px solid #ddd; padding-bottom: 0.4rem; }}
  h2 {{ font-size: 1.2rem; }} h3 {{ font-size: 1.05rem; }}
  p {{ margin-bottom: 0.75rem; text-align: justify; }}
  ul, ol {{ margin: 0.5rem 0 0.75rem 1.5rem; }} li {{ margin-bottom: 0.25rem; }}
  blockquote {{ border-left: 3px solid #aaa; padding-left: 1rem; color: #555; margin: 1rem 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; }}
  th {{ background: #f5f5f5; font-weight: bold; }}
  code {{ font-family: 'Courier New', monospace; font-size: 0.88rem; background: #f5f5f5; padding: 0.1rem 0.3rem; border-radius: 3px; }}
  pre code {{ display: block; padding: 0.75rem; overflow-x: auto; }}
  @media print {{ body {{ padding: 0; max-width: none; }} .no-print {{ display: none !important; }} }}
  .print-bar {{ position: fixed; bottom: 1.5rem; right: 1.5rem; }}
  .btn-print {{ padding: 0.6rem 1.2rem; background: #6366f1; color: #fff; border: none; border-radius: 8px; font-size: 0.9rem; font-weight: 600; cursor: pointer; }}
</style>
</head>
<body>
<div class="cover"><h1>{title}</h1><div class="meta">
{'<div>' + author + '</div>' if author else ''}
{'<div>' + date + '</div>' if date else ''}
{'<div>' + meta.get('institution','') + '</div>' if meta.get('institution') else ''}
</div></div>
{content_html}
<div class="print-bar no-print"><button class="btn-print" onclick="window.print()">Print / Save PDF</button></div>
</body></html>"""
    return Response(html, mimetype='text/html')


@app.route('/api/download/<job_id>/<fmt>')
def download(job_id, fmt):
    if fmt not in ('pdf', 'docx', 'md'):
        return jsonify({'error': 'Invalid format'}), 400
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    path = job.get(fmt)
    if not path or not Path(path).exists():
        return jsonify({'error': f'{fmt.upper()} not available'}), 404
    return send_file(path, as_attachment=True)


@app.route('/api/md/<job_id>')
def get_md(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    md_path = job.get('md')
    if not md_path or not Path(md_path).exists():
        return jsonify({'error': 'Markdown not available yet'}), 404
    raw = Path(md_path).read_text(encoding='utf-8')
    return jsonify({'content': raw})


@app.route('/api/save/<job_id>', methods=['POST'])
def save_md(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    md_path = job.get('md')
    if not md_path:
        return jsonify({'error': 'No markdown file'}), 404
    content = (request.json or {}).get('content', '')
    Path(md_path).write_text(content, encoding='utf-8')
    return jsonify({'ok': True})


@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """Chat with AI to modify current section content."""
    data = request.json or {}
    message = data.get('message', '').strip()
    current_content = data.get('current_content', '')
    section_name = data.get('section_name', '')
    language = data.get('language', 'zh')

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    lang_labels = {
        'zh': 'Chinese', 'en': 'English', 'de': 'German',
        'fr': 'French', 'es': 'Spanish', 'ja': 'Japanese',
    }
    lang_name = lang_labels.get(language, 'Chinese')

    prompt = f"""You are an academic writing assistant. The user is working on a research paper.

Current section: {section_name or 'Unknown'}
Language: {lang_name}

The current content of this section is:
---
{current_content[:8000]}
---

The user's instruction:
{message}

Apply the user's instruction to modify the content above. Return ONLY the modified content in markdown format. Keep the same academic tone and language ({lang_name}). Do not include any explanation or meta-text."""

    try:
        from utils.agent_runner import setup_model
        model = setup_model()
        response = model.generate_content(prompt)
        raw = response.text

        # Strip thinking if present
        import re
        heading_match = re.search(r'^(#{1,6}\s)', raw, re.MULTILINE)
        if heading_match and heading_match.start() > 0:
            content_part = raw[heading_match.start():]
            if len(content_part) > len(raw) * 0.3:
                raw = content_part

        return jsonify({'content': raw.strip()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sections/<job_id>')
def list_sections(job_id):
    """List available chapter files for a job."""
    output_dir = Path('/tmp') / f'opendraft_{job_id}' / 'drafts'
    if not output_dir.exists():
        return jsonify([])
    sections = []
    for f in sorted(output_dir.glob('*.md')):
        sections.append({
            'name': f.stem,
            'path': str(f),
            'size': f.stat().st_size,
        })
    return jsonify(sections)


@app.route('/api/sections/<job_id>/<section_name>')
def get_section(job_id, section_name):
    """Read a specific section's content."""
    output_dir = Path('/tmp') / f'opendraft_{job_id}' / 'drafts'
    # Find matching files, prefer formatted versions
    matches = [f for f in output_dir.glob('*.md') if section_name in f.stem]
    if not matches:
        return jsonify({'error': 'Section not found'}), 404
    # Prefer formatted_outline over plain outline
    best = next((f for f in matches if 'formatted' in f.stem), matches[0])
    return jsonify({'content': best.read_text(encoding='utf-8'), 'path': str(best)})


@app.route('/api/sections/<job_id>/<section_name>', methods=['PUT'])
def save_section(job_id, section_name):
    """Save modified section content back to file."""
    output_dir = Path('/tmp') / f'opendraft_{job_id}' / 'drafts'
    content = (request.json or {}).get('content', '')
    for f in output_dir.glob('*.md'):
        if section_name in f.stem:
            f.write_text(content, encoding='utf-8')
            return jsonify({'ok': True})
    return jsonify({'error': 'Section not found'}), 404


@app.route('/api/papers')
def list_papers():
    """List all papers (for 'My Papers' page)."""
    papers = _load_papers()
    # Update status for running jobs from in-memory state
    with _jobs_lock:
        for p in papers:
            job = _jobs.get(p['job_id'])
            if job and job['status'] not in ('done', 'error'):
                p['status'] = job['status']
    return jsonify(papers)


@app.route('/api/papers/<job_id>', methods=['DELETE'])
def delete_paper(job_id):
    """Delete a paper from the list."""
    papers = _load_papers()
    papers = [p for p in papers if p['job_id'] != job_id]
    _save_papers(papers)
    return jsonify({'ok': True})


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=8080, debug=False)
