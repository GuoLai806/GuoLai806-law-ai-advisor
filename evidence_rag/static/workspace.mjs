const decisionLabels = {
  SUPPORT: '资料支持该结论', CONTRADICT: '资料与该结论相矛盾',
  NOT_ENOUGH_INFO: '现有资料不足以支持结论', ANSWER: '综合回答'
};

export function newRun(mode, epoch, started) {
  return {mode, epoch, started, evidenceAt: null, ended: null, outcome: 'running',
    finalSeen: false, error: null, pending: '', decision: null, quotesChecked: false, bundle: null};
}

export function isCurrentRun(state, run) {
  return state.run === run && state.epoch === run.epoch;
}

export function acceptEvent(run, kind, data, now) {
  if (run.ended !== null) return;
  if (kind === 'evidence_ready') { run.bundle = data.bundle; run.evidenceAt = now; }
  if (kind === 'generation_retry' && !run.finalSeen) run.pending = '';
  if (kind === 'grounding_validation' && data.status === 'verbatim_quotes_checked') {
    run.quotesChecked = true; run.decision = data.decision;
  }
  if (kind === 'answer_delta' && !run.error && !run.finalSeen) {
    if (typeof data.delta !== 'string') throw new Error('回答数据不完整。');
    run.pending += data.delta;
  }
  if (kind === 'answer_rejected' || kind === 'error') {
    run.error = kind === 'answer_rejected' ? '答案引用未通过检查。' : '检索或回答服务未完成本次请求。';
    run.pending = ''; run.outcome = 'failed';
  }
  if (kind === 'completed' && !run.finalSeen) {
    run.finalSeen = true;
    run.outcome = run.error ? 'failed' : data.status === 'ready' && run.pending.trim() ? 'ready' : 'evidence_only';
    if (run.outcome !== 'ready') run.pending = '';
  }
}

export function finishRun(run, now, error = null) {
  run.ended = now;
  if (error) {
    run.outcome = error.name === 'AbortError' ? 'stopped' : 'failed';
    run.error = error.name === 'AbortError' ? null : error.message;
    run.pending = '';
  } else if (!run.finalSeen) {
    run.outcome = 'failed'; run.pending = ''; run.error = '连接提前结束，未取得最终答案。';
  }
}

function duration(start, end) {
  return Number.isFinite(start) && Number.isFinite(end) && end >= start ? end - start : null;
}

export function timingValues(run, now) {
  const end = run.ended ?? now;
  return {
    retrieval: run.evidenceAt !== null ? duration(run.started, run.evidenceAt) : run.ended === null ? duration(run.started, now) : null,
    generation: run.mode === 'explore' || run.evidenceAt === null ? null : duration(run.evidenceAt, end),
    total: duration(run.started, end), finished: run.ended !== null,
    retrievalComplete: run.evidenceAt !== null, generationComplete: run.outcome === 'ready'
  };
}

export function splitFinalAnswer(text, decision = null) {
  let body = text;
  const label = decisionLabels[decision] || '';
  if (label && decision !== 'ANSWER' && body.startsWith(decision + '\n')) body = body.slice(decision.length + 1);
  const marker = '\n\n证据原文：\n';
  const position = label ? body.indexOf(marker) : -1;
  return {label, body: (position < 0 ? body : body.slice(0, position)).trim(),
    quotes: position < 0 ? '' : body.slice(position + marker.length).trim()};
}

export function previewSummary(bundle) {
  if (bundle.degraded) return {status: '仅取得部分检索资料', message: '本次只检索资料，没有请求 AI 生成。', note: '部分检索服务不可用，当前结果可能不完整。'};
  if (!bundle.items.length) return {status: '未找到可用资料', message: '本次没有找到可用资料，未请求 AI 生成。', note: ''};
  return {status: '检索已完成', message: '本次只检索资料，没有请求 AI 生成。', note: ''};
}

export function extractFrames(buffer) {
  buffer = buffer.replace(/\r\n/g, '\n');
  const frames = [];
  let boundary;
  while ((boundary = buffer.indexOf('\n\n')) !== -1) {
    const lines = buffer.slice(0, boundary).split('\n');
    buffer = buffer.slice(boundary + 2);
    const kind = lines.find(line => line.startsWith('event: '));
    const payload = lines.filter(line => line.startsWith('data: ')).map(line => line.slice(6)).join('\n');
    if (kind && payload) frames.push({kind: kind.slice(7), data: JSON.parse(payload)});
  }
  return {frames, rest: buffer};
}

function boot() {
  const get = id => document.getElementById(id);
  const question = get('question'), ask = get('ask'), explore = get('explore'), stop = get('stop');
  const state = {epoch: 0, run: null, source: null, snapshot: null, timer: null};
  const sessionId = 'web-' + crypto.randomUUID();
  const setNotice = message => { get('notice').textContent = message; get('notice').classList.toggle('hidden', !message); };
  const busy = value => { ask.disabled = explore.disabled = value; stop.disabled = !value; };
  const seconds = value => value === null ? '—' : (value / 1000).toFixed(2) + ' 秒';

  function paintTiming(run) {
    const values = timingValues(run, performance.now());
    get('retrieval-time').textContent = values.retrievalComplete ? seconds(values.retrieval) : values.finished ? '未完成' : seconds(values.retrieval) + '（进行中）';
    get('generation-time').textContent = run.mode === 'explore' ? '未请求生成' : values.generation === null ? '尚未开始' : seconds(values.generation) + (values.finished ? values.generationComplete ? '' : '（未完成）' : '（进行中）');
    get('total-time').textContent = seconds(values.total) + (values.finished ? run.outcome === 'stopped' ? '（已停止）' : run.outcome === 'failed' ? '（失败）' : '' : '（进行中）');
  }

  function clearAnswer(message) {
    get('answer').textContent = message;
    get('decision').classList.add('hidden'); get('quotes-panel').classList.add('hidden');
    get('answer-quotes').textContent = ''; get('quotes-panel').open = false;
  }

  function invalidate() {
    state.run?.controller.abort(); state.source?.abort(); clearInterval(state.timer);
    state.epoch++; state.run = null; state.source = null;
    get('source-panel').classList.add('hidden');
  }

  async function hash(text) {
    const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
    return Array.from(new Uint8Array(bytes), x => x.toString(16).padStart(2, '0')).join('');
  }

  async function showSource(item, revision, epoch) {
    state.source?.abort(); const controller = new AbortController(); state.source = controller;
    const response = await fetch('/api/v2/units/' + encodeURIComponent(item.unit_id) + '?revision=' + encodeURIComponent(revision), {signal: controller.signal});
    if (!response.ok) throw new Error('该版本的原文暂时无法读取。');
    const source = await response.json();
    const part = source.parts?.find(part => part.part_no === item.part_no) || source;
    if (source.revision !== revision || await hash(part.text) !== item.content_sha256 || await hash(source.text) !== (item.document_sha256 || item.content_sha256)) throw new Error('原文版本或内容不一致，暂不展示。');
    if (epoch !== state.epoch || controller.signal.aborted) return;
    get('source-content').textContent = source.title + '\n\n' + source.text;
    get('source-panel').classList.remove('hidden');
    get('source-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
  }

  function displayEvidence(bundle, epoch) {
    const container = get('evidence'); container.replaceChildren();
    const preview = bundle.kind === 'exploration';
    const items = preview ? bundle.items : bundle.evidence;
    get('evidence-summary').textContent = '找到 ' + items.length + ' 条资料' + (bundle.degraded ? '（部分检索）' : '') + (preview ? ' · 本次只检索，尚未生成答案' : ' · 以下为本次检索资料');
    for (const [index, item] of items.entries()) {
      const article = document.createElement('article');
      const heading = document.createElement('h3'); heading.textContent = '[' + (item.citation_id || '资料' + (index + 1)) + '] ' + item.title;
      const text = preview ? item.preview : item.text;
      const excerpt = document.createElement('p'); excerpt.textContent = text.slice(0, 320) + (text.length > 320 || item.preview_truncated ? '…' : '');
      article.append(heading, excerpt);
      if (!preview && text.length > 320) {
        const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = '展开完整检索段落';
        const full = document.createElement('div'); full.className = 'full-text'; full.textContent = text;
        details.append(summary, full); article.append(details);
      }
      const button = document.createElement('button'); button.type = 'button'; button.className = 'secondary'; button.textContent = '查看完整原文';
      button.addEventListener('click', () => showSource(item, bundle.corpus_revision, epoch).catch(error => { if (epoch === state.epoch && error.name !== 'AbortError') setNotice(error.message); }));
      article.append(button); container.append(article);
    }
    if (!items.length) container.textContent = '没有找到可用资料。';
  }

  function paintResult(run) {
    if (run.outcome === 'ready') {
      const final = splitFinalAnswer(run.pending, run.decision);
      get('answer').textContent = final.body; get('decision').textContent = final.label;
      get('decision').classList.toggle('hidden', !final.label);
      get('answer-quotes').textContent = final.quotes; get('quotes-panel').classList.toggle('hidden', !final.quotes);
      get('status').textContent = '回答已完成';
      setNotice(run.quotesChecked ? '引用摘录已与原文核对；结论是否充分仍请结合资料判断。' : '回答已完成，请结合资料核对关键结论。');
    } else if (run.outcome === 'preview') {
      const preview = previewSummary(run.bundle);
      clearAnswer(preview.message); get('status').textContent = preview.status; setNotice(preview.note);
    } else if (run.outcome === 'running') {
      get('status').textContent = run.evidenceAt === null ? '正在检索资料…' : '正在生成并核验答案…';
      clearAnswer(run.evidenceAt === null ? '等待资料检索完成。' : '正在根据检索资料生成答案，完成后显示。');
    } else {
      const stopped = run.outcome === 'stopped';
      clearAnswer(stopped ? '已停止，未完成的答案不会展示。' : '本次没有取得可交付的 AI 答案，可先查看检索资料。');
      get('status').textContent = stopped ? '已停止' : '本次未完成'; setNotice(run.error || '');
      if (!run.bundle) { get('evidence').textContent = stopped ? '检索已停止，未取得资料。' : '本次未取得资料。'; get('evidence-summary').textContent = '检索未完成'; }
    }
  }

  async function send(mode) {
    const query = question.value.trim(); if (!query) return;
    const pinned = state.snapshot?.query === query ? state.snapshot.revision : null;
    invalidate(); state.snapshot = null;
    const run = newRun(mode, state.epoch, performance.now()); run.controller = new AbortController(); state.run = run;
    busy(true); clearAnswer('等待资料检索完成。'); setNotice('');
    get('evidence').textContent = '正在检索本次问题的资料…'; get('evidence-summary').textContent = '正在检索';
    get('timing-note').textContent = mode === 'explore' ? '本次只检索，不调用生成模型；再次问答会重新计时。' : '本次点击到完成的浏览器等待；检索含重排，生成含引用核验和网络等待。';
    paintResult(run); paintTiming(run); state.timer = setInterval(() => { if (isCurrentRun(state, run)) paintTiming(run); }, 250);
    try {
      const response = await fetch(mode === 'explore' ? '/api/v2/explore' : '/api/v2/chat/stream', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query, session_id: sessionId, corpus_revision: mode === 'explore' ? null : pinned}), signal: run.controller.signal
      });
      if (!response.ok) throw new Error(response.status === 401 || response.status === 403 ? '当前页面无法访问资料，请检查服务访问设置。' : '服务暂时不可用。');
      if (mode === 'explore') {
        const bundle = await response.json(); if (!isCurrentRun(state, run)) return;
        run.bundle = bundle; run.evidenceAt = performance.now(); run.finalSeen = true; run.outcome = 'preview';
        state.snapshot = {query, revision: bundle.corpus_revision}; displayEvidence(bundle, run.epoch);
      } else {
        const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
        for (;;) {
          const chunk = await reader.read(); if (!isCurrentRun(state, run)) return;
          buffer += chunk.done ? decoder.decode() : decoder.decode(chunk.value, {stream: true});
          const parsed = extractFrames(buffer); buffer = parsed.rest;
          for (const frame of parsed.frames) {
            acceptEvent(run, frame.kind, frame.data, performance.now());
            if (frame.kind === 'evidence_ready') displayEvidence(run.bundle, run.epoch);
          }
          // Streaming drafts stay buffered until the stream and its completion event finish.
          if (run.outcome !== 'ready') paintResult(run);
          if (chunk.done) break;
        }
      }
      finishRun(run, performance.now());
    } catch (error) {
      if (!isCurrentRun(state, run)) return;
      finishRun(run, performance.now(), error);
    } finally {
      if (isCurrentRun(state, run)) {
        clearInterval(state.timer); busy(false); paintResult(run); paintTiming(run);
      }
    }
  }

  get('question-form').addEventListener('submit', event => { event.preventDefault(); send('answer'); });
  explore.addEventListener('click', () => send('explore'));
  stop.addEventListener('click', () => state.run?.controller.abort());
  question.addEventListener('input', () => {
    invalidate(); state.snapshot = null; busy(false); clearAnswer('等待生成答案。'); setNotice('');
    get('status').textContent = '等待提问'; get('evidence').textContent = '问题已更改，请重新检索资料。'; get('evidence-summary').textContent = '等待重新检索';
    for (const id of ['retrieval-time', 'generation-time', 'total-time']) get(id).textContent = '—';
  });
}

if (typeof document !== 'undefined') boot();
