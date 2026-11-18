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
  const question = get('question'), ask = get('ask'), explore = get('explore'), stop = get('stop'), eventDate = get('event-date');
  const graphSearch = get('graph-search'), graphSeed = get('graph-seed'), graphPanel = get('graph-panel');
  const graphNext = get('graph-next');
  const state = {epoch: 0, run: null, source: null, snapshot: null, timer: null};
  const sessionId = 'web-' + crypto.randomUUID();
  const setNotice = message => { get('notice').textContent = message; get('notice').classList.toggle('hidden', !message); };
  const busy = value => { ask.disabled = explore.disabled = graphSearch.disabled = value; stop.disabled = !value; };
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
    graphPanel.classList.add('hidden');
  }

  async function hash(text) {
    const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
    return Array.from(new Uint8Array(bytes), x => x.toString(16).padStart(2, '0')).join('');
  }

  async function showSource(item, revision, epoch, span = null) {
    state.source?.abort(); const controller = new AbortController(); state.source = controller;
    const response = await fetch('/api/v2/units/' + encodeURIComponent(item.unit_id) + '?revision=' + encodeURIComponent(revision), {signal: controller.signal});
    if (!response.ok) throw new Error('该版本的原文暂时无法读取。');
    const source = await response.json();
    const part = source.parts?.find(part => part.part_no === item.part_no) || source;
    if (source.revision !== revision || await hash(part.text) !== item.content_sha256 || await hash(source.text) !== (item.document_sha256 || item.content_sha256)) throw new Error('原文版本或内容不一致，暂不展示。');
    if (epoch !== state.epoch || controller.signal.aborted) return;
    get('source-content').textContent = source.title + '\n\n' + source.text;
    const match = get('source-match');
    match.classList.toggle('hidden', span === null);
    if (span) {
      const excerpt = document.createElement('mark');
      excerpt.textContent = Array.from(source.text).slice(span[0], span[1]).join('');
      match.replaceChildren('关系依据：', excerpt);
    } else match.replaceChildren();
    get('source-panel').classList.remove('hidden');
    get('source-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
  }

  function displayGraph(bundle, epoch) {
    const graph = bundle.graph;
    const list = get('graph-paths');
    list.replaceChildren();
    graphPanel.classList.toggle('hidden', !graph);
    graphNext.classList.toggle('hidden', !graph?.next_path_offset);
    if (!graph) return;
    const reasons = {
      ambiguous_seed: '起点有歧义，请输入完整的实体名称或版本。',
      ambiguous_date: '日期不明确，请在上方填写具体的查询日期。',
      seed_not_found: '没有找到可核对的起点，请填写完整名称。',
      graph_unavailable: '这批资料暂无可用关系图，下面保留普通检索结果。',
      graph_disabled: '当前查询未启用关系检索，下面保留普通检索结果。',
      graph_changed: '关系图已更新，请重新检索。',
      source_revoked: '原文状态已变更，请重新检索。',
      graph_error: '关系查询未能完成，下面显示可用资料。'
    };
    let status;
    if (graph.traversal_status === 'exhausted') {
      status = graph.evidence_status === 'sufficient'
        ? '已查完当前图谱中有依据的依赖。'
        : '已查完可核对的关系，资料覆盖不完整。';
    } else if (graph.traversal_status === 'partial_budget') {
      status = '达到查询预算，当前展示已确认的部分关系。';
    } else {
      status = reasons[graph.stop_reason] || '关系查询未完成。';
    }
    const time = Number.isFinite(graph.graph_ms) ? ' · ' + graph.graph_ms.toFixed(2) + ' 毫秒' : '';
    const count = graph.paths.length ? ' · ' + graph.edge_count + ' 条关系 · 最深 ' + graph.max_depth_reached + ' 跳' : '';
    const date = graph.as_of ? ' · 查询日期 ' + graph.as_of : '';
    get('graph-status').textContent = status + count + date + time + (graph.paths_truncated ? ' · 当前展示 ' + graph.paths.length + ' 条' : '');
    for (const path of graph.paths) {
      const li = document.createElement('li');
      const title = document.createElement('strong');
      title.textContent = path.src_name + ' → ' + path.dst_name;
      const source = document.createElement('button');
      source.type = 'button'; source.className = 'secondary'; source.textContent = '查看原文';
      const label = path.citation_id ? ' [' + path.citation_id + ']' : '';
      li.append(title, ' · 第 ' + path.depth + ' 跳' + label + ' ', source);
      source.addEventListener('click', () => {
        const item = {unit_id: path.unit_id, content_sha256: path.content_sha256};
        showSource(item, bundle.corpus_revision, epoch, [path.char_start,path.char_end])
          .catch(error => { if (epoch === state.epoch && error.name !== 'AbortError') setNotice(error.message); });
      });
      list.append(li);
    }
  }

  async function loadMoreGraph() {
    const run = state.run, bundle = run?.bundle, graph = bundle?.graph;
    if (!graph?.next_path_offset || graphNext.disabled || !run.finalSeen) return;
    graphNext.disabled = true;
    try {
      const response = await fetch('/api/v2/retrieve', {
        method:'POST', headers:{'Content-Type':'application/json'}, signal:run.controller.signal,
        body:JSON.stringify({query:run.query, session_id:sessionId, event_date:graph.as_of,
          corpus_revision:bundle.corpus_revision, retrieval_mode:'graph_dependency', graph_seed:graph.seed_id,
          graph_path_offset:graph.next_path_offset, graph_expected_build_id:graph.graph_build_id})
      });
      if (!response.ok) throw new Error('关系图或资料版本已变更，请重新检索。');
      const page = await response.json();
      if (!isCurrentRun(state,run)) return;
      const next = page.graph;
      if (next?.graph_build_id !== graph.graph_build_id || next.as_of !== graph.as_of ||
          page.corpus_revision !== bundle.corpus_revision || !next.paths.length) {
        if (['graph_changed','source_revoked'].includes(next?.stop_reason)) {
          bundle.graph=next; displayGraph(bundle,run.epoch);
          if (next.stop_reason==='source_revoked') {
            get('evidence').replaceChildren(); get('evidence-summary').textContent='原文状态已变更';
            get('source-panel').classList.add('hidden'); clearAnswer('请重新检索。');
          }
        }
        throw new Error('未能读取下一页关系，请重新检索。');
      }
      const paths = graph.paths.concat(next.paths.map(({citation_id, ...path}) => path));
      bundle.graph = {...next, paths, paths_truncated:paths.length < next.edge_count};
      displayGraph(bundle,run.epoch);
    } catch (error) {
      if (isCurrentRun(state,run) && error.name!=='AbortError') setNotice(error.message);
    } finally {
      graphNext.disabled=false;
    }
  }

  function displayEvidence(bundle, epoch) {
    const container = get('evidence'); container.replaceChildren();
    const preview = bundle.kind === 'exploration';
    const items = preview ? bundle.items : bundle.evidence;
    const selection = bundle.query_plan?.revision_selection;
    const dateLine = selection ? '适用日期：' + selection.reference_date + (selection.selected ? ' · 版本：' + selection.selected : ' · 没有可核验的适用版本') : '';
    get('evidence-summary').textContent = '找到 ' + items.length + ' 条资料' + (bundle.degraded ? '（部分检索）' : '') + (preview ? ' · 本次只检索，尚未生成答案' : ' · 以下为本次检索资料') + (dateLine ? ' · ' + dateLine : '');
    displayGraph(bundle, epoch);
    for (const [index, item] of items.entries()) {
      const article = document.createElement('article');
      const heading = document.createElement('h3'); heading.textContent = '[' + (item.citation_id || '资料' + (index + 1)) + '] ' + item.title + (item.article_label ? ' · ' + item.article_label : '');
      const text = preview ? item.preview : item.text;
      const excerpt = document.createElement('p'); excerpt.textContent = text.slice(0, 320) + (text.length > 320 || item.preview_truncated ? '…' : '');
      article.append(heading);
      if (item.article_label || item.temporal) {
        const meta = document.createElement('p'); meta.className = 'muted';
        const validity = item.temporal?.validity ? ' · 效力：' + item.temporal.validity : '';
        const interval = item.temporal?.valid_from ? ' · 生效：' + item.temporal.valid_from + (item.temporal.valid_to ? ' 至 ' + item.temporal.valid_to : '') : '';
        meta.textContent = (item.article_label || '') + validity + interval; article.append(meta);
      }
      article.append(excerpt);
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
      setNotice(run.bundle?.graph?.uses_text_fallback ? '以下为普通检索结果，关系检索未完成。' :
        run.quotesChecked ? '引用摘录已与原文核对；结论是否充分仍请结合资料判断。' : '回答已完成，请结合资料核对关键结论。');
    } else if (run.outcome === 'preview') {
      const preview = previewSummary(run.bundle);
      clearAnswer(preview.message); get('status').textContent = preview.status; setNotice(preview.note);
    } else if (run.outcome === 'running') {
      get('status').textContent = run.evidenceAt === null ? '正在检索资料…' : '正在生成并核验答案…';
      clearAnswer(run.evidenceAt === null ? '等待资料检索完成。' : '正在根据检索资料生成答案，完成后显示。');
    } else if (run.mode === 'graph' && run.outcome === 'evidence_only') {
      clearAnswer('请查看逐跳关系和原文。本次没有生成可核验的 AI 总结。');
      get('status').textContent = '关系检索已返回'; setNotice('');
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
    run.query=query;
    busy(true); clearAnswer('等待资料检索完成。'); setNotice('');
    get('evidence').textContent = '正在检索本次问题的资料…'; get('evidence-summary').textContent = '正在检索';
    get('timing-note').textContent = mode === 'explore' ? '本次只检索，不调用生成模型；再次问答会重新计时。' : '本次点击到完成的浏览器等待；检索含重排，生成含引用核验和网络等待。';
    paintResult(run); paintTiming(run); state.timer = setInterval(() => { if (isCurrentRun(state, run)) paintTiming(run); }, 250);
    try {
      const response = await fetch(mode === 'explore' ? '/api/v2/explore' : '/api/v2/chat/stream', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query, session_id: sessionId, event_date: eventDate.value || null,
          corpus_revision: mode === 'explore' ? null : pinned,
          ...(mode === 'graph' ? {retrieval_mode:'graph_dependency', graph_seed:graphSeed.value.trim() || null} : {})}),
        signal: run.controller.signal
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
  graphSearch.addEventListener('click', () => send('graph'));
  graphNext.addEventListener('click', loadMoreGraph);
  fetch('/api/v2/models')
    .then(response => response.ok ? response.json() : {})
    .then(capabilities => {
      graphSearch.classList.toggle('hidden', !capabilities.graph_available);
      get('graph-input').classList.toggle('hidden', !capabilities.graph_available);
    }).catch(() => {});
  stop.addEventListener('click', () => state.run?.controller.abort());
  question.addEventListener('input', () => {
    invalidate(); state.snapshot = null; busy(false); clearAnswer('等待生成答案。'); setNotice('');
    get('status').textContent = '等待提问'; get('evidence').textContent = '问题已更改，请重新检索资料。'; get('evidence-summary').textContent = '等待重新检索';
    for (const id of ['retrieval-time', 'generation-time', 'total-time']) get(id).textContent = '—';
  });
}

if (typeof document !== 'undefined') boot();
