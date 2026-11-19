import test from 'node:test';
import assert from 'node:assert/strict';
import {newRun, acceptEvent, finishRun, timingValues, splitFinalAnswer, extractFrames, isCurrentRun, previewSummary, retrievalStages} from '../../evidence_rag/static/workspace.mjs';

test('replay preserves different source ranks and uses the recorded stage times', () => {
  const candidates = [
    {unit_id:'a', document_id:'alpha', rrf_score:0.035, reranker_score:0.82,
      sources:[{source:'bm25',rank:2,raw_score:-2},{source:'dense',rank:1,raw_score:0.9}]},
    {unit_id:'b', document_id:'beta', reranker_score:0,
      sources:[{source:'bm25',rank:1,raw_score:-4},{source:'dense',rank:2,raw_score:0.7}]},
  ];
  const stages = retrievalStages({corpus_revision:'r1', evidence:[{unit_id:'a',title:'Alpha'}],
    retrieval_trace:{dense_backend:'openviking', dense_hits:[{},{}], candidates,
      timings_ms:{lexical_ms:0,dense_ms:18.11,rank_and_rerank_ms:202.84}},
    graph:{as_of:'2026-11-19',paths:[],graph_build_id:'g',graph_ms:1.81,edge_count:0,
      traversal_status:'partial_budget',evidence_status:'incomplete'}});
  assert.deepEqual(stages.map(stage=>stage.id),['scope','lexical','dense','rank','graph','returned']);
  assert.deepEqual(stages[1].rows.map(row=>row[1]),['beta','alpha']);
  assert.deepEqual(stages[2].rows.map(row=>row[1]),['alpha','beta']);
  assert.equal(stages[3].rows[1].at(-1),'0.000000');
  assert.equal(stages[4].result,'达到预算 · 0 条已查');
  assert.equal(stages[4].duration,1.81);
});

test('replay distinguishes a missing trace from a recorded but unavailable service', () => {
  const missing = retrievalStages({evidence:[],corpus_revision:null});
  assert.deepEqual(missing.map(stage=>stage.id),['scope','returned']);
  const degraded = retrievalStages({kind:'exploration',items:[],corpus_revision:'r1',
    degraded_reasons:['dense_not_configured','reranker_failed'],
    retrieval_trace:{timings_ms:{lexical_ms:0.28,rank_and_rerank_ms:1}}});
  assert.deepEqual(degraded.map(stage=>stage.id),['scope','lexical','dense','rank','returned']);
  assert.equal(degraded[2].result,'语义服务不可用');
  assert.equal(degraded[3].title,'合并候选');
  assert.equal(degraded[3].result,'本次未记录候选');
});

test('real event stages preserve the whole answer and separate elapsed phases', () => {
  const run = newRun('answer', 1, 100);
  acceptEvent(run, 'evidence_ready', {bundle: {evidence: ['fixture']}}, 1600);
  acceptEvent(run, 'answer_delta', {delta: 'First '}, 2200);
  acceptEvent(run, 'answer_delta', {delta: 'answer [E1]'}, 2800);
  assert.equal(run.outcome, 'running');
  acceptEvent(run, 'completed', {status: 'ready'}, 3000);
  finishRun(run, 3100);
  assert.equal(run.pending, 'First answer [E1]');
  assert.deepEqual(timingValues(run, 9999), {retrieval: 1500, generation: 1500, total: 3000, finished: true, retrievalComplete: true, generationComplete: true});
});

test('preview does not report model generation and a new request starts its own clock', () => {
  const preview = newRun('explore', 1, 0);
  preview.evidenceAt = 4000; preview.finalSeen = true; preview.outcome = 'preview'; finishRun(preview, 4001);
  assert.equal(timingValues(preview, 10000).generation, null);
  const answer = newRun('answer', 2, 20000);
  assert.equal(timingValues(answer, 20001).total, 1);
  assert.equal(timingValues(answer, 20001).generation, null);
});

test('early disconnect, failure and cancellation cannot retain a final draft', () => {
  for (const error of [null, new Error('failed'), Object.assign(new Error('stop'), {name: 'AbortError'})]) {
    const run = newRun('answer', 1, 10);
    acceptEvent(run, 'answer_delta', {delta: 'unverified draft'}, 30);
    finishRun(run, 50, error);
    assert.equal(run.pending, ''); assert.notEqual(run.outcome, 'ready');
    assert.equal(timingValues(run, 999).retrieval, null);
    assert.equal(timingValues(run, 999).generation, null);
    assert.equal(timingValues(run, 999).total, 40);
  }
});

test('rejection followed by completion cannot become a success', () => {
  const run = newRun('answer', 1, 0);
  acceptEvent(run, 'answer_delta', {delta: 'draft'}, 1);
  acceptEvent(run, 'answer_rejected', {}, 2);
  acceptEvent(run, 'completed', {status: 'ready'}, 3); finishRun(run, 4);
  assert.equal(run.pending, ''); assert.equal(run.outcome, 'failed');
});

test('retry replaces the discarded draft without restarting the user clock', () => {
  const run = newRun('answer', 1, 0);
  acceptEvent(run, 'evidence_ready', {bundle: {}}, 100);
  acceptEvent(run, 'answer_delta', {delta: 'old'}, 120);
  acceptEvent(run, 'generation_retry', {}, 150);
  acceptEvent(run, 'answer_delta', {delta: 'new'}, 200);
  acceptEvent(run, 'completed', {status: 'ready'}, 220); finishRun(run, 250);
  assert.equal(run.pending, 'new'); assert.equal(timingValues(run, 999).total, 250);
});

test('only the bound request epoch may change the current page', () => {
  const old = newRun('answer', 1, 0), current = newRun('answer', 2, 10);
  const state = {run: current, epoch: 2};
  assert.equal(isCurrentRun(state, old), false); assert.equal(isCurrentRun(state, current), true);
  state.epoch = 3; assert.equal(isCurrentRun(state, current), false);
});

test('Chinese verdict and quotes are presentation-only; ungrounded prose is preserved', () => {
  const canonical = 'SUPPORT\nUseful answer [E1]\n\n证据原文：\n[E1] “exact original”';
  assert.deepEqual(splitFinalAnswer(canonical, 'SUPPORT'), {label: '资料支持该结论', body: 'Useful answer [E1]', quotes: '[E1] “exact original”'});
  assert.deepEqual(splitFinalAnswer(canonical), {label: '', body: canonical, quotes: ''});
  assert.equal(splitFinalAnswer('ANSWER\nThis is plain body', 'ANSWER').body, 'ANSWER\nThis is plain body');
});

test('split SSE frames wait for completion and retain unicode payloads', () => {
  const first = extractFrames('event: answer_delta\r\ndata: {"delta":"内容"}\r\n\r\nevent: completed\ndata: {"status":');
  assert.deepEqual(first.frames, [{kind: 'answer_delta', data: {delta: '内容'}}]);
  const second = extractFrames(first.rest + '"ready"}\n\n');
  assert.deepEqual(second.frames, [{kind: 'completed', data: {status: 'ready'}}]);
});

test('invalid clocks never fabricate a positive latency', () => {
  const run = newRun('answer', 1, 100); run.evidenceAt = 50; finishRun(run, 40);
  const values = timingValues(run, 500);
  assert.equal(values.retrieval, null); assert.equal(values.generation, null); assert.equal(values.total, null);
});

test('partial or empty previews cannot report a normal complete result', () => {
  assert.equal(previewSummary({degraded: false, items: [1]}).status, '检索已完成');
  const partial = previewSummary({degraded: true, items: [1]});
  assert.equal(partial.status, '仅取得部分检索资料');
  assert.match(partial.note, /不完整/);
  assert.equal(previewSummary({degraded: false, items: []}).status, '未找到可用资料');
});

test('legal date selection context is preserved in preview metadata', () => {
  const bundle = {degraded:false, kind:'exploration', corpus_revision:'civil-2020',
    query_plan:{event_date:'2015-01-01', revision_selection:{selected:'civil-old', reference_date:'2015-01-01'}},
    items:[{title:'民法典', article_label:'第一条', temporal:{validity:'valid', valid_from:'2010-01-01', valid_to:'2020-12-31'}}]};
  assert.equal(bundle.query_plan.revision_selection.selected, 'civil-old');
  assert.equal(bundle.items[0].temporal.validity, 'valid');
});
