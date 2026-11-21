import concurrent.futures, threading, hashlib
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    PreTrainedTokenizerFast,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
)
import evidence_rag.retrieval as r


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    p = tmp_path_factory.mktemp("tiny-xlm")
    tk = Tokenizer(
        models.WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "x": 4}, unk_token="<unk>"
        )
    )
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 0))
    PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=8192,
    ).save_pretrained(p)
    c = XLMRobertaConfig(
        vocab_size=5,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=12,
        max_position_embeddings=8194,
        pad_token_id=1,
        num_labels=1,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    torch.manual_seed(41)
    XLMRobertaForSequenceClassification(c).save_pretrained(p)
    return p


def test_real_tiny_model_forward(monkeypatch, tiny):
    monkeypatch.setitem(r.CrossEncoderReranker.MODELS, "bge", (str(tiny), "local"))
    ranker = r.CrossEncoderReranker(
        "bge", device="cpu", long_input="split_8192", batch_size=4
    )
    q = "x"
    text = " ".join(["x"] * 4091)
    p = "x\n" + text
    assert len(ranker.model.tokenizer(q, p, truncation=False)["input_ids"]) == 4097
    unit = {
        "unit_id": "u",
        "title": "x",
        "text": text,
        "document": {"doc_id": "d"},
        "parts": [
            {
                "part_no": 1,
                "text": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "locator": {"char_start": 0, "char_end": len(text)},
            }
        ],
    }
    obs = []
    old = ranker.model.forward

    def forward(inputs, **kw):
        obs.append(int(inputs["attention_mask"].sum()))
        return old(inputs, **kw)

    monkeypatch.setattr(ranker.model, "forward", forward)
    item, score = ranker.rank(q, [unit])[0]
    assert (
        obs == [4097]
        and ranker.model.max_length == 4096
        and item["rerank_window"]["char_end"] == len(text)
    )
    t = r._LONG_WINDOW_CONTEXT.set((ranker.model, 8192))
    try:
        expected = float(ranker.model.predict([(q, p)], batch_size=1)[0])
    finally:
        r._LONG_WINDOW_CONTEXT.reset(t)
    assert score == pytest.approx(expected, abs=1e-7) and 0 <= score <= 1
    with pytest.raises(r.RerankWindowExceeded):
        ranker.model.predict([(q, p)], batch_size=1)


class Base:
    max_length = 4096

    def preprocess(self, inputs, prompt=None, **kw):
        return {
            "attention_mask": torch.ones((len(inputs), inputs[0]), dtype=torch.int64)
        }


class Guard(r._GuardedWindow, Base):
    pass


def test_thread_isolation_and_recovery():
    g = Guard()
    barrier = threading.Barrier(2)

    def long():
        t = r._LONG_WINDOW_CONTEXT.set((g, 8192))
        try:
            barrier.wait()
            assert g.preprocess([8192])["attention_mask"].sum() == 8192
            barrier.wait()
            raise RuntimeError("forward error")
        finally:
            r._LONG_WINDOW_CONTEXT.reset(t)

    def normal():
        barrier.wait()
        with pytest.raises(r.RerankWindowExceeded):
            g.preprocess([4097])
        barrier.wait()

    with concurrent.futures.ThreadPoolExecutor(2) as p:
        a = p.submit(long)
        b = p.submit(normal)
        b.result()
        with pytest.raises(RuntimeError):
            a.result()
    assert r._LONG_WINDOW_CONTEXT.get() is None
    with pytest.raises(r.RerankWindowExceeded):
        g.preprocess([4097])
    t = r._LONG_WINDOW_CONTEXT.set((g, 8192))
    try:
        with pytest.raises(r.RerankWindowExceeded):
            g.preprocess([8193])
        with pytest.raises(r.RerankWindowExceeded):
            Guard().preprocess([4097])
    finally:
        r._LONG_WINDOW_CONTEXT.reset(t)


def test_unsupported():
    c = r.CrossEncoderReranker
    for kw in [
        dict(name="qwen", device="cpu"),
        dict(name="bge", device="cuda"),
        dict(name="bge", device="cpu", window_check="legacy"),
    ]:
        with pytest.raises(ValueError):
            c(long_input="split_8192", **kw)


def test_exception_resets_context():
    class Model:
        max_length = 4096
        default_prompt_name = None
        prompts = {}

        def preprocess(self, pairs, **kw):
            return {"attention_mask": torch.ones((1, 4097))}

        def predict(self, *a, **kw):
            raise RuntimeError("forward error")

    ranker = object.__new__(r.CrossEncoderReranker)
    ranker.model = Model()
    ranker.long_input = "split_8192"
    ranker._fits = lambda q, p: True
    with pytest.raises(RuntimeError):
        ranker._score_long_part("q", "", "long")
    assert r._LONG_WINDOW_CONTEXT.get() is None and ranker.model.max_length == 4096


def test_real_tokenizer_boundaries_and_batches(monkeypatch, tiny):
    monkeypatch.setitem(r.CrossEncoderReranker.MODELS, "bge", (str(tiny), "local"))
    ranker = r.CrossEncoderReranker(
        "bge", device="cpu", long_input="split_8192", batch_size=4
    )
    q = "x"
    calls = []

    def fake_predict(pairs, batch_size=1):
        # Exercise actual guarded preprocessing, but avoid expensive 8192-token attention.
        batch = ranker.model.preprocess(
            pairs, processing_kwargs=r._untruncated_processing()
        )
        lengths = batch["attention_mask"].sum(dim=1).tolist()
        calls.append((pairs, batch_size, lengths))
        return [0.5] * len(pairs)

    monkeypatch.setattr(ranker.model, "predict", fake_predict)

    def unit(size):
        text = " ".join(["x"] * (size - 6))
        assert (
            len(ranker.model.tokenizer(q, "x\n" + text, truncation=False)["input_ids"])
            == size
        )
        return {
            "unit_id": "u",
            "title": "x",
            "text": text,
            "document": {"doc_id": "d"},
            "parts": [
                {
                    "part_no": 1,
                    "text": text,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "locator": {"char_start": 0, "char_end": len(text)},
                }
            ],
        }

    for size in (4096, 4097, 8192, 8193):
        calls.clear()
        u = unit(size)
        out = ranker.rank(q, [u])
        lengths = [n for _, _, ns in calls for n in ns]
        if size <= 8192:
            assert lengths == [size]
        else:
            assert len(lengths) > 1 and all(x <= 4096 for x in lengths)
            w = out[0][0]["rerank_window"]
            assert "".join(u["text"][a:b] for a, b, _ in w["scored_spans"]) == u["text"]
        assert ranker.model.max_length == 4096 and r._LONG_WINDOW_CONTEXT.get() is None
    calls.clear()
    ranker.rank(q, [unit(100)])
    new = list(calls)
    calls.clear()
    ranker.long_input = "split"
    ranker.rank(q, [unit(100)])
    assert calls == new


def test_incompatible_capacity_fails(monkeypatch, tiny):
    import sentence_transformers

    original = sentence_transformers.CrossEncoder

    class Incompatible(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self[0].auto_model.config.max_position_embeddings = 4096

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Incompatible)
    monkeypatch.setitem(r.CrossEncoderReranker.MODELS, "bge", (str(tiny), "local"))
    with pytest.raises(ValueError, match="compatible"):
        r.CrossEncoderReranker("bge", device="cpu", long_input="split_8192")


def test_service_evidence_and_tie(monkeypatch, tiny):
    import asyncio, json

    monkeypatch.setitem(r.CrossEncoderReranker.MODELS, "bge", (str(tiny), "local"))
    ranker = r.CrossEncoderReranker(
        "bge", device="cpu", long_input="split_8192", batch_size=4
    )
    q = "x"
    a = " ".join(["x"] * 4091)
    b = "x x"
    seen = []

    def fake(pairs, batch_size=1):
        ranker.model.preprocess(pairs, processing_kwargs=r._untruncated_processing())
        seen.append((pairs, batch_size))
        return [0.5] * len(pairs)

    monkeypatch.setattr(ranker.model, "predict", fake)
    parts = []
    pos = 0
    for i, text in enumerate((a, b), 1):
        parts.append(
            {
                "part_no": i,
                "text": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "locator": {
                    "kind": "text",
                    "pages": [],
                    "char_start": pos,
                    "char_end": pos + len(text),
                },
            }
        )
        pos += len(text)
    full = a + b
    doc = {
        "unit_id": "u",
        "title": "x",
        "text": full,
        "sha256": hashlib.sha256(full.encode()).hexdigest(),
        "document": {
            "doc_id": "d",
            "source_url": "https://example.test",
            "source_kind": "benchmark",
        },
        "parts": parts,
        "structure": [],
        "locator": {"char_start": 0, "char_end": len(full)},
    }

    class Catalog:
        def active(self):
            return "rev"

        def revision(self, rev):
            return {
                "manifest": json.dumps(
                    {
                        "profile": "general",
                        "parser_version": "generic-documents-v2",
                        "verification_state": "provided",
                    }
                )
            }

        def temporal_version(self, rev):
            return None

        def lexical_scored(self, rev, query):
            return [("u", -1)]

        def exact_documents(self, rev, query, doc_id=None):
            return []

        def get(self, rev, uid):
            return doc

    service = r.RetrievalService(
        Catalog(), reranker=ranker, profile="general", general_rerank_top_n=24
    )
    response = asyncio.run(
        service.retrieve(q, revision="rev", return_retrieval_trace=True)
    )
    e = response["evidence"][0]
    assert (
        len(seen) == 2
        and seen[0][0] == [(q, "x\n" + a)]
        and seen[1][0] == [(q, "x\n" + b)]
    )
    assert (
        e["part_no"] == 1
        and e["text"] == a
        and e["source_locator"] == parts[0]["locator"]
        and e["content_sha256"] == parts[0]["sha256"]
        and e["rerank_window"]["part_sha256"] == parts[0]["sha256"]
        and (e["rerank_window"]["char_start"], e["rerank_window"]["char_end"])
        == (0, len(a))
    )
