from evidence_rag.catalog import Catalog


def test_pruning_is_scoped_to_frequent_single_english_terms(tmp_path):
    cat=Catalog(tmp_path/'db')
    try:
        cat.db.executemany('INSERT INTO unit_fts VALUES(?,?,?)',[('r',str(i),'the rare' if i<2 else 'the paper') for i in range(5)])
        assert cat.lexical_scored('r','THE rare',limit=2,filter_common=True)==cat.lexical_scored('r','rare',limit=2)
        # A short list of selective matches or no selective terms reuses the exact original query.
        for q in ('the rare','the unknownword','the',''):
            assert cat.lexical_scored('r',q,limit=40,filter_common=True)==cat.lexical_scored('r',q,limit=40)
        cat.db.execute('BEGIN')
        cat.db.executemany('INSERT INTO unit_fts VALUES(?,?,?)',[('another',str(i),'unrelated') for i in range(5)])
        # At exactly half of the index the token remains. The caller's transaction survives.
        assert cat.lexical_scored('r','the rare',limit=2,filter_common=True)==cat.lexical_scored('r','the rare',limit=2)
        assert cat.db.in_transaction
        cat.db.rollback()
        assert cat.lexical_scored('r','the rare',limit=2,filter_common=True)==cat.lexical_scored('r','rare',limit=2)
    finally:cat.db.close()
