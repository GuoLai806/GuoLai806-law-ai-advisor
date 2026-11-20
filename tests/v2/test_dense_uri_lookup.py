from evidence_rag.catalog import Catalog


def insert_unit(db, revision, uid, uri):
    db.execute(
        'INSERT INTO units(revision,unit_id,label,title,text,locator,sha256,uri,structure) '
        'VALUES(?,?,?,?,?,?,?,?,?)', (revision,uid,uid,'','','{}','hash',uri,'[]')
    )


def test_dense_lookup_is_scoped_order_independent_and_handles_legacy_duplicates():
    catalog = Catalog(':memory:')
    try:
        for revision, uid, uri in [
            ('r1','b','same'),('r1','a','same'),('r1','c','only-r1'),
            ('r2','other','same'),('r2','outside','only-r2'),
        ]:
            insert_unit(catalog.db,revision,uid,uri)
        hits = ['only-r2','same','only-r1','same',"x' OR 1=1 --"]
        old = [catalog.db.execute('SELECT unit_id FROM units WHERE revision=? AND uri=?',('r1',uri)).fetchone()
               for uri in hits]
        lookup = catalog.unit_ids_for_uris('r1',iter(hits))
        assert [lookup.get(uri) for uri in hits] == [r[0] if r else None for r in old]
        assert catalog.unit_ids_for_uris('r1',[]) == {}
    finally:
        catalog.db.close()


def test_dense_batch_reads_only_once_per_group_not_per_hit():
    catalog = Catalog(':memory:')
    try:
        uris = [f'u{i}' for i in range(257)]
        for uri in uris:
            insert_unit(catalog.db,'r',uri,uri)
        queries = []
        catalog.db.set_trace_callback(queries.append)
        assert len(catalog.unit_ids_for_uris('r',uris + uris)) == len(uris)
        assert len(queries) == 3
    finally:
        catalog.db.close()
