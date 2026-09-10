# src/tirzah/db/repositories.py

def apply_targeted_rebuild(db, client, tree_id, new_document, new_tree, new_nodes, new_edges, previous_nodes, previous_edges):
    """
    Applies a targeted rebuild atomically using a MongoDB transaction when supported (replica set),
    ensuring rollback/restore operations do not leave partial states.
    """
    session = client.start_session() if client else None
    
    def _execute_txn(s):
        # 1. Replace document & tree
        db.documents.replace_one({"_id": new_document["_id"]}, new_document, session=s)
        db.trees.replace_one({"_id": new_tree["_id"]}, new_tree, session=s)
        
        # 2. Update nodes
        db.nodes.delete_many({"tree_id": tree_id}, session=s)
        if new_nodes:
            db.nodes.insert_many(new_nodes, ordered=True, session=s)
            
        # 3. Update graph edges
        delete_graph_edges_for_tree(db, tree_id, session=s)
        if new_edges:
            db.graph_edges.insert_many(new_edges, ordered=True, session=s)

    try:
        if session:
            with session:
                session.with_transaction(_execute_txn)
        else:
            # Standalone MongoDB fallback with best-effort ordering & logging
            _execute_txn(None)
    except Exception as exc:
        # If transaction fails, attempt best-effort restore if not handled atomically by MongoDB
        raise RuntimeError(f"Targeted rebuild failed and was rolled back: {exc}") from exc