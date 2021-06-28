def batch_decorator(f):
    def body(bs):
        assert isinstance(bs, int) and bs > 0
        batch = []
        for out in f():
            batch.append(out)
            if len(batch) == bs:
                yield batch
                batch = []
    return body
