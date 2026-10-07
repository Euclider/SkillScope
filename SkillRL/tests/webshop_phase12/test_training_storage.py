import json
import torch


def test_training_archive_compression_is_byte_and_tensor_lossless(tmp_path):
    from phase1.archive import sha256_file
    from webshop_phase12.training_storage import compress_training_archive,load_training_archive
    path=tmp_path/'training_batch.pt'
    tensors={'input_ids':torch.tensor([[0,0,12,13,99,99]]),
             'attention_mask':torch.tensor([[0,0,1,1,1,0]]),
             'advantages':torch.tensor([[0.25,-0.125]]),'old_log_probs':torch.tensor([[-1.3,-2.5]])}
    torch.save({'schema_version':'phase2.exact_training_batch.v1','tensors':tensors},path)
    old_hash=sha256_file(path)
    (tmp_path/'manifest.json').write_text(json.dumps({'batch_sha256':old_hash}))
    compress_training_archive(tmp_path)
    assert not path.exists() and (tmp_path/'training_batch.pt.gz').is_file()
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['uncompressed_batch_sha256']==old_hash
    restored=load_training_archive(tmp_path)
    assert all(torch.equal(restored['tensors'][k],v) for k,v in tensors.items())


def test_legacy_uncompressed_training_archive_remains_loadable(tmp_path):
    from webshop_phase12.training_storage import load_training_archive
    torch.save({'schema_version':'phase2.exact_training_batch.v1','value':torch.tensor([7])},tmp_path/'training_batch.pt')
    (tmp_path/'manifest.json').write_text('{}')
    assert load_training_archive(tmp_path)['value'].item()==7
