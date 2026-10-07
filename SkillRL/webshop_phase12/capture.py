"""Retain actual training evidence; full-vocabulary readout is streamed later."""
from phase2.capture import mark_batch as native_mark,archive_batch as native_archive


def mark_batch(batch, *, root, update, full_vocab=True, copies=2):
    native_mark(batch,root=root,update=update,full_vocab=False,copies=copies)
    batch.meta_info['phase2_capture']['readout_storage_contract']='same-HF-four-condition-recompute-v1'


def archive_batch(batch,*,root,update,config):
    native_archive(batch,root=root,update=update,config=config)
    if config.phase2.get('compress_training_batch',False):
        from pathlib import Path
        from webshop_phase12.training_storage import compress_training_archive
        compress_training_archive(Path(root)/'batches'/f'u{update:04d}')
