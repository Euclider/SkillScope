"""Retain actual training evidence; full-vocabulary readout is streamed later."""
from phase2.capture import mark_batch as native_mark


def mark_batch(batch, *, root, update, full_vocab=True, copies=2):
    native_mark(batch,root=root,update=update,full_vocab=False,copies=copies)
    batch.meta_info['phase2_capture']['readout_storage_contract']='same-HF-four-condition-recompute-v1'
