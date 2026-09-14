"""Persistent CPT origin and strict resume identity, independent of CLI mode."""
from copy import deepcopy
from mixture import identity, sha256


SCHEMA = 'p529m-cpt-resume-v2'


def resolve_origin(initial_sha256=None, checkpoint=None):
    if checkpoint is not None:
        if initial_sha256 is not None:
            raise ValueError('resume and initialization are mutually exclusive')
        if checkpoint.get('resume_schema') != SCHEMA:
            raise ValueError('legacy checkpoint needs an explicit migration; origin cannot be inferred')
        origin = deepcopy(checkpoint['origin'])
        if origin.get('kind') != 'continued_pretraining':
            raise ValueError('unsupported training origin')
        sha256(origin['initial_checkpoint_sha256'])
        return origin
    return {'kind': 'continued_pretraining', 'initial_checkpoint_sha256': sha256(initial_sha256)}


def run_fingerprint(fields, origin):
    # Code, data, protocol, topology, schedules and origin belong in fields.
    # The current launch's --resume flag and transport paths do not.
    return identity({'schema': SCHEMA, 'origin': origin, 'fields': fields})


def verify_resume(checkpoint, fingerprint, reader, blocks_per_step, total_steps, world_size):
    resolve_origin(checkpoint=checkpoint)
    if checkpoint.get('run_fingerprint') != fingerprint:
        raise ValueError('resume run fingerprint mismatch')
    step = checkpoint['step']
    if type(step) is not int or not 0 <= step <= total_steps:
        raise ValueError('invalid checkpoint step')
    reader.verify_state(checkpoint['reader_state'], step * blocks_per_step)
    for name in ('python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'):
        if len(checkpoint[name]) != world_size:
            raise ValueError(f'wrong RNG topology: {name}')
    return step
