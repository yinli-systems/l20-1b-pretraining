"""Content-refine completed DCLM pool without touching the active web writer."""
from continuation_common import RUN, lock
from continuation_general_quality import reject_reason
from curate_continuation import curate


if __name__ == '__main__':
    handle = lock(RUN/'refine-dclm.lock')
    source = RUN/'curated-v2/dclm'
    curate('dclm',input_path=source/'manifest.json',directory=RUN/'refined-v3/dclm',
           input_inventory=source/'documents.sqlite',quality_filter=reject_reason)
    handle.close()
