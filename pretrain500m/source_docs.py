"""Read the already hash-verified FineWeb-Edu 10BT Parquet snapshot read-only."""
from pathlib import Path
import os
import random
import pyarrow.parquet as pq

REVISION='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
REVISIONS={'web':{'repo':'HuggingFaceFW/fineweb-edu','config':'sample-10BT','revision':REVISION}}
CODE_LANGUAGE_MIN_SCORES={}
ROOT=Path('/ssd/scxi253/corpus/fineweb-edu-10BT/sample/10BT')


def iter_source(source,seed=42,state_db=None,delete_completed_raw=False):
    if source!='web':
        raise ValueError('This isolated adapter only admits the verified English web source')
    if delete_completed_raw:
        raise ValueError('Shared corpus deletion is forbidden')
    paths=sorted(ROOT.glob('*.parquet'))
    if len(paths)!=14:
        raise RuntimeError(f'expected 14 verified source files, found {len(paths)}')
    file_index=os.getenv('P500M_SOURCE_FILE_INDEX')
    if file_index is None:
        random.Random(seed).shuffle(paths)
    else:
        index=int(file_index)
        if not 0<=index<len(paths):raise ValueError(f'invalid P500M_SOURCE_FILE_INDEX={index}')
        paths=[paths[index]]
    for path in paths:
        parquet=pq.ParquetFile(path,memory_map=True)
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                text=row.get('text')
                if not isinstance(text,str) or not text.strip():
                    continue
                if row.get('language')!='en' or float(row.get('language_score') or 0)<0.90:
                    continue
                if int(row.get('int_score') or 0)<3:
                    continue
                yield {'text':text,'source_path':str(path),'source_file_index':file_index}
