# Nested F2 quality-feature reuse

The DCLM-topup quality report reuses 45 feature/progress files from the earlier
expansion output and stores one new file in its own output. The frozen quality
scanner assumes every reused file lives in the immediate prior output, so it
fails the F2 append before processing new data. This wrapper keeps the frozen
scanner and its dependency/quality checks, then verifies each reused file at
the exact path in the SHA-bound prior report. It restricts those paths to the
two frozen prefix directories and rereads each feature SHA and completion
record. Any missing, changed, or unexpected feature fails closed. New F2 rows
remain unadmitted until family and packing gates pass.
