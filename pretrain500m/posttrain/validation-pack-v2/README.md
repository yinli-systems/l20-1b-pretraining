# Document-masked development pack v2

This packer applies the bound reserved-quality split patch and turns the final
family-disjoint development streams into five domain shards. Rejected reserved
families are removed, and their deterministic replacements are read from the
original train streams. Each document starts a new block sequence; masks remove
padding and all cross-document targets. The output supports the preregistered
equal-domain loss objective and cannot grant training admission.
