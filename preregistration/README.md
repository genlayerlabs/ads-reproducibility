# Preregistration manifests

`pilot.yaml` is deliberately marked mutable and cannot support a confirmatory
claim. `confirmatory.template.yaml` documents every field that must be frozen.
Before any confirmatory provider call, copy the template, replace every
placeholder, run `ads-study validate`, record `ads-study hash-manifest`, commit
the manifest, and create an immutable signed tag.

Changing a frozen manifest requires a new run identifier and new model calls.
