# Data layout

This release does not contain `data/core4d/` or SMPL-X model files.
Download them separately after checking their licenses. See
`../../THIRD_PARTY_LICENSES.md`.



Data files are excluded from Git.

```text
data/
├── core4d/
│   ├── metadata/
│   │   └── action_labels.json
│   ├── object_models/
│   └── motions/
├── robots/
│   └── unitree_g1/
└── smplx/
    └── models/
```

## Licenses

- CORE4D-derived files must retain the upstream attribution and license.
- SMPL-X model files require separate registration and must not be redistributed.
- Robot assets retain their upstream licenses.

The acquisition scripts intentionally download only metadata and selected
`join + box + obs0` sequences where possible.
