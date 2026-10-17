# Binary genomic-alteration prediction from pathology features

This folder contains twelve training scripts for gated-attention multiple-instance learning (ABMIL). Each script consumes precomputed tile embeddings, with 1536 features per tile, and trains one binary event classifier.

## Training scripts

| Target label column | Subtype-specific script | Pooled script |
|---|---|---|
| `AKT1` | `train_akt1_subtype.py` | `train_akt1_pooled.py` |
| `ERBB2_amp` | `train_erbb2_amp_subtype.py` | `train_erbb2_amp_pooled.py` |
| `ESR1` | `train_esr1_subtype.py` | `train_esr1_pooled.py` |
| `BRCA` | `train_brca_subtype.py` | `train_brca_pooled.py` |
| `PIK3CA` | `train_pik3ca_subtype.py` | `train_pik3ca_pooled.py` |
| `PTEN_mutdel` | `train_pten_mutdel_subtype.py` | `train_pten_mutdel_pooled.py` |

Subtype-specific scripts train separate models for `HR+HER2-`, `HER2+`, and `TNBC`. Pooled scripts combine subtypes. Event definitions must be established in the input data; the scripts map `MT` to 1 and other retained values to 0, and exclude the literal string `Not available`.

## Private configuration

Install the dependencies in `requirements.txt`. These are dependency names, not a validated or pinned reproduction environment.

Copy `config.example.json` to `config.local.json` and set the locations of your private files. Alternatively set the `ABMIL_CONFIG` environment variable to a private configuration file.

Run from this directory, for example:

```sh
python train_akt1_pooled.py
```

Required inputs:

- Split CSV: `IMAGE_ID_clean`, `Dataset_Split` (`Training` or `Validation`), target label column, and `Subtype` for subtype-specific runs. `PathoID` and `Cohort` are not required by the public scripts.
- Cohort XLSX tables: `image_dir`, plus an identifier column selected in order from `GIGA_feats_name`, `IMAGE_ID`, or `Sample_ID`. Input identifier fields are used locally to find feature files; changing field names is not anonymization.
- Feature files: `<feature_dir>/<image_dir>/tiles_feature/<IMAGE_ID_clean>.pt`, containing a dictionary with a `tile_embeds` tensor of shape `(number_of_tiles, 1536)`.


## Outputs

Outputs are written under `./results/<target>/<mode>/`, with subtype subdirectories for subtype runs. The scripts save `best_model.pt` and `training_log.txt`. The log contains aggregate loss, AUC, accuracy, F1 and thresholds. Console output includes aggregate sample counts and training metrics.
