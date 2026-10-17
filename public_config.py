
import json
import os
from pathlib import Path


def load_data_config(target_gene):
    config_path = Path(os.environ.get('ABMIL_CONFIG', Path(__file__).with_name('config.local.json')))
    if not config_path.is_file():
        raise ValueError('Create config.local.json from config.example.json, or set ABMIL_CONFIG.')
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError('The private input configuration could not be read as UTF-8 JSON.') from None
    if not isinstance(config, dict):
        raise ValueError('The private input configuration must be a JSON object.')
    base_dir = config_path.resolve().parent

    def resolve(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Input locations must be non-empty strings.')
        path = Path(value)
        return str(path if path.is_absolute() else base_dir / path)

    cohorts_by_target = config.get('cohort_files')
    if not isinstance(cohorts_by_target, dict):
        raise ValueError('cohort_files must map target genes to cohort tables.')
    cohorts = cohorts_by_target.get(target_gene)
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError('No cohort tables are configured for the selected target gene.')
    # Configured cohort keys can be private aliases. Do not propagate them into outputs.
    return {
        'global_split_file': resolve(config.get('global_split_file')),
        'h5_base_dir': resolve(config.get('feature_dir')),
        'cohort_files': {f'cohort_{i}': resolve(value) for i, value in enumerate(cohorts.values(), 1)},
    }
