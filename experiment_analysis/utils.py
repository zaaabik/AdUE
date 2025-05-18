from typing import Iterable

import pandas as pd
import wandb
import torch
import os
from tqdm.auto import tqdm

from src.utils.utils import get_adapter_path_from_ckpt_path, find_best_path_from_ckpt

METRIC_COLUMNS = ['lr-AdamW', 'test/acc', 'test/ece', 'test/loss', 'test/matthews_corr', 'test/roc-auc', 'train/acc',
                  'train/ece', 'train/loss', 'train/matthews_corr', 'train/roc-auc', 'trainer/global_step', 'val/acc',
                  'val/ece', 'val/loss', 'val/matthews_corr', 'val/optimize_metric', 'val/roc-auc']


def get_wandb_table(username: str, project_name: str) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{username}/{project_name}")

    summary_list, config_list, name_list = [], [], []
    for run in tqdm(runs):
        summary_list.append(run.summary._json_dict)
        name_list.append(run.name)
        config_list.append({
            **{
                k: v for k, v in run.config.items()
            }
        })

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })
    return runs_df


def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for key, value in d.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        row_dict = {
            'name': row['name'],
            **flatten_dict(row['summary']),
            **flatten_dict(row['config']),
        }
        rows.append(row_dict)

    return pd.DataFrame(rows)


def get_target_modules(x):
    try:
        return ' '.join(x)
    except:
        return ''


def fill_adapter_df(df: pd.DataFrame, hparams_fields: Iterable) -> pd.DataFrame:
    df = df.copy()
    if 'adapter/init_r' in df.columns:
        df['adapter/r'] = df['adapter/r'].fillna(df['adapter/init_r'])

    for hparams_field in hparams_fields:
        if hparams_field not in df.columns:
            df[hparams_field] = ''
        else:
            df[hparams_field] = df[hparams_field].fillna('')
    return df


def transform_adapter_df(df: pd.DataFrame) -> pd.DataFrame:
    df[
        'adapter/target_modules'
    ] = df[
        'adapter/target_modules'
    ].apply(get_target_modules)

    df[
        'model/model/freeze_parts'
    ] = df[
        'model/model/freeze_parts'
    ].apply(get_target_modules)

    df['model/model/freeze_parts'] = df['model/model/freeze_parts'].apply(lambda x: 'Full model' if x == '' else x)
    df.loc[df['model/model/freeze_parts'] == 'roberta', 'adapter/_target_'] = 'cls_layer_only'
    df['adapter/_target_'] = df['adapter/_target_'].fillna(df['model/model/freeze_parts'])

    df[METRIC_COLUMNS] = df[METRIC_COLUMNS].astype('float32')
    return df


def get_run_args(df: pd.DataFrame) -> pd.DataFrame:
    output_dirs = df['paths/output_dir']
    from omegaconf import OmegaConf
    overrides = []

    for output_dir in output_dirs:
        path = os.path.join(output_dir, '.hydra', 'overrides.yaml')
        if os.path.isfile(path):
            overrides_config = OmegaConf.load(
                os.path.join(output_dir, '.hydra', 'overrides.yaml')
            )
            overrides.append({
                 f'overrider/{k.split("=")[0]}': k.split("=")[1] for k in overrides_config
            })
        else:
            overrides.append({})

    df = pd.concat((
        df, pd.DataFrame(overrides)
    ), axis='columns')
    df = df[df['base_out_dir'].notna()]

    return df


def get_best_path_from_ckpt_path(ckpt_path: str) -> str:
    last_ckpt = torch.load(
        os.path.join(ckpt_path, 'last.ckpt'),
        map_location='cpu')
    best_model_path = find_best_path_from_ckpt(last_ckpt)
    best_adapter_path = get_adapter_path_from_ckpt_path(best_model_path)
    return best_adapter_path


def get_best_runs(
        df: pd.DataFrame,
        base_fields: list, hyper_params_fields: list
) -> pd.DataFrame:
    group_df = df.groupby(list(base_fields) + list(hyper_params_fields))
    group_df_with_concatenated_paths = group_df.first()
    group_df_with_concatenated_paths['callbacks/model_checkpoint/dirpath'] = pd.DataFrame(
        group_df[
            'callbacks/model_checkpoint/dirpath'
        ].apply(list)
    )

    group_df_with_concatenated_paths['overrider/experiment'] = group_df['overrider/experiment'].first()
    return group_df_with_concatenated_paths


def get_name_from_multi_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.index.map(lambda x: "_".join(f"{name}-{value}" for name, value in zip(df.index.names, x)))


def build_predict_command(experiment: str, dirpaths: list[str], output_folder: str,
                          additional_params: list[str] = None) -> str:
    command = f'nohup python src/predict.py -m {experiment} pretrained_model_path=\"{",".join(map(lambda x: x.split("uncertainty_estimation/")[1], dirpaths))}\" output_folder=\"{output_folder}\" best_model_from_callback=true'

    if additional_params:
        command += f' {" ".join(additional_params)}'
    return command


def build_eval_command(output_folder: str) -> str:
    return f'nohup python src/eval.py glob_predict_folder=\"{output_folder}/**/*/prediction.pkl\" +output_folder=\"{output_folder}\"'


def create_command_from_df(
        df: pd.DataFrame, num_files: int = 1,
        num_devices: int = 2, predict_params: list[str] = None
):
    df = df.copy()
    df['output_folder'] = get_name_from_multi_index(df)
    commands = []
    eval_commands = []
    env_vars = []
    file_nums = []
    res_path = []
    for i, (index, row) in enumerate(df.iterrows()):
        experiment = 'experiment=' + row['overrider/experiment']
        dirpaths = row['callbacks/model_checkpoint/dirpath']
        dataset_name = row['data/dataset/_target_'].split('.')[-1]
        output_folder = os.path.join(
            'prediction', row['base_architecture'], dataset_name, row['output_folder'].replace('/', '_')
        )

        commands.append(
            build_predict_command(
                experiment, dirpaths, output_folder, predict_params
            )
        )

        eval_commands.append(
            build_eval_command(
                output_folder
            )
        )
        file_nums.append(i % num_files)
        res_path.append(
            os.path.join(output_folder, 'results_v2.pkl')
        )

        env_vars.append(f'CUDA_VISIBLE_DEVICES={int(i % num_devices)}')
    df['predict_command'] = commands
    df['eval_command'] = eval_commands
    df['env_var'] = env_vars
    df['file_num'] = file_nums
    df['res_path'] = res_path
    return df


def create_predict_run_file_from_df(df: pd.DataFrame, output_path: str):
    df = df.copy().reset_index()
    file_nums = df['file_num'].value_counts()
    all_files = []
    os.makedirs(os.path.join(output_path, 'predict'), exist_ok=True)
    for file_num in file_nums.index:
        current_part = df[df['file_num'] == file_num]
        commands = list((current_part['env_var'] + ' ' + current_part['predict_command']).values + '\n')
        output_file_name = f'predict_{file_num}.sh'
        with open(os.path.join(output_path, 'predict', f'predict_{file_num}.sh'), 'w') as f:
            f.writelines(commands)

        all_files.append(output_file_name)

    with open(os.path.join(output_path, f'predict.sh'), 'w') as f:
        f.writelines([
            f'bash {output_path}/predict/' + file + ' &\n' for file in all_files
        ])


def flatten_df(df):
    row_oriented_df = pd.DataFrame(
        {col + '_' + metric_name: [df.loc[metric_name, col]] for metric_name in df.index for col in df.columns}
    )
    return row_oriented_df