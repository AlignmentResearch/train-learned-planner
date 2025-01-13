"""
Merge multiple Wandb runs into a single run.

Adapted from:
Collin McCarthy
https://github.com/collinmccarthy/wandb-scripts

Examples (original):
- Merge runs with run ids
    ```
    python wandb_merge.py \
    --wandb-entity=farai \
    --wandb-project=lp-cleanba \
    --merge_run_save_dir=./wandb \
    --verify_overlap_metric iter \
    --run-ids l7egn2nb frnuw7jp \
    --iter_range 51200,500736000,5120,3000000000 
    ```

- Merge runs with names `my_run` and `my_other_run`
    ```
    python wandb_merge.py \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project=$WANDB_PROJECT \
    --merge_run_save_dir=$SAVE_DIR \
    --verify_overlap_metric iter \
    --run-names my_run my_other_run
    ```
"""

import argparse
import os
import pprint
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import wandb
from tqdm import tqdm
from wandb.old.summary import SummarySubDict as OldSummarySubDict
from wandb.sdk.wandb_run import Run as SdkRun
from wandb.sdk.wandb_summary import SummarySubDict


def parse_args() -> Namespace:
    default_merged_run_tag = "merged-run"

    parser = argparse.ArgumentParser(description="Clean up Wandb runs remotely")
    parser.add_argument(
        "--wandb-entity",
        "--wandb_entity",
        default=os.environ.get("WANDB_ENTITY", None),
        help="Wandb entity name (e.g. username or team name)",
    )
    parser.add_argument(
        "--wandb-project",
        "--wandb_project",
        required=True,
        help="Wandb project name",
    )
    parser.add_argument(
        "--verify-overlap-metric",
        "--verify_overlap_metric",
        type=str,
        required=True,
        help="Logged key to use for verifying histories do not overlap, e.g. 'iter' or 'epoch'",
    )
    parser.add_argument(
        "--run-names",
        "--run_names",
        nargs="+",
        default=list(),
        help="Merge wandb runs that were split.",
    )
    parser.add_argument(
        "--run-ids",
        "--run_ids",
        nargs="+",
        default=list(),
        help="Merge wandb runs that were split.",
    )
    parser.add_argument(
        "--tag-partial-runs",
        "--tag_partial_runs",
        type=str,
        default="partial-run",
        help="Tag for original (partial) runs that have been merged, so we can filter them out",
    )
    parser.add_argument(
        "--tag-merged-run",
        "--tag_merged_run",
        type=str,
        default=default_merged_run_tag,
        help="Tag for merged (new) runs that are merged from partial runs.",
    )
    parser.add_argument(
        "--skip-run-ids",
        "--skip_run_ids",
        nargs="+",
        default=list(),
        help="Run ids to skip",
    )
    parser.add_argument(
        "--skip-run-tags",
        "--skip_run_tags",
        nargs="+",
        type=str,
        default=(default_merged_run_tag,),
        help="Tag for merged (new) runs that are merged from partial runs.",
    )
    parser.add_argument(
        "--set-run-name",
        "--set_run_name",
        type=str,
        help=("Run name for merged run. If not specified, the name of first run being merged" " will be used."),
    )
    parser.add_argument(
        "--merge-run-save-dir",
        "--merge_run_save_dir",
        type=str,
        help=(
            "Top-level run directory to save results for merged run. If None (default), will try"
            " to find `work_dir` in config (set for mmdetection models)"
        ),
    )
    parser.add_argument(
        "--iter_range",
        type=str,
        default="",
        help="Offset to start merging runs from",
    )
    parser.add_argument(
        "--mark-last-run-state",
        action="store_true",
        help="Mark the state using the last run's state",
    )
    args = parser.parse_args()

    if args.wandb_entity is None or len(args.wandb_entity) == 0:
        raise RuntimeError("Found empty string for --wandb-entity")

    if args.wandb_project is None or len(args.wandb_project) == 0:
        raise RuntimeError("Found empty string for --wandb-project")

    if args.merge_run_save_dir is not None and len(args.merge_run_save_dir) == 0:
        raise RuntimeError("Found empty string for --merge-run-save-dir")

    if len(args.run_names) == 0 and len(args.run_ids) == 0:
        raise RuntimeError("Expected either --run-names or --run-ids to query runs to merge")

    return args


def _get_run_tags(run) -> list[str]:
    if isinstance(run.tags, str):
        return [run.tags]
    elif isinstance(run.tags, Iterable):
        return list(run.tags)
    else:
        return []


def _backup_wandb_resume(save_dir: Union[Path, str]) -> None:
    wandb_resume = Path(save_dir, "wandb", "wandb-resume.json")
    wandb_resume_backup = Path(save_dir, "wandb", "wandb-resume_backup.json")
    if wandb_resume.exists():
        print(f"Backing up current resume file: {wandb_resume} -> {wandb_resume_backup}")
        shutil.copyfile(wandb_resume, wandb_resume_backup)
        wandb_resume.unlink()


def _remove_prefix_dir(filepath: Path, prefix_dir: Path) -> Path:
    filepath_no_prefix = str(filepath).replace(str(prefix_dir), "")
    if filepath_no_prefix.startswith(os.sep):
        filepath_no_prefix = filepath_no_prefix[len(os.sep) :]
    return Path(filepath_no_prefix)


def _find_wandb_dir(save_dir: Path) -> Optional[Path]:
    # Different ways of finding the wandb subdir:
    #   1. Search for wandb-resume.json
    #       - Should exist for our runs, but doesn't always generalize
    #   2. Search for 'wandb' folder name, top-down, and when we find the folder stop
    #       - Need to stop recursing because there maybe sub-folders within this also called 'wandb'
    #       - e.g. <save_dir>/subfolder/wandb/run-20240506_153614-s4k9nk96/files/code/tools/wandb'
    # We will choose #2

    # Get all potential wandb dirs with save_dir part removed
    wandb_folders: list[str] = [
        str(_remove_prefix_dir(filepath=f, prefix_dir=save_dir))
        for f in Path(save_dir).rglob("*")
        if f.is_dir() and f.name == "wandb"
    ]

    # Remove any wandb dirs that are subdirs of any others
    remove_indices: list[int] = []
    for idx, curr_path in enumerate(wandb_folders):
        if any(
            [
                # Need os.sep in startswith to prevent 'wandb_vis/wandb'.startswith('wandb') == True
                # Instead use 'wandb_vis/wandb'.startswith('wandb/') == False which is what we want
                curr_path.startswith(f"{other_path}{os.sep}")
                for other_path in wandb_folders
                if other_path != curr_path
            ]
        ):
            # Current path is a subdir of other path
            remove_indices.append(idx)

    wandb_dirs = [Path(save_dir, folder) for idx, folder in enumerate(wandb_folders) if idx not in remove_indices]

    if len(wandb_dirs) == 0:
        pass  # Keep save_dir
    elif len(wandb_dirs) == 1:
        save_dir = wandb_dirs[0].parent
    else:
        wandb_dirs_str = "\n  " + "\n  ".join([str(d) for d in wandb_dirs])
        raise RuntimeError(
            f"Found {len(wandb_dirs)} 'wandb' subdirs in work_dir={str(save_dir)}, expected 1." f"\nSubdirs: {wandb_dirs_str}"
        )

    return save_dir


def merge_runs(args: Namespace) -> None:
    api = wandb.Api()
    print(f"Querying Wandb entity: {args.wandb_entity}, project: {args.wandb_project}")
    runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}")

    if len(args.run_names) > 0:
        print(f"Searching {len(runs)} runs for run names matching {args.run_names}")
    if len(args.run_ids) > 0:
        print(f"Searching {len(runs)} runs for run ids matching {args.run_ids}")

    # Get matching runs
    matching_runs: list = []
    if len(args.run_ids) > 0 and len(args.run_names) == 0:
        for run_id in args.run_ids:
            run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{run_id}")
            matching_runs.append(run)
    else:
        for run in tqdm(runs, desc="Run"):
            if run.lastHistoryStep == -1:
                continue  # Empty run, possibly from previous merging attempt

            if len(args.skip_run_ids) > 0 and run.id in args.skip_run_ids:
                continue

            if args.skip_run_tags is not None:
                matching_tags = [tag for tag in args.skip_run_tags if tag in _get_run_tags(run)]
                if len(matching_tags) > 0:
                    continue

            if run.name in args.run_names or run.id in args.run_ids:
                matching_runs.append(run)

    if len(matching_runs) < 2:
        print(f"Found {len(matching_runs)} matching runs. Need >= 2 runs to merge, skipping merge.")
        return

    print(
        f"Found {len(matching_runs)} matching runs, verifying non-overlaping using metric" f" '{args.verify_overlap_metric}'"
    )

    # Get the start/end iterations for each run
    iter_ranges: list[dict[str, Union[int, float]]] = []
    if args.iter_range:
        iter_stops = list(map(int, args.iter_range.split(",")))
        for idx, run in enumerate(matching_runs):
            iter_ranges.append({"min": iter_stops[2 * idx], "max": iter_stops[2 * idx + 1]})
        sorted_tuples = list(zip(matching_runs, iter_ranges))
    else:
        for run in matching_runs:
            history = run.scan_history(keys=[args.verify_overlap_metric])
            iters = [row[args.verify_overlap_metric] for row in history]
            iter_ranges.append({"min": min(iters), "max": max(iters)})

        # Sort runs by min iteration
        sorted_tuples: list[tuple[Any, dict[str, Union[int, float]]]] = sorted(
            zip(matching_runs, iter_ranges), key=lambda pair: pair[1]["min"]
        )

        # Verify ranges don't overlap
        prev_run, prev_iter_range = sorted_tuples[0]
        for run, iter_range in sorted_tuples[1:]:
            if not iter_range["min"] >= prev_iter_range["max"]:
                raise RuntimeError(
                    f"Found overlapping runs {prev_run} and {run}, cannot merge."
                    f" Iter ranges using metric={args.verify_overlap_metric}: {prev_iter_range},"
                    f" {iter_range}."
                )
            prev_run, prev_iter_range = (run, iter_range)

    # Verify we want to continue
    matching_runs_dicts = [
        {
            "run": run.name,
            f"min('{args.verify_overlap_metric}')": iter_range["min"],
            f"max('{args.verify_overlap_metric}')": iter_range["max"],
        }
        for run, iter_range in sorted_tuples
    ]

    print(f"Merging {len(matching_runs)} runs into a new run:" f"\n{pprint.pformat(matching_runs_dicts, sort_dicts=False)}")
    response = input("Continue with merge? (y/N): ")
    if response.lower() == "y":
        print("Merging runs (response=y)")
    else:
        print("Skipping merge (response=n)")
        return

    # Create a new run
    matching_runs = [run for run, _iter_range in sorted_tuples]
    base_run = matching_runs[0]

    # Try to use 'work_dir' which is set for mmdetection/mmengine models
    # We may have re-named the folder, so use the current run name if it doesn't match
    work_dir = base_run.config.get("work_dir", None)
    if args.merge_run_save_dir is not None:
        save_dir = Path(args.merge_run_save_dir)
    elif work_dir is not None:
        if Path(work_dir).name != base_run.name:
            save_dir = Path(work_dir).with_name(base_run.name)
        else:
            save_dir = work_dir
    else:
        raise RuntimeError(
            "Missing both --merge-run-save-dir and 'work_dir' in config."
            " Must use --merge-run-save-dir if 'work_dir' doesn't exist in config."
        )

    # We may save in a sub-dir of work_dir, e.g. 'wandb_vis', search for 'wandb' dir
    if save_dir is not None and save_dir.exists():
        wandb_dir = _find_wandb_dir(save_dir=save_dir)
        if wandb_dir is not None:
            save_dir = wandb_dir

    save_dir = str(save_dir)

    new_tags = [args.tag_merged_run] if args.tag_merged_run is not None else []
    new_name = args.set_run_name if args.set_run_name is not None else base_run.name

    # Backup any existing wandb-resume.json and then use resume="auto"; this will create a
    #   a new run, and a new wandb-resume.json file so we can resume training with merged run
    _backup_wandb_resume(save_dir=save_dir)

    init_kwargs = dict(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=new_name,
        tags=new_tags,
        allow_val_change=True,
        resume="auto",
        dir=save_dir,
    )

    # Verify save dir
    save_dir = init_kwargs["dir"]
    if save_dir is None:
        raise RuntimeError(
            "Failed to find save dir for creating run. Run config is missing"
            " 'wandb_init_kwargs' and 'work_dir'. Must specify --into-save-dir"
            " and re-run."
        )
    elif not Path(save_dir).exists():
        raise RuntimeError(
            f"Attempting to re-use base work_dir={save_dir}, but directory does not exist."
            f" Pass in --into-save-dir to force a directory which will be created."
        )

    print(f"Creating new run for merging into with init() kwargs:" f"\n{pprint.pformat(init_kwargs, sort_dicts=False)}")
    combined_run: SdkRun = wandb.init(**init_kwargs)
    combined_run_id = combined_run.id
    print(f"Created new run with run.id={combined_run_id}")

    # Merge matching_runs into combined_run
    last_run_offset = sorted_tuples[0][1]["min"]
    for idx, (partial_run, iter_range) in enumerate(sorted_tuples):
        history = partial_run.scan_history()
        artifacts = partial_run.logged_artifacts()
        files = partial_run.files()
        header = f"[Run {idx + 1}/{len(matching_runs)}]"

        # Update config
        wandb.config.update(partial_run.config, allow_val_change=True)  # pyright: ignore[reportCallIssue]

        # Update tags for new run, adding new tags and --tag-combined-run
        partial_run_tags = _get_run_tags(partial_run)
        combined_run_tags = _get_run_tags(combined_run)
        add_combined_run_tags = [tag for tag in partial_run_tags if tag not in [combined_run_tags, args.tag_partial_runs]]
        combined_run.tags = combined_run_tags + add_combined_run_tags

        # Update tags for old run, adding --tag-partial-runs
        if args.tag_partial_runs not in partial_run_tags:
            # Don't see 'tags' attribute in apis.public.runs.Run but docs say this is valid
            # See https://docs.wandb.ai/guides/app/features/tags
            # fmt: off
            partial_run.tags = partial_run_tags + [args.tag_partial_runs]  # pyright: ignore[reportAttributeAccessIssue]
            # fmt: on
            partial_run.update()

        # Overwrite previous run summary
        for key, val in tqdm(
            partial_run.summary.items(),
            total=len(list(partial_run.summary.keys())),
            desc=f"{header}: Summary",
        ):
            if isinstance(val, (SummarySubDict, OldSummarySubDict)):
                # Need to convert to normal dict for logging, or get TypeError not JSON serializable
                val = dict(val)
            combined_run.summary[key] = val

        history = partial_run.history(samples=100000, pandas=False)

        for step, logged_dict in tqdm(enumerate(history), desc=f"{header}: History"):
            if logged_dict["_step"] >= iter_range["max"]:
                last_run_offset = logged_dict["_step"]
                break
            wandb_step = int(logged_dict.pop("_step") + (last_run_offset - iter_range["min"]))
            wandb.log(logged_dict, step=wandb_step)
            # time.sleep(0.001)

        # Log artifacts
        artifacts_dir = Path(save_dir, "merged_artifacts", partial_run.id)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for artifact in tqdm(artifacts, desc=f"{header}: Artifacts"):
            # Try to create a new version of same artifact; if wandb doesn't let us ignore it
            # Following https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version
            saved_artifact = wandb.use_artifact(artifact.name)
            try:
                draft_artifact = saved_artifact.new_draft()
                print(f"Logging artifact {draft_artifact.name}")
                wandb.log_artifact(draft_artifact)
            except ValueError as e:  # If type is `wandb-`, e.g. `wandb-history`, it's reserved
                if "reserved for internal use" in str(e):
                    pass

        # Log files
        files_dir = Path(save_dir, "merged_files", partial_run.id)
        files_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm(files, desc=f"{header}: Files"):
            result = file.download(root=files_dir, replace=True)
            filepath = result.name
            # Using policy="end" so any repeated files are only uploaded once (hopefully)
            wandb.save(filepath, base_path=Path(filepath).parent, policy="end")

    # Add list of merged runs to run summary
    assert wandb.run is not None
    wandb.run.summary["merged_run_ids"] = [run.id for run in matching_runs]

    # Mark run as finished with exit code according to the state of the last run
    exit_code = 0
    if args.mark_last_run_state:
        last_state = matching_runs[-1].state
        exit_code = 0 if last_state == "finished" else 1
        if last_state == "preempted":
            combined_run.mark_preempting()
    wandb.finish(exit_code=exit_code)


if __name__ == "__main__":
    args = parse_args()
    merge_runs(args)
